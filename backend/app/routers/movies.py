"""Danh mục phim: tìm kiếm & lọc, chi tiết, phim tương tự, phim cho onboarding."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, literal, select
from sqlalchemy.orm import Session

from app.auth import current_user, optional_user
from app.db import get_db
from app.models import Movie, NotInterested, Rating, Tag, User, UserGenre, Watched, WatchlistItem
from app.recsys.service import rec_service
from app.schemas import MovieOut

router = APIRouter(prefix="/movies", tags=["movies"])

MIN_COUNT_FOR_AVG = 20  # sắp theo điểm TB chỉ có nghĩa khi đủ lượt chấm


def has_genre(g: str):
    return (literal("|") + Movie.genres + literal("|")).like(f"%|{g}|%")


@router.get("/genres")
def genres(db: Session = Depends(get_db)) -> list[str]:
    names: set[str] = set()
    for (gs,) in db.execute(select(Movie.genres).distinct()):
        names.update(g for g in (gs or "").split("|") if g and g != "(no genres listed)")
    return sorted(names)


@router.get("")
def search(
    q: str = "",
    genre: list[str] = Query(default=[]),
    year_min: int | None = None,
    year_max: int | None = None,
    min_avg: float = 0,
    sort: Literal["rel", "avg", "year", "pop"] = "pop",
    page: int = Query(1, ge=1),
    page_size: int = Query(40, ge=1, le=100),
    user: User | None = Depends(optional_user),
    db: Session = Depends(get_db),
):
    """GET /movies?q=&genre=&year_min=&year_max=&min_avg=&sort= — gợi ý theo ràng buộc."""
    cond = []
    if q.strip():
        cond.append(Movie.title.ilike(f"%{q.strip()}%"))
    cond += [has_genre(g) for g in genre]
    if year_min is not None:
        cond.append(Movie.year >= year_min)
    if year_max is not None:
        cond.append(Movie.year <= year_max)
    if min_avg > 0:
        cond.append((Movie.rating_avg >= min_avg) | (Movie.rating_count == 0))
    if user is not None:
        cond.append(Movie.id.not_in(select(NotInterested.movie_id).where(NotInterested.user_id == user.id)))

    total = db.scalar(select(func.count()).select_from(Movie).where(*cond)) or 0
    offset = (page - 1) * page_size

    if sort == "rel" and user is not None:
        ids = list(db.scalars(select(Movie.id).where(*cond)))
        ids = rec_service.relevance_order(db, user.id, ids)[offset: offset + page_size]
        by_id = {m.id: m for m in db.scalars(select(Movie).where(Movie.id.in_(ids)))}
        movies = [by_id[i] for i in ids if i in by_id]
    else:
        order = {
            "avg": [case((Movie.rating_count >= MIN_COUNT_FOR_AVG, 0), else_=1), Movie.rating_avg.desc()],
            "year": [Movie.year.desc().nulls_last(), Movie.rating_count.desc()],
            "pop": [Movie.recent_count.desc(), Movie.rating_count.desc()],
            "rel": [Movie.recent_count.desc(), Movie.rating_count.desc()],
        }[sort]
        movies = list(db.scalars(select(Movie).where(*cond).order_by(*order, Movie.id).offset(offset).limit(page_size)))

    my = {}
    if user is not None and movies:
        my = dict(db.execute(select(Rating.movie_id, Rating.rating)
                             .where(Rating.user_id == user.id, Rating.movie_id.in_([m.id for m in movies]))).all())
    return {
        "total": total, "page": page, "page_size": page_size,
        "items": [{"movie": MovieOut.of(m), "my_rating": my.get(m.id)} for m in movies],
    }


@router.get("/onboarding")
def onboarding_pool(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=60),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Phim phổ biến để chấm ở onboarding, ưu tiên thể loại người dùng đã chọn."""
    gs = list(db.scalars(select(UserGenre.genre).where(UserGenre.user_id == user.id)))
    match = sum((case((has_genre(g), 1), else_=0) for g in gs), literal(0)) if gs else literal(0)
    stmt = (select(Movie).where(Movie.rating_count >= 5)
            .order_by((match > 0).desc(), Movie.recent_count.desc(), Movie.rating_count.desc(), Movie.id)
            .offset(offset).limit(limit))
    movies = list(db.scalars(stmt))
    my = dict(db.execute(select(Rating.movie_id, Rating.rating).where(Rating.user_id == user.id)).all())
    return {"items": [{"movie": MovieOut.of(m), "my_rating": my.get(m.id)} for m in movies]}


@router.get("/{movie_id}")
def detail(movie_id: int, user: User | None = Depends(optional_user), db: Session = Depends(get_db)):
    m = db.get(Movie, movie_id)
    if m is None:
        raise HTTPException(404, "Không tìm thấy phim")
    out: dict = {"movie": MovieOut.of(m), "me": None}
    if user is not None:
        uid = user.id
        r = db.get(Rating, (uid, movie_id))
        out["me"] = {
            "rating": r.rating if r else None,
            "rated_at": r.timestamp if r else None,
            "tags": [{"id": t.id, "tag": t.tag} for t in db.scalars(
                select(Tag).where(Tag.user_id == uid, Tag.movie_id == movie_id).order_by(Tag.timestamp))],
            "watchlist": db.get(WatchlistItem, (uid, movie_id)) is not None,
            "not_interested": db.get(NotInterested, (uid, movie_id)) is not None,
            "watched": db.get(Watched, (uid, movie_id)) is not None,
        }
    # tag phổ biến của cộng đồng làm gợi ý cho ô gắn tag
    out["popular_tags"] = list(db.scalars(
        select(Tag.tag).where(Tag.movie_id == movie_id).group_by(Tag.tag)
        .order_by(func.count().desc()).limit(6)))
    return out


@router.get("/{movie_id}/similar")
def similar(movie_id: int, k: int = Query(12, ge=1, le=50),
            user: User | None = Depends(optional_user), db: Session = Depends(get_db)):
    if db.get(Movie, movie_id) is None:
        raise HTTPException(404, "Không tìm thấy phim")
    return rec_service.similar(db, user.id if user else None, movie_id, k)
