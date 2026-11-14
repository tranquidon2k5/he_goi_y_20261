"""Tương tác tường minh: rating, tag, xem sau, không quan tâm, đã xem.

Mỗi thao tác ghi đồng thời (1) bảng trạng thái theo schema MovieLens và (2) một sự kiện
vào bảng events, rồi huỷ cache gợi ý của người dùng để lần tải trang kế tiếp có kết quả mới.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.auth import current_user
from app.db import get_db
from app.models import Event, Movie, NotInterested, Rating, Tag, User, Watched, WatchlistItem, now_ts
from app.recsys.service import rec_service
from app.schemas import MovieOut, RatingIn, TagIn

router = APIRouter(tags=["interactions"])


def log_event(db: Session, user_id: int, type_: str, movie_id: int | None = None, **payload) -> None:
    db.add(Event(event_id=uuid.uuid4().hex, user_id=user_id, type=type_, timestamp=now_ts(),
                 movie_id=movie_id, model_version=rec_service.version, payload=payload))


def require_movie(db: Session, movie_id: int) -> Movie:
    m = db.get(Movie, movie_id)
    if m is None:
        raise HTTPException(404, "Không tìm thấy phim")
    return m


def done(db: Session, user: User) -> dict:
    db.commit()
    rec_service.invalidate(user.id)
    return {"ok": True, "model_version": rec_service.version}


# ---------------------------------------------------------------- ratings
@router.post("/ratings")
def rate(body: RatingIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    require_movie(db, body.movie_id)
    r = db.get(Rating, (user.id, body.movie_id))
    if r is None:
        db.add(Rating(user_id=user.id, movie_id=body.movie_id, rating=body.rating, timestamp=now_ts(), source="app"))
    else:
        r.rating, r.timestamp = body.rating, now_ts()
    # chấm sao ngầm định là đã xem
    if db.get(Watched, (user.id, body.movie_id)) is None:
        db.add(Watched(user_id=user.id, movie_id=body.movie_id, timestamp=now_ts()))
    return done(db, user)


@router.delete("/ratings/{movie_id}")
def unrate(movie_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.execute(delete(Rating).where(Rating.user_id == user.id, Rating.movie_id == movie_id))
    log_event(db, user.id, "rating_delete", movie_id)
    return done(db, user)


@router.get("/me/ratings")
def my_ratings(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.execute(select(Rating, Movie).join(Movie, Movie.id == Rating.movie_id)
                      .where(Rating.user_id == user.id).order_by(Rating.timestamp.desc())).all()
    tags: dict[int, list[str]] = {}
    for t in db.scalars(select(Tag).where(Tag.user_id == user.id).order_by(Tag.timestamp)):
        tags.setdefault(t.movie_id, []).append(t.tag)
    return {"items": [{"movie": MovieOut.of(m), "rating": r.rating, "timestamp": r.timestamp,
                       "source": r.source, "tags": tags.get(m.id, [])} for r, m in rows]}


# ---------------------------------------------------------------- tags
@router.post("/movies/{movie_id}/tags")
def add_tag(movie_id: int, body: TagIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    require_movie(db, movie_id)
    tag = body.tag.strip()
    exists = db.scalar(select(Tag).where(Tag.user_id == user.id, Tag.movie_id == movie_id, Tag.tag == tag))
    if exists is None:
        t = Tag(user_id=user.id, movie_id=movie_id, tag=tag, timestamp=now_ts(), source="app")
        db.add(t)
        db.flush()
        tag_id = t.id
    else:
        tag_id = exists.id
    out = done(db, user)
    return out | {"id": tag_id, "tag": tag}


@router.delete("/movies/{movie_id}/tags/{tag_id}")
def remove_tag(movie_id: int, tag_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.execute(delete(Tag).where(Tag.id == tag_id, Tag.user_id == user.id, Tag.movie_id == movie_id))
    return done(db, user)


# ---------------------------------------------------------------- danh sách trạng thái
LISTS = {
    "watchlist": (WatchlistItem, "watchlist_add", "watchlist_remove"),
    "not-interested": (NotInterested, "not_interested", "not_interested_undo"),
    "watched": (Watched, "mark_watched", "unmark_watched"),
}


def _list_model(name: str):
    if name not in LISTS:
        raise HTTPException(404, "Danh sách không tồn tại")
    return LISTS[name]


@router.put("/me/{list_name}/{movie_id}")
def list_add(list_name: str, movie_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    model, ev_add, _ = _list_model(list_name)
    require_movie(db, movie_id)
    if db.get(model, (user.id, movie_id)) is None:
        db.add(model(user_id=user.id, movie_id=movie_id, timestamp=now_ts()))
        log_event(db, user.id, ev_add, movie_id)
    return done(db, user)


@router.delete("/me/{list_name}/{movie_id}")
def list_remove(list_name: str, movie_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    model, _, ev_rm = _list_model(list_name)
    res = db.execute(delete(model).where(model.user_id == user.id, model.movie_id == movie_id))
    if res.rowcount:
        log_event(db, user.id, ev_rm, movie_id)
    return done(db, user)


@router.get("/me/{list_name}")
def list_get(list_name: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    model, _, _ = _list_model(list_name)
    rows = db.execute(select(model, Movie).join(Movie, Movie.id == model.movie_id)
                      .where(model.user_id == user.id).order_by(model.timestamp.desc())).all()
    return {"items": [{"movie": MovieOut.of(m), "timestamp": x.timestamp} for x, m in rows]}
