"""Quản trị danh mục: thêm, sửa, xoá phim mới (chỉ admin).

Phim do quản trị thêm có `source = "app"` và movieId từ 10.000.001. Sau mỗi thay đổi, dịch vụ gợi ý nạp lại
danh mục: phim mới lên hàng "Mới thêm vào kho" ngay, và với mô hình tuần tự có nhánh nội dung thì được xếp hạng
nhờ embedding nội dung (thể loại + thập niên); embedding ID của nó được học ở lần fine-tune kế tiếp (PRD mục 5).
Chỉ sửa / xoá được phim nguồn "app" — phim MovieLens giữ nguyên để không lệch tập huấn luyện.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import admin_user
from app.config import get_settings
from app.db import SessionLocal, get_db
from app.models import IdSequence, Movie, now_ts
from app.recsys.service import rec_service
from app.schemas import MovieOut
from scripts.enrich_tmdb import apply as apply_tmdb
from scripts.enrich_tmdb import client_args, fetch

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(admin_user)])


def known_genres(db: Session) -> set[str]:
    names: set[str] = set()
    for (gs,) in db.execute(select(Movie.genres).where(Movie.source == "movielens").distinct()):
        names.update(g for g in (gs or "").split("|") if g and g != "(no genres listed)")
    return names


class MovieIn(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    year: int | None = Field(default=None, ge=1880, le=2100)
    genres: list[str] = Field(min_length=1)
    overview: str | None = Field(default=None, max_length=5000)
    tmdb_id: int | None = Field(default=None, ge=1)
    imdb_id: str | None = Field(default=None, max_length=20)
    trailer_key: str | None = Field(default=None, max_length=50)
    poster_path: str | None = Field(default=None, max_length=200)
    runtime: int | None = Field(default=None, ge=1, le=1000)
    directors: str | None = Field(default=None, max_length=500)
    cast: str | None = Field(default=None, max_length=1000)
    fetch_tmdb: bool = False  # lấy mô tả / poster / trailer / đạo diễn / diễn viên từ TMDB theo tmdb_id

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, v: str | None) -> str | None:
        if isinstance(v, str) and not v.strip():
            raise ValueError("Tên phim không được để trống")
        return v.strip() if isinstance(v, str) else v

    @field_validator("overview", "imdb_id", "trailer_key", "poster_path", "directors", "cast")
    @classmethod
    def blank_to_none(cls, v: str | None) -> str | None:
        return (v.strip() or None) if isinstance(v, str) else v


class MoviePatch(MovieIn):
    title: str | None = Field(default=None, min_length=1, max_length=500)
    genres: list[str] | None = Field(default=None, min_length=1)


def _reload_catalog() -> None:
    with SessionLocal() as db:
        rec_service.reload(db)


async def _enrich(m: Movie) -> str | None:
    """Trả thông báo lỗi (nếu có); không chặn việc thêm phim khi TMDB lỗi."""
    if not m.tmdb_id:
        return "Cần tmdb_id để lấy dữ liệu TMDB"
    try:
        headers, params = client_args()
    except RuntimeError as e:
        return str(e)
    try:
        async with httpx.AsyncClient(headers=headers, params=params, timeout=20) as client:
            d = await fetch(client, m.tmdb_id)
    except httpx.HTTPError as e:
        return f"TMDB lỗi: {e}"
    if d is None:
        return f"TMDB không có phim {m.tmdb_id}"
    apply_tmdb(m, d)
    return None


def next_movie_id(db: Session) -> int:
    """movieId mới cho phim do quản trị thêm — không bao giờ dùng lại id của phim đã xoá, vì id cũ có thể còn trong
    từ điển của checkpoint (embedding đã học) và trong bảng events."""
    floor = max(get_settings().app_movie_id_start - 1, db.scalar(select(func.max(Movie.id))) or 0)
    seq = db.get(IdSequence, "movie_id")
    if seq is None:
        seq = IdSequence(name="movie_id", value=floor)
        db.add(seq)
    seq.value = max(seq.value, floor) + 1
    return seq.value


def _check_genres(db: Session, genres: list[str]) -> str:
    bad = [g for g in genres if g not in known_genres(db)]
    if bad:
        raise HTTPException(422, f"Thể loại không hợp lệ: {', '.join(bad)}")
    return "|".join(dict.fromkeys(genres))


@router.get("/movies")
def list_movies(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    base = select(Movie).where(Movie.source == "app")
    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    rows = db.scalars(base.order_by(Movie.added_at.desc(), Movie.id.desc()).offset((page - 1) * page_size).limit(page_size))
    return {"total": total, "page": page, "page_size": page_size, "items": [MovieOut.of(m) for m in rows],
            "tmdb_enabled": bool(get_settings().tmdb_api_key)}


@router.post("/movies", status_code=201)
async def create_movie(body: MovieIn, tasks: BackgroundTasks, db: Session = Depends(get_db)):
    genres = _check_genres(db, body.genres)
    if body.tmdb_id and db.scalar(select(Movie.id).where(Movie.tmdb_id == body.tmdb_id)):
        raise HTTPException(409, f"Đã có phim với tmdb_id {body.tmdb_id}")
    fields = body.model_dump(exclude={"genres", "fetch_tmdb"})
    m = Movie(id=next_movie_id(db), genres=genres, source="app", added_at=now_ts(), rating_avg=0.0, rating_count=0,
              recent_count=0, **fields)
    warning = await _enrich(m) if body.fetch_tmdb else None
    db.add(m)
    db.commit()
    tasks.add_task(_reload_catalog)
    return {"movie": MovieOut.of(m), "warning": warning}


@router.patch("/movies/{movie_id}")
async def update_movie(movie_id: int, body: MoviePatch, tasks: BackgroundTasks, db: Session = Depends(get_db)):
    m = db.get(Movie, movie_id)
    if m is None or m.source != "app":
        raise HTTPException(404, "Không có phim do quản trị thêm với id này")
    data = body.model_dump(exclude_unset=True, exclude={"fetch_tmdb"})
    for k in ("title", "genres"):
        if k in data and data[k] is None:
            raise HTTPException(422, f"Trường {k} không được để trống")
    if "genres" in data:
        data["genres"] = _check_genres(db, data["genres"])
    if data.get("tmdb_id") and db.scalar(select(Movie.id).where(Movie.tmdb_id == data["tmdb_id"], Movie.id != movie_id)):
        raise HTTPException(409, f"Đã có phim với tmdb_id {data['tmdb_id']}")
    for k, v in data.items():
        setattr(m, k, v)
    warning = await _enrich(m) if body.fetch_tmdb else None
    db.commit()
    tasks.add_task(_reload_catalog)
    return {"movie": MovieOut.of(m), "warning": warning}


@router.delete("/movies/{movie_id}")
def delete_movie(movie_id: int, tasks: BackgroundTasks, db: Session = Depends(get_db)):
    m = db.get(Movie, movie_id)
    if m is None or m.source != "app":
        raise HTTPException(404, "Không có phim do quản trị thêm với id này")
    db.delete(m)  # rating / danh sách của phim này bị xoá theo (ON DELETE CASCADE)
    db.commit()
    tasks.add_task(_reload_catalog)
    return {"ok": True}
