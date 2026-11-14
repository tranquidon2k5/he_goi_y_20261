"""Ảnh chụp danh mục phim trong bộ nhớ, dùng chung cho mọi Recommender."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Movie


@dataclass
class Catalog:
    ids: np.ndarray  # (N,) movieId
    titles: list[str]
    years: np.ndarray  # (N,) 0 nếu không rõ
    genre_names: list[str]
    genre_matrix: np.ndarray  # (N, G) multi-hot float32
    rating_avg: np.ndarray
    rating_count: np.ndarray
    recent_count: np.ndarray
    added_at: np.ndarray
    is_new: np.ndarray  # (N,) bool — phim do quản trị thêm (source=app), chưa có trong MovieLens
    index: dict[int, int]  # movieId -> hàng

    def __len__(self) -> int:
        return len(self.ids)

    def row(self, movie_id: int) -> int | None:
        return self.index.get(movie_id)

    def genres_of(self, movie_id: int) -> list[str]:
        r = self.index.get(movie_id)
        if r is None:
            return []
        return [self.genre_names[g] for g in np.flatnonzero(self.genre_matrix[r])]

    @classmethod
    def from_db(cls, db: Session) -> Catalog:
        rows = db.execute(
            select(
                Movie.id, Movie.title, Movie.year, Movie.genres,
                Movie.rating_avg, Movie.rating_count, Movie.recent_count, Movie.added_at, Movie.source,
            ).order_by(Movie.id)
        ).all()
        return cls.from_rows(rows)

    @classmethod
    def from_rows(cls, rows) -> Catalog:
        genre_sets = []
        names: dict[str, int] = {}
        for r in rows:
            gs = [] if not r.genres or r.genres == "(no genres listed)" else r.genres.split("|")
            genre_sets.append(gs)
            for g in gs:
                names.setdefault(g, len(names))
        gm = np.zeros((len(rows), len(names)), dtype=np.float32)
        for i, gs in enumerate(genre_sets):
            for g in gs:
                gm[i, names[g]] = 1.0
        ids = np.array([r.id for r in rows], dtype=np.int64)
        return cls(
            ids=ids,
            titles=[r.title for r in rows],
            years=np.array([r.year or 0 for r in rows], dtype=np.int32),
            genre_names=list(names),
            genre_matrix=gm,
            rating_avg=np.array([r.rating_avg or 0 for r in rows], dtype=np.float32),
            rating_count=np.array([r.rating_count or 0 for r in rows], dtype=np.int64),
            recent_count=np.array([r.recent_count or 0 for r in rows], dtype=np.int64),
            added_at=np.array([r.added_at or 0 for r in rows], dtype=np.int64),
            is_new=np.array([r.source == "app" for r in rows], dtype=bool),
            index={int(m): i for i, m in enumerate(ids)},
        )
