"""Schema CSDL.

ratings / tags / movies / links giữ đúng trường của MovieLens (PRD mục 4) và có thêm
cột `source` ("movielens" | "app") để nối thẳng dữ liệu app vào tập huấn luyện.
Timestamp luôn là giây UTC, cùng đơn vị với MovieLens.
"""

import time

from sqlalchemy import JSON, BigInteger, Boolean, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def now_ts() -> int:
    return int(time.time())


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(100))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    onboarded: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[int] = mapped_column(BigInteger, default=now_ts)


class UserGenre(Base):
    __tablename__ = "user_genres"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    genre: Mapped[str] = mapped_column(String(50), primary_key=True)


class Movie(Base):
    __tablename__ = "movies"

    # movies.csv
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(String(500), index=True)
    genres: Mapped[str] = mapped_column(String(300), default="")  # "Action|Sci-Fi" như MovieLens
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    # links.csv
    imdb_id: Mapped[str | None] = mapped_column(String(20))
    tmdb_id: Mapped[int | None] = mapped_column(Integer)
    # làm giàu từ TMDB
    overview: Mapped[str | None] = mapped_column(Text)
    poster_path: Mapped[str | None] = mapped_column(String(200))
    backdrop_path: Mapped[str | None] = mapped_column(String(200))
    trailer_key: Mapped[str | None] = mapped_column(String(50))
    runtime: Mapped[int | None] = mapped_column(Integer)
    directors: Mapped[str | None] = mapped_column(String(500))
    cast: Mapped[str | None] = mapped_column(String(1000))
    # thống kê, tính lại bởi scripts/load_movielens.py và POST /lab/retrain
    rating_avg: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    rating_count: Mapped[int] = mapped_column(Integer, default=0, index=True)
    recent_count: Mapped[int] = mapped_column(Integer, default=0, index=True)
    source: Mapped[str] = mapped_column(String(20), default="movielens")
    added_at: Mapped[int] = mapped_column(BigInteger, default=now_ts)

    @property
    def genre_list(self) -> list[str]:
        if not self.genres or self.genres == "(no genres listed)":
            return []
        return self.genres.split("|")


class Rating(Base):
    """ratings.csv: userId, movieId, rating (0.5–5, bước 0.5), timestamp."""

    __tablename__ = "ratings"

    user_id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    rating: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[int] = mapped_column(BigInteger, default=now_ts, index=True)
    source: Mapped[str] = mapped_column(String(20), default="app")

    __table_args__ = (Index("ix_ratings_movie", "movie_id"), Index("ix_ratings_user_ts", "user_id", "timestamp"))


class Tag(Base):
    """tags.csv: userId, movieId, tag, timestamp."""

    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), index=True)
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"), index=True)
    tag: Mapped[str] = mapped_column(String(255))
    timestamp: Mapped[int] = mapped_column(BigInteger, default=now_ts)
    source: Mapped[str] = mapped_column(String(20), default="app")


class _UserMovieState:
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    timestamp: Mapped[int] = mapped_column(BigInteger, default=now_ts)


class WatchlistItem(_UserMovieState, Base):
    __tablename__ = "watchlist"


class NotInterested(_UserMovieState, Base):
    __tablename__ = "not_interested"


class Watched(_UserMovieState, Base):
    __tablename__ = "watched"


class Event(Base):
    """Sự kiện tiềm ẩn (PRD mục 4): impression, detail_view, watch, mark_watched,
    watchlist_add/remove, not_interested, search, ..."""

    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), index=True)
    session_id: Mapped[str | None] = mapped_column(String(64))
    type: Mapped[str] = mapped_column(String(40), index=True)
    timestamp: Mapped[int] = mapped_column(BigInteger, index=True)
    movie_id: Mapped[int | None] = mapped_column(Integer, index=True)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    model_version: Mapped[str | None] = mapped_column(String(50))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(50), unique=True)
    recommender: Mapped[str] = mapped_column(String(100))  # tên class Recommender
    kind: Mapped[str] = mapped_column(String(20))  # full | finetune | refresh
    # queued | running | live | pass | fail | archived
    status: Mapped[str] = mapped_column(String(20), index=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)  # vd. {"ndcg@10": 0.15}
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    data_info: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[int] = mapped_column(BigInteger, default=now_ts)
    deployed_at: Mapped[int | None] = mapped_column(BigInteger)
