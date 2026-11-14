"""Xuất dữ liệu ra đúng định dạng file của MovieLens để nối vào tập huấn luyện (PRD mục 4).

    uv run python -m scripts.export_training --out ../data/export            # chỉ dữ liệu app
    uv run python -m scripts.export_training --out ../data/export --all      # cả MovieLens + app

Sinh: ratings.csv, tags.csv, movies.csv, links.csv (cột y như MovieLens, thêm cột `source`)
và events.csv (sự kiện tiềm ẩn, payload dạng JSON) — trainer tự áp quy tắc chuyển đổi ở PRD mục 4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from app.db import engine
from app.models import Event, Movie, Rating, Tag


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--all", action="store_true", help="kèm dữ liệu nguồn movielens")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    def q(stmt):
        return pd.read_sql(stmt, engine)

    src = (lambda col: True) if a.all else (lambda col: col == "app")
    r = q(select(Rating.user_id.label("userId"), Rating.movie_id.label("movieId"), Rating.rating,
                 Rating.timestamp, Rating.source).where(src(Rating.source)).order_by(Rating.user_id, Rating.timestamp))
    r.to_csv(a.out / "ratings.csv", index=False)
    t = q(select(Tag.user_id.label("userId"), Tag.movie_id.label("movieId"), Tag.tag, Tag.timestamp, Tag.source)
          .where(src(Tag.source)).order_by(Tag.user_id, Tag.timestamp))
    t.to_csv(a.out / "tags.csv", index=False)

    m = q(select(Movie.id.label("movieId"), Movie.title, Movie.year, Movie.genres, Movie.imdb_id, Movie.tmdb_id, Movie.source))
    # movies.csv của MovieLens để năm trong ngoặc ở cuối tiêu đề
    m["title"] = m.apply(lambda x: f"{x.title} ({int(x.year)})" if pd.notna(x.year) else x.title, axis=1)
    m[["movieId", "title", "genres", "source"]].to_csv(a.out / "movies.csv", index=False)
    links = m[["movieId", "imdb_id", "tmdb_id"]].rename(columns={"imdb_id": "imdbId", "tmdb_id": "tmdbId"})
    links["imdbId"] = links["imdbId"].str.removeprefix("tt")
    links["tmdbId"] = links["tmdbId"].astype("Int64")
    links.to_csv(a.out / "links.csv", index=False)

    e = q(select(Event.event_id, Event.user_id, Event.session_id, Event.type, Event.timestamp, Event.movie_id,
                 Event.request_id, Event.model_version, Event.payload).order_by(Event.timestamp))
    e["payload"] = e["payload"].map(lambda p: p if isinstance(p, str) else json.dumps(p, ensure_ascii=False))
    e.to_csv(a.out / "events.csv", index=False)
    print(f"ratings {len(r):,} · tags {len(t):,} · movies {len(m):,} · events {len(e):,} → {a.out}")


if __name__ == "__main__":
    main()
