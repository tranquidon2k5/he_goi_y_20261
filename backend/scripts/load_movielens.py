"""Nạp MovieLens (ml-latest-small, ml-1m dạng csv, ml-32m, ...) vào CSDL.

Ví dụ:
    uv run python -m scripts.load_movielens --dir ../data/raw/ml-latest-small
    uv run python -m scripts.load_movielens --dir ../data/raw/ml-32m --since-year 2015
    uv run python -m scripts.load_movielens --dir ../data/raw/ml-32m --no-ratings   # chỉ danh mục + thống kê

Script xoá dữ liệu nguồn "movielens" cũ rồi nạp lại; dữ liệu nguồn "app" được giữ nguyên.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, insert, select, update

from app.config import get_settings
from app.db import SessionLocal, engine, init_db
from app.models import Movie, Rating, Tag

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
CHUNK = 200_000


def parse_title(t: str) -> tuple[str, int | None]:
    m = YEAR_RE.search(t.strip())
    if not m:
        return t.strip(), None
    return t[: m.start()].strip(), int(m.group(1))


def bulk(table, df: pd.DataFrame) -> None:
    if df.empty:
        return
    if engine.dialect.name == "postgresql":
        # COPY nhanh hơn executemany hàng chục lần với ML-32M
        cols = list(df.columns)
        raw = engine.raw_connection()
        try:
            with raw.cursor() as cur, cur.copy(f"COPY {table.name} ({', '.join(cols)}) FROM STDIN WITH CSV") as cp:
                buf = io.StringIO()
                df.to_csv(buf, header=False, index=False)
                cp.write(buf.getvalue())
            raw.commit()
        finally:
            raw.close()
        return
    with engine.begin() as conn:
        recs = df.to_dict("records")
        for i in range(0, len(recs), CHUNK):
            conn.execute(insert(table), recs[i: i + CHUNK])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--since-year", type=int, default=None, help="chỉ giữ rating/tag từ năm này (cắt theo thời gian)")
    ap.add_argument("--no-ratings", action="store_true", help="không nạp từng rating, chỉ thống kê")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252
    d: Path = args.dir
    t0 = time.time()
    init_db()

    movies = pd.read_csv(d / "movies.csv")
    parsed = movies["title"].map(parse_title)
    movies["title"] = parsed.map(lambda x: x[0])
    movies["year"] = parsed.map(lambda x: x[1]).astype("Int64")
    if (d / "links.csv").exists():
        links = pd.read_csv(d / "links.csv", dtype={"imdbId": str})
        movies = movies.merge(links, on="movieId", how="left")
    else:
        movies["imdbId"], movies["tmdbId"] = None, None

    ratings = pd.read_csv(d / "ratings.csv", dtype={"userId": "int64", "movieId": "int32", "rating": "float32", "timestamp": "int64"})
    if args.since_year:
        since = int(pd.Timestamp(f"{args.since_year}-01-01", tz="UTC").timestamp())
        ratings = ratings[ratings["timestamp"] >= since]
    print(f"ratings: {len(ratings):,} dòng · {ratings['userId'].nunique():,} người dùng")
    if ratings["userId"].max() >= get_settings().app_user_id_start:
        raise SystemExit("userId của MovieLens chạm dải userId của app — tăng CINEREC_APP_USER_ID_START")

    window = get_settings().trending_window_seconds
    recent_cut = ratings["timestamp"].max() - window
    g = ratings.groupby("movieId")
    stats = pd.DataFrame({
        "rating_avg": g["rating"].mean(),
        "rating_count": g.size(),
        "recent_count": ratings[ratings["timestamp"] >= recent_cut].groupby("movieId").size(),
    }).fillna({"recent_count": 0})
    movies = movies.merge(stats, left_on="movieId", right_index=True, how="left").fillna(
        {"rating_avg": 0.0, "rating_count": 0, "recent_count": 0})

    now = int(time.time())
    mdf = pd.DataFrame({
        "id": movies["movieId"].astype(int),
        "title": movies["title"],
        "genres": movies["genres"],
        "year": movies["year"],
        "imdb_id": movies["imdbId"].map(lambda x: f"tt{x}" if isinstance(x, str) and x else None),
        "tmdb_id": movies["tmdbId"].astype("Int64"),
        "rating_avg": movies["rating_avg"].astype(float),
        "rating_count": movies["rating_count"].astype(int),
        "recent_count": movies["recent_count"].astype(int),
        "source": "movielens",
        "added_at": now,
    })

    with SessionLocal() as db:
        db.execute(delete(Rating).where(Rating.source == "movielens"))
        db.execute(delete(Tag).where(Tag.source == "movielens"))
        db.commit()
        existing = set(db.scalars(select(Movie.id)))

    # Upsert, KHÔNG xoá phim: xoá sẽ cascade sang rating/danh sách của người dùng app,
    # và giữ nguyên metadata TMDB đã lấy trước đó.
    mdf = mdf.astype(object).where(mdf.notna(), None)
    is_old = mdf["id"].isin(existing)
    bulk(Movie.__table__, mdf[~is_old])
    upd = mdf[is_old].drop(columns=["source", "added_at"]).to_dict("records")
    with SessionLocal() as db:
        for i in range(0, len(upd), CHUNK):
            db.execute(update(Movie), upd[i: i + CHUNK])
        db.commit()
    print(f"movies: {len(mdf):,} ({int((~is_old).sum()):,} mới, {int(is_old.sum()):,} cập nhật)")

    if not args.no_ratings:
        rdf = pd.DataFrame({"user_id": ratings["userId"], "movie_id": ratings["movieId"],
                            "rating": ratings["rating"].astype(float), "timestamp": ratings["timestamp"],
                            "source": "movielens"})
        rdf = rdf[rdf["movie_id"].isin(set(mdf["id"]))]
        for i in range(0, len(rdf), 2_000_000):
            bulk(Rating.__table__, rdf.iloc[i: i + 2_000_000])
            print(f"  ratings đã nạp {min(i + 2_000_000, len(rdf)):,}/{len(rdf):,}")

        if (d / "tags.csv").exists():
            tags = pd.read_csv(d / "tags.csv", dtype={"tag": str}).dropna(subset=["tag"])
            if args.since_year:
                tags = tags[tags["timestamp"] >= since]
            tdf = pd.DataFrame({"user_id": tags["userId"], "movie_id": tags["movieId"], "tag": tags["tag"].str.slice(0, 255),
                                "timestamp": tags["timestamp"], "source": "movielens"})
            tdf = tdf[tdf["movie_id"].isin(set(mdf["id"]))]
            bulk(Tag.__table__, tdf)
            print(f"tags: {len(tdf):,}")

    print(f"xong sau {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
