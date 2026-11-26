"""Lấy mô tả, poster, trailer, đạo diễn, diễn viên từ TMDB qua links.csv (tmdbId).

    set CINEREC_TMDB_API_KEY=...   (API key v3 hoặc API Read Access Token v4)
    uv run python -m scripts.enrich_tmdb --limit 2000          # 2000 phim phổ biến nhất chưa có dữ liệu

Ưu tiên mô tả tiếng Việt, thiếu thì lấy tiếng Anh. Dữ liệu TMDB phải ghi nguồn theo điều khoản của TMDB.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx
from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import Movie

API = "https://api.themoviedb.org/3"


def pick_trailer(videos: list[dict]) -> str | None:
    yt = [v for v in videos if v.get("site") == "YouTube"]
    for kind in ("Trailer", "Teaser"):
        for v in sorted(yt, key=lambda v: not v.get("official")):
            if v.get("type") == kind:
                return v["key"]
    return yt[0]["key"] if yt else None


async def fetch(client: httpx.AsyncClient, tmdb_id: int) -> dict | None:
    params = {"append_to_response": "videos,credits", "language": "vi-VN", "include_video_language": "vi,en,null"}
    r = await client.get(f"{API}/movie/{tmdb_id}", params=params)
    if r.status_code == 404:
        return None
    if r.status_code == 429:
        await asyncio.sleep(2)
        return await fetch(client, tmdb_id)
    r.raise_for_status()
    data = r.json()
    if not data.get("overview"):
        en = await client.get(f"{API}/movie/{tmdb_id}", params={"language": "en-US"})
        if en.is_success:
            data["overview"] = en.json().get("overview")
    return data


def apply(m: Movie, d: dict) -> None:
    """Ghi dữ liệu TMDB vào một phim; trường TMDB bỏ trống thì giữ giá trị đang có."""
    m.overview = d.get("overview") or m.overview or ""
    m.poster_path = d.get("poster_path") or m.poster_path
    m.backdrop_path = d.get("backdrop_path") or m.backdrop_path
    m.runtime = d.get("runtime") or m.runtime
    m.trailer_key = pick_trailer((d.get("videos") or {}).get("results", [])) or m.trailer_key
    credits = d.get("credits") or {}
    m.directors = ", ".join(c["name"] for c in credits.get("crew", []) if c.get("job") == "Director")[:500] or m.directors
    m.cast = ", ".join(c["name"] for c in credits.get("cast", [])[:8])[:1000] or m.cast


def client_args() -> tuple[dict, dict]:
    """(headers, params) cho httpx: hỗ trợ cả API key v3 và Read Access Token v4."""
    key = get_settings().tmdb_api_key
    if not key:
        raise RuntimeError("Thiếu CINEREC_TMDB_API_KEY")
    if key.startswith("eyJ"):  # token v4
        return {"Authorization": f"Bearer {key}"}, {}
    return {}, {"api_key": key}


async def run(limit: int, concurrency: int) -> None:
    try:
        headers, params = client_args()
    except RuntimeError as e:
        raise SystemExit(str(e)) from e

    with SessionLocal() as db:
        todo = list(db.execute(select(Movie.id, Movie.tmdb_id)
                               .where(Movie.tmdb_id.is_not(None), Movie.overview.is_(None))
                               .order_by(Movie.rating_count.desc()).limit(limit)))
    print(f"{len(todo)} phim cần làm giàu")
    sem = asyncio.Semaphore(concurrency)
    results: dict[int, dict | None] = {}

    async with httpx.AsyncClient(headers=headers, params=params, timeout=20) as client:
        async def one(mid: int, tid: int):
            async with sem:
                try:
                    results[mid] = await fetch(client, tid)
                except httpx.HTTPError as e:
                    print(f"  lỗi movieId={mid}: {e}")
        await asyncio.gather(*(one(m, t) for m, t in todo))

    with SessionLocal() as db:
        n = 0
        for mid, d in results.items():
            m = db.get(Movie, mid)
            if d is None:
                m.overview = ""  # đánh dấu đã thử để không gọi lại
                continue
            apply(m, d)
            n += 1
        db.commit()
    print(f"đã cập nhật {n} phim")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    asyncio.run(run(a.limit, a.concurrency))


if __name__ == "__main__":
    main()
