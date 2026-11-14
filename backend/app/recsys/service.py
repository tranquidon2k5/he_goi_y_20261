"""Tầng phục vụ gợi ý: ghép các hàng trang chủ, cold-start, lọc, giải thích, cache.

Độc lập với thuật toán — mọi lời gọi mô hình đi qua interface `Recommender`.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Movie, ModelVersion, NotInterested, Rating, Tag, User, UserGenre, Watched, WatchlistItem, now_ts
from app.recsys import registry
from app.recsys.base import Interaction, Recommender, UserContext
from app.recsys.catalog import Catalog
from app.recsys.heuristic import popularity_scores
from app.schemas import MovieOut

ROW_SIZE = 12
CACHE_TTL = 300


@dataclass
class _Live:
    catalog: Catalog
    model: Recommender
    version: str
    pop: np.ndarray


class RecService:
    def __init__(self) -> None:
        self._live: _Live | None = None
        self._lock = threading.Lock()
        self._cache: dict[int, tuple[float, dict]] = {}
        self.latencies_ms: deque[float] = deque(maxlen=2000)

    # ------------------------------------------------------------------ nạp mô hình
    def reload(self, db: Session) -> str:
        """Nạp lại danh mục + phiên bản đang live. Gọi lúc khởi động, sau retrain/rollback."""
        with self._lock:
            mv = db.scalar(select(ModelVersion).where(ModelVersion.status == "live").order_by(ModelVersion.id.desc()))
            if mv is None:
                mv = ModelVersion(
                    version="v0.1.0", recommender=registry.DEFAULT_RECOMMENDER, kind="refresh", status="live",
                    notes="Bản tạm (độ phổ biến + thể loại) — thuật toán chính chưa chốt.", deployed_at=now_ts(),
                )
                db.add(mv)
                db.commit()
            catalog = Catalog.from_db(db)
            model = registry.create(mv.recommender, mv.config)
            model.load(catalog, mv.config.get("artifact_dir"))
            self._live = _Live(catalog, model, mv.version, popularity_scores(catalog))
            self._cache.clear()
            return mv.version

    @property
    def live(self) -> _Live:
        if self._live is None:
            raise RuntimeError("RecService chưa được nạp")
        return self._live

    @property
    def version(self) -> str:
        return self.live.version

    def invalidate(self, user_id: int) -> None:
        self._cache.pop(user_id, None)

    def p95_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        return float(np.percentile(np.array(self.latencies_ms), 95))

    # ------------------------------------------------------------------ ngữ cảnh người dùng
    def build_context(self, db: Session, user_id: int) -> UserContext:
        ratings = db.execute(select(Rating.movie_id, Rating.rating, Rating.timestamp).where(Rating.user_id == user_id)).all()
        rated = {r.movie_id for r in ratings}
        hist: list[Interaction] = [Interaction(r.movie_id, r.timestamp, "rating", r.rating) for r in ratings]
        # PRD mục 4: cặp (user, phim) có rating thì rating quyết định
        for model, kind in ((Watched, "mark_watched"), (WatchlistItem, "watchlist_add")):
            for mid, ts in db.execute(select(model.movie_id, model.timestamp).where(model.user_id == user_id)).all():
                if mid not in rated:
                    hist.append(Interaction(mid, ts, kind))
        seen = {i.movie_id for i in hist}
        for mid, ts in db.execute(select(Tag.movie_id, Tag.timestamp).where(Tag.user_id == user_id)).all():
            if mid not in seen:
                seen.add(mid)
                hist.append(Interaction(mid, ts, "tag"))
        hist.sort(key=lambda i: i.timestamp)

        ni = set(db.scalars(select(NotInterested.movie_id).where(NotInterested.user_id == user_id)))
        watched = set(db.scalars(select(Watched.movie_id).where(Watched.user_id == user_id)))
        genres = set(db.scalars(select(UserGenre.genre).where(UserGenre.user_id == user_id)))
        return UserContext(user_id=user_id, history=hist, genres=genres, exclude=rated | ni | watched)

    # ------------------------------------------------------------------ helpers
    def _trending(self, exclude: set[int], genres: set[str] | None, k: int) -> list[int]:
        cat = self.live.catalog
        score = cat.recent_count.astype(np.float64) + 1e-3 * self.live.pop
        mask = cat.rating_count >= 5
        if genres:
            gi = [cat.genre_names.index(g) for g in genres if g in cat.genre_names]
            if gi:
                mask &= cat.genre_matrix[:, gi].sum(axis=1) > 0
        score = np.where(mask, score, -np.inf)
        order = np.argsort(-score)
        out = []
        for i in order:
            if not np.isfinite(score[i]):
                break
            mid = int(cat.ids[i])
            if mid not in exclude:
                out.append(mid)
                if len(out) >= k:
                    break
        return out

    def _new_movies(self, exclude: set[int], k: int) -> list[int]:
        cat = self.live.catalog
        idx = np.flatnonzero(cat.is_new)
        idx = idx[np.argsort(-cat.added_at[idx])]
        return [int(cat.ids[i]) for i in idx if int(cat.ids[i]) not in exclude][:k]

    @staticmethod
    def _movies(db: Session, ids: list[int]) -> dict[int, MovieOut]:
        if not ids:
            return {}
        return {m.id: MovieOut.of(m) for m in db.scalars(select(Movie).where(Movie.id.in_(set(ids))))}

    # ------------------------------------------------------------------ API
    def home(self, db: Session, user: User) -> dict:
        hit = self._cache.get(user.id)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]
        t0 = time.perf_counter()
        live = self.live
        s = get_settings()
        ctx = self.build_context(db, user.id)
        n = ctx.n_interactions
        cold = n < s.cold_start_threshold
        request_id = uuid.uuid4().hex
        rows: list[dict] = []  # {key,title,subtitle,kind,ids,reasons}
        used: set[int] = set()

        personal: list[int] = []
        reasons: dict[int, int] = {}
        if not cold:
            personal = [x.movie_id for x in live.model.recommend(ctx, 300) if x.movie_id not in ctx.exclude]
            for mid in personal[: ROW_SIZE + 1]:
                why = live.model.explain(ctx, mid)
                if why is not None:
                    reasons[mid] = why

        trending = self._trending(ctx.exclude, ctx.genres if cold else None, ROW_SIZE + 1)
        hero_id = (personal or trending or [None])[0]
        if hero_id is not None:
            used.add(hero_id)

        if not cold:
            fy = [m for m in personal if m not in used][:ROW_SIZE]
            rows.append({"key": "for_you", "kind": "for_you", "title": "Dành cho bạn",
                         "subtitle": "Chọn riêng theo sở thích của bạn", "ids": fy})
            # "Vì bạn đã xem X": 2 phim gần nhất được chấm ≥ 4 (hoặc đánh dấu đã xem)
            anchors = [i.movie_id for i in reversed(ctx.history)
                       if (i.rating is None and i.kind == "mark_watched") or (i.rating or 0) >= s.positive_rating]
            for a in list(dict.fromkeys(anchors))[:2]:
                sims = [x.movie_id for x in live.model.similar(a, ROW_SIZE, ctx.exclude)]
                if sims:
                    rows.append({"key": f"because_{a}", "kind": "because", "anchor": a, "title": None,
                                 "subtitle": "Có thể bạn cũng thích", "ids": sims})

        tr = [m for m in trending if m != hero_id][:ROW_SIZE]
        rows.append({"key": "trending", "kind": "trending",
                     "title": "Thịnh hành — theo thể loại bạn chọn" if cold else "Thịnh hành",
                     "subtitle": "Được chấm nhiều nhất gần đây", "ids": tr})

        for g in sorted(ctx.genres)[:2]:
            gi = live.catalog.genre_names.index(g) if g in live.catalog.genre_names else None
            if gi is None:
                continue
            ids = [m for m in personal if live.catalog.genre_matrix[live.catalog.row(m), gi] > 0][:ROW_SIZE]
            if len(ids) < ROW_SIZE:
                ids += [m for m in self._trending(ctx.exclude, {g}, ROW_SIZE * 2) if m not in ids][: ROW_SIZE - len(ids)]
            rows.append({"key": f"genre_{g}", "kind": "genre", "genre": g, "title": g, "subtitle": "", "ids": ids})

        fresh = self._new_movies(ctx.exclude, ROW_SIZE)
        if fresh:
            rows.append({"key": "new", "kind": "new", "title": "Mới thêm vào kho",
                         "subtitle": "Phim vừa có mặt trên CineRec", "ids": fresh})

        all_ids = [hero_id] if hero_id else []
        for r in rows:
            all_ids += r["ids"] + ([r["anchor"]] if "anchor" in r else [])
        all_ids += list(reasons.values())
        mv = self._movies(db, all_ids)

        def item(mid: int, with_reason: bool) -> dict | None:
            if mid not in mv:
                return None
            out = {"movie": mv[mid]}
            if with_reason and mid in reasons and reasons[mid] in mv:
                out["reason"] = {"movie_id": reasons[mid], "title": mv[reasons[mid]].title}
            return out

        out_rows = []
        for r in rows:
            items = [x for x in (item(m, r["kind"] == "for_you") for m in r["ids"]) if x]
            if r["kind"] == "because":
                anchor = mv.get(r["anchor"])
                if not anchor:
                    continue
                r["title"] = f"Vì bạn đã xem {anchor.title}"
            out_rows.append({k: r[k] for k in ("key", "kind", "title", "subtitle")} | {"items": items})

        hero = None
        if hero_id and hero_id in mv:
            hero = item(hero_id, not cold)
            if "reason" in hero:
                rid = hero["reason"]["movie_id"]
                hero["reason"]["rating"] = next((i.rating for i in ctx.history if i.movie_id == rid), None)
                hero["reason"]["shared_genres"] = sorted(set(mv[rid].genres) & set(mv[hero_id].genres))

        payload = {
            "request_id": request_id, "model_version": live.version, "cold": cold,
            "n_interactions": n, "threshold": s.cold_start_threshold, "hero": hero, "rows": out_rows,
        }
        self.latencies_ms.append((time.perf_counter() - t0) * 1000)
        self._cache[user.id] = (time.time(), payload)
        return payload

    def similar(self, db: Session, user_id: int | None, movie_id: int, k: int = ROW_SIZE) -> dict:
        t0 = time.perf_counter()
        exclude: set[int] = set()
        if user_id is not None:
            exclude = set(db.scalars(select(NotInterested.movie_id).where(NotInterested.user_id == user_id)))
        ids = [x.movie_id for x in self.live.model.similar(movie_id, k, exclude)]
        mv = self._movies(db, ids)
        self.latencies_ms.append((time.perf_counter() - t0) * 1000)
        return {
            "request_id": uuid.uuid4().hex, "model_version": self.version,
            "items": [{"movie": mv[i]} for i in ids if i in mv],
        }

    def relevance_order(self, db: Session, user_id: int, ids: list[int]) -> list[int]:
        """Sắp lại kết quả tìm kiếm theo "Phù hợp với bạn"."""
        ctx = self.build_context(db, user_id)
        ctx.exclude = set()
        ranked = self.live.model.recommend(ctx, len(self.live.catalog))
        pos = {x.movie_id: i for i, x in enumerate(ranked)}
        return sorted(ids, key=lambda m: pos.get(m, len(pos)))


rec_service = RecService()
