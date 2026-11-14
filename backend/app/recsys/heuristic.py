"""Recommender TẠM THỜI — chỉ để app chạy được trong lúc nhóm chưa chốt thuật toán.

Không phải baseline của báo cáo. Cách chấm:
- độ phổ biến: trung bình Bayes của rating + số lượt chấm gần đây (log);
- mức hợp thể loại: vector sở thích thể loại tính từ lịch sử (rating lệch quanh 2.75)
  cộng thể loại chọn ở onboarding;
- phim tương tự: Jaccard trên thể loại + một chút độ phổ biến và độ gần về năm.
Thay bằng mô hình thật bằng cách viết một Recommender mới (xem base.py).
"""

from __future__ import annotations

import numpy as np

from app.recsys.base import Recommender, Scored, UserContext
from app.recsys.catalog import Catalog

KIND_WEIGHT = {"mark_watched": 0.5, "watch": 0.5, "watchlist_add": 0.3, "tag": 0.3}


def popularity_scores(cat: Catalog, prior: float = 50.0) -> np.ndarray:
    """Điểm phổ biến trong [0, 1]; dùng cho cả hàng "Thịnh hành"."""
    n = cat.rating_count.astype(np.float64)
    if len(n) == 0:
        return np.zeros(0)
    total = n.sum()
    mean = float((cat.rating_avg * n).sum() / total) if total else 3.5
    bayes = (prior * mean + cat.rating_avg * n) / (prior + n)
    recent = np.log1p(cat.recent_count.astype(np.float64))
    norm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)  # noqa: E731
    return 0.5 * norm(bayes) + 0.5 * norm(recent)


class HeuristicRecommender(Recommender):
    name = "heuristic"

    def load(self, catalog: Catalog, artifact_dir: str | None = None) -> None:
        self.cat = catalog
        self.pop = popularity_scores(catalog)
        self.min_count = int(self.config.get("min_count", 5))
        # phim quá ít lượt chấm chỉ được gợi ý nếu là phim mới thêm vào kho (cold-start phía item)
        self.eligible = (catalog.rating_count >= self.min_count) | catalog.is_new
        gm = catalog.genre_matrix
        self.genre_norm = gm / np.maximum(gm.sum(axis=1, keepdims=True), 1)

    def _affinity(self, ctx: UserContext) -> np.ndarray:
        g = np.zeros(self.cat.genre_matrix.shape[1], dtype=np.float64)
        for it in ctx.history:
            r = self.cat.row(it.movie_id)
            if r is None:
                continue
            w = (it.rating - 2.75) / 2.25 if it.rating is not None else KIND_WEIGHT.get(it.kind, 0.3)
            g += w * self.cat.genre_matrix[r]
        if np.abs(g).max(initial=0) > 0:
            g /= np.abs(g).max()
        for name in ctx.genres:
            if name in self.cat.genre_names:
                g[self.cat.genre_names.index(name)] += 0.5
        return g

    def _top(self, scores: np.ndarray, k: int, exclude: set[int]) -> list[Scored]:
        scores = np.where(self.eligible, scores, -np.inf)
        for mid in exclude:
            r = self.cat.row(mid)
            if r is not None:
                scores[r] = -np.inf
        k_eff = min(k, int(np.isfinite(scores).sum()))
        if k_eff <= 0:
            return []
        top = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top = top[np.argsort(-scores[top])]
        return [Scored(int(self.cat.ids[i]), float(scores[i])) for i in top]

    def recommend(self, ctx: UserContext, k: int) -> list[Scored]:
        aff = self.genre_norm @ self._affinity(ctx)
        return self._top(self.pop + 0.8 * aff, k, ctx.exclude)

    def _sim_vector(self, movie_id: int) -> np.ndarray | None:
        r = self.cat.row(movie_id)
        if r is None:
            return None
        gm = self.cat.genre_matrix
        inter = gm @ gm[r]
        union = gm.sum(axis=1) + gm[r].sum() - inter
        jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        y0 = self.cat.years[r]
        year = np.where((self.cat.years > 0) & (y0 > 0), np.exp(-np.abs(self.cat.years - y0) / 10.0), 0)
        return jac + 0.15 * self.pop + 0.1 * year

    def similar(self, movie_id: int, k: int, exclude: set[int]) -> list[Scored]:
        s = self._sim_vector(movie_id)
        if s is None:
            return []
        return self._top(s, k, exclude | {movie_id})

    def explain(self, ctx: UserContext, movie_id: int) -> int | None:
        r = self.cat.row(movie_id)
        if r is None:
            return None
        g = self.cat.genre_matrix[r]
        best, best_s = None, 0.0
        for it in ctx.history:
            if it.rating is not None and it.rating < 4.0:
                continue
            h = self.cat.row(it.movie_id)
            if h is None:
                continue
            hg = self.cat.genre_matrix[h]
            union = (g + hg > 0).sum()
            s = float((g * hg).sum() / union) if union else 0.0
            s += 1e-9 * it.timestamp  # hoà thì ưu tiên phim mới tương tác hơn
            if s > best_s:
                best, best_s = it.movie_id, s
        return best
