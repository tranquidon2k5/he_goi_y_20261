"""Recommender tuần tự (họ SASRec) — phục vụ checkpoint huấn luyện bởi `ml/train.py`.

Thư mục artifact (config.artifact_dir) gồm:
    model.pt            checkpoint (hp, spec, state_dict) — xem ml/train.py: cmd_final
    item_ids.npy        movieId của các chỉ số phim 1..N trong mô hình
    features.npy        đặc trưng nội dung (N + 1, F); feature_meta.json: từ điển thể loại / tag

Người dùng = chuỗi tương tác dương gần nhất (rating >= 4, đã xem, xem sau, xem trailer đủ lâu) —
không cần embedding riêng, nên người dùng mới của app được cá nhân hoá ngay mà không học lại (PRD mục 5).
Phim mới thêm vào kho (source = app) chưa có trong từ điển: nếu mô hình có nhánh nội dung thì biểu diễn
bằng phép chiếu đặc trưng nội dung (thể loại + thập niên), nên vẫn được xếp hạng (PRD mục 3, cold-start phía phim).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.recsys.base import Recommender, Scored, UserContext
from app.recsys.catalog import Catalog
from app.recsys.heuristic import HeuristicRecommender

POSITIVE_KINDS = {"mark_watched", "watchlist_add", "watch"}


class SeqRecommender(Recommender):
    name = "seqrec"

    def load(self, catalog: Catalog, artifact_dir: str | None = None) -> None:
        from ml.data import featurize
        from ml.models import SeqRec

        if not artifact_dir:
            raise ValueError("seqrec cần config.artifact_dir")
        d = Path(artifact_dir)
        if not d.is_absolute():
            d = Path(__file__).resolve().parents[2] / d  # tương đối so với backend/
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        ckpt = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        hp, spec = ckpt["hp"], ckpt["spec"]
        feats = torch.as_tensor(np.load(d / "features.npy")) if spec.get("content") else None
        self.model = SeqRec(ckpt["n_items"], d=hp["d"], blocks=hp["blocks"], heads=hp["heads"], drop=0.0,
                            maxlen=hp["maxlen"], ligr=spec.get("ligr", False), features=feats, fuse=hp.get("fuse", "add"))
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.eval()
        self.maxlen = hp["maxlen"]
        self.ckpt_name = ckpt["name"]
        item_ids = np.load(d / "item_ids.npy")
        self.vocab = {int(m): i + 1 for i, m in enumerate(item_ids)}  # movieId -> chỉ số trong mô hình

        self.cat = catalog
        self.fallback = HeuristicRecommender(self.config)
        self.fallback.load(catalog)

        # Biểu diễn cho từng hàng của danh mục app; hàng không biểu diễn được = NaN (không gợi ý)
        with torch.no_grad():
            R = self.model.item_repr()
            rep = torch.full((len(catalog), hp["d"]), float("nan"))
            rows = [r for r, m in enumerate(catalog.ids) if int(m) in self.vocab]
            rep[rows] = R[[self.vocab[int(catalog.ids[r])] for r in rows]]
            self.n_content_only = 0
            if feats is not None:
                meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
                new_rows = [r for r in np.flatnonzero(catalog.is_new) if int(catalog.ids[r]) not in self.vocab]
                if new_rows:
                    f = torch.as_tensor(np.stack([featurize("|".join(catalog.genres_of(int(catalog.ids[r]))),
                                                            int(catalog.years[r]), meta) for r in new_rows]))
                    c = self.model.proj(f)
                    if self.model.fuse == "gate":
                        g = torch.sigmoid(self.model.gate(torch.cat([torch.zeros_like(c), c], -1)))
                        c = (1 - g) * c
                    rep[new_rows] = c
                    self.n_content_only = len(new_rows)
        self.known = ~torch.isnan(rep[:, 0])
        self.rep = torch.nan_to_num(rep)
        self.rep_norm = F.normalize(self.rep, dim=-1)

    # ------------------------------------------------------------------ helpers
    def _sequence(self, ctx: UserContext) -> list[int]:
        seq = []
        for it in ctx.history:  # đã sắp theo thời gian
            positive = (it.rating >= 4.0) if it.rating is not None else it.kind in POSITIVE_KINDS
            if positive and it.movie_id in self.vocab:
                seq.append(self.vocab[it.movie_id])
        return seq[-self.maxlen:]

    def _top(self, scores: torch.Tensor, k: int, exclude: set[int]) -> list[Scored]:
        scores = scores.masked_fill(~self.known, float("-inf"))
        for mid in exclude:
            r = self.cat.row(mid)
            if r is not None:
                scores[r] = float("-inf")
        k_eff = min(k, int(torch.isfinite(scores).sum()))
        if k_eff <= 0:
            return []
        v, i = scores.topk(k_eff)
        return [Scored(int(self.cat.ids[j]), float(s)) for s, j in zip(v.tolist(), i.tolist())]

    # ------------------------------------------------------------------ API
    @torch.no_grad()
    def user_vector(self, ctx: UserContext) -> torch.Tensor | None:
        seq = self._sequence(ctx)
        if not seq:
            return None
        x = torch.zeros(1, self.maxlen, dtype=torch.long)
        x[0, self.maxlen - len(seq):] = torch.tensor(seq)
        return self.model.user_vec(x)[0]

    def recommend(self, ctx: UserContext, k: int) -> list[Scored]:
        u = self.user_vector(ctx)
        if u is None:  # chưa có tương tác dương nào mô hình biết → bản tạm
            return self.fallback.recommend(ctx, k)
        return self._top(self.rep @ u, k, ctx.exclude)

    def similar(self, movie_id: int, k: int, exclude: set[int]) -> list[Scored]:
        r = self.cat.row(movie_id)
        if r is None or not bool(self.known[r]):
            return self.fallback.similar(movie_id, k, exclude)
        return self._top(self.rep_norm @ self.rep_norm[r], k, exclude | {movie_id})

    def explain(self, ctx: UserContext, movie_id: int) -> int | None:
        """Phim dương trong lịch sử có embedding gần phim được gợi ý nhất (PRD mục 3)."""
        r = self.cat.row(movie_id)
        if r is None or not bool(self.known[r]):
            return None
        best, best_s = None, -2.0
        for it in ctx.history:
            positive = (it.rating >= 4.0) if it.rating is not None else it.kind in POSITIVE_KINDS
            h = self.cat.row(it.movie_id)
            if not positive or h is None or not bool(self.known[h]):
                continue
            s = float(self.rep_norm[h] @ self.rep_norm[r])
            if s > best_s:
                best, best_s = it.movie_id, s
        return best
