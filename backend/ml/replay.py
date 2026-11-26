"""Trình diễn vòng học (PRD mục 5): phát lại cửa sổ test của MovieLens như một luồng sự kiện.

Cửa sổ test được chia thành K đoạn liên tiếp theo thời gian. Với mỗi đoạn k:
  1. chấm NDCG@10 trên tương tác đầu tiên của từng người dùng trong đoạn k (lịch sử = mọi tương tác trước đoạn),
     cho hai mô hình: "tĩnh" (checkpoint gốc, không học thêm) và "cập nhật" (đã fine-tune qua các đoạn 0..k-1);
  2. fine-tune mô hình "cập nhật" trên dữ liệu của đoạn k (chuỗi của người dùng hoạt động trong đoạn)
     trộn với một mẫu replay người dùng cũ — đúng quy trình mức 2 của app (ml/finetune.py: train_steps).
Cả hai mô hình đều đã dùng lịch sử mới làm đầu vào (mức 1, tức thời); chênh lệch giữa chúng là phần đóng góp
của fine-tune (mức 2).

    python -m ml.replay --data DIR --ckpt results/proposed/model.pt --out results/replay.json --chunks 4
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ml.data import Dataset
from ml.finetune import build_model, ndcg10, train_steps


def run(ds: Dataset, ckpt: dict, chunks: int = 4, epochs: int = 2, lr_scale: float = 0.3, replay_users: int = 3000,
        seed: int = 0, log=print) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    static = build_model(ckpt, ds.features, drop=0.0).to(dev).eval()
    model = copy.deepcopy(build_model(ckpt, ds.features)).to(dev)
    L = ckpt["hp"]["maxlen"]
    bounds = np.quantile(ds.ts[ds.split == 2], np.linspace(0, 1, chunks + 1)).astype(np.int64)
    bounds[-1] += 1
    ubounds = np.searchsorted(ds.user, np.arange(ds.n_users + 1))

    def before(cut: int) -> list[np.ndarray]:
        return [ds.item[a:b][ds.ts[a:b] < cut] for a, b in zip(ubounds[:-1], ubounds[1:])]

    rows = []
    for k in range(chunks):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        hist_all = before(lo)
        m = (ds.ts >= lo) & (ds.ts < hi)
        u, it = ds.user[m], ds.item[m]
        first = np.r_[True, u[1:] != u[:-1]]
        u, it = u[first], it[first]
        keep = np.array([len(hist_all[x]) > 0 for x in u], dtype=bool)
        hist, tgt = [hist_all[x] for x in u[keep]], it[keep]
        nd_static, n = ndcg10(static, hist, tgt, L)
        nd_upd, _ = ndcg10(model, hist, tgt, L)
        row = {"chunk": k, "from": lo, "to": hi, "cases": n, "static_ndcg@10": nd_static, "updated_ndcg@10": nd_upd,
               "finetunes_before": k}
        log(f"đoạn {k}: {n} ca · tĩnh {nd_static:.4f} · cập nhật {nd_upd:.4f}")
        if k < chunks - 1:  # học từ đoạn k để phục vụ đoạn k + 1
            t0 = time.time()
            upto = before(hi)
            active = np.unique(ds.user[m])
            others = np.setdiff1d(np.arange(ds.n_users), active)
            replay = rng.choice(others, size=min(replay_users, len(others)), replace=False)
            train_steps(model, [upto[x] for x in active] + [upto[x] for x in replay], ckpt, epochs, lr_scale, log)
            row["finetune_seconds"] = round(time.time() - t0, 1)
            row["finetune_users"] = int(len(active))
        rows.append(row)
    s = np.array([r["static_ndcg@10"] for r in rows[1:]])
    d = np.array([r["updated_ndcg@10"] for r in rows[1:]])
    w = np.array([r["cases"] for r in rows[1:]])
    return {"chunks": rows, "epochs": epochs, "lr_scale": lr_scale, "replay_users": replay_users,
            "weighted_static_ndcg@10": float((s * w).sum() / w.sum()) if len(w) else None,
            "weighted_updated_ndcg@10": float((d * w).sum() / w.sum()) if len(w) else None}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()
    ds = Dataset.load(a.data)
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    res = run(ds, ckpt, chunks=a.chunks, epochs=a.epochs)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "chunks"}, indent=1))


if __name__ == "__main__":
    main()
