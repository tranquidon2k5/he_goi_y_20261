"""Đóng gói checkpoint thành thư mục artifact cho dịch vụ gợi ý (app/recsys/seqrec.py).

    python -m ml.export --data ../data/processed/ml32m_2020 --ckpt results/proposed/model.pt \
        --out artifacts/proposed-v1 [--metrics results/proposed/test_seed0.json]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def export(data: Path, ckpt: Path, out: Path, metrics: Path | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(data / "interactions.npz")
    np.save(out / "item_ids.npy", z["item_ids"])
    for f in ("features.npy", "feature_meta.json", "meta.json"):
        shutil.copy2(data / f, out / f)
    if ckpt.resolve() != (out / "model.pt").resolve():
        shutil.copy2(ckpt, out / "model.pt")
    if metrics and metrics.exists():
        shutil.copy2(metrics, out / "metrics.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--metrics", type=Path)
    a = ap.parse_args()
    print(json.dumps({"artifact_dir": str(export(a.data, a.ckpt, a.out, a.metrics))}))


if __name__ == "__main__":
    main()
