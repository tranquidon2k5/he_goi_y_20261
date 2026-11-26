"""Đăng ký một checkpoint huấn luyện offline (ml/train.py trên GPU/Colab) thành phiên bản trong model_versions.

    uv run python -m scripts.register_model --artifact-dir artifacts/proposed-v1 --version v1.0.0 --live

- Chấm NDCG@10 trên tập cổng (cửa sổ test MovieLens trong CSDL) để so được với các lần fine-tune sau.
- Gắn kèm metric test offline (metrics.json do ml.export chép từ results/<model>/test_seed0.json).
- --live: đưa lên phục vụ ngay (phiên bản live cũ chuyển sang "pass", rollback được).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sqlalchemy import select, update

from app.db import SessionLocal, init_db
from app.models import ModelVersion, now_ts
from app.routers.lab import BACKEND_DIR, export_positives
from ml.finetune import gate_eval

OFFLINE_KEYS = ("ndcg@10", "recall@10", "ndcg@20", "recall@20", "mrr", "coverage@10", "diversity@10", "novelty@10")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--notes", default=None)
    a = ap.parse_args()
    init_db()
    art = Path(a.artifact_dir)
    full = art if art.is_absolute() else BACKEND_DIR / art
    ckpt_meta = json.loads((full / "metrics.json").read_text(encoding="utf-8")) if (full / "metrics.json").exists() else {}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "positives.csv"
        export_positives(p)
        gate = gate_eval(full, pd.read_csv(p))
    metrics = {"ndcg@10": gate["gate_ndcg@10"]} | gate | {f"test_{k}": ckpt_meta[k] for k in OFFLINE_KEYS if k in ckpt_meta}
    import torch

    ck = torch.load(full / "model.pt", map_location="cpu", weights_only=False)
    config = {"artifact_dir": str(art).replace("\\", "/"), "model": ck["name"], "hp": ck["hp"], "epochs": ck.get("epochs")}
    with SessionLocal() as db:
        if db.scalar(select(ModelVersion).where(ModelVersion.version == a.version)):
            raise SystemExit(f"Phiên bản {a.version} đã tồn tại")
        if a.live:
            db.execute(update(ModelVersion).where(ModelVersion.status == "live").values(status="pass"))
        db.add(ModelVersion(version=a.version, recommender="seqrec", kind="full", status="live" if a.live else "pass",
                            metrics=metrics, config=config, deployed_at=now_ts() if a.live else None,
                            notes=a.notes or f"Huấn luyện offline trên GPU (ml/train.py): {ck['name']}, {ck.get('epochs')} epoch",
                            data_info={"dataset": json.loads((full / "meta.json").read_text(encoding="utf-8"))}))
        db.commit()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
