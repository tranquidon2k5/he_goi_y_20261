"""Trang Lab: phiên bản mô hình, metric, học lại, rollback.

Ba mức học lại (PRD mục 5):
- refresh  — tính lại thống kê phim (độ phổ biến, "Thịnh hành") và nạp lại mô hình đang chạy.
- finetune — mô hình tuần tự (seqrec): fine-tune tăng dần trên CPU từ checkpoint đang chạy với dữ liệu
             mới của app + replay MovieLens, rồi qua cổng đánh giá (ml/finetune.py) mới được triển khai.
- full     — huấn luyện lại toàn bộ chạy offline trên GPU (ml/train.py, Colab), rồi đăng ký checkpoint
             bằng scripts/register_model.py; trong API chỉ ghi nhận yêu cầu.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from app.auth import admin_user
from app.db import SessionLocal, engine, get_db
from app.models import Event, ModelVersion, Rating, User, now_ts
from app.recsys import registry
from app.recsys.catalog import Catalog
from app.recsys.service import rec_service
from app.stats import recompute_movie_stats

router = APIRouter(prefix="/lab", tags=["lab"], dependencies=[Depends(admin_user)])

POSITIVE_EVENTS = ("mark_watched", "watchlist_add", "watch")
BACKEND_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS = BACKEND_DIR / "artifacts"
EXPERIMENTS = ARTIFACTS / "experiments" / "summary.json"


def version_out(v: ModelVersion) -> dict:
    return {
        "version": v.version, "recommender": v.recommender, "kind": v.kind, "status": v.status,
        "metrics": v.metrics, "config": v.config, "data_info": v.data_info, "notes": v.notes,
        "created_at": v.created_at, "deployed_at": v.deployed_at,
    }


def _bump(version: str) -> str:
    try:
        major, minor, patch = (int(x) for x in version.lstrip("v").split("."))
    except ValueError:
        return version + ".1"
    return f"v{major}.{minor}.{patch + 1}"


def _online_metrics(db: Session) -> list[dict]:
    """CTR theo hàng gợi ý: lượt click (detail_view có source_row) / số phim đã hiển thị."""
    shown: dict[str, int] = {}
    for (p,) in db.execute(select(Event.payload).where(Event.type == "impression")):
        row = (p or {}).get("row")
        if row:
            shown[row] = shown.get(row, 0) + len((p or {}).get("movie_ids", []))
    clicks: dict[str, int] = {}
    for (p,) in db.execute(select(Event.payload).where(Event.type == "detail_view")):
        row = (p or {}).get("source_row")
        if row:
            clicks[row] = clicks.get(row, 0) + 1
    return sorted(({"row": r, "impressions": n, "clicks": clicks.get(r, 0), "ctr": clicks.get(r, 0) / n if n else 0}
                   for r, n in shown.items()), key=lambda x: -x["impressions"])


@router.get("/models")
def models(db: Session = Depends(get_db)):
    versions = list(db.scalars(select(ModelVersion).order_by(ModelVersion.id.desc())))
    live = next((v for v in versions if v.status == "live"), None)
    since = (live.deployed_at or live.created_at) if live else 0
    pending = (db.scalar(select(func.count()).select_from(Rating).where(Rating.source == "app", Rating.timestamp > since)) or 0) \
        + (db.scalar(select(func.count()).select_from(Event).where(Event.type.in_(POSITIVE_EVENTS), Event.timestamp > since)) or 0)
    p95 = rec_service.p95_ms()
    return {
        "live": version_out(live) if live else None,
        "versions": [version_out(v) for v in versions],
        "recommenders": sorted(registry.RECOMMENDERS),
        "stats": {
            "p95_ms": round(p95, 1) if p95 is not None else None,
            "pending_interactions": pending,
            "app_users": db.scalar(select(func.count()).select_from(User)) or 0,
            "app_ratings": db.scalar(select(func.count()).select_from(Rating).where(Rating.source == "app")) or 0,
            "events": db.scalar(select(func.count()).select_from(Event)) or 0,
        },
        "online": _online_metrics(db),
        "offline": [{"version": v.version, "recommender": v.recommender, **v.metrics} for v in versions if v.metrics],
        # bảng so sánh mô hình đề xuất với baseline (ml/train.py report → artifacts/experiments/summary.json)
        "experiments": json.loads(EXPERIMENTS.read_text(encoding="utf-8")) if EXPERIMENTS.exists() else None,
    }


class RetrainIn(BaseModel):
    kind: Literal["refresh", "finetune", "full"] = "refresh"
    recommender: str | None = None


def export_positives(path: Path) -> int:
    """Tương tác dương dạng (user_id, movie_id, timestamp, source) cho fine-tune — PRD mục 4:
    rating >= 4; với người dùng app thêm "đã xem" và "xem sau" (rating quyết định nếu có cả hai)."""
    q = """
        SELECT user_id, movie_id, timestamp, source FROM ratings WHERE rating >= 4
        UNION ALL
        SELECT w.user_id, w.movie_id, w.timestamp, 'app' FROM watched w
          WHERE NOT EXISTS (SELECT 1 FROM ratings r WHERE r.user_id = w.user_id AND r.movie_id = w.movie_id)
        UNION ALL
        SELECT w.user_id, w.movie_id, w.timestamp, 'app' FROM watchlist w
          WHERE NOT EXISTS (SELECT 1 FROM ratings r WHERE r.user_id = w.user_id AND r.movie_id = w.movie_id)
            AND NOT EXISTS (SELECT 1 FROM watched x WHERE x.user_id = w.user_id AND x.movie_id = w.movie_id)
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(q), conn)
    df = df.drop_duplicates(["user_id", "movie_id"])
    df.to_csv(path, index=False)
    return len(df)


def export_new_movies(path: Path) -> int:
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT id AS movie_id, genres, year FROM movies WHERE source = 'app'"), conn)
    df.to_csv(path, index=False)
    return len(df)


def _finetune(mv: ModelVersion, parent: ModelVersion) -> tuple[dict, str]:
    """Chạy ml.finetune trong tiến trình con (không chặn API). Trả (metrics, artifact_dir mới)."""
    src = Path(parent.config["artifact_dir"])
    src = src if src.is_absolute() else BACKEND_DIR / src
    out = ARTIFACTS / mv.version
    with tempfile.TemporaryDirectory() as tmp:
        inter, new = Path(tmp) / "positives.csv", Path(tmp) / "new_movies.csv"
        export_positives(inter)
        export_new_movies(new)
        proc = subprocess.run(
            [sys.executable, "-m", "ml.finetune", "--in", str(src), "--out", str(out),
             "--interactions", str(inter), "--new-movies", str(new)],
            cwd=BACKEND_DIR, capture_output=True, text=True, encoding="utf-8", timeout=3600,
        )
    m = re.search(r"^RESULT (.*)$", proc.stdout, re.M)
    if proc.returncode != 0 or not m:
        raise RuntimeError((proc.stderr or proc.stdout)[-800:])
    return json.loads(m.group(1)), str(out.relative_to(BACKEND_DIR)).replace("\\", "/")


def _run_retrain(version: str) -> None:
    db = SessionLocal()
    try:
        mv = db.scalar(select(ModelVersion).where(ModelVersion.version == version))
        mv.status = "running"
        db.commit()
        promoted, parent = False, None
        try:
            info = recompute_movie_stats(db)
            parent = db.scalar(select(ModelVersion).where(ModelVersion.status == "live"))
            if mv.kind == "finetune" and mv.recommender == "seqrec" and parent and parent.config.get("artifact_dir"):
                metrics, art = _finetune(mv, parent)
                mv.metrics = {"ndcg@10": metrics["gate_ndcg@10"]} | metrics
                mv.config = dict(mv.config) | {"artifact_dir": art}
                f4 = lambda x: "—" if x is None else f"{x:.4f}"  # noqa: E731
                if not metrics["passed"]:
                    mv.status = "fail"
                    mv.notes = ("Không qua cổng: không có ca đánh giá" if metrics["gate_basis"] == "none" else
                                f"Không qua cổng: NDCG@10 {f4(metrics['gate_ndcg@10'])} < "
                                f"{f4(metrics['live_gate_ndcg@10'])} × (1 − {metrics['tolerance']})")
                    mv.data_info = mv.data_info | info
                    db.commit()
                    return
                mv.notes = (f"Fine-tune từ {parent.version}: NDCG@10 cổng {f4(metrics['live_gate_ndcg@10'])} → "
                            f"{f4(metrics['gate_ndcg@10'])} ({metrics['gate_basis']}), {metrics['new_movies']} phim mới")
            # nạp thử trước khi đổi trạng thái: lỗi nạp mô hình không được làm mất phiên bản đang live
            registry.create(mv.recommender, mv.config).load(Catalog.from_db(db), mv.config.get("artifact_dir"))
            db.execute(update(ModelVersion).where(ModelVersion.status == "live").values(status="pass"))
            mv.status, mv.deployed_at = "live", now_ts()
            mv.data_info = mv.data_info | info
            db.commit()
            promoted = True
            rec_service.reload(db)
        except Exception as e:  # noqa: BLE001
            db.rollback()
            mv.status, mv.notes = "fail", f"{mv.notes or ''}\nLỗi: {e}".strip()
            if promoted and parent is not None:  # đã đổi trạng thái mà nạp lỗi → trả lại bản live cũ
                parent.status = "live"
            db.commit()
            if promoted:
                rec_service.reload(db)
    finally:
        db.close()


@router.post("/retrain")
def retrain(body: RetrainIn, tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if db.scalar(select(ModelVersion).where(ModelVersion.status.in_(("queued", "running")))):
        raise HTTPException(409, "Đang có một lần học lại chạy")
    live = db.scalar(select(ModelVersion).where(ModelVersion.status == "live"))
    name = body.recommender or (live.recommender if live else registry.DEFAULT_RECOMMENDER)
    if name not in registry.RECOMMENDERS:
        raise HTTPException(400, f"Recommender chưa đăng ký: {name}")
    if body.kind == "full":
        raise HTTPException(400, "Huấn luyện lại toàn bộ chạy offline trên GPU: python -m ml.train tune/final, "
                                 "rồi scripts/register_model.py")
    if body.kind == "finetune" and name != "seqrec":
        raise HTTPException(400, f"Fine-tune chỉ áp dụng cho mô hình tuần tự (seqrec), không cho {name}")
    if body.kind == "finetune" and not (live and live.recommender == "seqrec" and live.config.get("artifact_dir")):
        raise HTTPException(400, "Fine-tune cần một checkpoint seqrec đang live — đăng ký bằng scripts/register_model.py")
    note = None
    versions = list(db.scalars(select(ModelVersion.version)))
    new = _bump(max(versions, key=lambda v: tuple(int(x) for x in v.lstrip("v").split(".") if x.isdigit()))) if versions else "v0.1.0"
    mv = ModelVersion(version=new, recommender=name, kind=body.kind, status="queued",
                      config=dict(live.config) if live and live.recommender == name else {}, notes=note)
    db.add(mv)
    db.commit()
    tasks.add_task(_run_retrain, new)
    return version_out(mv)


class RollbackIn(BaseModel):
    version: str


@router.post("/rollback")
def rollback(body: RollbackIn, db: Session = Depends(get_db)):
    target = db.scalar(select(ModelVersion).where(ModelVersion.version == body.version))
    if target is None:
        raise HTTPException(404, "Không có phiên bản này")
    if target.status not in ("pass", "live"):
        raise HTTPException(400, "Chỉ rollback về phiên bản đã qua cổng đánh giá")
    db.execute(update(ModelVersion).where(ModelVersion.status == "live").values(status="pass"))
    target.status, target.deployed_at = "live", now_ts()
    db.commit()
    rec_service.reload(db)
    return version_out(target)
