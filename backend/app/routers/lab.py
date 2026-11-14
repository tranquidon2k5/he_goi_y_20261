"""Trang Lab: phiên bản mô hình, metric, học lại, rollback.

Khi chưa có thuật toán chính, "retrain" chỉ tính lại thống kê phim (độ phổ biến) và nạp lại
bản tạm thành một phiên bản mới — đủ để chạy thử toàn bộ vòng: tạo phiên bản → live → rollback.
Trainer thật (fine-tune / huấn luyện lại + cổng đánh giá) sẽ cắm vào `_run_retrain`.
"""

from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.auth import admin_user
from app.db import SessionLocal, get_db
from app.models import Event, ModelVersion, Rating, User, now_ts
from app.recsys import registry
from app.recsys.service import rec_service
from app.stats import recompute_movie_stats

router = APIRouter(prefix="/lab", tags=["lab"], dependencies=[Depends(admin_user)])

POSITIVE_EVENTS = ("mark_watched", "watchlist_add", "watch")


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
        # bảng so sánh offline — điền khi đã chạy thí nghiệm (model_versions.metrics)
        "offline": [{"version": v.version, "recommender": v.recommender, **v.metrics} for v in versions if v.metrics],
    }


class RetrainIn(BaseModel):
    kind: Literal["refresh", "finetune", "full"] = "refresh"
    recommender: str | None = None


def _run_retrain(version: str) -> None:
    db = SessionLocal()
    try:
        mv = db.scalar(select(ModelVersion).where(ModelVersion.version == version))
        mv.status = "running"
        db.commit()
        try:
            info = recompute_movie_stats(db)
            # TODO(thuật toán): gọi trainer của mô hình chính ở đây (fine-tune / full), chấm cổng đánh giá
            # trên cửa sổ giữ riêng, ghi mv.metrics; nếu không qua cổng → mv.status = "fail" và return.
            db.execute(update(ModelVersion).where(ModelVersion.status == "live").values(status="pass"))
            mv.status, mv.deployed_at = "live", now_ts()
            mv.data_info = mv.data_info | info
            db.commit()
            rec_service.reload(db)
        except Exception as e:  # noqa: BLE001
            db.rollback()
            mv.status, mv.notes = "fail", f"{mv.notes or ''}\nLỗi: {e}".strip()
            db.commit()
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
    if body.kind != "refresh" and name == registry.DEFAULT_RECOMMENDER:
        note = "Thuật toán chính chưa chốt — chỉ tính lại thống kê độ phổ biến."
    else:
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
