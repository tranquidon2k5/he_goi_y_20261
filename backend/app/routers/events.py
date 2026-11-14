"""POST /events — nhận một lô sự kiện tiềm ẩn từ web app (PRD mục 4)."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import current_user
from app.db import get_db
from app.models import Event, User, now_ts
from app.recsys.service import rec_service
from app.schemas import EventsIn

router = APIRouter(tags=["events"])

CLIENT_EVENT_TYPES = {"impression", "detail_view", "watch", "trailer_play", "search", "click"}
# các sự kiện này làm thay đổi đầu vào của mô hình → huỷ cache gợi ý
INVALIDATING = {"watch"}


@router.post("/events")
async def ingest(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    # navigator.sendBeacon gửi text/plain nên tự parse body thay vì để FastAPI validate theo Content-Type
    body = EventsIn.model_validate_json(await request.body())
    incoming = [e for e in body.events if e.type in CLIENT_EVENT_TYPES]
    ids = [e.event_id for e in incoming]
    known = set(db.scalars(select(Event.event_id).where(Event.event_id.in_(ids)))) if ids else set()
    now = now_ts()
    accepted = 0
    for e in incoming:
        if e.event_id in known:
            continue  # idempotent: client có thể gửi lại cùng lô
        known.add(e.event_id)
        ts = e.timestamp if e.timestamp and abs(e.timestamp - now) < 86400 else now
        db.add(Event(event_id=e.event_id, user_id=user.id, session_id=e.session_id, type=e.type, timestamp=ts,
                     movie_id=e.movie_id, request_id=e.request_id, model_version=e.model_version, payload=e.payload))
        accepted += 1
    db.commit()
    if any(e.type in INVALIDATING for e in incoming):
        rec_service.invalidate(user.id)
    return {"accepted": accepted, "dropped": len(body.events) - accepted}
