"""Tính lại thống kê phim (điểm TB, số lượt chấm, số lượt chấm gần đây) từ bảng ratings."""

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Rating


def recompute_movie_stats(db: Session) -> dict:
    max_ts = db.scalar(select(func.max(Rating.timestamp))) or 0
    since = max_ts - get_settings().trending_window_seconds
    db.execute(text("""
        UPDATE movies SET
          rating_avg   = COALESCE((SELECT AVG(r.rating) FROM ratings r WHERE r.movie_id = movies.id), 0),
          rating_count = (SELECT COUNT(*) FROM ratings r WHERE r.movie_id = movies.id),
          recent_count = (SELECT COUNT(*) FROM ratings r WHERE r.movie_id = movies.id AND r.timestamp >= :since)
    """), {"since": since})
    db.commit()
    return {"max_timestamp": int(max_ts), "trending_since": int(since)}
