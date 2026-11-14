from typing import Literal

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth import current_user
from app.db import get_db
from app.models import User
from app.recsys.service import rec_service

router = APIRouter(tags=["recommendations"])


@router.get("/recommendations")
def recommendations(surface: Literal["home"] = "home", user: User = Depends(current_user),
                    db: Session = Depends(get_db)):
    """GET /recommendations?surface=home — các hàng gợi ý của trang chủ, kèm model_version và request_id."""
    return rec_service.home(db, user)
