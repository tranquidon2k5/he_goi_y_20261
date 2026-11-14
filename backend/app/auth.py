import time

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import User

COOKIE = "cinerec_token"


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hashed.encode())


def next_user_id(db: Session) -> int:
    start = get_settings().app_user_id_start
    cur = db.scalar(select(func.max(User.id)))
    return max(start, (cur or 0) + 1)


def issue_cookie(resp: Response, user_id: int) -> None:
    s = get_settings()
    token = jwt.encode({"sub": str(user_id), "exp": int(time.time()) + s.jwt_ttl_hours * 3600}, s.jwt_secret, "HS256")
    resp.set_cookie(COOKIE, token, httponly=True, samesite="lax", secure=s.cookie_secure,
                    max_age=s.jwt_ttl_hours * 3600, path="/")


def clear_cookie(resp: Response) -> None:
    resp.delete_cookie(COOKIE, path="/")


def _user_from_request(request: Request, db: Session) -> User | None:
    token = request.cookies.get(COOKIE)
    if not token:
        auth = request.headers.get("authorization", "")
        token = auth.removeprefix("Bearer ").strip() or None
    if not token:
        return None
    try:
        data = jwt.decode(token, get_settings().jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    return db.get(User, int(data["sub"]))


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = _user_from_request(request, db)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Chưa đăng nhập")
    return user


def optional_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    return _user_from_request(request, db)


def admin_user(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Chỉ admin")
    return user
