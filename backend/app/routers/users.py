"""Đăng ký, đăng nhập, hồ sơ, xuất và xoá dữ liệu cá nhân."""

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.auth import clear_cookie, current_user, hash_password, issue_cookie, next_user_id, verify_password
from app.config import get_settings
from app.db import get_db
from app.models import Event, Movie, NotInterested, Rating, Tag, User, UserGenre, Watched, WatchlistItem
from app.recsys.service import rec_service
from app.schemas import LoginIn, ProfileUpdateIn, RegisterIn, UserOut

router = APIRouter(tags=["users"])


def user_out(db: Session, u: User) -> UserOut:
    return UserOut(
        id=u.id, email=u.email, display_name=u.display_name, is_admin=u.is_admin, onboarded=u.onboarded,
        genres=sorted(db.scalars(select(UserGenre.genre).where(UserGenre.user_id == u.id))),
        n_ratings=db.scalar(select(func.count()).select_from(Rating).where(Rating.user_id == u.id)) or 0,
    )


@router.post("/auth/register", response_model=UserOut)
def register(body: RegisterIn, resp: Response, db: Session = Depends(get_db)):
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "Email đã được dùng")
    admins = {e.strip().lower() for e in get_settings().admin_emails.split(",") if e.strip()}
    u = User(id=next_user_id(db), email=email, password_hash=hash_password(body.password),
             display_name=body.display_name.strip(), is_admin=email in admins)
    db.add(u)
    db.commit()
    issue_cookie(resp, u.id)
    return user_out(db, u)


@router.post("/auth/login", response_model=UserOut)
def login(body: LoginIn, resp: Response, db: Session = Depends(get_db)):
    u = db.scalar(select(User).where(User.email == body.email.lower()))
    if not u or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "Sai email hoặc mật khẩu")
    issue_cookie(resp, u.id)
    return user_out(db, u)


@router.post("/auth/logout")
def logout(resp: Response):
    clear_cookie(resp)
    return {"ok": True}


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return user_out(db, user)


@router.patch("/me", response_model=UserOut)
def update_me(body: ProfileUpdateIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if body.display_name is not None:
        user.display_name = body.display_name.strip()
    if body.onboarded is not None:
        user.onboarded = body.onboarded
    if body.genres is not None:
        db.execute(delete(UserGenre).where(UserGenre.user_id == user.id))
        db.add_all(UserGenre(user_id=user.id, genre=g) for g in dict.fromkeys(body.genres))
    db.commit()
    rec_service.invalidate(user.id)
    return user_out(db, user)


@router.get("/me/export")
def export_me(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Tải dữ liệu của tôi — CSV gộp ratings, tags, danh sách, sự kiện."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["kind", "movieId", "title", "value", "timestamp"])
    mids: set[int] = set()
    for model in (Rating, Tag, WatchlistItem, NotInterested, Watched):
        mids.update(db.scalars(select(model.movie_id).where(model.user_id == user.id)))
    titles = dict(db.execute(select(Movie.id, Movie.title).where(Movie.id.in_(mids))).all()) if mids else {}
    for r in db.scalars(select(Rating).where(Rating.user_id == user.id)):
        w.writerow(["rating", r.movie_id, titles.get(r.movie_id, ""), r.rating, r.timestamp])
    for t in db.scalars(select(Tag).where(Tag.user_id == user.id)):
        w.writerow(["tag", t.movie_id, titles.get(t.movie_id, ""), t.tag, t.timestamp])
    for model, kind in ((WatchlistItem, "watchlist"), (NotInterested, "not_interested"), (Watched, "watched")):
        for x in db.scalars(select(model).where(model.user_id == user.id)):
            w.writerow([kind, x.movie_id, titles.get(x.movie_id, ""), "", x.timestamp])
    for e in db.scalars(select(Event).where(Event.user_id == user.id).order_by(Event.timestamp)):
        w.writerow([f"event:{e.type}", e.movie_id or "", titles.get(e.movie_id, "") if e.movie_id else "", "", e.timestamp])
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="cinerec_{user.id}.csv"'})


@router.delete("/me")
def delete_me(resp: Response, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Xoá tài khoản và toàn bộ dữ liệu (PRD mục 3, yêu cầu phi chức năng)."""
    uid = user.id
    for model in (Rating, Tag, Event, WatchlistItem, NotInterested, Watched, UserGenre):
        db.execute(delete(model).where(model.user_id == uid))
    db.delete(user)
    db.commit()
    rec_service.invalidate(uid)
    clear_cookie(resp)
    return {"ok": True}
