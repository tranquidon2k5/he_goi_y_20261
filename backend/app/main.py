from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db import SessionLocal, init_db
from app.recsys.service import rec_service
from app.routers import admin, events, interactions, lab, movies, recommendations, users


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    with SessionLocal() as db:
        rec_service.reload(db)
    yield


app = FastAPI(title="CineRec API", version="0.1.0", lifespan=lifespan)

# users phải đứng trước interactions để /me/export không bị /me/{list_name} bắt
for r in (users, movies, interactions, events, recommendations, lab, admin):
    app.include_router(r.router)


@app.get("/health")
def health():
    return {"ok": True, "model_version": rec_service.version}
