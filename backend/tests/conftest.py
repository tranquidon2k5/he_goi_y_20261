import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["CINEREC_DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["CINEREC_ADMIN_EMAILS"] = "admin@test.vn"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import SessionLocal, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Movie, Rating  # noqa: E402

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Animation", "Horror", "Romance", "Thriller"]


def _seed() -> None:
    init_db()
    with SessionLocal() as db:
        if db.get(Movie, 1):
            return
        for i in range(1, 61):
            gs = "|".join({GENRES[i % 8], GENRES[(i * 3) % 8]})
            db.add(Movie(id=i, title=f"Phim {i}", year=1990 + i % 30, genres=gs, source="movielens"))
        db.add(Movie(id=999, title="Phim mới", year=2026, genres="Drama|Romance", source="app"))
        db.flush()
        # 40 người dùng MovieLens giả, mỗi người chấm 20 phim
        for u in range(1, 41):
            for j in range(20):
                mid = (u * 7 + j * 3) % 60 + 1
                if db.get(Rating, (u, mid)) is None:
                    db.add(Rating(user_id=u, movie_id=mid, rating=float(1 + (u + j) % 5), timestamp=1_600_000_000 + u * 100 + j,
                                  source="movielens"))
        db.commit()
        from app.stats import recompute_movie_stats
        recompute_movie_stats(db)


@pytest.fixture(scope="session")
def client():
    _seed()
    with TestClient(app) as c:
        yield c


def register(client: TestClient, email: str) -> TestClient:
    client.cookies.clear()
    r = client.post("/auth/register", json={"email": email, "password": "secret123", "display_name": "Minh Anh"})
    assert r.status_code == 200, r.text
    return client
