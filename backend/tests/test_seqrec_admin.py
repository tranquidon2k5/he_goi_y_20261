"""Quản trị thêm phim mới + phục vụ mô hình tuần tự (seqrec) + fine-tune qua cổng đánh giá.

Chạy sau test_flow.py (thứ tự chữ cái) vì đổi mô hình đang live.
"""

import json

import numpy as np
import torch

from app.db import SessionLocal
from app.models import ModelVersion
from app.recsys.service import rec_service
from ml.data import featurize
from ml.models import SeqRec
from tests.conftest import GENRES, register


def as_admin(client):
    client.cookies.clear()
    if client.post("/auth/login", json={"email": "admin@test.vn", "password": "secret123"}).status_code != 200:
        register(client, "admin@test.vn")
    return client


def make_artifact(path, n_items=60, content=True):
    """Checkpoint nhỏ chưa huấn luyện cho danh mục test (movieId 1..60)."""
    meta = {"genres": GENRES, "genre_index": {g: i for i, g in enumerate(GENRES)}, "tags": [], "tag_index": {}, "idf": []}
    feats = np.vstack([np.zeros((1, len(GENRES) + 7), np.float32)] +
                      [featurize("|".join({GENRES[i % 8], GENRES[(i * 3) % 8]}), 1990 + i % 30, meta)[None] for i in range(1, n_items + 1)])
    hp = {"d": 16, "blocks": 1, "heads": 1, "maxlen": 20, "batch": 64, "lr": 1e-3, "dropout": 0.1, "wd": 0.0,
          "n_neg": 16, "cl_weight": 0.1, "tau": 0.5, "crop": 0.6, "mask": 0.3, "reorder": 0.6, "fuse": "add"}
    spec = {"family": "seq", "loss": "ss", "content": content, "cl": True}
    torch.manual_seed(0)
    m = SeqRec(n_items, d=16, blocks=1, heads=1, maxlen=20, features=torch.as_tensor(feats) if content else None)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"name": "proposed", "hp": hp, "spec": spec, "epochs": 1, "state_dict": m.state_dict(), "n_items": n_items},
               path / "model.pt")
    np.save(path / "item_ids.npy", np.arange(1, n_items + 1))
    np.save(path / "features.npy", feats)
    (path / "feature_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (path / "meta.json").write_text(json.dumps({"t_test": 1_600_002_010}), encoding="utf-8")  # user 20 vắt qua mốc


def test_admin_adds_movie(client):
    register(client, "viewer@test.vn")
    assert client.get("/admin/movies").status_code == 403
    assert client.post("/admin/movies", json={"title": "X", "genres": ["Drama"]}).status_code == 403

    as_admin(client)
    r = client.post("/admin/movies", json={"title": "Phim quản trị", "year": 2026, "genres": ["Sci-Fi", "Drama"],
                                          "overview": "Mô tả", "trailer_key": "abc123"})
    assert r.status_code == 201, r.text
    m = r.json()["movie"]
    assert m["id"] >= 10_000_001 and m["is_new"] and m["genres"] == ["Sci-Fi", "Drama"]
    assert client.post("/admin/movies", json={"title": "  ", "genres": ["Drama"]}).status_code == 422
    assert client.post("/admin/movies", json={"title": "Y", "genres": ["Không có"]}).status_code == 422
    assert client.post("/admin/movies", json={"title": "Y", "genres": []}).status_code == 422
    # TMDB: không có khoá thì vẫn thêm được, kèm cảnh báo
    r2 = client.post("/admin/movies", json={"title": "Có TMDB", "genres": ["Comedy"], "tmdb_id": 42, "fetch_tmdb": True})
    assert r2.status_code == 201 and "TMDB" in r2.json()["warning"]
    assert client.post("/admin/movies", json={"title": "Trùng", "genres": ["Comedy"], "tmdb_id": 42}).status_code == 409

    listed = client.get("/admin/movies").json()
    assert listed["items"][0]["id"] == r2.json()["movie"]["id"] and listed["total"] >= 3  # gồm phim 999 của fixture

    # sửa / xoá chỉ áp dụng cho phim nguồn app
    assert client.patch(f"/admin/movies/{m['id']}", json={"title": "Phim quản trị (bản đạo diễn)"}).json()["movie"]["title"] \
        == "Phim quản trị (bản đạo diễn)"
    assert client.patch("/admin/movies/1", json={"title": "hack"}).status_code == 404
    assert client.delete("/admin/movies/1").status_code == 404
    # trường bắt buộc không được đặt null; tmdb_id không được trùng phim khác
    assert client.patch(f"/admin/movies/{m['id']}", json={"genres": None}).status_code == 422
    assert client.patch(f"/admin/movies/{m['id']}", json={"title": None}).status_code == 422
    assert client.patch(f"/admin/movies/{m['id']}", json={"tmdb_id": 42}).status_code == 409
    # xoá phim có id lớn nhất rồi thêm phim mới: id không được dùng lại
    deleted = r2.json()["movie"]["id"]
    assert client.delete(f"/admin/movies/{deleted}").json()["ok"]
    r3 = client.post("/admin/movies", json={"title": "Không rõ năm", "genres": ["Horror"]})
    assert r3.status_code == 201 and r3.json()["movie"]["id"] > deleted and r3.json()["movie"]["year"] is None

    # phim mới lên hàng "Mới thêm vào kho" của người xem ngay (dịch vụ gợi ý đã nạp lại danh mục)
    register(client, "watcher@test.vn")
    client.patch("/me", json={"genres": ["Sci-Fi"]})
    for mid in range(10, 16):
        client.post("/ratings", json={"movie_id": mid, "rating": 4.5})
    new_row = next(r for r in client.get("/recommendations").json()["rows"] if r["kind"] == "new")
    assert [it["movie"]["id"] for it in new_row["items"]][:2] == [r3.json()["movie"]["id"], m["id"]]
    assert client.get(f"/movies/{m['id']}").json()["movie"]["trailer_key"] == "abc123"


def test_seqrec_serving_and_finetune(client, tmp_path):
    # fine-tune cần một checkpoint seqrec đang live; bản tạm heuristic thì bị từ chối, không làm mất bản live
    as_admin(client)
    assert client.post("/lab/retrain", json={"kind": "finetune", "recommender": "seqrec"}).status_code == 400
    assert client.get("/lab/models").json()["live"]["recommender"] == "heuristic"

    art = tmp_path / "art_v1"
    make_artifact(art)
    with SessionLocal() as db:
        db.query(ModelVersion).filter(ModelVersion.status == "live").update({"status": "pass"})
        db.add(ModelVersion(version="v1.0.0", recommender="seqrec", kind="full", status="live",
                            metrics={"ndcg@10": 0.01}, config={"artifact_dir": str(art)}))
        db.commit()
        rec_service.reload(db)
    assert rec_service.version == "v1.0.0"
    model = rec_service.live.model
    # phim mới (source app, ngoài từ điển) vẫn có biểu diễn nhờ nội dung
    cat = rec_service.live.catalog
    assert bool(model.known[cat.row(999)]) and model.n_content_only >= 1

    register(client, "seq@test.vn")
    client.patch("/me", json={"genres": ["Drama"]})
    rated = [3, 8, 13, 21, 34, 55]
    for mid in rated:
        client.post("/ratings", json={"movie_id": mid, "rating": 5})
    home = client.get("/recommendations").json()
    assert home["model_version"] == "v1.0.0" and not home["cold"]
    fy = next(r for r in home["rows"] if r["kind"] == "for_you")["items"]
    assert len(fy) >= 10 and not {it["movie"]["id"] for it in fy} & set(rated)
    assert all(it["reason"]["movie_id"] in rated for it in fy if "reason" in it) and any("reason" in it for it in fy)
    sim = client.get("/movies/8/similar").json()
    assert sim["model_version"] == "v1.0.0" and 8 not in {it["movie"]["id"] for it in sim["items"]}

    # fine-tune: chạy ml.finetune trong tiến trình con, ghi metric cổng, live hoặc bị chặn
    as_admin(client)
    r = client.post("/lab/retrain", json={"kind": "finetune"})
    assert r.status_code == 200, r.text
    v = next(x for x in client.get("/lab/models").json()["versions"] if x["version"] == r.json()["version"])
    assert v["status"] in ("live", "fail"), v
    assert {"gate_ndcg@10", "live_gate_ndcg@10", "passed", "new_movies"} <= set(v["metrics"])
    assert v["metrics"]["new_movies"] >= 1  # phim do quản trị thêm được nối vào từ điển
    if v["status"] == "live":
        assert rec_service.version == v["version"]
        assert client.get("/recommendations").json()["model_version"] == v["version"]
    assert client.post("/lab/retrain", json={"kind": "full"}).status_code == 400

    # trả lại bản tạm cho các test khác
    with SessionLocal() as db:
        db.query(ModelVersion).filter(ModelVersion.status == "live").update({"status": "pass"})
        db.query(ModelVersion).filter(ModelVersion.version == "v0.1.0").update({"status": "live"})
        db.commit()
        rec_service.reload(db)
