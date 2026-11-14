import uuid

from tests.conftest import register


def ids_in(rows):
    return {it["movie"]["id"] for row in rows for it in row["items"]}


def test_user_journey(client):
    register(client, "user@test.vn")
    me = client.get("/me").json()
    assert me["id"] >= 1_000_001 and not me["onboarded"]

    client.patch("/me", json={"genres": ["Sci-Fi", "Comedy"]})
    pool = client.get("/movies/onboarding").json()["items"]
    assert pool and any(g in pool[0]["movie"]["genres"] for g in ("Sci-Fi", "Comedy"))

    # cold-start: < 5 tương tác → chỉ có Thịnh hành theo thể loại, không có "Dành cho bạn"
    home = client.get("/recommendations?surface=home").json()
    assert home["cold"] and home["model_version"]
    assert "for_you" not in {r["kind"] for r in home["rows"]}

    rated = [it["movie"]["id"] for it in pool[:5]]
    for mid, r in zip(rated, [5, 4.5, 4, 2.5, 3]):
        assert client.post("/ratings", json={"movie_id": mid, "rating": r}).status_code == 200
    assert client.post("/ratings", json={"movie_id": rated[0], "rating": 4.3}).status_code == 422

    home = client.get("/recommendations?surface=home").json()
    assert not home["cold"]
    kinds = [r["kind"] for r in home["rows"]]
    assert kinds[0] == "for_you" and "because" in kinds and "trending" in kinds
    # không gợi ý lại phim đã chấm
    assert not ids_in(home["rows"]) & set(rated)
    assert home["hero"]["movie"]["id"] not in rated
    fy = home["rows"][0]["items"]
    assert any("reason" in it for it in fy)

    # "Không quan tâm" → biến mất ngay ở lần tải kế tiếp
    victim = fy[0]["movie"]["id"]
    client.put(f"/me/not-interested/{victim}")
    home2 = client.get("/recommendations?surface=home").json()
    assert victim not in ids_in(home2["rows"])
    assert victim not in {i["movie"]["id"] for i in client.get("/movies?page_size=100").json()["items"]}

    # chi tiết + tag + xem sau
    d = client.get(f"/movies/{rated[0]}").json()
    assert d["me"]["rating"] == 5
    t = client.post(f"/movies/{rated[0]}/tags", json={"tag": "kết bất ngờ"}).json()
    client.put(f"/me/watchlist/{rated[1]}")
    d = client.get(f"/movies/{rated[0]}").json()
    assert d["me"]["tags"][0]["tag"] == "kết bất ngờ"
    client.delete(f"/movies/{rated[0]}/tags/{t['id']}")
    assert client.get("/me/watchlist").json()["items"][0]["movie"]["id"] == rated[1]

    sim = client.get(f"/movies/{rated[0]}/similar").json()
    assert sim["items"] and rated[0] not in {i["movie"]["id"] for i in sim["items"]}

    # tìm kiếm & lọc
    res = client.get("/movies", params={"genre": ["Sci-Fi"], "year_min": 2000, "sort": "rel"}).json()
    assert all("Sci-Fi" in i["movie"]["genres"] and i["movie"]["year"] >= 2000 for i in res["items"])
    assert client.get("/movies", params={"q": "Phim 1"}).json()["total"] >= 1

    # sự kiện: idempotent theo event_id
    ev = {"event_id": uuid.uuid4().hex, "type": "impression", "request_id": home2["request_id"],
          "payload": {"row": "for_you", "movie_ids": [1, 2, 3]}}
    assert client.post("/events", json={"events": [ev, ev]}).json()["accepted"] == 1
    assert client.post("/events", json={"events": [ev]}).json()["accepted"] == 0

    # lịch sử: xoá rating
    assert len(client.get("/me/ratings").json()["items"]) == 5
    client.delete(f"/ratings/{rated[4]}")
    assert len(client.get("/me/ratings").json()["items"]) == 4

    assert client.get("/me/export").text.startswith("kind,movieId")
    assert client.get("/lab/models").status_code == 403

    assert client.delete("/me").json()["ok"]
    assert client.get("/me").status_code == 401
    assert client.post("/auth/login", json={"email": "user@test.vn", "password": "secret123"}).status_code == 401


def test_lab_retrain_and_rollback(client):
    register(client, "admin@test.vn")
    lab = client.get("/lab/models").json()
    first = lab["live"]["version"]
    r = client.post("/lab/retrain", json={"kind": "refresh"})
    assert r.status_code == 200
    lab = client.get("/lab/models").json()
    assert lab["live"]["version"] != first
    assert client.get("/health").json()["model_version"] == lab["live"]["version"]
    rb = client.post("/lab/rollback", json={"version": first}).json()
    assert rb["status"] == "live"
    assert client.get("/health").json()["model_version"] == first


def test_new_movie_reaches_users(client):
    register(client, "fresh@test.vn")
    client.patch("/me", json={"genres": ["Drama"]})
    for mid in range(1, 7):
        client.post("/ratings", json={"movie_id": mid, "rating": 4})
    rows = client.get("/recommendations").json()["rows"]
    new_row = next(r for r in rows if r["kind"] == "new")
    assert new_row["items"][0]["movie"]["id"] == 999 and new_row["items"][0]["movie"]["is_new"]
