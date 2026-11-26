"""Tiền xử lý MovieLens thành bộ dữ liệu đóng băng dùng chung cho mọi thí nghiệm (PRD mục 6).

1. Giữ rating >= 4.0 làm tương tác dương; cắt theo thời gian (--since-year).
2. Lọc k-core lặp: phim >= 10 tương tác, người dùng >= 5.
3. Chia theo mốc thời gian toàn cục: 5% tương tác cuối là test, 5% liền trước là validation.
4. Từ điển phim = phim có tương tác trong train (chỉ số 1..N, 0 là padding).
5. Đặc trưng nội dung: thể loại multi-hot + thập niên + TF-IDF của tag (chỉ tag trước mốc validation).

Script deterministic: chạy lại trên máy khác (Colab) cho đúng cùng ánh xạ phim.

    python -m ml.data --raw ../data/raw/ml-32m --out ../data/processed/ml32m_2020 --since-year 2020
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
DECADES = [1960, 1970, 1980, 1990, 2000, 2010, 2020]  # bucket i: năm < DECADES[i] (cuối: >= 2020)
N_TAGS = 300


def kcore(df: pd.DataFrame, min_item: int, min_user: int) -> pd.DataFrame:
    while True:
        ic = df["movieId"].map(df["movieId"].value_counts())
        uc = df["userId"].map(df["userId"].value_counts())
        keep = (ic >= min_item) & (uc >= min_user)
        if keep.all():
            return df
        df = df[keep]


def parse_year(title: str) -> int:
    m = YEAR_RE.search(str(title).strip())
    return int(m.group(1)) if m else 0


def decade_onehot(year: int) -> np.ndarray:
    v = np.zeros(len(DECADES), dtype=np.float32)
    if year > 0:
        v[min(int(np.searchsorted(DECADES, year, side="right")), len(DECADES) - 1)] = 1.0
    return v


def featurize(genres: str, year: int, meta: dict, tag_counts: dict[str, int] | None = None) -> np.ndarray:
    """Vector đặc trưng nội dung của một phim — dùng cả khi tiền xử lý lẫn khi app thêm phim mới."""
    g = np.zeros(len(meta["genres"]), dtype=np.float32)
    for x in (genres or "").split("|"):
        if x in meta["genre_index"]:
            g[meta["genre_index"][x]] = 1.0
    t = np.zeros(len(meta["tags"]), dtype=np.float32)
    for tag, c in (tag_counts or {}).items():
        j = meta["tag_index"].get(tag)
        if j is not None:
            t[j] = np.log1p(c) * meta["idf"][j]
    if t.any():
        t /= np.linalg.norm(t)
    return np.concatenate([g, decade_onehot(year), t])


def build(raw: Path, out: Path, since_year: int, positive: float = 4.0, min_item: int = 10, min_user: int = 5,
          val_frac: float = 0.05, test_frac: float = 0.05, until_year: int | None = None) -> dict:  # PATCH_V2
    out.mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(raw / "ratings.csv", dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"})
    n_raw = len(r)
    r = r[r["rating"] >= positive]
    since = int(pd.Timestamp(f"{since_year}-01-01", tz="UTC").timestamp())
    r = r[r["timestamp"] >= since]
    if until_year:  # lát cắt kết thúc trước năm này (để xác nhận trên khoảng thời gian không trùng)
        r = r[r["timestamp"] < int(pd.Timestamp(f"{until_year}-01-01", tz="UTC").timestamp())]
    n_pos = len(r)
    r = kcore(r, min_item, min_user)
    ts = r["timestamp"].to_numpy()
    t_test = int(np.quantile(ts, 1 - test_frac))
    t_val = int(np.quantile(ts, 1 - test_frac - val_frac))

    item_ids = np.sort(r.loc[r["timestamp"] < t_val, "movieId"].unique())
    n_before_vocab = len(r)
    r = r[r["movieId"].isin(item_ids)]
    item_index = pd.Series(np.arange(1, len(item_ids) + 1, dtype=np.int32), index=item_ids)
    user_ids = np.sort(r["userId"].unique())
    user_index = pd.Series(np.arange(len(user_ids), dtype=np.int32), index=user_ids)
    r = r.sort_values(["userId", "timestamp", "movieId"], kind="stable")
    split = np.where(r["timestamp"] < t_val, 0, np.where(r["timestamp"] < t_test, 1, 2)).astype(np.int8)
    np.savez_compressed(
        out / "interactions.npz",
        user=user_index.loc[r["userId"]].to_numpy(np.int32), item=item_index.loc[r["movieId"]].to_numpy(np.int32),
        ts=r["timestamp"].to_numpy(np.int64), split=split, user_ids=user_ids.astype(np.int64),
        item_ids=item_ids.astype(np.int64),
    )

    # ---------- đặc trưng nội dung
    movies = pd.read_csv(raw / "movies.csv").set_index("movieId").loc[item_ids]
    genres = sorted({g for gs in movies["genres"] for g in str(gs).split("|") if g and g != "(no genres listed)"})
    tags = pd.read_csv(raw / "tags.csv", dtype={"userId": "int32", "movieId": "int32", "tag": "str", "timestamp": "int64"})
    tags = tags[(tags["timestamp"] < t_val) & tags["movieId"].isin(item_ids)].dropna(subset=["tag"])
    tags = tags.assign(tag=tags["tag"].str.strip().str.lower())
    tags = tags[tags["tag"].str.len() > 0].drop_duplicates(["userId", "movieId", "tag"])
    df = tags.groupby("tag")["movieId"].nunique().sort_values(ascending=False, kind="stable")
    top = df.head(N_TAGS)
    top = top.sort_index(kind="stable").sort_values(ascending=False, kind="stable")  # hoà thì theo chữ cái
    vocab = list(top.index)
    idf = np.log(len(item_ids) / top.to_numpy()).astype(np.float32)
    meta = {"genres": genres, "genre_index": {g: i for i, g in enumerate(genres)}, "decades": DECADES,
            "tags": vocab, "tag_index": {t: i for i, t in enumerate(vocab)}, "idf": idf.tolist()}
    counts = tags[tags["tag"].isin(set(vocab))].groupby(["movieId", "tag"]).size()
    per_movie: dict[int, dict[str, int]] = {}
    for (mid, tag), c in counts.items():
        per_movie.setdefault(int(mid), {})[tag] = int(c)
    years = [parse_year(t) for t in movies["title"]]
    feats = np.stack([featurize(movies.loc[m, "genres"], y, meta, per_movie.get(int(m)))
                      for m, y in zip(item_ids, years)])
    feats = np.vstack([np.zeros((1, feats.shape[1]), np.float32), feats]).astype(np.float32)
    np.save(out / "features.npy", feats)
    (out / "feature_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"idx": np.arange(1, len(item_ids) + 1), "movieId": item_ids, "title": movies["title"].to_numpy(),
                  "genres": movies["genres"].to_numpy(), "year": years}).to_csv(out / "items.csv", index=False)

    info = {
        "since_year": since_year, "until_year": until_year, "positive_threshold": positive, "kcore": [min_item, min_user],
        "t_val": t_val, "t_test": t_test, "raw_ratings": n_raw, "positives_after_cut": n_pos,
        "after_kcore": n_before_vocab, "interactions": int(len(r)), "users": int(len(user_ids)), "items": int(len(item_ids)),
        "split_sizes": {k: int((split == i).sum()) for i, k in enumerate(("train", "val", "test"))},
        "density": float(len(r) / (len(user_ids) * len(item_ids))), "n_features": int(feats.shape[1]),
        "items_with_tags": int(sum(1 for m in item_ids if int(m) in per_movie)),
    }
    (out / "meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


# ---------------------------------------------------------------------------------- nạp lại
@dataclass
class Dataset:
    n_users: int
    n_items: int
    user: np.ndarray
    item: np.ndarray
    ts: np.ndarray
    split: np.ndarray
    item_ids: np.ndarray
    features: np.ndarray  # (n_items + 1, F), hàng 0 = padding
    genre_mat: np.ndarray  # (n_items + 1, G) dùng cho độ đa dạng
    meta: dict
    path: Path

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        path = Path(path)
        z = np.load(path / "interactions.npz")
        feats = np.load(path / "features.npy")
        fmeta = json.loads((path / "feature_meta.json").read_text(encoding="utf-8"))
        return cls(n_users=len(z["user_ids"]), n_items=len(z["item_ids"]), user=z["user"], item=z["item"], ts=z["ts"],
                   split=z["split"], item_ids=z["item_ids"], features=feats, genre_mat=feats[:, : len(fmeta["genres"])],
                   meta=json.loads((path / "meta.json").read_text(encoding="utf-8")), path=path)

    def sequences(self, max_split: int) -> list[np.ndarray]:
        """Chuỗi phim theo thời gian của từng người dùng, chỉ gồm tương tác có split <= max_split."""
        m = self.split <= max_split
        u, it = self.user[m], self.item[m]
        bounds = np.searchsorted(u, np.arange(self.n_users + 1))  # user đã được sắp tăng dần
        return [it[bounds[i]: bounds[i + 1]] for i in range(self.n_users)]

    def eval_cases(self, phase: str) -> tuple[np.ndarray, np.ndarray]:
        """(users, target): target = tương tác đầu tiên của người dùng trong cửa sổ val/test,
        chỉ lấy người dùng đã có lịch sử trước cửa sổ đó."""
        target_split = {"val": 1, "test": 2}[phase]
        hist_users = np.unique(self.user[self.split < target_split])
        m = self.split == target_split
        u, it = self.user[m], self.item[m]
        first = np.r_[True, u[1:] != u[:-1]]
        u, it = u[first], it[first]
        keep = np.isin(u, hist_users)
        return u[keep], it[keep]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--since-year", type=int, default=2020)
    ap.add_argument("--until-year", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(build(Path(a.raw), Path(a.out), a.since_year, until_year=a.until_year), indent=2))


if __name__ == "__main__":
    main()
