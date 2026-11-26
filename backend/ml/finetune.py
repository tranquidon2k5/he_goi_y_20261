"""Fine-tune tăng dần + cổng đánh giá (PRD mục 5, mức 2). Chạy được trên CPU của server.

- Dữ liệu: tương tác dương mới của app + một mẫu người dùng MovieLens cũ (replay) để tránh quên.
- Phim mới thêm vào kho được nối vào từ điển; embedding ID của nó khởi tạo bằng 0, nên biểu diễn ban đầu
  = phép chiếu đặc trưng nội dung, rồi được học trong lần fine-tune này (PRD mục 5, "Phim mới").
- Cổng: mô hình đang chạy và mô hình mới được chấm trên cùng tập giữ riêng — cửa sổ test của MovieLens
  (cả hai chưa thấy) + phim dương cuối cùng của mỗi người dùng app. Mô hình mới chỉ qua cổng khi
  NDCG@10 không giảm quá `tolerance` (tương đối).

Gọi từ app/routers/lab.py qua subprocess để không chặn API:
    python -m ml.finetune --in artifacts/v1 --out artifacts/v2 --interactions positives.csv --new-movies new_movies.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.data import featurize
from ml.models import SeqRec, augment_gpu, info_nce, seq_loss
from ml.train import make_windows, pad_last


def load(artifact: Path):
    ckpt = torch.load(artifact / "model.pt", map_location="cpu", weights_only=False)
    item_ids = np.load(artifact / "item_ids.npy")
    feats = np.load(artifact / "features.npy")
    meta = json.loads((artifact / "feature_meta.json").read_text(encoding="utf-8"))
    data_meta = json.loads((artifact / "meta.json").read_text(encoding="utf-8"))
    return ckpt, item_ids, feats, meta, data_meta


def build_model(ckpt: dict, feats: np.ndarray, drop: float | None = None) -> SeqRec:
    hp, spec = ckpt["hp"], ckpt["spec"]
    m = SeqRec(ckpt["n_items"], d=hp["d"], blocks=hp["blocks"], heads=hp["heads"],
               drop=hp["dropout"] if drop is None else drop, maxlen=hp["maxlen"], ligr=spec.get("ligr", False),
               features=torch.as_tensor(feats) if spec.get("content") else None, fuse=hp.get("fuse", "add"))
    m.load_state_dict(ckpt["state_dict"], strict=False)
    return m


def extend_vocab(ckpt: dict, item_ids: np.ndarray, feats: np.ndarray, meta: dict, new_movies: pd.DataFrame):
    """Nối phim mới vào cuối từ điển. Trả (ckpt, item_ids, feats) mới; token mask dời xuống cuối."""
    new_movies = new_movies[~new_movies["movie_id"].isin(set(item_ids.tolist()))]
    if new_movies.empty:
        return ckpt, item_ids, feats, 0
    k = len(new_movies)
    n = ckpt["n_items"]
    sd = dict(ckpt["state_dict"])
    w = sd["item.weight"]  # (n + 2, d): 0 pad, 1..n phim, n + 1 mask
    sd["item.weight"] = torch.cat([w[: n + 1], torch.zeros(k, w.shape[1]), w[n + 1:]])
    # CSV đọc lại năm / thể loại trống thành NaN (NaN là "truthy" nên không dùng `y or 0` được)
    f_new = np.stack([featurize(g if isinstance(g, str) else "", 0 if pd.isna(y) else int(y), meta)
                      for g, y in zip(new_movies["genres"], new_movies["year"])])
    ck = dict(ckpt) | {"state_dict": sd, "n_items": n + k}
    return ck, np.concatenate([item_ids, new_movies["movie_id"].to_numpy(np.int64)]), np.vstack([feats, f_new]).astype(np.float32), k


def gate_cases(inter: pd.DataFrame, vocab: dict[int, int], t_test: int):
    """Tập cổng: (1) MovieLens — phim dương đầu tiên từ mốc test, lịch sử = mọi phim dương trước mốc;
    (2) app — phim dương cuối cùng của mỗi người dùng có >= 2 phim dương.
    Trả (hist_ml, tgt_ml, chuỗi ML trước mốc theo user, hist_app, tgt_app, chuỗi app dùng để fine-tune)."""
    inter = inter[inter["movie_id"].isin(vocab.keys())].sort_values(["user_id", "timestamp", "movie_id"], kind="stable")
    inter = inter.assign(idx=inter["movie_id"].map(vocab))
    ml = inter[inter["source"] == "movielens"]
    app = inter[inter["source"] == "app"]
    hist_g, tgt_g = [], []
    ml_before = {u: g["idx"].to_numpy() for u, g in ml[ml["timestamp"] < t_test].groupby("user_id")}
    for u, g in ml[ml["timestamp"] >= t_test].groupby("user_id"):
        if u in ml_before:
            hist_g.append(ml_before[u])
            tgt_g.append(int(g["idx"].iloc[0]))
    app_hist, app_tgt, app_train = [], [], []
    for _, g in app.groupby("user_id"):
        s = g["idx"].to_numpy()
        if len(s) >= 2:
            app_hist.append(s[:-1])
            app_tgt.append(int(s[-1]))
            app_train.append(s[:-1])
        else:
            app_train.append(s)
    return hist_g, np.array(tgt_g), ml_before, app_hist, np.array(app_tgt), app_train


def gate_eval(artifact: Path, inter: pd.DataFrame) -> dict:
    """NDCG@10 của một artifact trên tập cổng (dùng khi đăng ký checkpoint huấn luyện offline)."""
    ckpt, item_ids, feats, _, data_meta = load(artifact)
    model = build_model(ckpt, feats, drop=0.0)
    vocab = {int(m): i + 1 for i, m in enumerate(item_ids)}
    hist_g, tgt_g, _, app_hist, app_tgt, _ = gate_cases(inter, vocab, data_meta["t_test"])
    nd, n = ndcg10(model, hist_g, tgt_g, ckpt["hp"]["maxlen"])
    nd_app, n_app = ndcg10(model, app_hist, app_tgt, ckpt["hp"]["maxlen"])
    return {"gate_ndcg@10": None if np.isnan(nd) else nd, "gate_cases": n,
            "app_ndcg@10": None if np.isnan(nd_app) else nd_app, "app_cases": n_app}


@torch.no_grad()
def ndcg10(model: SeqRec, hist: list[np.ndarray], targets: np.ndarray, maxlen: int) -> tuple[float, int]:
    if len(targets) == 0:
        return float("nan"), 0
    model.eval()
    dev = next(model.parameters()).device
    x = torch.as_tensor(pad_last(hist, maxlen), device=dev)
    out = []
    for st in range(0, len(x), 512):
        s = model.score_all(x[st: st + 512])
        s[:, 0] = float("-inf")
        for i, h in enumerate(hist[st: st + 512]):
            s[i, torch.as_tensor(h.astype(np.int64), device=dev)] = float("-inf")
        t = torch.as_tensor(targets[st: st + 512].astype(np.int64), device=dev)
        rank = (s > s.gather(1, t[:, None])).sum(1).cpu().numpy()
        out.append(np.where(rank < 10, 1 / np.log2(rank + 2), 0.0))
    return float(np.concatenate(out).mean()), len(targets)


def train_steps(model: SeqRec, seqs: list[np.ndarray], ckpt: dict, epochs: int, lr_scale: float, log=print) -> None:
    """Huấn luyện tiếp `model` (đúng hàm mất mát của checkpoint) trên các chuỗi cho trước, learning rate thấp."""
    hp, spec = ckpt["hp"], ckpt["spec"]
    dev = next(model.parameters()).device
    seqs = [s for s in seqs if len(s) >= 2]
    if not seqs:
        return
    X, Y = make_windows(seqs, hp["maxlen"])
    X, Y = torch.as_tensor(X, device=dev), torch.as_tensor(Y, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"] * lr_scale, weight_decay=hp["wd"])
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), device=dev)
        tot, nb = 0.0, 0
        for i in range(0, len(X), hp["batch"]):
            x, y = X[perm[i: i + hp["batch"]]], Y[perm[i: i + hp["batch"]]]
            loss = seq_loss(model, model.encode(x), y, spec["loss"], hp.get("n_neg", 1), hp.get("t", 0.75))
            if spec.get("cl"):
                a1 = augment_gpu(x, ckpt["n_items"] + 1, hp["crop"], hp["mask"], hp["reorder"])
                a2 = augment_gpu(x, ckpt["n_items"] + 1, hp["crop"], hp["mask"], hp["reorder"])
                loss = loss + hp["cl_weight"] * info_nce(model.user_vec(a1), model.user_vec(a2), hp["tau"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        log(f"  epoch {ep + 1}: loss {tot / max(nb, 1):.4f} ({len(X)} cửa sổ)")
    model.eval()


def run(art_in: Path, art_out: Path, inter: pd.DataFrame, new_movies: pd.DataFrame, epochs: int = 2,
        lr_scale: float = 0.3, replay_users: int = 3000, tolerance: float = 0.05, seed: int = 0, log=print) -> dict:
    """inter: user_id, movie_id, timestamp, source ("movielens" | "app") — chỉ các tương tác dương."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    ckpt, item_ids, feats, meta, data_meta = load(art_in)
    ckpt, item_ids, feats, n_new = extend_vocab(ckpt, item_ids, feats, meta, new_movies)
    live_ext = build_model(ckpt, feats, drop=0.0)  # mô hình đang chạy, từ điển đã mở rộng (phim mới = nội dung)
    model = build_model(ckpt, feats)
    hp, spec = ckpt["hp"], ckpt["spec"]
    vocab = {int(m): i + 1 for i, m in enumerate(item_ids)}

    hist_g, tgt_g, ml_before, app_hist, app_tgt, app_train = gate_cases(inter, vocab, data_meta["t_test"])

    # ---- dữ liệu fine-tune: toàn bộ dữ liệu app (trừ phim giữ cho cổng) + mẫu replay MovieLens (< t_test)
    users = np.array(list(ml_before))
    replay = rng.choice(users, size=min(replay_users, len(users)), replace=False) if len(users) else []
    seqs = [ml_before[u] for u in replay] + app_train
    seqs = [s for s in seqs if len(s) >= 2]
    log(f"fine-tune: {len(seqs)} chuỗi ({len(app_train)} người dùng app, {len(replay)} replay), "
        f"{n_new} phim mới, {epochs} epoch")
    train_steps(model, seqs, ckpt, epochs, lr_scale, log)

    L = hp["maxlen"]
    live_ml, n_ml = ndcg10(live_ext, hist_g, np.array(tgt_g), L)
    cand_ml, _ = ndcg10(model, hist_g, np.array(tgt_g), L)
    live_app, n_app = ndcg10(live_ext, app_hist, np.array(app_tgt), L)
    cand_app, _ = ndcg10(model, app_hist, np.array(app_tgt), L)
    # cổng dựa trên cửa sổ test MovieLens; không có thì dựa trên dữ liệu app; không có cả hai → chặn
    if n_ml:
        gate, gate_live, basis = cand_ml, live_ml, "movielens_test_window"
    elif n_app:
        gate, gate_live, basis = cand_app, live_app, "app_last_item"
    else:
        gate, gate_live, basis = None, None, "none"
    passed = gate is not None and gate >= gate_live * (1 - tolerance)
    clean = lambda x: None if x is None or np.isnan(x) else float(x)  # noqa: E731
    metrics = {
        "gate_ndcg@10": clean(gate), "live_gate_ndcg@10": clean(gate_live), "gate_basis": basis,
        "gate_cases": n_ml, "ml_ndcg@10": clean(cand_ml), "live_ml_ndcg@10": clean(live_ml),
        "app_ndcg@10": clean(cand_app), "live_app_ndcg@10": clean(live_app), "app_cases": n_app,
        "passed": bool(passed), "tolerance": tolerance, "new_movies": n_new,
        "finetune_sequences": len(seqs), "seconds": round(time.time() - t0, 1),
    }
    log(f"cổng: live {live_ml:.4f} -> mới {cand_ml:.4f} ({n_ml} ca) · app {live_app:.4f} -> {cand_app:.4f} ({n_app} ca)"
        f" · {'QUA' if passed else 'KHÔNG QUA'}")

    art_out.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt | {"state_dict": {k: v.detach() for k, v in model.state_dict().items()},
                       "parent": str(art_in), "finetune": metrics}, art_out / "model.pt")
    np.save(art_out / "item_ids.npy", item_ids)
    np.save(art_out / "features.npy", feats)
    for f in ("feature_meta.json", "meta.json"):
        (art_out / f).write_text((art_in / f).read_text(encoding="utf-8"), encoding="utf-8")
    (art_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="art_in", required=True, type=Path)
    ap.add_argument("--out", dest="art_out", required=True, type=Path)
    ap.add_argument("--interactions", required=True, type=Path, help="parquet/csv: user_id, movie_id, timestamp, source")
    ap.add_argument("--new-movies", type=Path, help="csv: movie_id, genres, year")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--replay-users", type=int, default=3000)
    ap.add_argument("--tolerance", type=float, default=0.05)
    a = ap.parse_args()
    read = lambda p: pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)  # noqa: E731
    new = read(a.new_movies) if a.new_movies else pd.DataFrame(columns=["movie_id", "genres", "year"])
    m = run(a.art_in, a.art_out, read(a.interactions), new, epochs=a.epochs, replay_users=a.replay_users,
            tolerance=a.tolerance)
    print("RESULT " + json.dumps(m))


if __name__ == "__main__":
    main()
