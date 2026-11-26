"""Huấn luyện + đánh giá + tune (PRD mục 7–9).

Giao thức:
- phase "val": huấn luyện trên train, dừng sớm theo NDCG@10 trên validation (dùng khi tune).
- phase "test": huấn luyện lại trên train + val với số epoch tốt nhất tìm được, chấm test đúng một lần.
- Xếp hạng trên toàn bộ danh mục, loại phim đã có trong lịch sử; không lấy mẫu âm khi đánh giá.
- Mục tiêu = tương tác đầu tiên của người dùng trong cửa sổ val/test.

    python -m ml.train tune  --data DIR --out results --models sasrec,gsasrec --trials 6
    python -m ml.train final --data DIR --out results --models sasrec,gsasrec --seeds 0,1,2
    python -m ml.train report --data DIR --out results
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml.data import Dataset
from ml.models import SeqRec, XSimGCL, augment_gpu, info_nce, seq_loss

# Mô hình = (họ, đặc tả cố định). Hyperparameter tune nằm ở `space()`.
MODEL_SPECS: dict[str, dict] = {
    # ablation của mô hình đề xuất (PRD mục 7)
    "sasrec": {"family": "seq", "loss": "bce", "label": "SASRec gốc (BCE, 1 mẫu âm) — ablation 1"},
    "sasrec_ss": {"family": "seq", "loss": "ss", "label": "+ sampled softmax — ablation 2"},
    "sasrec_ss_content": {"family": "seq", "loss": "ss", "content": True, "label": "+ đặc trưng nội dung — ablation 3"},
    "proposed": {"family": "seq", "loss": "ss", "content": True, "cl": True, "label": "Mô hình đề xuất (+ học tương phản)"},
    # ablation 4 có kiểm soát: đúng cấu hình tốt nhất của ablation 3, chỉ thêm học tương phản (cmd_ablate)
    "sasrec_ss_content_cl": {"family": "seq", "loss": "ss", "content": True, "cl": True,
                             "label": "Ablation 3 + học tương phản (cùng cấu hình)"},
    # baseline (PRD mục 8)
    "cl4srec": {"family": "seq", "loss": "bce", "cl": True, "label": "CL4SRec (2022)"},
    "gsasrec": {"family": "seq", "loss": "gbce", "label": "gSASRec (2023)"},
    "esasrec": {"family": "seq", "loss": "ss", "ligr": True, "label": "eSASRec (2025)"},
    # PATCH_V2 — thiết kế sau khi có kết quả v1: nền gBCE (thành phần tốt nhất) + nội dung (có ích) + CL tuỳ chọn
    "proposed_v2": {"family": "seq", "loss": "gbce", "content": True, "cl": True, "warm": "gsasrec",
                    "label": "Đề xuất v2: gSASRec + nội dung (+ học tương phản tuỳ chọn)"},
    "xsimgcl": {"family": "gcl", "label": "XSimGCL (2023)"},
}
BASELINES = ["cl4srec", "gsasrec", "esasrec", "xsimgcl"]
# Cố định cho mọi mô hình tuần tự để so sánh công bằng và vừa ngân sách thời gian
SEQ_FIXED = {"d": 64, "blocks": 2, "heads": 2, "maxlen": 100, "batch": 256}
GCL_FIXED = {"d": 64, "batch": 8192, "reg": 1e-4, "cl_nodes": 4096}
CL_AUG = {"crop": 0.6, "mask": 0.3, "reorder": 0.6}
# Lần thử "mặc định" chung, xếp hàng thêm cho MỌI mô hình (giá trị theo bài báo gốc) — với ít lần thử,
# chỉ bốc ngẫu nhiên thì một mô hình có thể thua chỉ vì không bốc trúng learning rate hợp lý.
ANCHOR = {"lr": 1e-3, "dropout": 0.2, "wd": 0.0, "n_neg": 256, "t": 0.75, "cl_weight": 0.1, "tau": 1.0, "fuse": "add"}
ANCHOR_GCL = {"lr": 1e-3, "layers": 2, "eps": 0.2, "cl_weight": 0.2, "tau": 0.2}
GATE_BIAS = 8.0  # khởi tạo ấm: sigmoid(8) ≈ 0,9997 → biểu diễn phim ≈ embedding ID của mô hình gốc
BASE_ROOT: Path | None = None  # thư mục cache mô hình gốc cho khởi tạo ấm / xếp hạng lại (đặt trong main)
RERANK_DAYS = (30, 90, 365)
RERANK_W = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


def space(trial, name: str) -> dict:
    spec = MODEL_SPECS[name]
    if spec["family"] == "gcl":
        return GCL_FIXED | {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "layers": trial.suggest_int("layers", 1, 3),
            "eps": trial.suggest_float("eps", 0.05, 0.3),
            "cl_weight": trial.suggest_float("cl_weight", 0.01, 0.5, log=True),
            "tau": trial.suggest_float("tau", 0.05, 1.0, log=True),
        }
    hp = SEQ_FIXED | {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "wd": trial.suggest_categorical("wd", [0.0, 1e-4, 1e-3, 1e-2]),
    }
    if spec["loss"] == "gbce":
        hp["n_neg"] = trial.suggest_int("n_neg", 16, 256, log=True)
        hp["t"] = trial.suggest_float("t", 0.0, 1.0)
    if spec["loss"] == "ss":
        hp["n_neg"] = trial.suggest_int("n_neg", 64, 1024, log=True)
    if spec.get("cl"):
        if spec.get("warm"):  # v2: học tương phản là tuỳ chọn, λ rất nhỏ hoặc tắt
            hp["cl_weight"] = trial.suggest_categorical("cl_weight", [0.0, 0.001, 0.01])
            hp["tau"] = 1.0
        else:
            hp["cl_weight"] = trial.suggest_float("cl_weight", 0.01, 0.5, log=True)
            hp["tau"] = trial.suggest_float("tau", 0.05, 1.0, log=True)
        hp |= CL_AUG
    if spec.get("content"):
        hp["fuse"] = "gate" if spec.get("warm") else trial.suggest_categorical("fuse", ["add", "gate"])
    return hp


def device() -> torch.device:
    # is_available() có thể True khi CUDA_VISIBLE_DEVICES="" đã ẩn mọi GPU → kiểm cả số thiết bị
    return torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")


def pad_last(seqs: list[np.ndarray], L: int) -> np.ndarray:
    out = np.zeros((len(seqs), L), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-L:]
        if len(s):
            out[i, L - len(s):] = s
    return out


def make_windows(seqs: list[np.ndarray], L: int) -> tuple[np.ndarray, np.ndarray]:
    """Cắt chuỗi dài thành các cửa sổ L+1 phim (từ cuối về đầu) để không bỏ phí lịch sử cũ."""
    X, Y = [], []
    for s in seqs:
        end = len(s)
        while end > 1:
            start = max(0, end - L - 1)
            w = s[start:end]
            x = np.zeros(L, np.int64)
            y = np.zeros(L, np.int64)
            x[L - len(w) + 1:] = w[:-1]
            y[L - len(w) + 1:] = w[1:]
            X.append(x)
            Y.append(y)
            if start == 0:
                break
            end = start + 1
    return np.stack(X), np.stack(Y)


# ------------------------------------------------------------------------------------- đánh giá
class EvalCtx:
    def __init__(self, ds: Dataset, phase: str, dev: torch.device):
        self.ds, self.phase, self.dev = ds, phase, dev
        hist_split = 0 if phase == "val" else 1
        self.users, self.targets = ds.eval_cases(phase)
        seqs = ds.sequences(hist_split)
        self.hist = [seqs[u] for u in self.users]
        self.hist_len = np.array([len(h) for h in self.hist])
        train_pop = np.bincount(ds.item[ds.split <= hist_split], minlength=ds.n_items + 1).astype(np.float64)
        self.pop = train_pop
        n_train_users = len(np.unique(ds.user[ds.split <= hist_split]))
        self.self_info = torch.tensor(-np.log2((train_pop + 1) / (n_train_users + 1)), dtype=torch.float32, device=dev)
        order = np.argsort(-train_pop[1:]) + 1
        bucket = np.zeros(ds.n_items + 1, dtype=np.int8)
        bucket[order[int(0.2 * ds.n_items): int(0.5 * ds.n_items)]] = 1
        bucket[order[int(0.5 * ds.n_items):]] = 2
        self.pop_bucket = bucket  # 0 head (20% phim phổ biến nhất), 1 mid (30%), 2 tail (50%)
        g = torch.tensor(ds.genre_mat, dtype=torch.float32, device=dev)
        self.genres = F.normalize(g, dim=-1)


@torch.no_grad()
def evaluate(score_fn, ctx: EvalCtx, item_emb: torch.Tensor | None = None, mmr: tuple[float, ...] = (),
             batch: int = 1024, full: bool = True) -> dict:
    """score_fn(idx: np.ndarray) -> (b, n_items + 1) điểm cho các ca đánh giá idx."""
    n = len(ctx.users)
    N = ctx.ds.n_items
    ranks = np.zeros(n, dtype=np.int64)
    tops = []
    mmr_tops = {lam: [] for lam in mmr}
    for st in range(0, n, batch):
        idx = np.arange(st, min(n, st + batch))
        s = score_fn(idx).float()
        s[:, 0] = float("-inf")
        lens = [len(ctx.hist[i]) for i in idx]
        rows = torch.as_tensor(np.repeat(np.arange(len(idx)), lens), device=s.device)
        cols = torch.as_tensor(np.concatenate([ctx.hist[i] for i in idx]).astype(np.int64), device=s.device)
        s[rows, cols] = float("-inf")
        tgt = torch.as_tensor(ctx.targets[idx].astype(np.int64), device=s.device)
        ts = s.gather(1, tgt[:, None])
        ranks[idx] = (s > ts).sum(1).cpu().numpy()
        if full:
            top = s.topk(100, dim=1).indices
            tops.append(top[:, :10])
            for lam in mmr:
                mmr_tops[lam].append(_mmr(s, top, item_emb, lam))
    hit10 = ranks < 10
    ndcg10 = np.where(hit10, 1 / np.log2(ranks + 2), 0.0)
    out = {
        "n_cases": int(n),
        "ndcg@10": float(ndcg10.mean()), "recall@10": float(hit10.mean()),
        "ndcg@20": float(np.where(ranks < 20, 1 / np.log2(ranks + 2), 0).mean()), "recall@20": float((ranks < 20).mean()),
        "mrr": float((1 / (ranks + 1)).mean()),
    }
    if not full:
        return out
    top10 = torch.cat(tops)
    out |= _beyond(top10, ctx)
    out["_per_user_ndcg10"] = ndcg10
    out["by_history_len"] = _bucketed(ndcg10, np.digitize(ctx.hist_len, [6, 21, 51]), ["1-5", "6-20", "21-50", ">50"])
    out["by_target_popularity"] = _bucketed(ndcg10, ctx.pop_bucket[ctx.targets], ["head", "mid", "tail"])
    for lam, lst in mmr_tops.items():
        t10 = torch.cat(lst)
        tgt = torch.as_tensor(ctx.targets.astype(np.int64), device=t10.device)
        pos = (t10 == tgt[:, None]).float()
        nd = (pos / torch.log2(torch.arange(2, 12, device=t10.device).float())).sum(1)
        out[f"mmr_{lam}"] = {"ndcg@10": float(nd.mean()), "recall@10": float(pos.sum(1).mean())} | _beyond(t10, ctx)
    return out


def _mmr(s: torch.Tensor, top: torch.Tensor, emb: torch.Tensor, lam: float) -> torch.Tensor:
    """Xếp hạng lại top-100 bằng MMR: (1 - lam) * độ liên quan - lam * độ giống lớn nhất với phim đã chọn."""
    rel = s.gather(1, top)
    rel = (rel - rel.min(1, keepdim=True).values) / (rel.max(1, keepdim=True).values - rel.min(1, keepdim=True).values + 1e-9)
    E = F.normalize(emb[top], dim=-1)  # (b, 100, d)
    sim = E @ E.transpose(1, 2)
    b = top.shape[0]
    chosen = torch.zeros_like(rel, dtype=torch.bool)
    maxsim = torch.zeros_like(rel)
    picks = []
    ar = torch.arange(b, device=s.device)
    for step in range(10):
        score = (1 - lam) * rel - lam * maxsim
        score = score.masked_fill(chosen, float("-inf"))
        j = score.argmax(1)
        picks.append(top[ar, j])
        chosen[ar, j] = True
        maxsim = torch.maximum(maxsim, sim[ar, j]) if step else sim[ar, j]
    return torch.stack(picks, 1)


def _beyond(top10: torch.Tensor, ctx: EvalCtx) -> dict:
    G = ctx.genres[top10]  # (n, 10, g)
    sim = G @ G.transpose(1, 2)
    ild = 1 - (sim.sum((1, 2)) - sim.diagonal(dim1=1, dim2=2).sum(1)) / 90
    return {
        "coverage@10": float(len(torch.unique(top10)) / ctx.ds.n_items),
        "diversity@10": float(ild.mean()),
        "novelty@10": float(ctx.self_info[top10].mean()),
    }


def _bucketed(vals: np.ndarray, bucket: np.ndarray, names: list[str]) -> dict:
    return {nm: {"n": int((bucket == i).sum()), "ndcg@10": float(vals[bucket == i].mean()) if (bucket == i).any() else None}
            for i, nm in enumerate(names)}


# ------------------------------------------------------------------------------------- huấn luyện
def train(name: str, hp: dict, ds: Dataset, phase: str, seed: int, max_epochs: int = 60, patience: int = 5,
          fixed_epochs: int | None = None, log=print, time_limit: float | None = None):
    """Trả (model, eval_ctx, best_val_ndcg, best_epoch, epoch_seconds)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    ctx = EvalCtx(ds, phase, dev)
    fn = _train_gcl if MODEL_SPECS[name]["family"] == "gcl" else _train_seq
    return fn(name, hp, ds, phase, seed, ctx, dev, max_epochs, patience, fixed_epochs, log, time_limit)


def _loop(step_epoch, eval_fn, model, max_epochs, patience, fixed_epochs, log, time_limit, eval_first=False):
    best, best_ep, best_state, bad = -1.0, 0, None, 0
    t0 = time.time()
    epochs = max_epochs if fixed_epochs is None else fixed_epochs
    if eval_first and fixed_epochs is None:  # khởi tạo ấm: điểm xuất phát cũng là một ứng viên (epoch 0)
        best = eval_fn()
        best_state = copy.deepcopy(model.state_dict())
        log(f"  epoch 0 (khởi tạo ấm) val ndcg@10 {best:.4f}")
    ep = 0
    for ep in range(1, epochs + 1):
        te = time.time()
        loss = step_epoch()
        if fixed_epochs is not None:
            log(f"  epoch {ep} loss {loss:.4f} ({time.time() - te:.1f}s)")
            continue
        nd = eval_fn()
        log(f"  epoch {ep} loss {loss:.4f} val ndcg@10 {nd:.4f} ({time.time() - te:.1f}s)")
        if nd > best:
            best, best_ep, bad = nd, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
        if time_limit and time.time() - t0 > time_limit:
            log("  dừng: hết thời gian cho phép của lần thử")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, best_ep, (time.time() - t0) / max(ep, 1)


def _train_seq(name, hp, ds, phase, seed, ctx, dev, max_epochs, patience, fixed_epochs, log, time_limit):
    spec = MODEL_SPECS[name]
    L = hp["maxlen"]
    seqs = ds.sequences(0 if phase == "val" else 1)
    X, Y = make_windows(seqs, L)
    X, Y = torch.as_tensor(X, device=dev), torch.as_tensor(Y, device=dev)
    feats = torch.as_tensor(ds.features) if spec.get("content") else None
    model = SeqRec(ds.n_items, d=hp["d"], blocks=hp["blocks"], heads=hp["heads"], drop=hp["dropout"], maxlen=L,
                   ligr=spec.get("ligr", False), features=feats, fuse=hp.get("fuse", "add")).to(dev)
    warm = bool(hp.get("warm") and spec.get("warm"))
    if warm:  # khởi tạo từ mô hình gốc đã huấn luyện (cùng phase, cùng seed), cổng nghiêng hẳn về embedding ID
        model.load_state_dict(ensure_base(spec["warm"], ds, phase, seed, log), strict=False)
        with torch.no_grad():
            model.gate.weight.zero_()
            model.gate.bias.fill_(GATE_BIAS)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    eval_in = torch.as_tensor(pad_last(ctx.hist, L), device=dev)
    bs = hp["batch"]

    def step_epoch():
        model.train()
        perm = torch.randperm(len(X), device=dev)
        tot, nb = 0.0, 0
        for i in range(0, len(X), bs):
            idx = perm[i: i + bs]
            x, y = X[idx], Y[idx]
            loss = seq_loss(model, model.encode(x), y, spec["loss"], hp.get("n_neg", 1), hp.get("t", 0.75))
            if spec.get("cl") and hp.get("cl_weight", 0) > 0:
                a1 = augment_gpu(x, ds.n_items + 1, hp["crop"], hp["mask"], hp["reorder"])
                a2 = augment_gpu(x, ds.n_items + 1, hp["crop"], hp["mask"], hp["reorder"])
                loss = loss + hp["cl_weight"] * info_nce(model.user_vec(a1), model.user_vec(a2), hp["tau"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        return tot / nb

    def scores(idx):
        return model.score_all(eval_in[torch.as_tensor(idx, device=dev)])

    def eval_fn():
        model.eval()
        return evaluate(scores, ctx, full=False)["ndcg@10"]

    best, best_ep, sec = _loop(step_epoch, eval_fn, model, max_epochs, patience, fixed_epochs, log, time_limit,
                               eval_first=warm)
    model.eval()
    model._scores = scores  # noqa: SLF001
    model._item_emb = lambda: model.item_repr()[: ds.n_items + 1]  # noqa: SLF001
    return model, ctx, best, best_ep, sec


def _train_gcl(name, hp, ds, phase, seed, ctx, dev, max_epochs, patience, fixed_epochs, log, time_limit):
    m = ds.split <= (0 if phase == "val" else 1)
    u, it = ds.user[m].astype(np.int64), ds.item[m].astype(np.int64)
    adj = XSimGCL.build_adj(u, it, ds.n_users, ds.n_items).to(dev)
    model = XSimGCL(ds.n_users, ds.n_items, adj, d=hp["d"], layers=hp["layers"], eps=hp["eps"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    U, I = torch.as_tensor(u, device=dev), torch.as_tensor(it, device=dev)
    bs = hp["batch"]
    cache: dict = {}

    def nce1(a, b, tau):
        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        return F.cross_entropy(a @ b.T / tau, torch.arange(len(a), device=a.device))

    def step_epoch():
        model.train()
        cache.clear()
        perm = torch.randperm(len(U), device=dev)
        tot, nb = 0.0, 0
        for i in range(0, len(U), bs):
            idx = perm[i: i + bs]
            bu, bi = U[idx], I[idx]
            bj = torch.randint(1, ds.n_items + 1, bi.shape, device=dev)
            final, cl = model.propagate(True)
            fu, fi = model.split(final)
            cu, ci = model.split(cl)
            bpr = -F.logsigmoid((fu[bu] * fi[bi]).sum(-1) - (fu[bu] * fi[bj]).sum(-1)).mean()
            ub, ib = torch.unique(bu), torch.unique(bi)
            cap = hp.get("cl_nodes", 4096)  # giới hạn kích thước ma trận tương phản
            ub = ub[torch.randperm(len(ub), device=dev)[:cap]]
            ib = ib[torch.randperm(len(ib), device=dev)[:cap]]
            cl_loss = nce1(fu[ub], cu[ub], hp["tau"]) + nce1(fi[ib], ci[ib], hp["tau"])
            e = model.emb
            reg = (e[bu].pow(2).sum() + e[ds.n_users + bi].pow(2).sum() + e[ds.n_users + bj].pow(2).sum()) / len(bu)
            loss = bpr + hp["cl_weight"] * cl_loss + hp["reg"] * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        return tot / nb

    def embeddings():
        if "e" not in cache:
            with torch.no_grad():
                cache["e"] = model.split(model.propagate(False)[0])
        return cache["e"]

    def scores(idx):
        fu, fi = embeddings()
        return fu[torch.as_tensor(ctx.users[idx].astype(np.int64), device=dev)] @ fi.T

    def eval_fn():
        model.eval()
        cache.clear()
        return evaluate(scores, ctx, full=False)["ndcg@10"]

    best, best_ep, sec = _loop(step_epoch, eval_fn, model, max_epochs, patience, fixed_epochs, log, time_limit)
    model.eval()
    cache.clear()
    model._scores = scores  # noqa: SLF001
    model._item_emb = lambda: embeddings()[1]  # noqa: SLF001
    return model, ctx, best, best_ep, sec


def ensure_base(base: str, ds: Dataset, phase: str, seed: int, log=print) -> dict:
    """state_dict của `base` với cấu hình tốt nhất, huấn luyện đúng phase + seed (val: chỉ train; test: train + val).
    Dùng cho khởi tạo ấm và xếp hạng lại; huấn luyện một lần rồi lưu cache trong BASE_ROOT."""
    p = BASE_ROOT / f"{base}_{phase}_s{seed}.pt"
    if p.exists():
        return torch.load(p, map_location=device())
    final = BASE_ROOT.parent / base / f"model_seed{seed}.pt"
    if phase == "test" and final.exists():  # checkpoint của `final` = đúng cấu hình, số epoch, seed, train + val
        return torch.load(final, map_location=device(), weights_only=False)["state_dict"]
    best = json.loads((BASE_ROOT.parent / base / "best.json").read_text(encoding="utf-8"))
    log(f"  huấn luyện mô hình gốc {base} ({phase}, seed {seed}, {best['best_epoch']} epoch)")
    m, *_ = train(base, best["hp"], ds, phase, seed, fixed_epochs=best["best_epoch"], log=lambda s: None)
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, p)
    return sd


def load_trained(name: str, hp: dict, ds: Dataset, phase: str, seed: int, state: dict):
    """Dựng mô hình + bộ đánh giá cho `phase` rồi nạp trọng số đã huấn luyện (không huấn luyện thêm)."""
    hp = {k: v for k, v in hp.items() if k != "warm"}
    model, ctx, *_ = train(name, hp, ds, phase, seed, fixed_epochs=0, log=lambda s: None)
    model.load_state_dict(state)
    if hasattr(model, "adj"):  # XSimGCL: xoá cache embedding cũ
        model.train()
        model.eval()
    return model, ctx


def recency_score(ds: Dataset, phase: str, days: int) -> torch.Tensor:
    """log(1 + số tương tác trong `days` ngày trước mốc của phase), chuẩn hoá z — chỉ dùng dữ liệu trước mốc."""
    cut = ds.meta["t_val"] if phase == "val" else ds.meta["t_test"]
    m = (ds.split < (1 if phase == "val" else 2)) & (ds.ts >= cut - days * 86400)
    v = np.log1p(np.bincount(ds.item[m], minlength=ds.n_items + 1).astype(np.float32))
    return torch.as_tensor((v - v.mean()) / (v.std() + 1e-9))


def reranked(score_fn, rec: torch.Tensor, w: float):
    """Xếp hạng lại (PRD mục 7): điểm mô hình chuẩn hoá z theo từng người dùng + w × độ phổ biến gần đây."""
    if w == 0:
        return score_fn

    def f(idx):
        s = score_fn(idx).float()
        s = (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-9)
        return s + w * rec.to(s.device)
    return f


# ------------------------------------------------------------------------------------- lệnh
def _logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def cmd_tune(a) -> None:
    import optuna

    ds = Dataset.load(a.data)
    for name in a.models.split(","):
        out = Path(a.out) / name
        log = _logger(out / "tune.log")
        log(f"=== tune {name}: {a.trials} lần thử TPE + {int(a.anchor)} lần thử mặc định, tối đa {a.max_epochs} epoch, "
            f"kiên nhẫn {a.patience}")
        tf = out / "trials.json"
        trials = json.loads(tf.read_text(encoding="utf-8")) if tf.exists() else []

        def objective(trial):
            hp = space(trial, name)
            log(f"trial {trial.number}: {json.dumps(hp)}")
            _, _, best, ep, sec = train(name, hp, ds, "val", seed=0, max_epochs=a.max_epochs, patience=a.patience,
                                        log=log, time_limit=a.trial_minutes * 60)
            trial.set_user_attr("best_epoch", ep)
            trials.append({"number": len(trials), "hp": hp, "val_ndcg@10": best, "best_epoch": ep, "sec_per_epoch": sec,
                           "kind": "tpe"})
            tf.write_text(json.dumps(trials, indent=2), encoding="utf-8")
            log(f"trial {trial.number} -> val ndcg@10 {best:.4f} (epoch {ep})")
            torch.cuda.empty_cache()
            return best

        done = [t | {"kind": "tpe"} for t in trials if t.get("kind", "tpe") == "tpe"][: a.trials]
        if len(done) < a.trials:
            # Tiếp tục một lần tune dở: TPE cùng seed sinh lại đúng chuỗi tham số cũ; báo lại kết quả cũ cho
            # các lần thử đã chạy rồi chỉ chạy phần còn thiếu.
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            for t in done:
                tr = study.ask()
                hp = space(tr, name)
                if any(abs(hp[k] - t["hp"][k]) > 1e-12 if isinstance(hp[k], float) else hp[k] != t["hp"][k] for k in hp):
                    raise RuntimeError(f"{name}: chuỗi tham số TPE không khớp lần chạy cũ — xoá {tf} để tune lại")
                study.tell(tr, t["val_ndcg@10"])
            trials[:] = done + [t for t in trials if t.get("kind") == "anchor"]
            study.optimize(objective, n_trials=a.trials - len(done))
        if a.anchor and not any(t.get("kind") == "anchor" for t in trials):
            if MODEL_SPECS[name].get("warm"):  # khởi tạo ấm từ cấu hình tốt nhất của mô hình gốc, chưa bật CL
                base = json.loads((Path(a.out) / MODEL_SPECS[name]["warm"] / "best.json").read_text(encoding="utf-8"))
                hp = base["hp"] | {"cl_weight": 0.0, "tau": 1.0, "fuse": "gate", "warm": True} | CL_AUG
            else:
                fixed = ANCHOR_GCL if MODEL_SPECS[name]["family"] == "gcl" else ANCHOR
                hp = space(optuna.trial.FixedTrial(fixed), name)
            log(f"trial mặc định: {json.dumps(hp)}")
            _, _, best_v, ep, sec = train(name, hp, ds, "val", seed=0, max_epochs=a.max_epochs, patience=a.patience,
                                          log=log, time_limit=a.trial_minutes * 60)
            trials.append({"number": len(trials), "hp": hp, "val_ndcg@10": best_v, "best_epoch": ep, "sec_per_epoch": sec,
                           "kind": "anchor"})
            tf.write_text(json.dumps(trials, indent=2), encoding="utf-8")
            log(f"trial mặc định -> val ndcg@10 {best_v:.4f} (epoch {ep})")
            torch.cuda.empty_cache()
        best = max(trials, key=lambda t: t["val_ndcg@10"])
        (out / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        log(f"=== best {name}: {best['val_ndcg@10']:.4f} {json.dumps(best['hp'])}")


def cmd_ablate(a) -> None:
    """Ablation có kiểm soát: lấy cấu hình tốt nhất của --base, thêm thành phần của mô hình đích với giá trị
    mặc định (ANCHOR), huấn luyện không trần thời gian, ghi trials.json / best.json để `final` dùng như mọi mô hình."""
    ds = Dataset.load(a.data)
    base = json.loads((Path(a.out) / a.base / "best.json").read_text(encoding="utf-8"))
    for name in a.models.split(","):
        out = Path(a.out) / name
        log = _logger(out / "tune.log")
        hp = dict(base["hp"])
        if MODEL_SPECS[name].get("cl"):
            hp |= {"cl_weight": ANCHOR["cl_weight"], "tau": ANCHOR["tau"]} | CL_AUG
        log(f"=== ablate {name} từ {a.base}: {json.dumps(hp)}")
        _, _, best_v, ep, sec = train(name, hp, ds, "val", seed=0, max_epochs=a.max_epochs, patience=a.patience, log=log)
        rec = {"number": 0, "hp": hp, "val_ndcg@10": best_v, "best_epoch": ep, "sec_per_epoch": sec, "kind": f"ablate:{a.base}"}
        (out / "trials.json").write_text(json.dumps([rec], indent=2), encoding="utf-8")
        (out / "best.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        log(f"=== {name}: val ndcg@10 {best_v:.4f} (epoch {ep})")


def cmd_final(a) -> None:
    ds = Dataset.load(a.data)
    for name in a.models.split(","):
        out = Path(a.out) / name
        log = _logger(out / "final.log")
        best = json.loads((out / "best.json").read_text(encoding="utf-8"))
        hp = best["hp"]
        epochs = best["best_epoch"] if hp.get("warm") else max(1, best["best_epoch"])  # khởi tạo ấm: 0 epoch hợp lệ
        for seed in (int(s) for s in a.seeds.split(",")):
            log(f"=== final {name} seed {seed}: train+val, {epochs} epoch")
            t0 = time.time()
            model, ctx, *_ = train(name, hp, ds, "test", seed=seed, fixed_epochs=epochs, log=log)
            train_sec = time.time() - t0
            emb = model._item_emb()  # noqa: SLF001
            t1 = time.time()
            res = evaluate(model._scores, ctx, item_emb=emb, mmr=(0.2,))  # noqa: SLF001
            res["eval_ms_per_user"] = (time.time() - t1) * 1000 / max(1, res["n_cases"])
            res["train_seconds"] = train_sec
            np.save(out / f"per_user_ndcg10_seed{seed}.npy", res.pop("_per_user_ndcg10"))
            (out / f"test_seed{seed}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            log(f"test seed {seed}: ndcg@10 {res['ndcg@10']:.4f} recall@10 {res['recall@10']:.4f} mrr {res['mrr']:.4f}")
            ck = {"name": name, "hp": hp, "spec": MODEL_SPECS[name], "epochs": epochs,
                  "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "n_items": ds.n_items}
            torch.save(ck, out / f"model_seed{seed}.pt")
            if seed == 0:
                torch.save(ck, out / "model.pt")
            del model
            torch.cuda.empty_cache()


def cmd_rerank(a) -> None:
    """Xếp hạng lại theo độ phổ biến gần đây: chọn (số ngày, w) trên val bằng mô hình chỉ học trên train (seed 0),
    rồi chấm test bằng checkpoint cuối của các seed có sẵn. Ghi results/<model>/rerank.json."""
    ds = Dataset.load(a.data)
    root = Path(a.out)
    for name in a.models.split(","):
        out = root / name
        log = _logger(out / "rerank.log")
        hp = json.loads((out / "best.json").read_text(encoding="utf-8"))["hp"]
        model, ctx = load_trained(name, hp, ds, "val", 0, ensure_base(name, ds, "val", 0, log))
        grid = {}
        for days in RERANK_DAYS:
            rec = recency_score(ds, "val", days)
            for w in RERANK_W:
                grid[f"{days}:{w}"] = evaluate(reranked(model._scores, rec, w), ctx, full=False)["ndcg@10"]
        pick = max(grid, key=lambda k: (grid[k], -float(k.split(":")[1])))  # hoà thì chọn w nhỏ hơn
        days, w = int(pick.split(":")[0]), float(pick.split(":")[1])
        log(f"=== {name}: chọn {days} ngày, w = {w} · val {grid['30:0.0']:.4f} -> {grid[pick]:.4f}")
        res = {"days": days, "w": w, "val_base": grid["30:0.0"], "val_rerank": grid[pick], "val_grid": grid, "test": []}
        del model
        for seed in range(3):
            f = out / f"model_seed{seed}.pt"
            f = f if f.exists() else (out / "model.pt" if seed == 0 else None)
            if f is None:
                continue
            ck = torch.load(f, map_location="cpu", weights_only=False)
            model, ctx = load_trained(name, hp, ds, "test", seed, ck["state_dict"])
            rr = evaluate(reranked(model._scores, recency_score(ds, "test", days), w), ctx)
            np.save(out / f"rerank_per_user_ndcg10_seed{seed}.npy", rr.pop("_per_user_ndcg10"))
            res["test"].append({"seed": seed} | {k: rr[k] for k in ("ndcg@10", "recall@10", "mrr", "coverage@10",
                                                                     "novelty@10", "by_history_len")})
            log(f"test seed {seed}: rerank ndcg@10 {rr['ndcg@10']:.4f}")
            del model
            torch.cuda.empty_cache()
        (out / "rerank.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


def cmd_confirm(a) -> None:
    """Xác nhận trên bộ dữ liệu khác: giữ nguyên siêu tham số đã chọn ở --source, chỉ chọn lại số epoch bằng
    dừng sớm trên val của bộ mới (seed 0). Ghi best.json để `final` / `report` dùng như thường."""
    ds = Dataset.load(a.data)
    for name in a.models.split(","):
        out = Path(a.out) / name
        log = _logger(out / "tune.log")
        hp = json.loads((Path(a.source) / name / "best.json").read_text(encoding="utf-8"))["hp"]
        log(f"=== confirm {name}: {json.dumps(hp)}")
        _, _, v, ep, sec = train(name, hp, ds, "val", seed=0, max_epochs=a.max_epochs, patience=a.patience, log=log)
        rec = {"number": 0, "hp": hp, "val_ndcg@10": v, "best_epoch": ep, "sec_per_epoch": sec, "kind": "confirm"}
        (out / "trials.json").write_text(json.dumps([rec], indent=2), encoding="utf-8")
        (out / "best.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        log(f"=== {name}: val ndcg@10 {v:.4f} (epoch {ep})")


def popularity_reference(ds: Dataset) -> dict:
    """Mốc tham chiếu không học: độ phổ biến (toàn bộ / 365 ngày trước mốc test), chấm cùng giao thức test."""
    ctx = EvalCtx(ds, "test", torch.device("cpu"))
    seen = ds.split <= 1
    out = {}
    for name, days in (("Phổ biến (toàn bộ train + val)", None), ("Phổ biến 365 ngày gần nhất", 365)):
        m = seen if days is None else seen & (ds.ts >= ds.meta["t_test"] - days * 86400)
        s = torch.as_tensor(np.bincount(ds.item[m], minlength=ds.n_items + 1).astype(np.float32))
        s = s + 1e-6 * torch.as_tensor(np.bincount(ds.item[seen], minlength=ds.n_items + 1).astype(np.float32))
        r = evaluate(lambda idx: s.expand(len(idx), -1).clone(), ctx)
        r.pop("_per_user_ndcg10")
        out[name] = r
    return out


def cmd_report(a) -> None:
    from scipy import stats

    root = Path(a.out)
    rows, per_user = {}, {}
    for name in MODEL_SPECS:
        runs = sorted((root / name).glob("test_seed*.json"))
        if not runs:
            continue
        rs = [json.loads(p.read_text(encoding="utf-8")) for p in runs]
        keys = ["ndcg@10", "recall@10", "ndcg@20", "recall@20", "mrr", "coverage@10", "diversity@10", "novelty@10"]
        agg = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in keys}
        agg["seeds"] = len(rs)
        agg["by_history_len"] = {b: float(np.mean([r["by_history_len"][b]["ndcg@10"] or 0 for r in rs])) for b in rs[0]["by_history_len"]}
        agg["by_target_popularity"] = {b: float(np.mean([r["by_target_popularity"][b]["ndcg@10"] or 0 for r in rs])) for b in rs[0]["by_target_popularity"]}
        agg["mmr_0.2"] = {k: float(np.mean([r["mmr_0.2"][k] for r in rs])) for k in rs[0]["mmr_0.2"]}
        agg["train_seconds"] = float(np.mean([r["train_seconds"] for r in rs]))
        agg["eval_ms_per_user"] = float(np.mean([r["eval_ms_per_user"] for r in rs]))
        best = json.loads((root / name / "best.json").read_text(encoding="utf-8"))
        agg["val_ndcg@10"], agg["hp"], agg["best_epoch"] = best["val_ndcg@10"], best["hp"], best["best_epoch"]
        trials = json.loads((root / name / "trials.json").read_text(encoding="utf-8"))
        agg["n_trials"] = len(trials)
        rows[name] = agg
        per_user[name] = np.mean([np.load(root / name / f"per_user_ndcg10_seed{p.stem.split('seed')[1]}.npy") for p in runs], 0)
    summary = {"models": rows}
    if "proposed" in rows:
        cands = [b for b in BASELINES if b in rows]
        if cands:
            strongest = max(cands, key=lambda b: rows[b]["ndcg@10"][0])
            t, p = stats.ttest_rel(per_user["proposed"], per_user[strongest])
            summary["significance"] = {"vs": strongest, "t": float(t), "p": float(p),
                                       "mean_diff": float((per_user["proposed"] - per_user[strongest]).mean())}
        for abl in ("sasrec", "sasrec_ss", "sasrec_ss_content"):
            if abl in rows:
                t, p = stats.ttest_rel(per_user["proposed"], per_user[abl])
                summary.setdefault("ablation_tests", {})[abl] = {"t": float(t), "p": float(p)}
    if "sasrec_ss_content_cl" in rows and "sasrec_ss_content" in rows:  # tác dụng riêng của học tương phản
        t, p = stats.ttest_rel(per_user["sasrec_ss_content_cl"], per_user["sasrec_ss_content"])
        summary["cl_controlled"] = {"t": float(t), "p": float(p), "mean_diff": float(
            (per_user["sasrec_ss_content_cl"] - per_user["sasrec_ss_content"]).mean())}
    if "proposed_v2" in rows and "gsasrec" in rows:
        t, p = stats.ttest_rel(per_user["proposed_v2"], per_user["gsasrec"])
        summary["significance_v2"] = {"vs": "gsasrec", "t": float(t), "p": float(p),
                                      "mean_diff": float((per_user["proposed_v2"] - per_user["gsasrec"]).mean())}
    rr = {}
    for name in rows:
        f = root / name / "rerank.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            seeds = [t["seed"] for t in r["test"]]
            base = [json.loads((root / name / f"test_seed{s}.json").read_text(encoding="utf-8"))["ndcg@10"] for s in seeds]
            rr[name] = {"days": r["days"], "w": r["w"], "val_base": r["val_base"], "val_rerank": r["val_rerank"],
                        "seeds": seeds, "test_base": float(np.mean(base)),
                        "test_rerank": float(np.mean([t["ndcg@10"] for t in r["test"]])),
                        "test_rerank_std": float(np.std([t["ndcg@10"] for t in r["test"]])),
                        "by_history_len": {b: float(np.mean([t["by_history_len"][b]["ndcg@10"] or 0 for t in r["test"]]))
                                           for b in r["test"][0]["by_history_len"]}}
    if rr:
        summary["rerank"] = rr
    # mô hình tốt nhất theo validation (dùng để chọn mô hình phục vụ app — không chọn theo test)
    if rows:
        summary["best_on_val"] = max(rows, key=lambda n: rows[n]["val_ndcg@10"])
    ds = Dataset.load(a.data)
    summary["data"] = ds.meta
    summary["labels"] = {k: v["label"] for k, v in MODEL_SPECS.items()}
    summary["reference"] = popularity_reference(ds)
    summary["protocol"] = {
        "split": "chia theo mốc thời gian toàn cục: 90% train, 5% val, 5% test",
        "target": "tương tác dương đầu tiên của người dùng trong cửa sổ test; lịch sử = mọi tương tác trước đó",
        "ranking": "xếp hạng trên toàn bộ danh mục, loại phim đã có trong lịch sử",
        "tuning": "Optuna TPE, cùng số lần thử cho mọi mô hình, dừng sớm theo NDCG@10 trên val",
        "final": "huấn luyện lại trên train + val với số epoch tốt nhất, trung bình qua các seed",
        "fixed": {"seq": SEQ_FIXED, "gcl": GCL_FIXED, "cl_aug": CL_AUG},
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {m: rows[k][m] for m in ("ndcg@10", "recall@10", "mrr")} for k in rows}, indent=1))
    print(json.dumps(summary.get("significance"), indent=1))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "ablate", "final", "rerank", "confirm", "report"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default=",".join(MODEL_SPECS))
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--trial-minutes", type=float, default=8.0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--anchor", action="store_true", help="thêm một lần thử cấu hình mặc định chung cho mỗi mô hình")
    ap.add_argument("--base", default="sasrec_ss_content", help="ablate: mô hình lấy cấu hình gốc")
    ap.add_argument("--source", default="results", help="confirm: thư mục kết quả lấy siêu tham số")
    a = ap.parse_args()
    global BASE_ROOT
    BASE_ROOT = Path(a.out) / "_base"
    {"tune": cmd_tune, "ablate": cmd_ablate, "final": cmd_final, "rerank": cmd_rerank, "confirm": cmd_confirm,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
