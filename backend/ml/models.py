"""Mô hình: họ SASRec (SASRec, gSASRec, eSASRec, CL4SRec, mô hình đề xuất) và XSimGCL.

Mọi biến thể tuần tự dùng chung `SeqRec`; khác nhau ở hàm mất mát, kiểu block,
đặc trưng nội dung và học tương phản (xem MODEL_SPECS trong train.py).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------- họ SASRec
class Block(nn.Module):
    """Block Transformer pre-norm. `ligr=True`: residual có cổng học được (LiGR, dùng trong eSASRec)."""

    def __init__(self, d: int, heads: int, drop: float, ligr: bool):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(drop), nn.Linear(d * 2, d))
        self.drop = nn.Dropout(drop)
        self.ligr = ligr
        if ligr:
            self.g1, self.g2 = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a = self.drop(self.attn(h, h, h, attn_mask=mask, need_weights=False)[0])
        x = x + (torch.sigmoid(self.g1(h)) * a if self.ligr else a)
        h = self.ln2(x)
        f = self.drop(self.ff(h))
        return x + (torch.sigmoid(self.g2(h)) * f if self.ligr else f)


class SeqRec(nn.Module):
    """Transformer một chiều dự đoán phim kế tiếp. Chỉ số 0 = padding, n_items + 1 = token mask (cho CL)."""

    def __init__(self, n_items: int, d: int = 64, blocks: int = 2, heads: int = 2, drop: float = 0.2,
                 maxlen: int = 200, ligr: bool = False, features: torch.Tensor | None = None, fuse: str = "add"):
        super().__init__()
        self.n_items, self.d, self.maxlen, self.heads = n_items, d, maxlen, heads
        self.item = nn.Embedding(n_items + 2, d, padding_idx=0)
        nn.init.normal_(self.item.weight, std=d ** -0.5)
        with torch.no_grad():
            self.item.weight[0].zero_()
        self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList(Block(d, heads, drop, ligr) for _ in range(blocks))
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
        self.fuse = fuse if features is not None else None
        if features is not None:
            f = torch.cat([features, torch.zeros(1, features.shape[1])])  # hàng cho token mask
            self.register_buffer("features", f, persistent=False)
            self.proj = nn.Linear(features.shape[1], d, bias=False)
            if fuse == "gate":
                self.gate = nn.Linear(2 * d, d)

    def item_repr(self, ids: torch.Tensor | None = None) -> torch.Tensor:
        """Biểu diễn phim = embedding ID (+ phép chiếu đặc trưng nội dung). Dùng chung cho đầu vào và đầu ra."""
        e = self.item.weight if ids is None else self.item(ids)
        if self.fuse is None:
            return e
        c = self.proj(self.features if ids is None else self.features[ids])
        if self.fuse == "gate":
            g = torch.sigmoid(self.gate(torch.cat([e, c], -1)))
            return g * e + (1 - g) * c
        return e + c

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq (B, L) đệm trái bằng 0 -> trạng thái ẩn (B, L, d)."""
        B, L = seq.shape
        pad = seq == 0
        x = self.item_repr(seq) * math.sqrt(self.d) + self.pos.weight[self.maxlen - L:]
        x = self.drop(x) * (~pad).unsqueeze(-1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), 1)
        mask = causal.unsqueeze(0) | pad.unsqueeze(1)  # chặn tương lai và padding
        mask = mask & ~torch.eye(L, dtype=torch.bool, device=seq.device)  # luôn thấy chính nó → không có hàng toàn -inf
        mask = mask.repeat_interleave(self.heads, 0)
        for b in self.blocks:
            x = b(x, mask)
        return self.ln(x)

    def user_vec(self, seq: torch.Tensor) -> torch.Tensor:
        return self.encode(seq)[:, -1]

    def score_all(self, seq: torch.Tensor) -> torch.Tensor:
        return self.user_vec(seq) @ self.item_repr()[: self.n_items + 1].T


def seq_loss(model: SeqRec, h: torch.Tensor, pos: torch.Tensor, kind: str, n_neg: int, t: float = 0.75) -> torch.Tensor:
    """h (B, L, d), pos (B, L) phim đích (0 = bỏ qua).

    bce  — SASRec gốc: 1 mẫu âm ngẫu nhiên cho mỗi vị trí.
    gbce — gSASRec: n_neg mẫu âm riêng cho mỗi vị trí, hệ số beta hiệu chỉnh độ quá tự tin (tham số t).
    ss   — sampled softmax với n_neg mẫu âm dùng chung trong batch (eSASRec, mô hình đề xuất).

    Hàm mất mát dạng sigmoid (bce, gbce) phải lấy mẫu âm riêng từng vị trí: nếu dùng chung, vài phim bị cả batch
    đẩy xuống cùng lúc và mô hình không học được (đã kiểm: gBCE 1 mẫu âm chung, beta = 1, val NDCG@10 0,0027 so với
    0,0426 của BCE mẫu âm riêng). Sampled softmax chuẩn hoá theo hàng nên dùng chung không sao.
    """
    m = pos > 0
    h, pos = h[m], pos[m]
    W = model.item_repr()
    sp = (h * W[pos]).sum(-1)
    N = model.n_items
    if kind == "bce":
        neg = torch.randint(1, N + 1, pos.shape, device=pos.device)
        sn = (h * W[neg]).sum(-1)
        return (-F.logsigmoid(sp) - F.logsigmoid(-sn)).mean()
    if kind == "gbce":
        neg = torch.randint(1, N + 1, (len(pos), n_neg), device=pos.device)
        sn = torch.einsum("pd,pkd->pk", h, W[neg]).masked_fill(neg == pos[:, None], -1e4)
        alpha = n_neg / (N - 1)
        beta = alpha * (t * (1 - 1 / alpha) + 1 / alpha)
        return (-beta * F.logsigmoid(sp) - F.logsigmoid(-sn).sum(-1)).mean()
    neg = torch.randint(1, N + 1, (n_neg,), device=pos.device)
    sn = h @ W[neg].T  # (P, n_neg)
    sn = sn.masked_fill(neg.unsqueeze(0) == pos.unsqueeze(1), float("-inf"))
    if kind == "ss":
        logits = torch.cat([sp.unsqueeze(1), sn], 1)
        return F.cross_entropy(logits, torch.zeros(len(sp), dtype=torch.long, device=sp.device))
    raise ValueError(kind)


def info_nce(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    """NT-Xent: cặp (a_i, b_i) dương, 2B - 2 mẫu còn lại trong batch là âm."""
    z = F.normalize(torch.cat([a, b]), dim=-1)
    sim = z @ z.T / tau
    n = len(a)
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(a.device)
    return F.cross_entropy(sim, target)


def augment(seqs: np.ndarray, mask_id: int, rng: np.random.Generator, crop: float, mask: float, reorder: float) -> np.ndarray:
    """Tăng cường của CL4SRec: mỗi chuỗi chọn ngẫu nhiên crop / mask / reorder. seqs (B, L) đệm trái."""
    B, L = seqs.shape
    out = np.zeros_like(seqs)
    ops = rng.integers(0, 3, B)
    for i in range(B):
        s = seqs[i][seqs[i] > 0]
        n = len(s)
        if n > 1:
            if ops[i] == 0:
                k = max(1, int(crop * n))
                st = rng.integers(0, n - k + 1)
                s = s[st: st + k]
            elif ops[i] == 1:
                k = int(mask * n)
                if k:
                    s = s.copy()
                    s[rng.choice(n, k, replace=False)] = mask_id
            else:
                k = max(2, int(reorder * n))
                st = rng.integers(0, n - k + 1)
                s = s.copy()
                s[st: st + k] = rng.permutation(s[st: st + k])
        out[i, L - len(s):] = s
    return out


def augment_gpu(seqs: torch.Tensor, mask_id: int, crop: float, mask: float, reorder: float) -> torch.Tensor:
    """Bản vector hoá của `augment` (chạy trên GPU). Mask chọn mỗi vị trí với xác suất `mask` thay vì đúng k vị trí."""
    B, L = seqs.shape
    dev = seqs.device
    n = (seqs > 0).sum(1)  # độ dài thật
    pos = torch.arange(L, device=dev).expand(B, L)
    first = L - n  # vị trí bắt đầu (đệm trái)
    op = torch.randint(0, 3, (B,), device=dev)
    # crop: giữ k phim liên tiếp, dồn về bên phải
    k = torch.clamp((crop * n).long(), min=1)
    st = first + (torch.rand(B, device=dev) * (n - k + 1).float()).long()
    src = st[:, None] + pos - (L - k)[:, None]
    cropped = torch.where(pos >= (L - k)[:, None], seqs.gather(1, src.clamp(0, L - 1)), torch.zeros_like(seqs))
    # mask
    m = (torch.rand(B, L, device=dev) < mask) & (seqs > 0)
    masked = seqs.masked_fill(m, mask_id)
    # reorder: xáo một đoạn liên tiếp dài k2
    k2 = torch.clamp((reorder * n).long(), min=2)
    st2 = first + (torch.rand(B, device=dev) * (n - k2 + 1).clamp(min=1).float()).long()
    inside = (pos >= st2[:, None]) & (pos < (st2 + k2)[:, None])
    key = torch.where(inside, st2[:, None].float() + torch.rand(B, L, device=dev) * k2[:, None].float(), pos.float())
    reordered = seqs.gather(1, key.argsort(1))
    out = torch.where((op == 0)[:, None], cropped, torch.where((op == 1)[:, None], masked, reordered))
    return torch.where((n > 1)[:, None], out, seqs)


# ----------------------------------------------------------------------------------- XSimGCL
class XSimGCL(nn.Module):
    """LightGCN + nhiễu ngẫu nhiên đều có dấu ở mỗi tầng; tương phản giữa tầng cuối và một tầng trung gian."""

    def __init__(self, n_users: int, n_items: int, adj: torch.Tensor, d: int = 64, layers: int = 2,
                 eps: float = 0.2, cl_layer: int = 1):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.emb = nn.Parameter(torch.empty(n_users + n_items + 1, d))
        nn.init.xavier_uniform_(self.emb)
        self.register_buffer("adj", adj, persistent=False)
        self.layers, self.eps, self.cl_layer = layers, eps, cl_layer

    def propagate(self, perturb: bool):
        ego = self.emb
        layers, cl = [], None
        for k in range(self.layers):
            ego = torch.sparse.mm(self.adj, ego)
            if perturb:
                ego = ego + torch.sign(ego) * F.normalize(torch.rand_like(ego), dim=-1) * self.eps
            layers.append(ego)
            if k == self.cl_layer - 1:
                cl = ego
        final = torch.stack(layers).mean(0)
        return final, cl

    def split(self, x: torch.Tensor):
        return x[: self.n_users], x[self.n_users:]  # phim: hàng 0 = padding

    @staticmethod
    def build_adj(users: np.ndarray, items: np.ndarray, n_users: int, n_items: int) -> torch.Tensor:
        n = n_users + n_items + 1
        r = np.concatenate([users, items + n_users])
        c = np.concatenate([items + n_users, users])
        deg = np.bincount(r, minlength=n).astype(np.float32)
        dinv = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
        v = dinv[r] * dinv[c]
        return torch.sparse_coo_tensor(np.vstack([r, c]), v, (n, n)).coalesce()
