"""In các bảng markdown cho docs/experiments.md từ results/summary.json (+ replay.json nếu có).

    python -m ml.tables --summary results/summary.json [--replay results/replay.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ORDER = ["proposed_v2", "proposed", "esasrec", "gsasrec", "cl4srec", "xsimgcl", "sasrec_ss_content_cl", "sasrec_ss_content", "sasrec_ss", "sasrec"]
MAIN = ["ndcg@10", "recall@10", "ndcg@20", "recall@20", "mrr"]
BEYOND = ["coverage@10", "diversity@10", "novelty@10"]


def f(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--replay")
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text(encoding="utf-8"))
    M, L = s["models"], s["labels"]
    names = [n for n in ORDER if n in M]
    best = {c: max(M[n][c][0] for n in names) for c in MAIN + BEYOND}

    def cell(n: str, c: str, d: int = 4) -> str:
        m, sd = M[n][c]
        v = f"{f(m, d)} ± {f(sd, d)}"
        return f"**{v}**" if m == best[c] else v

    print("### Bảng chính (test, trung bình ± độ lệch chuẩn qua 3 seed)\n")
    print("| Mô hình | " + " | ".join(MAIN) + " | val NDCG@10 | epoch |")
    print("| --- | " + " | ".join("---:" for _ in MAIN) + " | ---: | ---: |")
    for n in names:
        print(f"| {L[n]} | " + " | ".join(cell(n, c) for c in MAIN) + f" | {f(M[n]['val_ndcg@10'])} | {M[n]['best_epoch']} |")
    for n, r in s["reference"].items():
        print(f"| *Tham chiếu: {n}* | " + " | ".join(f(r[c]) for c in MAIN) + " | — | — |")

    print("\n### Ngoài độ chính xác (top-10)\n")
    print("| Mô hình | Độ phủ danh mục | Độ đa dạng | Độ mới | + MMR λ=0,2: NDCG@10 | đa dạng | độ phủ |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for n in names:
        r = M[n]["mmr_0.2"]
        print(f"| {L[n]} | {cell(n, 'coverage@10', 3)} | {cell(n, 'diversity@10', 3)} | {cell(n, 'novelty@10', 2)} | "
              f"{f(r['ndcg@10'])} | {f(r['diversity@10'], 3)} | {f(r['coverage@10'], 3)} |")
    for n, r in s["reference"].items():
        print(f"| *{n}* | {f(r['coverage@10'], 3)} | {f(r['diversity@10'], 3)} | {f(r['novelty@10'], 2)} | — | — | — |")

    print("\n### Tách theo nhóm (NDCG@10)\n")
    hb = list(M[names[0]]["by_history_len"])
    pb = list(M[names[0]]["by_target_popularity"])
    print("| Mô hình | " + " | ".join(f"lịch sử {b}" for b in hb) + " | " + " | ".join(f"phim đích {b}" for b in pb) + " |")
    print("| --- | " + " | ".join("---:" for _ in hb + pb) + " |")
    for n in names:
        print(f"| {L[n]} | " + " | ".join(f(M[n]["by_history_len"][b]) for b in hb) + " | "
              + " | ".join(f(M[n]["by_target_popularity"][b]) for b in pb) + " |")
    for n, r in s["reference"].items():
        print(f"| *{n}* | " + " | ".join(f(r["by_history_len"][b]["ndcg@10"] or 0) for b in hb) + " | "
              + " | ".join(f(r["by_target_popularity"][b]["ndcg@10"] or 0) for b in pb) + " |")

    print("\n### Kiểm định\n")
    if "significance" in s:
        g = s["significance"]
        print(f"- Đề xuất vs {L[g['vs']]}: chênh lệch NDCG@10 trung bình {f(g['mean_diff'])}, t = {f(g['t'], 2)}, p = {g['p']:.3g}")
    for k, v in s.get("ablation_tests", {}).items():
        print(f"- Đề xuất vs {L[k]}: t = {f(v['t'], 2)}, p = {v['p']:.3g}")
    if "significance_v2" in s:
        g = s["significance_v2"]
        print(f"- Đề xuất v2 vs gSASRec: chênh lệch {f(g['mean_diff'])}, t = {f(g['t'], 2)}, p = {g['p']:.3g}")

    if "rerank" in s:
        print("\n### Xếp hạng lại theo độ phổ biến gần đây (w, số ngày chọn trên val)\n")
        print("| Mô hình | số ngày | w | val trước → sau | test trước → sau | seed | lịch sử 1-5 sau |")
        print("| --- | ---: | ---: | --- | --- | ---: | ---: |")
        for n in names:
            r = s["rerank"].get(n)
            if r:
                print(f"| {L[n]} | {r['days']} | {r['w']} | {f(r['val_base'])} → {f(r['val_rerank'])} | "
                      f"{f(r['test_base'])} → {f(r['test_rerank'])} | {len(r['seeds'])} | {f(r['by_history_len']['1-5'])} |")

    print("\n### Siêu tham số tốt nhất\n")
    for n in names:
        hp = {k: v for k, v in M[n]["hp"].items() if k not in ("d", "blocks", "heads", "maxlen", "batch", "reg", "cl_nodes",
                                                                 "crop", "mask", "reorder")}
        hp = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in hp.items()}
        print(f"- {L[n]}: `{json.dumps(hp)}` · {M[n]['n_trials']} lần thử · huấn luyện cuối {M[n]['train_seconds']:.0f} s/seed")

    if a.replay and Path(a.replay).exists():
        r = json.loads(Path(a.replay).read_text(encoding="utf-8"))
        print("\n### Phát lại luồng\n")
        print("| Đoạn | Thời gian | Số ca | Tĩnh | Cập nhật | Fine-tune sau đoạn |")
        print("| --- | --- | ---: | ---: | ---: | --- |")
        for c in r["chunks"]:
            d0 = datetime.fromtimestamp(c["from"], timezone.utc).strftime("%d/%m")
            d1 = datetime.fromtimestamp(c["to"], timezone.utc).strftime("%d/%m/%Y")
            ft = f"{c['finetune_users']} người dùng, {c['finetune_seconds']} s" if "finetune_users" in c else "—"
            print(f"| {c['chunk']} | {d0}–{d1} | {c['cases']} | {f(c['static_ndcg@10'])} | {f(c['updated_ndcg@10'])} | {ft} |")
        print(f"\nTrung bình có trọng số các đoạn 1..K−1: tĩnh {f(r['weighted_static_ndcg@10'])}, "
              f"cập nhật {f(r['weighted_updated_ndcg@10'])}")


if __name__ == "__main__":
    main()
