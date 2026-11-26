# CineRec — Hệ gợi ý phim (IT4613)

Web app xem phim có gợi ý cá nhân hoá, đồng thời là bệ thử cho các mô hình đề xuất.
Đặc tả: [docs/PRD — Hệ gợi ý phim (IT4613).md](docs/), giao diện: [docs/UI_design.html](docs/UI_design.html),
kết quả thí nghiệm: [docs/experiments.md](docs/experiments.md), nhật ký: [docs/journal.md](docs/journal.md).

> **Trạng thái (26/11/2026):** app chạy đủ 6 màn hình + trang Quản trị phim, phục vụ **đề xuất v2** (phiên bản v1.1.0, chọn theo NDCG@10 validation cao nhất), có fine-tune + cổng đánh giá + rollback ở trang Lab.
> Thí nghiệm (PRD mục 7–9, chi tiết: [docs/experiments.md](docs/experiments.md)):
> - Đề xuất v1 (SASRec + sampled softmax + nội dung + học tương phản) **thua** baseline mạnh nhất gSASRec (test NDCG@10 0,0239 so với 0,0422).
> - **Đề xuất v2** (gSASRec + đặc trưng nội dung, khởi tạo ấm) **ngang gSASRec** trên cả hai lát cắt thời gian (0,0448 so với 0,0422, p = 0,11; xác nhận 2017–2019: 0,0596 so với 0,0614, p = 0,052), với độ phủ danh mục cao hơn hẳn (0,272 so với 0,172).
> - Thêm bước xếp hạng lại theo độ phổ biến gần đây: v2 đạt **0,0554**, vượt gSASRec có ý nghĩa (p < 0,001), đổi bằng độ phủ thấp hơn.

## Cấu trúc

```
backend/   FastAPI + SQLAlchemy (SQLite cho dev, PostgreSQL cho production)
  app/recsys/base.py       ← interface Recommender: chỗ cắm thuật toán
  app/recsys/seqrec.py     ← phục vụ mô hình tuần tự (họ SASRec) từ checkpoint, suy luận CPU
  app/recsys/heuristic.py  ← bản tạm (độ phổ biến + thể loại), dùng khi chưa có checkpoint / làm dự phòng
  app/recsys/registry.py   ← đăng ký thuật toán
  app/recsys/service.py    ← ghép hàng trang chủ, cold-start, lọc, giải thích, cache
  app/routers/admin.py     ← quản trị thêm / sửa / xoá phim mới
  app/routers/lab.py       ← phiên bản mô hình, học lại (refresh / fine-tune + cổng đánh giá), rollback
  ml/                      ← thí nghiệm: tiền xử lý, mô hình, tune, đánh giá, fine-tune, phát lại luồng
  scripts/                 ← nạp MovieLens, làm giàu TMDB, xuất dữ liệu, đăng ký checkpoint
web/       Next.js 16 (App Router) — gọi API qua rewrite /api/* (cùng origin, cookie httpOnly)
data/      raw/ MovieLens tải về, processed/ bộ dữ liệu đóng băng (không commit)
```

## Chạy nhanh (dev, SQLite)

Cần: Python ≥ 3.11 + [uv](https://docs.astral.sh/uv/), Node ≥ 20. `uv sync` cài PyTorch bản CUDA 12.8 (chạy được cả trên CPU).

```bash
# 1. dữ liệu: ML-32M (PRD mục 6)
mkdir -p data/raw && cd data/raw
curl -LO https://files.grouplens.org/datasets/movielens/ml-32m.zip && unzip ml-32m.zip
cd ../..

# 2. backend (cổng 8000)
cd backend
cp .env.example .env            # sửa CINEREC_ADMIN_EMAILS nếu muốn
uv sync
uv run python -m scripts.load_movielens --dir ../data/raw/ml-32m --since-year 2020   # ~5 phút với SQLite
uv run uvicorn app.main:app --port 8000

# 3. web (cổng 3000) — terminal khác
cd web
npm install
npm run dev
```

Mở http://localhost:3000, đăng ký. Đăng ký bằng email nằm trong `CINEREC_ADMIN_EMAILS` để thấy **Quản trị** và **Lab**.
Tài liệu API tự sinh: http://localhost:8000/docs.

Không có checkpoint thì app dùng bản tạm `heuristic`. Để phục vụ mô hình đề xuất, xem mục **Đưa mô hình vào app** bên dưới.

### Dùng PostgreSQL

```bash
docker compose up -d db
# trong backend/.env:
# CINEREC_DATABASE_URL=postgresql+psycopg://cinerec:cinerec@localhost:5432/cinerec
uv run python -m scripts.load_movielens --dir ../data/raw/ml-32m --since-year 2020   # dùng COPY
```

### Poster, mô tả, trailer (TMDB)

```bash
# CINEREC_TMDB_API_KEY trong backend/.env
uv run python -m scripts.enrich_tmdb --limit 2000      # 2000 phim phổ biến nhất chưa có dữ liệu
```

Phim chưa có dữ liệu TMDB hiển thị poster thay thế dựng từ tiêu đề. Dữ liệu TMDB phải ghi nguồn theo điều khoản của TMDB.

## Thí nghiệm (PRD mục 6–9)

Toàn bộ nằm trong `backend/ml/`, chạy được ở local hoặc Colab (chỉ cần torch, numpy, pandas, scipy, optuna).
Kết quả và cách đọc: [docs/experiments.md](docs/experiments.md).

| Bước | Lệnh | Ghi chú |
| --- | --- | --- |
| Bộ dữ liệu đóng băng | `python -m ml.data --raw ../data/raw/ml-32m --out ../data/processed/ml32m_2020 --since-year 2020` | rating ≥ 4, k-core 10/5, chia thời gian 90/5/5, đặc trưng nội dung; deterministic |
| Tune | `python -m ml.train tune --data $D --out results --trials 4 --anchor --max-epochs 40 --patience 4 --trial-minutes 6` | 4 lần thử Optuna TPE + 1 cấu hình mặc định chung, cùng ngân sách cho 8 mô hình |
| Huấn luyện cuối + test | `python -m ml.train final --data $D --out results --seeds 0,1,2` | train + val, số epoch tốt nhất, chấm test một lần mỗi seed |
| Đề xuất v2 + xếp hạng lại | `python -m ml.train tune --data $D --out results --models gsasrec --trials 9 --anchor ...` rồi `--models proposed_v2 --trials 4 --anchor`; `python -m ml.train rerank --data $D --out results --models <các mô hình>` | v2 khởi tạo ấm từ gSASRec; xếp hạng lại chọn (số ngày, w) trên val |
| Xác nhận lát cắt khác | `python -m ml.data ... --since-year 2017 --until-year 2020`; `python -m ml.train confirm --data $D2 --out results_c --source results --models gsasrec,sasrec,proposed_v2` rồi `final` | giữ siêu tham số, chỉ chọn lại số epoch |
| Phát lại luồng | `python -m ml.replay --data $D --ckpt results/gsasrec/model.pt --out results/replay_gsasrec.json` | trình diễn vòng học (PRD mục 5) |
| Tổng hợp | `python -m ml.train report --data $D --out results` | bảng, kiểm định t, mốc tham chiếu → `results/summary.json` |

Mô hình (`--models`): `proposed_v2`, `proposed`, `sasrec`, `sasrec_ss`, `sasrec_ss_content` (ablation), `cl4srec`, `gsasrec`, `esasrec`, `xsimgcl` (baseline).

**Chạy trên Colab.** Repo có `.mcp.json` khai báo [colab-mcp](https://github.com/googlecolab/colab-mcp); khi dùng Claude Code, mở một notebook Colab (runtime GPU), chép `backend/ml/` vào `/content/cinerec/backend/ml`, tải ML-32M trong notebook và chạy các lệnh trên. Kết quả kéo về bằng `google.colab.files.download`.

## Đưa mô hình vào app

```bash
cd backend
# results/ từ bước thí nghiệm (vd. giải nén results.zip tải từ Colab)
# mô hình phục vụ = results/summary.json["best_on_val"] (chọn theo validation; hiện là đề xuất v2)
uv run python -m ml.export --data ../data/processed/ml32m_2020 --ckpt results/gsasrec/model.pt \
    --out artifacts/gsasrec-v1 --metrics results/gsasrec/test_seed0.json
uv run python -m scripts.register_model --artifact-dir artifacts/proposed-v2 --version v1.1.0 --live
mkdir -p artifacts/experiments && cp results/summary.json artifacts/experiments/   # bảng so sánh trên trang Lab
```

`register_model` chấm NDCG@10 của checkpoint trên tập cổng (cửa sổ test MovieLens trong CSDL), ghi một bản ghi `model_versions`
(`recommender = "seqrec"`) và đưa lên phục vụ; khởi động lại API hoặc bấm "Làm mới thống kê" ở Lab để nạp.

## Vòng học lại (trang Lab)

| Mức (PRD mục 5) | Cách chạy | Làm gì |
| --- | --- | --- |
| 1. Tức thời | tự động | người dùng = chuỗi tương tác gần nhất; mỗi tương tác mới huỷ cache, lần tải trang kế tiếp đã đổi gợi ý |
| 2. Fine-tune | nút **Fine-tune ngay** | tiến trình con `ml.finetune` trên CPU: dữ liệu mới của app + replay 3.000 người dùng MovieLens, 2 epoch; phim mới được nối vào từ điển; **cổng**: NDCG@10 trên cửa sổ test MovieLens không giảm quá 5% so với bản đang chạy → mới triển khai, ngược lại ghi `fail` |
| 3. Huấn luyện lại toàn bộ | offline trên GPU | `ml.train` rồi `scripts.register_model` |

Mọi lần chạy ghi vào `model_versions`; rollback về bất kỳ phiên bản "qua cổng" nào bằng một nút.

## Quản trị phim mới

Trang **Quản trị** (`/admin`, chỉ admin): thêm / sửa / xoá phim do quản trị thêm (`source = "app"`, movieId từ 10.000.001).
Nhập tên, năm, thể loại (bắt buộc — mô hình dùng để xếp hạng phim mới), mô tả, đạo diễn, diễn viên, mã trailer YouTube;
có TMDB id và khoá TMDB thì lấy tự động mô tả / poster / trailer / đạo diễn / diễn viên.

Sau khi lưu, dịch vụ gợi ý nạp lại danh mục: phim lên đầu hàng “Mới thêm vào kho” ngay; mô hình tuần tự xếp hạng nó nhờ
embedding nội dung (thể loại + thập niên) cho tới lần fine-tune kế tiếp, khi phim được nối vào từ điển và học embedding ID
từ tương tác. Phim MovieLens không sửa / xoá được để không lệch tập huấn luyện.

API: `GET/POST /admin/movies`, `PATCH/DELETE /admin/movies/{id}`.

## Dữ liệu app thu được

| Lớp | Bảng | Ghi chú |
| --- | --- | --- |
| Tường minh, trùng schema MovieLens | `ratings`, `tags`, `movies` (+ `links`) | cột `source` = `movielens` \| `app`; userId của app từ 1.000.001 |
| Trạng thái | `watchlist`, `not_interested`, `watched`, `user_genres` | |
| Tiềm ẩn | `events` | `impression` (hàng, vị trí, request_id, model_version), `detail_view` (hàng + vị trí nguồn), `trailer_play`, `search`, `watchlist_add/remove`, `not_interested`, `mark_watched`, ... |
| Mô hình | `model_versions` | phiên bản, trạng thái, metric, cấu hình |

Xuất ra đúng định dạng file MovieLens để nối vào tập huấn luyện:

```bash
uv run python -m scripts.export_training --out ../data/export          # chỉ dữ liệu app
uv run python -m scripts.export_training --out ../data/export --all    # cả MovieLens
```

## Cắm thêm thuật toán

1. Viết lớp con của `Recommender` (`backend/app/recsys/base.py`): `load(catalog, artifact_dir)`, `recommend(ctx, k)`,
   `similar(movie_id, k, exclude)`, tuỳ chọn `explain(ctx, movie_id)`. Ví dụ đầy đủ: `app/recsys/seqrec.py`.
2. Đăng ký trong `registry.py`.
3. Thêm bản ghi `model_versions` với `recommender=<tên>` và `config={"artifact_dir": ...}` (xem `scripts/register_model.py`).

Tầng service đã lo: cold-start dạng switching (< 5 tương tác → “Thịnh hành” theo thể loại), lọc phim đã chấm / đã xem / “Không quan tâm”, cache theo người dùng và huỷ khi có tương tác mới, gắn `model_version` + `request_id` vào mọi danh sách, đo p95.

## Kiểm thử

```bash
cd backend && uv run pytest        # luồng người dùng; quản trị thêm phim; phục vụ seqrec; fine-tune qua cổng; Lab
cd web && npx next typegen && npx tsc --noEmit && npm run lint && npm run build
```

`npx next typegen` sinh kiểu route (`LayoutProps`, ...) vào `.next/` — cần chạy trước `tsc` trên bản clone mới.
