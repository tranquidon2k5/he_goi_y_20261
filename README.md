# CineRec — Hệ gợi ý phim (IT4613)

Web app xem phim có gợi ý cá nhân hoá, đồng thời là bệ thử cho các mô hình đề xuất.
Đặc tả: [docs/PRD — Hệ gợi ý phim (IT4613).md](docs/), giao diện: [docs/UI_design.html](docs/UI_design.html).

> **Trạng thái:** app đã chạy đủ 6 màn hình và vòng dữ liệu. **Thuật toán gợi ý chính chưa chốt** —
> app đang dùng một bản tạm (`HeuristicRecommender`: độ phổ biến + thể loại). Bản tạm này
> **không phải baseline** cho báo cáo.

## Cấu trúc

```
backend/   FastAPI + SQLAlchemy (SQLite cho dev, PostgreSQL cho production)
  app/recsys/base.py       ← interface Recommender: chỗ cắm thuật toán
  app/recsys/heuristic.py  ← bản tạm
  app/recsys/registry.py   ← đăng ký thuật toán
  app/recsys/service.py    ← ghép hàng trang chủ, cold-start, lọc, giải thích, cache
  scripts/                 ← nạp MovieLens, làm giàu TMDB, xuất dữ liệu huấn luyện
web/       Next.js 16 (App Router) — gọi API qua rewrite /api/* (cùng origin, cookie httpOnly)
data/raw/  MovieLens tải về (không commit)
```

## Chạy nhanh (dev, SQLite)

Cần: Python ≥ 3.11 + [uv](https://docs.astral.sh/uv/), Node ≥ 20.

```bash
# 1. dữ liệu — bản nhỏ để dev; bản chính là ml-32m (PRD mục 6)
mkdir -p data/raw && cd data/raw
curl -LO https://files.grouplens.org/datasets/movielens/ml-latest-small.zip && unzip ml-latest-small.zip
cd ../..

# 2. backend (cổng 8000)
cd backend
cp .env.example .env            # sửa CINEREC_ADMIN_EMAILS nếu muốn
uv sync
uv run python -m scripts.load_movielens --dir ../data/raw/ml-latest-small
uv run uvicorn app.main:app --port 8000 --reload

# 3. web (cổng 3000) — terminal khác
cd web
npm install
npm run dev
```

Mở http://localhost:3000, đăng ký. Đăng ký bằng email nằm trong `CINEREC_ADMIN_EMAILS` để thấy trang **Lab**.
Tài liệu API tự sinh: http://localhost:8000/docs.

### Dùng PostgreSQL và ML-32M

```bash
docker compose up -d db
# trong backend/.env:
# CINEREC_DATABASE_URL=postgresql+psycopg://cinerec:cinerec@localhost:5432/cinerec
uv run python -m scripts.load_movielens --dir ../data/raw/ml-32m --since-year 2015   # dùng COPY, cắt theo thời gian
```

### Poster, mô tả, trailer (TMDB)

```bash
# CINEREC_TMDB_API_KEY trong backend/.env
uv run python -m scripts.enrich_tmdb --limit 2000      # 2000 phim phổ biến nhất chưa có dữ liệu
```

Phim chưa có dữ liệu TMDB hiển thị poster thay thế dựng từ tiêu đề. Dữ liệu TMDB phải ghi nguồn theo điều khoản của TMDB.

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

## Cắm thuật toán gợi ý khi đã chốt

1. Viết lớp con của `Recommender` (`backend/app/recsys/base.py`) với 3 hàm:
   - `load(catalog, artifact_dir)` — nạp checkpoint, dựng chỉ mục (vd. FAISS);
   - `recommend(ctx, k)` — `ctx.history` là chuỗi tương tác theo thời gian (rating, đã xem, xem sau, tag), hợp với mô hình tuần tự kiểu SASRec mà không cần huấn luyện lại cho người dùng mới;
   - `similar(movie_id, k, exclude)`, và tuỳ chọn `explain(ctx, movie_id)` cho dòng “Vì bạn đã xem X”.
2. Đăng ký trong `registry.py`.
3. Thêm một bản ghi `model_versions` với `recommender=<tên>` và `config={"artifact_dir": ...}`, hoặc gọi `POST /lab/retrain {"recommender": "<tên>"}`.
4. Trainer thật (fine-tune, cổng đánh giá NDCG@10, ghi `metrics`) cắm vào `_run_retrain` trong `backend/app/routers/lab.py`. Trang Lab tự hiển thị biểu đồ và bảng so sánh khi `metrics` có dữ liệu.

Tầng service đã lo: cold-start dạng switching (< 5 tương tác → “Thịnh hành” theo thể loại), lọc phim đã chấm / đã xem / “Không quan tâm”, cache theo người dùng và huỷ khi có tương tác mới, gắn `model_version` + `request_id` vào mọi danh sách, đo p95.

## Kiểm thử

```bash
cd backend && uv run pytest        # luồng đầy đủ: đăng ký → cold-start → chấm → gợi ý → xoá tài khoản; Lab
cd web && npx tsc --noEmit && npm run lint && npm run build
```
