# Nhật ký phát triển — CineRec (IT4613)

## 2026-11-14 — Dựng app, chưa làm thuật toán gợi ý

**Mục tiêu phiên:** đọc PRD + UI_design, xây toàn bộ app (giai đoạn 4 trong PRD mục 10). Thuật toán gợi ý chính **chưa chốt** nên chỉ dựng chỗ cắm và một bản tạm để app chạy được.

### Quyết định

| Quyết định | Lý do |
| --- | --- |
| Stack theo giả định PRD mục 11: Next.js + FastAPI + PostgreSQL | Chưa có yêu cầu khác |
| Backend chạy được cả SQLite (dev) lẫn Postgres | Docker chưa chạy trên máy; SQLite cho phép chạy ngay, không cần cài gì |
| Interface `Recommender` cắm-rút (`backend/app/recsys/base.py`) | Đổi thuật toán chỉ cần viết một lớp con + đăng ký vào `registry.py`, không sửa API hay UI |
| Người dùng biểu diễn bằng chuỗi tương tác (`UserContext.history`) | Khớp hướng tuần tự (SASRec) ở PRD mục 7; mô hình không tuần tự vẫn dùng được |
| Bản tạm `HeuristicRecommender` (độ phổ biến Bayes + mức hợp thể loại; phim tương tự = Jaccard thể loại) | Chỉ để app có gợi ý; **không phải baseline** của báo cáo |
| Cold-start, lọc, giải thích, cache, gắn `model_version`/`request_id` nằm ở tầng service | Độc lập với thuật toán, mọi mô hình sau này dùng lại |
| Web gọi API qua rewrite `/api/*` cùng origin | Cookie đăng nhập httpOnly hoạt động mà không cần CORS |
| Các trang là client component, CSS port thẳng từ `UI_design.html` | Giao diện tương tác dày như mock; giữ nguyên token màu |
| "Phim mới" = `source = "app"` (do quản trị thêm), không phải `rating_count = 0` | Khi cắt theo năm, nhiều phim cũ MovieLens có 0 rating — không phải phim mới |
| Script nạp MovieLens upsert phim, không xoá-rồi-chèn | Xoá phim sẽ cascade xoá rating/danh sách của người dùng app |

### Đã làm

**Backend (`backend/`)**
- Schema: `users`, `user_genres`, `movies` (movies.csv + links.csv + cột làm giàu TMDB + thống kê), `ratings`, `tags` (đúng trường MovieLens, thêm cột `source`), `watchlist`, `not_interested`, `watched`, `events`, `model_versions`. userId của app từ 1.000.001.
- API theo PRD mục 5: `/auth/*`, `/me` (sửa hồ sơ, xuất CSV, xoá tài khoản), `/movies` (tìm + lọc + 4 cách sắp xếp), `/movies/onboarding`, `/movies/{id}`, `/movies/{id}/similar`, `/ratings`, tag, `/me/{watchlist|not-interested|watched}`, `POST /events` (theo lô, idempotent theo `event_id`, nhận cả `sendBeacon`), `/recommendations?surface=home`, `/lab/models|retrain|rollback`.
- Mỗi thao tác ghi bảng trạng thái + một sự kiện, rồi huỷ cache gợi ý của người dùng → gợi ý đổi ngay ở lần tải trang kế tiếp.
- Lab: `retrain` hiện chỉ tính lại thống kê phim và tạo phiên bản mới (chạy nền); `rollback` về phiên bản đã qua cổng; thống kê p95, tương tác chờ học, CTR theo hàng (từ impression + detail_view).
- Scripts: `load_movielens.py` (pandas, COPY cho Postgres, `--since-year`, `--no-ratings`), `enrich_tmdb.py` (mô tả vi/en, poster, trailer, đạo diễn, diễn viên), `export_training.py` (xuất ratings/tags/movies/links/events đúng định dạng MovieLens).
- Test tích hợp `tests/test_flow.py` (3 test): đăng ký → cold-start → chấm 5 phim → có "Dành cho bạn" + lý do, không gợi ý lại phim đã chấm, "Không quan tâm" biến mất ngay, tag, xem sau, tìm kiếm, sự kiện idempotent, xoá rating, xoá tài khoản; Lab retrain + rollback; phim mới lên hàng "Mới thêm vào kho".

**Web (`web/`, Next.js 16.4)**
- 6 màn hình + đăng nhập/đăng ký: Onboarding, Trang chủ, Chi tiết phim, Khám phá, Hồ sơ, Lab (chỉ admin).
- Bảo vệ route: chưa đăng nhập → `/login`; chưa onboarding → `/onboarding`; Lab chỉ admin.
- Ghi sự kiện tiềm ẩn (`src/lib/events.ts`): gom lô, gửi mỗi 5 giây hoặc `sendBeacon` khi rời trang. Impression ghi một lần khi hàng hiện ≥ 40% trên màn hình; detail_view ghi hàng + vị trí nguồn.
- Chấm sao dùng được bằng bàn phím (←/→, Enter); poster thay thế dựng từ tiêu đề khi thiếu dữ liệu TMDB; nhãn thể loại tiếng Việt.

**Khác:** `docker-compose.yml` (Postgres 16), `backend/.env.example`, `.gitignore`, `README.md` (cách chạy + cách cắm thuật toán).

### Kiểm chứng

- `uv run pytest`: 3/3 pass. `tsc`, `eslint`, `next build`: sạch.
- Chạy thật với `ml-latest-small` (9.742 phim, 100.836 rating), đi qua cả 6 màn hình bằng Playwright:
  - Sau khi chấm Inception 5★, hero đổi sang Blade Runner 2049 với lý do "Vì bạn đã chấm Inception 5 sao — cùng Khoa học viễn tưởng", xuất hiện hàng "Vì bạn đã xem Inception".
  - Impression lưu đủ `row`, `request_id`, `model_version`; detail_view lưu `source_row`, `position`.
  - Lab: retrain v0.1.0 → v0.1.1, rollback hoạt động; p95 ≈ 17 ms (ngưỡng 300 ms).
  - Khổ 390px không tràn ngang.

### Lỗi gặp và đã sửa

- PowerShell 5.1 đọc file UTF-8 bằng mã ANSI làm hỏng dấu tiếng Việt ở 4 file → sửa bằng script Python; từ đó chỉ sửa file bằng Edit/Python.
- Console Windows (cp1252) làm script in tiếng Việt bị lỗi → `sys.stdout.reconfigure(encoding="utf-8")`.
- Next 16 bật `cacheComponents`: `usePathname` trong Providers chặn prerender → tách `RouteGuard`, bọc Header và các trang dùng `useSearchParams`/`useParams` trong `<Suspense>`.
- Lint `react-hooks/set-state-in-effect` → `useApi` tính `loading` từ khoá đã tải thay vì `setLoading` trong effect.
- Trang Khám phá: debounce + đồng bộ URL làm mất ký tự đang gõ → dùng ref ghi giá trị do chính trang đẩy lên URL.
- Chấm sao bằng phím: bấm nhanh đọc state cũ → functional update.
- Giao diện: số thứ hạng đè tên phim, tên lặp hai lần khi không có poster, hero trùng #1 "Thịnh hành", menu xuống dòng ở màn hẹp.
- `email-validator` từ chối tên miền `.local` → ví dụ admin đổi sang `admin@cinerec.vn`.

### Còn lại / việc tiếp theo

- [ ] **Chốt thuật toán** (PRD mục 7, mục 11) rồi viết `Recommender` thật; cắm trainer + cổng đánh giá NDCG@10 vào `_run_retrain` (`backend/app/routers/lab.py`, có TODO).
- [ ] Chạy thử với PostgreSQL + ML-32M (chưa kiểm vì Docker chưa chạy).
- [ ] Lấy khoá TMDB, chạy `scripts/enrich_tmdb.py` để có poster/mô tả/trailer.
- [ ] Sự kiện `watch` (số giây xem, tỉ lệ hoàn thành) cần YouTube IFrame API; hiện chỉ có `trailer_play` và nút "Đã xem".
- [ ] Trang quản trị thêm phim mới (hiện chỉ thêm được trực tiếp vào CSDL với `source = "app"`).
- [ ] Khởi tạo git repo và commit lần đầu.
