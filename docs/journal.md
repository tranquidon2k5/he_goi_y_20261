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

- [x] **Chốt thuật toán** (PRD mục 7, mục 11) rồi viết `Recommender` thật; cắm trainer + cổng đánh giá NDCG@10 vào `_run_retrain` — xong 26/11.
- [ ] Chạy thử với PostgreSQL + ML-32M (chưa kiểm vì Docker chưa chạy).
- [ ] Lấy khoá TMDB, chạy `scripts/enrich_tmdb.py` để có poster/mô tả/trailer.
- [ ] Sự kiện `watch` (số giây xem, tỉ lệ hoàn thành) cần YouTube IFrame API; hiện chỉ có `trailer_play` và nút "Đã xem".
- [x] Trang quản trị thêm phim mới — xong 26/11.
- [x] Khởi tạo git repo và commit lần đầu.

## 2026-11-26 — Thuật toán, thí nghiệm, vòng học lại, quản trị phim

**Mục tiêu phiên:** làm giai đoạn 1–3 của PRD (dữ liệu, baseline, mô hình đề xuất + ablation), cắm mô hình vào app, nối vòng học lại thật, thêm use case quản trị thêm phim. Hạn: trong ngày.

**Trả lời PRD mục 11 (nhóm chốt hôm nay):** xem phim = trailer + nút "Đã xem"; tài nguyên = Google Colab (qua colab-mcp), không dùng GPU máy cá nhân; hướng tuần tự SASRec; rating ≥ 4 là dương; stack giữ nguyên; vài chục người dùng thật; giảng viên chưa duyệt đề tài — chạy thử hướng đề xuất xem có tốt không.

### Quyết định

| Quyết định | Lý do |
| --- | --- |
| Cắt ML-32M từ 2020 (2,21 triệu tương tác dương, 27.673 người dùng, 9.693 phim) | Bản từ 2015 (6,4 triệu) không vừa 8 mô hình × tune × 3 seed trên một T4 trong một buổi |
| Tự cài 8 mô hình trong một khung PyTorch, không dùng RecBole | Cùng một bộ đánh giá full-ranking theo thời gian; RecBole khó cài trên môi trường mới và có mặc định ẩn (PRD mục 9) |
| Tiền xử lý deterministic, chạy ở cả local và Colab, so md5 từng mảng | Chỉ phải chuyển checkpoint (vài MB) từ Colab về, không chuyển dữ liệu |
| Tune 4 lần thử TPE + 1 lần thử cấu hình mặc định chung cho mọi mô hình | Lượt đầu chỉ 4 lần thử TPE: eSASRec bốc phải learning rate thấp (val 0,0162), với cấu hình mặc định lên 0,0303 — so sánh lượt đầu đo độ may, không đo mô hình |
| Thêm mốc tham chiếu "phổ biến 365 ngày" vào báo cáo | Trên val nó đạt 0,030, ngang các mô hình học — chia theo thời gian toàn cục có độ trôi mạnh, cần mốc này để đọc kết quả |
| Phục vụ trên CPU (`SeqRecommender`), người dùng = chuỗi tương tác dương gần nhất | Người dùng mới không cần học lại (mức 1, PRD mục 5); p95 vẫn dưới 300 ms |
| Fine-tune chạy trong tiến trình con trên CPU; cổng = cửa sổ test MovieLens trong CSDL | Không chặn API; cả bản live lẫn bản mới đều chưa thấy cửa sổ này |
| Phim do quản trị thêm có movieId từ 10.000.001, chỉ sửa / xoá được phim nguồn app | Không trùng movieId MovieLens, không làm lệch tập huấn luyện |
| Phim mới chưa có trong từ điển mô hình → biểu diễn bằng phép chiếu nội dung; fine-tune nối vào từ điển với embedding ID = 0 | Đúng PRD mục 5 "Phim mới" |

### Đã làm

- `backend/ml/`: `data.py` (bộ dữ liệu đóng băng + đặc trưng nội dung), `models.py` (họ SASRec, LiGR, gBCE, sampled softmax, tăng cường CL trên GPU, XSimGCL), `train.py` (tune Optuna có tiếp tục + lần thử mặc định, huấn luyện cuối, đánh giá full-ranking + độ phủ / đa dạng / độ mới + MMR + tách nhóm, báo cáo + kiểm định t), `finetune.py` (mức 2 + cổng), `replay.py` (phát lại luồng), `export.py`, `tables.py`.
- App: `app/recsys/seqrec.py`, đăng ký `seqrec`; `_run_retrain` chạy fine-tune thật + cổng; `scripts/register_model.py`; Lab có nút "Fine-tune ngay" và bảng so sánh thí nghiệm, tách nhóm, MMR.
- Quản trị phim: `app/routers/admin.py` (`/admin/movies` thêm / sửa / xoá, tuỳ chọn lấy TMDB), trang web `/admin`, link "Quản trị" cho admin, chặn route.
- Test: `tests/test_seqrec_admin.py` (quản trị thêm phim end-to-end, phục vụ seqrec + giải thích, fine-tune qua cổng). Tổng 5/5 pass.

### Lỗi gặp và đã sửa

- `curl` của Git Bash trên Windows không tải được grouplens (SEC_E_UNTRUSTED_ROOT) → tải bằng httpx (certifi).
- Tăng cường CL chạy vòng Python trên 2 CPU của Colab: 19 s/epoch → vector hoá trên GPU.
- Lần chạy đầu: chia GPU với XSimGCL làm mô hình đề xuất chậm 37 s/epoch → độ dài chuỗi 200 → 100 cho mọi mô hình tuần tự.
- Tập cổng rỗng làm NDCG = NaN, JSON không serialize được → cổng không có ca đánh giá thì chặn.
- Hai phim thêm cùng một giây: hàng "Mới thêm vào kho" xếp theo id tăng dần → thêm thứ tự phụ id giảm dần.
- Script Python ghi file bằng chế độ text trên Windows đổi LF → CRLF, md5 với Colab lệch → chuẩn hoá LF cho `ml/`.
- **gSASRec không học (val ≈ 0,001):** mẫu âm dùng chung trong batch phá hàm mất mát sigmoid (kiểm chéo: β = 1, 1 mẫu âm chung → 0,0027 so với 0,0426 của BCE mẫu âm riêng) → gBCE dùng mẫu âm riêng từng vị trí như bài gốc, tune lại gSASRec từ đầu.
- **Tune 4 lần thử thuần ngẫu nhiên không công bằng** (eSASRec bốc toàn lr thấp: 0,0162) → thêm 1 lần thử cấu hình mặc định chung cho mọi mô hình (eSASRec lên 0,0303).
- Khởi động tiến trình nền trên Colab bằng `!nohup … &` giữ kernel tới khi tiến trình xong → chỉ dùng `%%bash` cho tiến trình nền.
- Rà soát độc lập phát hiện 4 lỗi, đã sửa + có test hồi quy:
  - fine-tune hỏng khi có phim quản trị thêm không rõ năm (NaN qua CSV);
  - xoá phim có id lớn nhất rồi thêm phim mới thì id bị dùng lại và thừa kế embedding cũ → bộ đếm `id_sequences` chỉ tăng;
  - nạp mô hình lỗi sau khi đã đổi trạng thái làm mất bản live → nạp thử trước, lỗi thì trả lại bản live cũ; fine-tune khi chưa có checkpoint seqrec bị từ chối;
  - PATCH phim với `title`/`genres` = null trả 500, không kiểm trùng `tmdb_id` → 422 / 409.

### Kết quả

Test NDCG@10 (3 seed, chi tiết ở [experiments.md](experiments.md)): gSASRec **0,0422** · SASRec gốc 0,0344 · ablation 3 (sampled softmax + nội dung) 0,0330 · XSimGCL 0,0264 · eSASRec 0,0250 · **mô hình đề xuất 0,0239** · ablation 3 + CL 0,0223 · ablation 2 0,0220 · CL4SRec 0,0205 · mốc phổ biến 365 ngày 0,0249.

- Mô hình đề xuất không vượt baseline mạnh nhất (−0,0184, p < 0,001). Học tương phản gây giảm có ý nghĩa ngay cả khi giữ nguyên cấu hình; đổi lại độ phủ 0,149 → 0,209 và độ mới tăng.
- App phục vụ gSASRec (val cao nhất). Cổng tính từ CSDL app tái lập đúng số test (0,0410). Fine-tune qua Lab: 74 s trên CPU, cổng 0,0407 → 0,0421, qua cổng; đã rollback về v1.0.0 (checkpoint đã báo cáo) sau khi thử.
- Phát lại luồng: fine-tune nâng gSASRec 0,0343 → 0,0385 (+12%); mô hình đề xuất không cải thiện.
- p95 API gợi ý 17,9 ms. Backend 5/5 test; web `tsc` / `lint` / `build` sạch.

### Còn lại / việc tiếp theo

- [ ] Tune đủ 30 lần thử (PRD mục 9) cho gSASRec, SASRec, ablation 3, mô hình đề xuất; thử gSASRec + đặc trưng nội dung.
- [ ] Sampled softmax: mẫu âm riêng / in-batch + hiệu chỉnh logQ; học tương phản với λ nhỏ hơn.
- [ ] Đối chiếu baseline với số công bố trên ML-1M; vector mô tả TMDB (cần khoá TMDB).
- [ ] Sự kiện `watch` (YouTube IFrame API); chạy thử PostgreSQL.
- [ ] Hỏi giảng viên duyệt hướng đề tài với kết quả hiện tại.


## 2026-11-26 (chiều) — Cải tiến: đề xuất v2, xếp hạng lại, xác nhận trên lát cắt khác

**Mục tiêu:** tìm hướng để mô hình đề xuất ít nhất bằng gSASRec.

| Quyết định | Lý do |
| --- | --- |
| v2 = gSASRec + nhánh nội dung có cổng + CL tuỳ chọn (λ ∈ {0; 0,001; 0,01}) | Theo số liệu v1: gBCE là nền tốt nhất, nội dung có ích, CL có hại |
| Một lần thử khởi tạo ấm từ gSASRec, cổng nghiêng về ID, chấm cả epoch 0 | Sàn validation = gSASRec, nên "ít nhất bằng" có cơ sở trên val |
| gSASRec được thêm 5 lần thử TPE | v2 thừa hưởng phần tune của gSASRec → giữ ngân sách bằng nhau |
| Xếp hạng lại theo độ phổ biến gần đây, chọn (số ngày, w) trên val bằng mô hình chỉ học trên train | Đúng bước xếp hạng lại của PRD mục 7; nhắm điểm yếu ở người dùng lịch sử ngắn |
| Xác nhận trên ML-32M 2017–2019 | Test 2020+ đã bị xem; cần lát cắt chưa dùng để kiểm v2 |
| Kết quả v1 giữ nguyên trong báo cáo, v2 ghi rõ là thiết kế sau khi xem kết quả | Tránh "chọn theo test" |

**Kết quả:**
- v2 test NDCG@10 0,0448 so với gSASRec 0,0422 (p = 0,11).
- Xác nhận 2017–2019: 0,0596 so với 0,0614 (p = 0,052).
- Vậy v2 **ngang** gSASRec. Độ phủ 0,272 so với 0,172; tune chọn λ = 0.
- Xếp hạng lại: v2 0,0554 (+0,0132 so với gSASRec, p < 0,001); độ phủ giảm 0,272 → 0,047.
- gSASRec rất nhạy cấu hình: chỉ 1/10 lần thử học tốt (0,047), 9 lần còn lại ~0,014.

**Lỗi gặp:** `torch.cuda.is_available()` trả True dù `CUDA_VISIBLE_DEVICES=""` (không còn GPU nào) → `device()` kiểm thêm `device_count()`.

**Đưa vào app:** v2 được đăng ký thành phiên bản v1.1.0 (live), vì có NDCG@10 validation cao nhất. NDCG@10 trên tập cổng tính từ CSDL app là 0,0424, khớp số test offline seed 0 (0,0426). gSASRec v1.0.0 vẫn trong lịch sử, rollback được. Xếp hạng lại **chưa** bật trong app vì làm độ phủ giảm mạnh — để nhóm quyết định.
