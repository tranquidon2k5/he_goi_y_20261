# Báo cáo thí nghiệm — CineRec (IT4613)

Ngày chạy: 26/11/2026 · GPU: Tesla T4 16 GB (Google Colab) · Mã: `backend/ml/` · Số liệu gốc: `backend/artifacts/experiments/`

## Tóm tắt

**Cập nhật sau cải tiến (mục 6).** Đề xuất v2 = gSASRec + đặc trưng nội dung, khởi tạo từ gSASRec. Kết quả:
- **Độ chính xác ngang gSASRec trên cả hai lát cắt thời gian.**
  - Bộ chính: test NDCG@10 0,0448 so với 0,0422 (p = 0,11).
  - Lát cắt 2017–2019: 0,0596 so với 0,0614 (p = 0,052).
- **Độ phủ danh mục và độ mới cao hơn hẳn:** 0,272 so với 0,172.
- **Thêm bước xếp hạng lại theo độ phổ biến gần đây (PRD mục 7):** v2 lên 0,0554, **vượt gSASRec có ý nghĩa thống kê** (+0,0132, p < 0,001), nhưng đổi bằng độ phủ thấp đi nhiều.
- **Tune đã loại học tương phản** (λ = 0).

Phần dưới đây là kết quả của mô hình đề xuất ban đầu (v1), giữ nguyên.

**Mô hình đề xuất v1 không vượt baseline mạnh nhất.** Trên tập test chia theo thời gian, mô hình đề xuất (SASRec + sampled softmax + đặc trưng nội dung + học tương phản) đạt NDCG@10 **0,0239 ± 0,0019**. gSASRec đạt **0,0422 ± 0,0010**. Chênh lệch theo người dùng là −0,0184 (kiểm định t cặp đôi, p < 0,001). Mô hình đề xuất cũng thấp hơn SASRec gốc (0,0344), và không khác có ý nghĩa với eSASRec (p = 0,57) hay XSimGCL (p = 0,31).

| Câu hỏi | Kết quả (test NDCG@10, 3 seed) |
| --- | --- |
| Mô hình tốt nhất | gSASRec 0,0422 — cao nhất ở mọi metric độ chính xác; cũng có val cao nhất (0,0470) |
| Sampled softmax thay BCE (ablation 1 → 2) | **giảm** 0,0344 → 0,0220 (p < 0,001) |
| Thêm đặc trưng nội dung (2 → 3) | **tăng** 0,0220 → 0,0330 (p < 0,001) |
| Thêm học tương phản, giữ nguyên cấu hình (3 → 3 + CL) | **giảm** 0,0330 → 0,0223 (p < 0,001) |
| Học tương phản đổi lại được gì | độ phủ danh mục 0,149 → **0,209**, độ mới 4,80 → **5,86**; NDCG@10 với phim đích độ phổ biến trung bình 0,0063 → **0,0124** |
| Fine-tune theo luồng (PRD mục 5) | gSASRec 0,0343 → **0,0385** (+12%); mô hình đề xuất 0,0242 → 0,0235 (không cải thiện) |
| Mốc không học: phổ biến 365 ngày | 0,0249 — CL4SRec, ablation 2, ablation 3 + CL và mô hình đề xuất **không vượt** mốc này |

**Mô hình chạy trong app là đề xuất v2** (v1.1.0). Nó được chọn vì có NDCG@10 validation cao nhất (0,0539); con số test không tham gia vào lựa chọn. Tập cổng tính lại từ CSDL của app cho NDCG@10 0,0424, khớp số test offline seed 0 (0,0426). Trước đó app chạy gSASRec (v1.0.0, cổng 0,0410 so với test 0,0410), vẫn rollback được. Fine-tune trên CPU qua trang Lab (thử với gSASRec) mất 74 giây và nâng NDCG@10 trên tập cổng từ 0,0407 lên 0,0421.

**Đọc kết quả thận trọng.**
- Mỗi mô hình chỉ có 5 lần thử tune (PRD dự kiến 30).
- Tập test có 2.664 ca, một mục tiêu mỗi người dùng.
- Hướng đề xuất ở PRD mục 7 dựa trên eSASRec, trong đó sampled softmax là phần "bắt buộc". Trên dữ liệu này, sampled softmax với mẫu âm đều dùng chung lại kém BCE. Kết luận này chỉ đúng với cách cài đặt và ngân sách ở đây.

## 1. Dữ liệu

MovieLens 32M, làm đúng các bước ở PRD mục 6 (`python -m ml.data`, chạy deterministic — bản trên Colab và bản local trùng md5 từng mảng).

| Bước | Kết quả |
| --- | --- |
| Rating gốc | 32.000.204 |
| Giữ rating ≥ 4.0, từ 01/01/2020 | 2.293.342 tương tác dương |
| k-core lặp (phim ≥ 10, người dùng ≥ 5) | 2.216.521 |
| Bỏ phim không có tương tác nào trong train | **2.212.576 tương tác · 27.673 người dùng · 9.693 phim** (mật độ 0,82%) |
| Chia theo mốc thời gian toàn cục | train 1.994.862 · val 109.504 (từ 24/04/2023) · test 108.210 (từ 29/07/2023) |
| Đặc trưng nội dung (326 chiều) | 19 thể loại multi-hot + 7 thập niên + TF-IDF của 300 tag phổ biến nhất (chỉ tag trước mốc val; 9.020/9.693 phim có tag) |

**Vì sao cắt từ 2020.** Bản từ 2015 có 6,4 triệu tương tác dương và 72 nghìn người dùng — không vừa ngân sách 8 mô hình × tune × 3 seed trong một buổi trên một GPU T4. Cắt theo thời gian (không lấy mẫu người dùng) giữ nguyên tính chất tuần tự và dải phim mới nhất mà app cần.

**Chưa có vector mô tả phim (sentence embedding).** PRD mục 6 dự kiến mã hoá mô tả TMDB; dự án chưa có khoá TMDB nên nhánh nội dung chỉ dùng thể loại, thập niên và tag.

## 2. Giao thức đánh giá

- **Ca đánh giá:** với mỗi người dùng có tương tác trong cửa sổ test và có lịch sử trước đó, mục tiêu là **tương tác dương đầu tiên** trong cửa sổ; đầu vào là toàn bộ lịch sử trước mốc test. Test có 2.664 ca, val có 2.964 ca.
- **Xếp hạng trên toàn bộ 9.693 phim**, loại phim đã có trong lịch sử; không lấy mẫu âm khi đánh giá (PRD mục 8).
- **Metric:** NDCG@10 (chính), Recall@10/20, NDCG@20, MRR; độ phủ danh mục@10, độ đa dạng@10 (1 − cosine thể loại trung bình giữa các cặp trong top-10), độ mới@10 (−log₂ độ phổ biến).
- **Quy trình:** tune trên train → val (dừng sớm theo NDCG@10 val); cấu hình tốt nhất được huấn luyện lại trên train + val với đúng số epoch tốt nhất, chấm test **một lần** cho mỗi seed; báo cáo trung bình ± độ lệch chuẩn qua 3 seed.
- **Kiểm định:** t cặp đôi theo người dùng trên NDCG@10 (trung bình 3 seed của từng người dùng) giữa mô hình đề xuất và baseline mạnh nhất.
- **Mốc tham chiếu không học:** độ phổ biến trên toàn bộ train + val, và độ phổ biến trong 365 ngày trước mốc test. Hai mốc này không nằm trong bảng so sánh của PRD nhưng cần để đọc kết quả: chia theo thời gian toàn cục có độ trôi mạnh (mục 6).

## 3. Mô hình

Tất cả cài đặt trong một khung PyTorch chung (`ml/models.py`, `ml/train.py`) để dùng chung dữ liệu, cách chia, vòng huấn luyện và bộ đánh giá. Họ SASRec dùng chung backbone Transformer pre-norm hai chiều đầu ra gắn trọng số với embedding phim; chuỗi dài được cắt thành các cửa sổ 100 phim để không bỏ phí lịch sử cũ.

| Mô hình | Vai trò | Cài đặt |
| --- | --- | --- |
| SASRec gốc | ablation 1 | BCE, 1 mẫu âm ngẫu nhiên mỗi vị trí |
| + sampled softmax | ablation 2 | cross-entropy trên mục tiêu + n mẫu âm đều dùng chung trong batch |
| + đặc trưng nội dung | ablation 3 | biểu diễn phim = embedding ID + chiếu tuyến tính đặc trưng nội dung (cộng, hoặc cổng học được) |
| **Mô hình đề xuất** | ablation 4 | + học tương phản: 2 view tăng cường (crop / mask / reorder) của mỗi chuỗi, InfoNCE, `L = L_ss + λ·L_InfoNCE` |
| CL4SRec (2022) | baseline | SASRec BCE + cùng mô-đun tương phản |
| gSASRec (2023) | baseline | gBCE: n mẫu âm, hệ số β hiệu chỉnh quá tự tin theo tham số t |
| eSASRec (2025) | baseline | sampled softmax + block LiGR (residual có cổng) |
| XSimGCL (2023) | baseline | LightGCN + nhiễu đều có dấu mỗi tầng, tương phản tầng cuối – tầng 1, BPR |

**Khác với bản gốc / PRD (ghi rõ để tái lập):**
- Không dùng RecBole: tự cài đặt để cùng một bộ đánh giá full-ranking theo thời gian; các mô hình là bản rút gọn theo bài báo, chưa đối chiếu số trên ML-1M (PRD mục 6, "bộ phụ") vì thiếu thời gian.
- eSASRec: chỉ lấy hai thành phần cốt lõi (sampled softmax + gated residual kiểu LiGR), không có các thủ thuật huấn luyện khác của bài.
- Sampled softmax (ablation 2–4, eSASRec, mô hình đề xuất) dùng mẫu âm chung trong batch để tiết kiệm bộ nhớ.
  gSASRec dùng mẫu âm **riêng từng vị trí** như bài gốc — xem mục "Sự cố gSASRec" bên dưới.
- XSimGCL: batch 8.192 và giới hạn 4.096 nút mỗi lượt tương phản để một epoch vừa thời gian; vẫn ~86 giây/epoch khi chia GPU nên mỗi lần thử chỉ chạy được vài epoch.
- Tăng cường của CL được vector hoá trên GPU; "mask" chọn từng vị trí với xác suất γ thay vì đúng ⌊γn⌋ vị trí.

**Sự cố gSASRec (đã sửa, tune lại).** Lượt tune đầu, gSASRec dùng mẫu âm chung trong batch và chỉ đạt val NDCG@10 ≈ 0,001 –
0,012 (ngang xếp hạng ngẫu nhiên / độ phổ biến), dù loss giảm đều. Kiểm tra chéo: với 1 mẫu âm dùng chung và t = 0 (β = 1,
về lý thuyết trùng SASRec BCE) val chỉ 0,0027, trong khi SASRec BCE với mẫu âm riêng từng vị trí đạt 0,0426. Nguyên nhân:
với hàm mất mát sigmoid, mẫu âm dùng chung khiến vài phim bị toàn bộ ~25 nghìn vị trí của batch đẩy xuống cùng một bước;
sampled softmax chuẩn hoá theo hàng nên không bị. gBCE đã chuyển sang mẫu âm riêng từng vị trí và gSASRec được tune lại
từ đầu với cùng ngân sách 4 + 1 lần thử. Kết quả lượt hỏng không được dùng.

## 4. Tuning

Mỗi mô hình có **cùng 5 lần thử**: 4 lần do Optuna TPE (seed 42) đề xuất + 1 lần thử **cấu hình mặc định chung**
lấy theo bài báo gốc (lr 1e-3, dropout 0,2, weight decay 0, 256 mẫu âm, t = 0,75, λ = 0,1, τ = 1, ghép nội dung kiểu cộng;
XSimGCL: lr 1e-3, 2 tầng, ε = 0,2, λ = 0,2, τ = 0,2). Mỗi lần thử tối đa 40 epoch, dừng sớm khi NDCG@10 val không tăng
sau 4 epoch, **trần 6 phút** (2 tiến trình chạy song song trên cùng một T4).

**Vì sao thêm lần thử mặc định.** Ở lượt đầu chỉ có 4 lần thử TPE — với ít lần thử như vậy TPE thực chất là bốc ngẫu nhiên.
eSASRec bốc phải 4 learning rate thấp (1,9e-4 … 5,6e-4), dừng sớm ở val 0,0162, trong khi mô hình đề xuất tình cờ bốc được
lr 7,9e-3 (val 0,0324). So sánh như vậy đo độ may khi bốc tham số chứ không đo mô hình. Lần thử mặc định được thêm cho **mọi**
mô hình (kể cả mô hình đề xuất); 4 lần thử TPE cũ được giữ nguyên, không bỏ kết quả nào (`ml.train tune --anchor`
tiếp tục đúng chuỗi tham số TPE cũ và kiểm tra khớp).

| Phạm vi | Cố định | Tìm kiếm |
| --- | --- | --- |
| Họ SASRec | d = 64, 2 block, 2 head, độ dài chuỗi 100, batch 256 | learning rate 1e-4…1e-2 (log), dropout 0,1…0,5, weight decay {0, 1e-4, 1e-3, 1e-2} |
| gSASRec | | số mẫu âm 16…256, t 0…1 |
| Sampled softmax | | số mẫu âm 64…1024 |
| Học tương phản | crop 0,6 · mask 0,3 · reorder 0,6 | λ 0,01…0,5, τ 0,05…1 |
| Nội dung | | cách ghép: cộng / cổng |
| XSimGCL | d = 64, batch 8.192, L2 1e-4 | learning rate, số tầng 1…3, ε 0,05…0,3, λ, τ |

**Lệch khỏi PRD mục 9 (do hạn một buổi):** 5 thay vì 30 lần thử (4 lần thử TPE gần như là tìm ngẫu nhiên), kiên nhẫn 4 thay vì 10, cố định kích thước embedding / số block / độ dài chuỗi thay vì tìm kiếm, thêm trần thời gian mỗi lần thử. Mọi mô hình chịu cùng các giới hạn này.

## 5. Kết quả

### 5.1 Bảng chính (test, trung bình ± độ lệch chuẩn qua 3 seed)

In đậm là giá trị tốt nhất của mỗi cột. Cột "val" là NDCG@10 validation của cấu hình được chọn; cột "epoch" là số epoch dùng khi huấn luyện lại trên train + val.

| Mô hình | NDCG@10 | Recall@10 | NDCG@20 | Recall@20 | MRR | val | epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Mô hình đề xuất** | 0,0239 ± 0,0019 | 0,0488 ± 0,0054 | 0,0325 ± 0,0022 | 0,0832 ± 0,0066 | 0,0241 ± 0,0011 | 0,0324 | 10 |
| gSASRec (2023) | **0,0422 ± 0,0010** | **0,0825 ± 0,0027** | **0,0520 ± 0,0008** | **0,1214 ± 0,0020** | **0,0382 ± 0,0006** | **0,0470** | 20 |
| XSimGCL (2023) | 0,0264 ± 0,0007 | 0,0537 ± 0,0016 | 0,0348 ± 0,0004 | 0,0870 ± 0,0009 | 0,0248 ± 0,0003 | 0,0270 | 11 |
| eSASRec (2025) | 0,0250 ± 0,0029 | 0,0479 ± 0,0048 | 0,0317 ± 0,0045 | 0,0744 ± 0,0113 | 0,0245 ± 0,0030 | 0,0303 | 16 |
| CL4SRec (2022) | 0,0205 ± 0,0006 | 0,0414 ± 0,0014 | 0,0282 ± 0,0007 | 0,0719 ± 0,0018 | 0,0214 ± 0,0006 | 0,0274 | 12 |
| *Ablation 1 — SASRec gốc (BCE)* | 0,0344 ± 0,0004 | 0,0704 ± 0,0012 | 0,0445 ± 0,0004 | 0,1106 ± 0,0012 | 0,0319 ± 0,0002 | 0,0426 | 11 |
| *Ablation 2 — + sampled softmax* | 0,0220 ± 0,0028 | 0,0402 ± 0,0031 | 0,0282 ± 0,0031 | 0,0649 ± 0,0053 | 0,0225 ± 0,0030 | 0,0242 | 16 |
| *Ablation 3 — + đặc trưng nội dung* | 0,0330 ± 0,0003 | 0,0669 ± 0,0028 | 0,0418 ± 0,0004 | 0,1019 ± 0,0037 | 0,0306 ± 0,0007 | 0,0443 | 16 |
| *Ablation 3 + học tương phản (cùng cấu hình)* | 0,0223 ± 0,0026 | 0,0444 ± 0,0062 | 0,0300 ± 0,0030 | 0,0753 ± 0,0074 | 0,0225 ± 0,0018 | 0,0341 | 17 |
| Tham chiếu: phổ biến (toàn bộ) | 0,0212 | 0,0405 | 0,0270 | 0,0634 | 0,0209 | — | — |
| Tham chiếu: phổ biến 365 ngày | 0,0249 | 0,0507 | 0,0320 | 0,0788 | 0,0236 | — | — |

Kiểm định t cặp đôi theo người dùng (NDCG@10, trung bình 3 seed của mỗi người dùng, 2.664 ca):

| So sánh | Chênh lệch | t | p |
| --- | ---: | ---: | ---: |
| Đề xuất − gSASRec (baseline mạnh nhất) | −0,0184 | −8,10 | < 0,001 |
| Đề xuất − eSASRec | −0,0011 | −0,57 | 0,57 |
| Đề xuất − XSimGCL | −0,0025 | −1,02 | 0,31 |
| Đề xuất − CL4SRec | +0,0034 | 1,98 | 0,048 |
| gSASRec − SASRec gốc | +0,0078 | 4,94 | < 0,001 |

### 5.2 Ablation

| Bước | NDCG@10 | So với bước trước | p |
| --- | ---: | ---: | ---: |
| 1. SASRec gốc (BCE, 1 mẫu âm) | 0,0344 | | |
| 2. + sampled softmax | 0,0220 | −0,0124 | < 0,001 |
| 3. + đặc trưng nội dung | 0,0330 | +0,0111 | < 0,001 |
| 4. + học tương phản — mô hình đề xuất (tune riêng) | 0,0239 | −0,0092 | < 0,001 |
| 4'. + học tương phản — giữ đúng cấu hình bước 3 | 0,0223 | −0,0108 | < 0,001 |

- **Sampled softmax (bước 2) đi ngược kết quả của bài eSASRec.** Có thể do cài đặt ở đây: mẫu âm đều dùng chung trong batch, không hiệu chỉnh logQ, và chỉ 5 lần thử (cấu hình tốt nhất là cấu hình mặc định). eSASRec dùng cùng hàm mất mát nên cũng thấp (0,0250).
- **Đặc trưng nội dung (bước 3)** bù lại gần hết phần mất ở bước 2. Đây là thành phần riêng của nhóm có tác dụng rõ nhất.
- **Học tương phản (bước 4)** làm giảm độ chính xác cả khi tune riêng lẫn khi giữ nguyên cấu hình, nên nguyên nhân không phải do trần thời gian hay do bốc tham số. Bước 4' chạy không có trần thời gian, 17 epoch.
- **MMR (λ = 0,2)** hầu như không đổi NDCG@10, độ đa dạng hay độ phủ ở mọi mô hình: với embedding phim của các mô hình này, top-100 vốn đã ít trùng nhau. Bước xếp hạng lại cần λ lớn hơn hoặc độ tương tự theo thể loại mới có tác dụng; chưa thử vì hết thời gian.

### 5.3 Ngoài độ chính xác (top-10)

| Mô hình | Độ phủ danh mục | Độ đa dạng | Độ mới |
| --- | ---: | ---: | ---: |
| Mô hình đề xuất | 0,146 | 0,573 | 4,78 |
| gSASRec | 0,172 | 0,605 | 4,82 |
| XSimGCL | 0,066 | 0,665 | 2,97 |
| eSASRec | 0,041 | 0,652 | 3,70 |
| CL4SRec | 0,167 | 0,590 | 4,82 |
| Ablation 1 / 2 / 3 | 0,192 / 0,028 / 0,149 | 0,614 / 0,667 / 0,586 | 4,90 / 3,59 / 4,80 |
| Ablation 3 + học tương phản | **0,209** | 0,567 | **5,86** |
| Phổ biến 365 ngày | 0,010 | 0,672 | 1,94 |

Các mô hình dùng sampled softmax với mẫu âm chung (eSASRec, ablation 2) và XSimGCL thiên mạnh về phim phổ biến: độ phủ thấp, độ mới thấp. Độ đa dạng thể loại cao của chúng chủ yếu do các phim phổ biến thuộc nhiều thể loại khác nhau. Học tương phản kéo danh sách về phía phim ít phổ biến hơn.

### 5.4 Cold-start: tách theo độ dài lịch sử và độ phổ biến của phim đích (NDCG@10)

| Mô hình | lịch sử 1–5 | 6–20 | 21–50 | > 50 | phim đích head (20%) | mid (30%) | tail (50%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mô hình đề xuất | 0,0475 | 0,0290 | 0,0303 | 0,0206 | 0,0320 | 0,0037 | 0,0000 |
| gSASRec | 0,0677 | **0,0479** | **0,0465** | **0,0395** | **0,0573** | 0,0035 | 0,0000 |
| XSimGCL | **0,0918** | 0,0301 | 0,0380 | 0,0206 | 0,0363 | 0,0000 | 0,0000 |
| eSASRec | 0,0410 | 0,0357 | 0,0260 | 0,0228 | 0,0342 | 0,0010 | 0,0000 |
| CL4SRec | 0,0547 | 0,0286 | 0,0273 | 0,0164 | 0,0279 | 0,0015 | 0,0000 |
| Ablation 3 | 0,0676 | 0,0359 | 0,0325 | 0,0319 | 0,0438 | 0,0063 | 0,0009 |
| Ablation 3 + học tương phản | 0,0294 | 0,0211 | 0,0292 | 0,0201 | 0,0276 | **0,0124** | 0,0004 |
| Phổ biến 365 ngày | 0,0773 | 0,0182 | 0,0296 | 0,0230 | 0,0343 | 0,0000 | 0,0000 |

- Hầu như mọi mô hình chỉ đoán trúng phim đích thuộc nhóm phổ biến. Phim đuôi dài (50% danh mục) gần như không bao giờ vào top-10.
- **Lịch sử ngắn (1–5 phim):** độ phổ biến và XSimGCL mạnh nhất. Điều này ủng hộ cách cold-start dạng switching của app ("Thịnh hành" khi < 5 tương tác).
- PRD kỳ vọng nội dung và học tương phản giúp nhiều nhất ở chuỗi ngắn và phim ít tương tác. Nội dung có giúp phim độ phổ biến trung bình (0,0063 so với 0,0010 của ablation 2). Học tương phản giúp phim độ phổ biến trung bình nhưng làm giảm ở chuỗi ngắn.

### 5.5 Vòng học lại: phát lại luồng (PRD mục 5)

Cửa sổ test chia 4 đoạn liên tiếp. Sau mỗi đoạn, mô hình "cập nhật" được fine-tune 2 epoch trên chuỗi của người dùng hoạt động trong đoạn đó, cộng 3.000 người dùng replay, với learning rate × 0,3. Đây đúng là quy trình `ml/finetune.py` mà app dùng. Mô hình "tĩnh" giữ nguyên checkpoint. Cả hai đều nhận lịch sử mới làm đầu vào (mức 1, tức thời).

| Đoạn (2023) | Số ca | gSASRec tĩnh | gSASRec cập nhật | Đề xuất tĩnh | Đề xuất cập nhật |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 · 29/07–15/08 | 1.191 | 0,0400 | 0,0400 | 0,0247 | 0,0247 |
| 1 · 15/08–03/09 | 1.170 | 0,0426 | **0,0473** | 0,0301 | 0,0302 |
| 2 · 03/09–21/09 | 1.127 | 0,0283 | **0,0328** | 0,0201 | 0,0198 |
| 3 · 21/09–13/10 | 1.256 | 0,0320 | **0,0355** | 0,0224 | 0,0206 |
| Trung bình có trọng số (đoạn 1–3) | | 0,0343 | **0,0385** | 0,0242 | 0,0235 |

Mỗi lần fine-tune gSASRec mất 6–17 giây trên T4. Fine-tune nâng gSASRec ở cả ba đoạn (+12% trung bình). Mô hình đề xuất không hưởng lợi: learning rate tune được của nó rất cao (7,9e-3), mức × 0,3 vẫn có thể làm lệch, và nó cần tune riêng cho fine-tune (PRD mục 9 đã dự kiến).

### 5.6 Chi phí

| Mô hình | Huấn luyện cuối / seed (T4, có chia GPU) | Ghi chú |
| --- | ---: | --- |
| SASRec gốc | 49 s | 11 epoch |
| Ablation 2 / 3 | 62 s / 158 s | |
| gSASRec | 317 s | mẫu âm riêng từng vị trí (26 mẫu) |
| eSASRec | 235 s | |
| CL4SRec / đề xuất / ablation 3 + CL | 340 / 505 / 814 s | 3 lượt encode mỗi bước |
| XSimGCL | 777 s | lan truyền toàn đồ thị mỗi bước |

Phục vụ trong app trên CPU: p95 của API gợi ý 17,9 ms (ngưỡng 300 ms). Fine-tune mức 2 trên CPU: 74 giây cả vòng, gồm xuất dữ liệu, huấn luyện, cổng và nạp lại.

## 6. Cải tiến sau kết quả v1: đề xuất v2 và xếp hạng lại

Phần này được **thiết kế sau khi đã xem kết quả test của v1**. Để hạn chế việc "chọn theo test":
- mọi lựa chọn mới chỉ dựa trên validation;
- kết quả v1 ở mục 5 giữ nguyên;
- v2 được kiểm lại trên một lát cắt thời gian khác chưa dùng tới (mục 6.3).

### 6.1 Thiết kế

Số liệu mục 5 cho thấy gBCE là nền tốt nhất, đặc trưng nội dung có ích, còn học tương phản có hại.

- **Đề xuất v2** = gSASRec (gBCE, mẫu âm riêng từng vị trí) + nhánh nội dung ghép bằng cổng + học tương phản tuỳ chọn (λ ∈ {0; 0,001; 0,01}, để tune tự chọn).
- **Khởi tạo ấm.** Một lần thử khởi tạo v2 từ gSASRec đã huấn luyện, với cổng đặt nghiêng hẳn về embedding ID (sigmoid(8) ≈ 0,9997). Epoch 0 cũng được chấm và tính như một ứng viên, nên trên validation v2 không thể kém gSASRec: epoch 0 cho đúng 0,0470 của gSASRec.
- **Ngân sách bằng nhau.** v2 thừa hưởng phần tune của gSASRec, nên gSASRec được thêm 5 lần thử TPE: tổng 10, ngang 5 (gSASRec) + 5 (v2). Cả 5 lần thêm đều chỉ ~0,014; cấu hình tốt nhất không đổi.
- **Xếp hạng lại theo độ phổ biến gần đây** (bước xếp hạng lại của PRD mục 7). Điểm cuối = điểm mô hình chuẩn hoá z theo từng người dùng + w × log(1 + số tương tác trong N ngày trước mốc), chuẩn hoá z. N ∈ {30, 90, 365} và w ∈ {0 … 2} được chọn trên validation bằng mô hình chỉ học trên train, rồi áp nguyên lên test. Bước này áp cho cả 10 cấu hình.

### 6.2 Kết quả (test, bộ chính ML-32M 2020+, 3 seed)

| Mô hình | NDCG@10 | Recall@10 | MRR | Độ phủ@10 | Độ mới | val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Đề xuất v2** | **0,0448 ± 0,0018** | **0,0863** | **0,0409** | **0,272** | **5,41** | **0,0539** |
| gSASRec | 0,0422 ± 0,0010 | 0,0825 | 0,0382 | 0,172 | 4,82 | 0,0470 |
| SASRec gốc | 0,0344 ± 0,0004 | 0,0704 | 0,0319 | 0,192 | 4,90 | 0,0426 |
| Đề xuất v1 | 0,0239 ± 0,0019 | 0,0488 | 0,0241 | 0,146 | 4,78 | 0,0324 |

| So sánh (t cặp đôi theo người dùng) | Chênh lệch NDCG@10 | p |
| --- | ---: | ---: |
| v2 − gSASRec | +0,0026 | 0,11 |
| v2 − SASRec gốc | +0,0104 | < 0,001 |
| v2 − đề xuất v1 | +0,0209 | < 0,001 |

- **v2 đạt mức "ít nhất bằng" gSASRec**: cao hơn 0,0026 nhưng không có ý nghĩa thống kê.
- Cấu hình được chọn trên val là lần thử khởi tạo ấm, thêm 16 epoch, **λ = 0 (tắt học tương phản)**. Trong 4 lần thử huấn luyện từ đầu, tốt nhất đạt val 0,0417.
- v2 tốt hơn gSASRec rõ ở **phạm vi gợi ý**:
  - độ phủ danh mục 0,272 so với 0,172, cao nhất trong mọi mô hình;
  - độ mới 5,41 so với 4,82;
  - NDCG@10 với phim đích độ phổ biến trung bình 0,0153 so với 0,0035.
- Đó đúng là vai trò PRD đặt cho nhánh nội dung: giúp phim ít tương tác.

### 6.3 Xác nhận trên lát cắt khác (ML-32M 2017–2019)

Lát cắt này không trùng thời gian với bộ chính: 2,21 triệu tương tác, 28.471 người dùng, 9.048 phim, cùng tiền xử lý và cùng cách chia. Siêu tham số giữ nguyên như ở bộ chính; chỉ số epoch được chọn lại bằng dừng sớm trên val của lát cắt mới.

| Mô hình | test NDCG@10 (3 seed) | val | epoch |
| --- | ---: | ---: | ---: |
| gSASRec | **0,0614 ± 0,0040** | 0,0698 | 38 |
| Đề xuất v2 | 0,0596 ± 0,0033 | **0,0722** | 3 |
| SASRec gốc | 0,0424 ± 0,0032 | 0,0579 | 14 |
| Phổ biến 365 ngày | 0,0165 | — | — |

- v2 − gSASRec = −0,0019 (p = 0,052): **không khác có ý nghĩa**. Thứ tự đảo so với bộ chính, nên kết luận vững là **v2 ngang gSASRec**, không phải "v2 tốt hơn".
- Cả hai vượt SASRec rõ trên cả hai lát cắt, tức lợi thế của gBCE được xác nhận.
- Trên lát cắt này, dừng sớm chọn v2 chỉ thêm 3 epoch sau khởi tạo ấm. Nhánh nội dung có ít thời gian học, nên mức lợi riêng của nội dung ở đây nhỏ.

### 6.4 Xếp hạng lại theo độ phổ biến gần đây

| Mô hình | Cửa sổ, w (chọn trên val) | val trước → sau | test trước → sau |
| --- | --- | --- | --- |
| **Đề xuất v2** | 30 ngày, 0,5 | 0,0530 → 0,0597 | **0,0448 → 0,0554** (3 seed) |
| gSASRec | 30 ngày, 1,0 | 0,0470 → 0,0529 | 0,0422 → 0,0488 (3 seed) |
| SASRec gốc | 90 ngày, 0,5 | 0,0426 → 0,0504 | 0,0348 → 0,0474 (seed 0) |
| Ablation 3 | 30 ngày, 0,5 | 0,0443 → 0,0489 | 0,0326 → 0,0439 (seed 0) |
| Đề xuất v1 | 90 ngày, 1,0 | 0,0324 → 0,0437 | 0,0265 → 0,0377 (seed 0) |
| Ablation 3 + CL | 30 ngày, 1,0 | 0,0341 → 0,0495 | 0,0244 → 0,0391 (seed 0) |
| CL4SRec | 90 ngày, 1,0 | 0,0274 → 0,0399 | 0,0197 → 0,0338 (seed 0) |
| eSASRec | 90 ngày, 1,0 | 0,0303 → 0,0373 | 0,0233 → 0,0310 (seed 0) |
| Ablation 2 | 90 ngày, 1,0 | 0,0242 → 0,0340 | 0,0229 → 0,0297 (seed 0) |
| XSimGCL | 30 ngày, 2,0 | 0,0263 → 0,0301 | 0,0258 → 0,0296 (seed 0) |

Các mô hình ở lượt v1 chỉ lưu checkpoint seed 0, nên cột test của chúng là 1 seed; gSASRec và v2 có đủ 3 seed.

**Xếp hạng lại tăng độ chính xác ở mọi mô hình.**
- v2 + xếp hạng lại là cấu hình tốt nhất toàn bộ thí nghiệm:
  - so với v2 không xếp hạng lại: +0,0106, p < 0,001;
  - so với gSASRec + xếp hạng lại: +0,0066, p < 0,001;
  - so với gSASRec: +0,0132, p < 0,001.
- Phần tăng lớn nhất nằm ở người dùng lịch sử ngắn (1–5 phim): v2 0,0700 → 0,0920.

**Cái giá phải trả là phạm vi gợi ý.**
- Độ phủ danh mục của v2 giảm từ 0,272 xuống 0,047; độ mới từ 5,41 xuống 3,56.
- Danh sách dồn về phim đang thịnh hành.
- Với app, đây là đánh đổi cần cân nhắc: có thể dùng w nhỏ hơn, hoặc chỉ xếp hạng lại cho người dùng lịch sử ngắn.

**Kết luận phần cải tiến.**
- v2 (gSASRec + nội dung, không học tương phản) **ngang gSASRec về độ chính xác trên cả hai lát cắt và rộng hơn hẳn về độ phủ / độ mới**.
- Khi thêm bước xếp hạng lại của PRD, hệ thống đề xuất **vượt gSASRec có ý nghĩa thống kê**.
- Học tương phản, thành phần mà PRD kỳ vọng nhất, bị tune loại bỏ (λ = 0).

## 7. Hạn chế và việc nên làm tiếp

- **Ngân sách tune nhỏ** (5 lần thử, trần 6 phút). Kết quả của các mô hình chậm (có học tương phản, XSimGCL) nhạy với ngân sách nhất. Nên chạy đủ 30 lần thử theo PRD cho ít nhất gSASRec, SASRec, ablation 3 và mô hình đề xuất.
- **Sampled softmax:** thử mẫu âm riêng từng vị trí hoặc in-batch, và thêm hiệu chỉnh logQ, trước khi kết luận về eSASRec và bước 2.
- **Học tương phản:** thử λ nhỏ hơn (0,001–0,01) và tăng cường nhẹ hơn. Hiện nó đổi độ chính xác lấy độ phủ.
- **Tổ hợp chưa thử:** gSASRec + đặc trưng nội dung. Đây là ứng viên tự nhiên cho "mô hình đề xuất" kế tiếp, vì hai thành phần có tác dụng rõ nhất là gBCE và nội dung.
- **Chưa làm:** đối chiếu cài đặt baseline với số công bố trên ML-1M (PRD mục 6); vector mô tả TMDB.
- **Phạm vi:** một bộ dữ liệu (ML-32M từ 2020), một cách chia, một mục tiêu mỗi người dùng.

## Tái lập

```bash
cd backend
uv sync --all-groups
# 1. dữ liệu (local hoặc Colab — cùng kết quả)
uv run python -m ml.data --raw ../data/raw/ml-32m --out ../data/processed/ml32m_2020 --since-year 2020
# 2. tune + huấn luyện cuối (GPU)
D=../data/processed/ml32m_2020
python -m ml.train tune  --data $D --out results --trials 4 --anchor --max-epochs 40 --patience 4 --trial-minutes 6
python -m ml.train final --data $D --out results --seeds 0,1,2
python -m ml.replay --data $D --ckpt results/gsasrec/model.pt --out results/replay_gsasrec.json --chunks 4
# 3. tổng hợp (CPU) và đưa vào app
uv run python -m ml.train report --data $D --out results
# mô hình phục vụ = summary.json["best_on_val"] (chọn theo validation, ở đây là đề xuất v2)
uv run python -m ml.export --data $D --ckpt results/proposed_v2/model.pt --out artifacts/proposed-v2 --metrics results/proposed_v2/test_seed0.json
uv run python -m scripts.register_model --artifact-dir artifacts/proposed-v2 --version v1.1.0 --live
```
