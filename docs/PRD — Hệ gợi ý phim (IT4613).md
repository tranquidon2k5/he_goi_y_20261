# PRD — Hệ gợi ý phim (IT4613)

Nov 14, 2026 · @ducanh

## 1. Tổng quan

Sản phẩm là một trang xem phim trên web có gợi ý cá nhân hoá, đồng thời là bệ thử để so sánh mô hình đề xuất với các baseline trên cùng một bộ dữ liệu. App khởi động bằng MovieLens, ghi lại hành vi người dùng theo đúng schema của dataset, và tự học lại khi có dữ liệu mới.

**Bài toán.** Kho phim có hàng chục nghìn tựa; người dùng không biết xem gì tiếp theo. Sở thích dài hạn (thể loại quen thuộc) và ngắn hạn (vừa xem một phim, muốn phim tương tự) cùng tồn tại và thay đổi theo thời gian.

**Mục tiêu sản phẩm**

- Người dùng mới nhận được gợi ý hợp lý ngay sau bước onboarding.
- Gợi ý thay đổi ngay sau mỗi tương tác, không phải chờ huấn luyện lại.
- Mọi tương tác được lưu ở dạng nối thẳng được vào tập huấn luyện.
- Backend tự fine-tune, đánh giá và thay mô hình khi đủ dữ liệu mới.

**Mục tiêu học thuật**

- Một mô hình đề xuất chính, so sánh công bằng với các baseline công bố sau 2020.
- Mọi mô hình đều được tune hyperparameter với cùng ngân sách, cùng cách chia dữ liệu.
- Có ablation cho từng thành phần của mô hình đề xuất.

**Tiêu chí thành công (đề xuất, cần nhóm chốt)**

| Tiêu chí | Ngưỡng |
| --- | --- |
| NDCG@10 của mô hình đề xuất trên tập test | Không thấp hơn baseline mạnh nhất; khác biệt có kiểm định thống kê |
| Độ trễ API gợi ý (p95) | Dưới 300 ms |
| Người dùng mới có danh sách cá nhân hoá | Sau tối đa 5 phim được chấm ở onboarding |
| Vòng học lại | Chạy tự động từ sự kiện đến mô hình mới, có cổng đánh giá và rollback |

**Ngoài phạm vi.** Phát phim có bản quyền, app mobile, thanh toán, kiểm duyệt nội dung.

## 2. Người dùng và user stories

App phục vụ ba nhóm: người xem mới, người xem thường xuyên, và nhóm phát triển cần theo dõi mô hình.

| Nhóm | Nhu cầu chính | App đáp ứng bằng |
| --- | --- | --- |
| Người xem mới | Có gợi ý dù chưa có lịch sử | Onboarding chọn thể loại và chấm vài phim; hàng “Thịnh hành” |
| Người xem thường xuyên | Gợi ý bám theo phim vừa xem, không lặp lại phim cũ | Mô hình chuỗi cập nhật theo từng tương tác; lọc phim đã xem |
| Nhóm phát triển, giảng viên | Thấy mô hình nào đang chạy, tốt đến đâu | Trang Lab: phiên bản mô hình, metric offline, lịch sử học lại |

**User stories**

1. Là người mới, tôi chọn thể loại yêu thích và chấm ít nhất 5 phim để trang chủ có hàng “Dành cho bạn”.
2. Là người xem, tôi mở một phim và thấy “Phim tương tự” cùng lý do gợi ý (“Vì bạn đã xem X”).
3. Là người xem, tôi chấm sao, gắn tag, thêm vào danh sách xem sau, hoặc bấm “Không quan tâm”; gợi ý đổi ngay ở lần tải trang kế tiếp.
4. Là người xem, tôi lọc phim theo thể loại, năm, điểm trung bình khi đã biết mình muốn gì.
5. Là người xem, tôi xem và sửa lịch sử của mình (xoá một rating sai).
6. Là thành viên nhóm, tôi xem trang Lab để biết lần học lại gần nhất, metric trước và sau, và rollback nếu cần.

## 3. Yêu cầu chức năng của app

App có sáu màn hình; mỗi màn hình vừa phục vụ người xem vừa sinh ra một loại tín hiệu cho mô hình.

| Màn hình | Chức năng | Kỹ thuật gợi ý phía sau |
| --- | --- | --- |
| Onboarding | Chọn thể loại, chấm tối thiểu 5 phim phổ biến | Khởi tạo chuỗi tương tác cho người mới |
| Trang chủ | Các hàng: “Dành cho bạn”, “Vì bạn đã xem X”, “Thịnh hành”, theo thể loại | Mô hình chính; item-to-item; độ phổ biến gần đây |
| Chi tiết phim | Poster, mô tả, trailer, chấm sao 0.5–5, tag, xem sau, “Không quan tâm”, “Phim tương tự” | Tương đồng embedding item |
| Tìm kiếm và lọc | Tìm theo tên; lọc thể loại, năm, điểm | Gợi ý dựa trên ràng buộc (knowledge-based) |
| Hồ sơ | Lịch sử xem và chấm, sửa hoặc xoá, danh sách xem sau | Minh bạch: người dùng sửa được dữ liệu của mình |
| Lab (nội bộ) | Phiên bản mô hình, metric offline, trạng thái học lại, rollback | Vận hành mô hình |

**Cold-start.** Dùng lai kiểu chuyển dịch (switching): người dùng có dưới 5 tương tác nhận “Thịnh hành” lọc theo thể loại đã chọn; từ 5 tương tác trở lên chuyển sang mô hình chính. Phim mới chưa có tương tác được xếp hạng nhờ đặc trưng nội dung.

**Xem phim.** Giả định: trang chi tiết phát trailer nhúng và có nút “Đã xem”; app không phát phim đầy đủ.

**Giải thích gợi ý.** Mỗi phim trong “Dành cho bạn” kèm một lý do ngắn, lấy từ phim trong lịch sử có embedding gần nhất.

**Yêu cầu phi chức năng**

- API gợi ý trả về trong 300 ms (p95); kết quả được cache theo người dùng và huỷ cache khi có sự kiện mới.
- Không gợi ý lại phim đã chấm hoặc đã bấm “Không quan tâm”.
- Mỗi danh sách ghi kèm phiên bản mô hình đã sinh ra nó.
- Mật khẩu được băm; người dùng xoá được tài khoản và dữ liệu của mình.

## 4. Thu thập dữ liệu

App ghi hai lớp dữ liệu: lớp tường minh trùng schema MovieLens để nối thẳng vào tập huấn luyện, và lớp tiềm ẩn để bổ sung tín hiệu mà dataset không có.

**Ánh xạ sang các file của MovieLens**

| File MovieLens | Trường | App thu thập bằng |
| --- | --- | --- |
| ratings.csv | userId, movieId, rating (0.5–5, bước 0.5), timestamp (giây UTC) | Thao tác chấm sao |
| tags.csv | userId, movieId, tag, timestamp | Ô gắn tag ở trang chi tiết |
| movies.csv | movieId, title, genres | Danh mục phim; phim mới do quản trị thêm |
| links.csv | movieId, imdbId, tmdbId | Khoá để lấy poster, mô tả, trailer từ TMDB |

Người dùng của app nhận userId từ 1.000.001 trở lên để không trùng với userId của MovieLens. Cột `source` (movielens hoặc app) phân biệt nguồn.

**Sự kiện tiềm ẩn**

| Sự kiện | Trường riêng | Dùng để |
| --- | --- | --- |
| impression | danh sách movieId, vị trí, tên hàng, model\_version, request\_id | Tính CTR online; lấy mẫu âm đã hiển thị; đo thiên lệch vị trí |
| detail\_view | movieId, hàng và vị trí nguồn, request\_id | Tín hiệu quan tâm yếu |
| watch | movieId, số giây xem, tỉ lệ hoàn thành | Tín hiệu dương khi vượt ngưỡng |
| mark\_watched | movieId | Tín hiệu dương |
| watchlist\_add, watchlist\_remove | movieId | Ý định xem |
| not\_interested | movieId | Tín hiệu âm; loại khỏi gợi ý |
| search | chuỗi tìm, bộ lọc | Nhu cầu tức thời; phân tích |

Mọi sự kiện có chung: event\_id, user\_id, session\_id, timestamp (giây UTC, cùng đơn vị với MovieLens).

**Quy tắc chuyển thành dữ liệu huấn luyện**

- Rating từ 4.0 trở lên là tương tác dương, giống cách xử lý MovieLens ở mục 6.
- watch vượt 50% thời lượng, mark\_watched và watchlist\_add là tương tác dương với trọng số tin cậy thấp hơn rating; ba trọng số này là hyperparameter.
- Nếu một cặp (user, phim) có cả rating và tín hiệu tiềm ẩn thì rating quyết định.
- impression không được click chỉ dùng làm mẫu âm tuỳ chọn và cho đánh giá online, không đưa vào so sánh offline với baseline.

**Hồ sơ người dùng.** Chỉ lưu tên hiển thị, email, thể loại đã chọn. MovieLens 32M không có nhân khẩu học nên app không hỏi tuổi hay giới tính.

## 5. Kiến trúc và backend

Backend có hai đường: đường phục vụ (ghi sự kiện, trả gợi ý) và đường học (fine-tune, đánh giá, thay mô hình).

&#91;embedded content: kiến trúc · 5 thành phần, 1 vòng học lại\]

Sự kiện đi từ app vào cơ sở dữ liệu; Trainer đọc dữ liệu mới và chỉ đẩy mô hình sang dịch vụ gợi ý khi mô hình đạt cổng đánh giá.

**Thành phần (stack là giả định, xem mục 11)**

| Thành phần | Công nghệ đề xuất | Trách nhiệm |
| --- | --- | --- |
| Web app | Next.js | Giao diện; gửi sự kiện theo lô |
| API backend | FastAPI | Xác thực, danh mục phim, nhận sự kiện, gọi dịch vụ gợi ý |
| Cơ sở dữ liệu | PostgreSQL | users, movies, ratings, tags, events, model\_versions |
| Dịch vụ gợi ý | PyTorch + FAISS | Nạp checkpoint đang dùng, suy luận top-K, xếp hạng lại, cache |
| Trainer | PyTorch, chạy theo lịch | Fine-tune, huấn luyện lại toàn bộ, tune, đánh giá |
| Kho mô hình | MLflow hoặc thư mục có phiên bản | Lưu checkpoint, cấu hình, metric; rollback |

**Ba mức học khi có dữ liệu mới**

1. Tức thời, không huấn luyện. Mô hình chuỗi biểu diễn người dùng bằng chính dãy phim đã tương tác, nên sự kiện mới chỉ cần nối vào dãy và suy luận lại. Người dùng mới không cần embedding riêng.
2. Fine-tune tăng dần. Kích hoạt khi số tương tác mới vượt ngưỡng hoặc theo lịch hằng đêm. Huấn luyện tiếp từ checkpoint hiện tại trên dữ liệu mới trộn với một mẫu dữ liệu cũ (replay) để tránh quên, learning rate thấp, vài epoch.
3. Huấn luyện lại toàn bộ. Định kỳ hoặc khi thêm nhiều phim mới; có thể kèm tune lại hyperparameter.

**Cổng đánh giá.** Sau mức 2 và 3, mô hình mới và mô hình đang chạy cùng được chấm trên một cửa sổ thời gian gần nhất đã giữ riêng. Mô hình mới chỉ được dùng khi NDCG@10 không giảm quá ngưỡng cho phép; mọi lần chạy được ghi vào bảng model\_versions.

**Phim mới.** Phim thêm vào danh mục được tính embedding nội dung ngay; embedding ID của nó khởi tạo từ embedding nội dung và được học ở lần fine-tune kế tiếp.

**Trình diễn vòng học.** Số người dùng thật của một đồ án môn học rất nhỏ so với MovieLens. Để chứng minh vòng học hoạt động, giữ lại giai đoạn cuối của MovieLens và phát lại như một luồng sự kiện, rồi báo cáo metric qua từng lần fine-tune.

**API chính**

| Endpoint | Mục đích |
| --- | --- |
| POST /events | Nhận một lô sự kiện |
| POST /ratings, DELETE /ratings/{movieId} | Chấm sao, xoá rating |
| GET /recommendations?surface=home | Các hàng gợi ý của trang chủ |
| GET /movies/{id}/similar | Phim tương tự |
| GET /movies?genre=&year= | Tìm kiếm và lọc |
| GET /lab/models, POST /lab/retrain, POST /lab/rollback | Vận hành mô hình |

## 6. Bộ dữ liệu

Dùng [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) làm bộ chính cho cả app và thí nghiệm, làm giàu nội dung bằng TMDB qua links.csv.

**Vì sao ML-32M.** Đây là bản benchmark ổn định mới nhất của GroupLens: 32 triệu rating và 2 triệu lượt gắn tag trên 87.585 phim của 200.948 người dùng, thu thập tháng 10/2023. Kho phim đủ mới để app trông như một trang xem phim thật, và có tmdbId để lấy poster, mô tả, trailer. Giấy phép chỉ cho dùng phi thương mại và yêu cầu trích dẫn ([README](https://files.grouplens.org/datasets/movielens/ml-32m-README.html)).

**Tiền xử lý**

1. Giữ rating từ 4.0 trở lên làm tương tác dương; sắp xếp theo timestamp thành chuỗi của từng người dùng.
2. Lọc k-core: phim có ít nhất 10 tương tác, người dùng có ít nhất 5.
3. Cắt theo thời gian nếu tài nguyên hạn chế, ví dụ chỉ lấy tương tác từ 2015 trở đi; kích thước cuối cùng chốt sau bước khảo sát dữ liệu.
4. Lấy từ TMDB cho mỗi phim: mô tả, poster, trailer, diễn viên, đạo diễn. Mã hoá mô tả thành vector bằng một mô hình sentence embedding; thể loại mã hoá multi-hot.

**Chia tập.** Chia theo mốc thời gian toàn cục: khoảng 5% tương tác cuối là test, cửa sổ liền trước là validation, phần còn lại là train. Cách này không rò rỉ tương lai và khớp với cách app học lại theo thời gian.

**Bộ phụ.** MovieLens 1M chỉ dùng để kiểm tra cài đặt baseline có khớp số đã công bố hay không, theo giao thức leave-one-out quen thuộc trong các bài báo.

**Hạn chế cần nêu trong báo cáo**

- Timestamp của MovieLens là lúc chấm điểm, không phải lúc xem; người dùng có thể chấm không theo thứ tự xem ([gSASRec, mục 6.1](https://arxiv.org/pdf/2308.07192)). Dữ liệu app thu được sẽ có thứ tự xem thật.
- MovieLens chỉ có rating và tag; các sự kiện tiềm ẩn ở mục 4 chỉ đến từ app.
- Dữ liệu TMDB cần khoá API và ghi nguồn theo điều khoản của TMDB.

## 7. Hướng thuật toán đề xuất

Đề xuất một hướng duy nhất: gợi ý tuần tự dựa trên SASRec, bổ sung đặc trưng nội dung phim và học tương phản, theo sau là một bước xếp hạng lại.

**Vì sao mô hình tuần tự hợp với app này**

- Người dùng được biểu diễn bằng dãy phim đã tương tác, không có embedding riêng. Người dùng mới của app nhận gợi ý cá nhân hoá mà không cần huấn luyện lại; MF hay LightGCN phải học lại để có embedding cho họ.
- Dãy theo thời gian nắm được cả sở thích dài hạn và ngắn hạn.
- Bám sát nội dung môn học: gợi ý theo phiên, lai ghép, học tương phản (CL4Rec, CoSeRec).

**Thành phần của mô hình**

| Thành phần | Nội dung | Giải quyết |
| --- | --- | --- |
| Khung | Transformer một chiều kiểu SASRec, dự đoán phim kế tiếp | Sở thích thay đổi theo thời gian |
| Hàm mất mát | Sampled softmax với nhiều mẫu âm thay cho BCE một mẫu âm | SASRec gốc huấn luyện yếu vì quá ít mẫu âm |
| Embedding phim | Embedding ID cộng với phép chiếu của đặc trưng nội dung (thể loại, vector mô tả, tag) | Phim mới và phim ít tương tác |
| Học tương phản | Hai view của cùng một chuỗi (crop, mask, reorder), loss InfoNCE phụ trợ | Dữ liệu thưa, chuỗi ngắn của người dùng app |
| Xếp hạng lại | Lọc phim đã xem và “Không quan tâm”; MMR để tăng đa dạng; chuyển sang “Thịnh hành” khi chuỗi quá ngắn | Trải nghiệm người dùng, cold-start |

Hàm mục tiêu khi huấn luyện:

```latex
\mathcal{L} = \mathcal{L}_{\text{sampled softmax}} + \lambda \, \mathcal{L}_{\text{InfoNCE}}
```

**Căn cứ thực nghiệm.** Bài [eSASRec (RecSys 2025)](https://arxiv.org/pdf/2508.06450) cho thấy riêng việc đổi sang sampled softmax đã nâng NDCG@10 của SASRec trên ML-20M từ 0,1192 lên 0,1527 khi chia theo thời gian. Vì vậy sampled softmax là phần bắt buộc; nội dung và học tương phản là đóng góp của nhóm, cần ablation để chứng minh.

**Ablation dự kiến**

1. SASRec gốc (BCE, một mẫu âm).
2. Thêm sampled softmax.
3. Thêm đặc trưng nội dung.
4. Thêm học tương phản (mô hình đầy đủ).
5. Mô hình đầy đủ có và không có bước xếp hạng lại, đo thêm độ đa dạng và độ phủ.

**Phân tích riêng cho cold-start.** Báo cáo metric tách theo độ dài chuỗi người dùng và theo độ phổ biến của phim, vì đây là nơi nội dung và học tương phản được kỳ vọng giúp nhiều nhất.

## 8. Baseline và kế hoạch so sánh

So sánh với 4 baseline bắt buộc và 1 baseline tuỳ chọn, tất cả công bố sau 2020; tất cả dùng chung dữ liệu, cách chia và ngân sách tune.

| Nhóm | Mô hình | Năm | Mức | Lý do chọn |
| --- | --- | --- | --- | --- |
| Tuần tự + tương phản | CL4SRec | 2022 | Bắt buộc | Có trong bài giảng; gần nhất với mô hình đề xuất |
| CF đồ thị + tương phản | XSimGCL | 2023 | Bắt buộc | Đại diện không tuần tự; học tương phản trên đồ thị |
| Tuần tự | gSASRec | 2023 | Bắt buộc | Sửa hiện tượng quá tự tin do lấy mẫu âm |
| Tuần tự | eSASRec | 2025 | Bắt buộc | SASRec + sampled softmax + tầng LiGR; baseline mạnh và đơn giản |
| Tuần tự | HSTU hoặc FuXi-α | 2024–2025 | Tuỳ chọn | NDCG cao nhất trên ML-20M trong bài eSASRec; nặng, chỉ làm nếu đủ GPU |

Các mô hình từ 2020 trở về trước không nằm trong bảng so sánh. SASRec gốc chỉ xuất hiện như bước đầu của ablation ở mục 7, và “Thịnh hành” chỉ là phương án cold-start của app.

**Nguyên tắc so sánh công bằng**

- Cùng tập train, validation, test và cùng bước tiền xử lý.
- Xếp hạng trên toàn bộ danh mục phim, không lấy mẫu âm khi đánh giá.
- Mỗi mô hình được tune với cùng số lần thử (mục 9).
- Dùng một thư viện chung cho baseline, đề xuất [RecBole](https://github.com/RUCAIBox/RecBole); các mô hình chưa có sẵn lấy từ mã nguồn gốc của tác giả và ghi rõ trong báo cáo.

**Lưu ý khi đọc kết quả.** Một [nghiên cứu tái lập năm 2025](https://arxiv.org/pdf/2408.03873) cho thấy thứ hạng giữa các mô hình tuần tự thay đổi theo cách tiền xử lý và cài đặt. Vì thế kết quả của nhóm có thể khác số trong bài gốc; điều cần bảo đảm là giao thức nhất quán.

## 9. Đánh giá và tuning hyperparameter

Chỉ số chính là NDCG@10 trên tập test chia theo thời gian; hyperparameter được chọn trên validation và tập test chỉ được chấm một lần cho cấu hình tốt nhất.

**Chỉ số**

| Loại | Chỉ số | Ghi chú |
| --- | --- | --- |
| Độ chính xác xếp hạng | Recall@K, NDCG@K, MRR với K = 10 và 20 | Theo bài giảng đánh giá của môn |
| Ngoài độ chính xác | Độ phủ danh mục, độ đa dạng trong danh sách, độ mới | Phục vụ mục tiêu hướng người dùng |
| Online (trong app) | CTR theo hàng gợi ý, tỉ lệ phim được chấm từ 4 sao | Chỉ mang tính minh hoạ vì ít người dùng |
| Hệ thống | Thời gian huấn luyện, độ trễ suy luận | Để bàn về chi phí triển khai |

Kết quả cuối báo cáo trung bình và độ lệch chuẩn qua 3 seed, kèm kiểm định t cặp đôi theo người dùng giữa mô hình đề xuất và baseline mạnh nhất.

**Quy trình tune**

1. Khai báo trước không gian tìm kiếm cho từng mô hình và ghi vào báo cáo.
2. Tìm bằng tối ưu Bayes (TPE, qua Optuna).
3. Cùng ngân sách cho mọi mô hình học sâu, đề xuất 30 lần thử; cắt sớm lần thử kém.
4. Dừng sớm theo NDCG@10 trên validation, kiên nhẫn 10 epoch.
5. Với dữ liệu lớn, tune trên một mẫu người dùng rồi huấn luyện cấu hình tốt nhất trên toàn bộ.
6. Lưu mọi lần thử (cấu hình, metric, seed) để vẽ độ nhạy hyperparameter trong báo cáo.

Nếu dùng công cụ tune có sẵn của RecBole, cần đặt tường minh số lần thử và điều kiện dừng: một [nghiên cứu năm 2025](https://arxiv.org/pdf/2508.21180v1) chỉ ra các mặc định ẩn của nó làm kết quả khó tái lập.

**Không gian tìm kiếm đề xuất**

| Phạm vi | Hyperparameter | Khoảng |
| --- | --- | --- |
| Mọi mô hình | Learning rate | 1e-4 đến 1e-2, thang log |
| Mọi mô hình | Kích thước embedding | 32, 64, 128, 256 |
| Mọi mô hình | Weight decay | 0 đến 1e-2 |
| XSimGCL | Số tầng; độ lớn nhiễu; trọng số và nhiệt độ tương phản | 1–4; 0,05–0,3; 0,01–0,5 và 0,05–1 |
| Họ SASRec (CL4SRec, gSASRec, eSASRec, mô hình đề xuất) | Số block, số head | 1–3; 1, 2, 4 |
| Họ SASRec | Dropout | 0,1–0,5 |
| Họ SASRec | Độ dài chuỗi tối đa | 50, 100, 200 |
| gSASRec | Số mẫu âm; hệ số hiệu chỉnh t | 16–256; 0–1 |
| Sampled softmax (eSASRec, mô hình đề xuất) | Số mẫu âm | 64–1024 |
| Học tương phản (CL4SRec, mô hình đề xuất) | Trọng số λ, nhiệt độ τ | 0,01–0,5; 0,05–1 |
| Học tương phản | Tỉ lệ crop, mask, reorder | 0,1–0,7 |
| Đặc trưng nội dung | Cách ghép với embedding ID | Cộng, nối, cổng học được |
| Xếp hạng lại | Hệ số đa dạng MMR | 0–0,5 |
| Tín hiệu tiềm ẩn của app | Trọng số watch, mark\_watched, watchlist | 0,2–1 |
| Fine-tune tăng dần | Learning rate, tỉ lệ replay, số epoch | Tune bằng luồng phát lại ở mục 5 |

## 10. Lộ trình và rủi ro

Công việc chia thành năm giai đoạn nối tiếp; chưa gắn ngày vì chưa biết hạn nộp.

1. Dữ liệu. Tải ML-32M, khảo sát, tiền xử lý, chia tập, lấy metadata TMDB. Kết quả: bộ dữ liệu đóng băng dùng chung cho mọi thí nghiệm.
2. Baseline. Chạy và tune nhóm baseline bắt buộc; đối chiếu với số công bố trên ML-1M. Kết quả: bảng baseline.
3. Mô hình đề xuất. Cài đặt, tune, ablation. Kết quả: bảng so sánh và checkpoint tốt nhất.
4. App. Backend, cơ sở dữ liệu, giao diện, thu sự kiện, dịch vụ gợi ý. Có thể làm song song với giai đoạn 2 và 3 bằng một mô hình tạm.
5. Vòng học lại và báo cáo. Trainer, cổng đánh giá, trang Lab, thí nghiệm phát lại luồng; viết báo cáo, README, slide 15 phút.

**Rủi ro**

| Rủi ro | Ảnh hưởng | Phương án |
| --- | --- | --- |
| ML-32M quá nặng cho GPU miễn phí | Tune không kịp | Cắt theo thời gian; tune trên mẫu người dùng; lùi về ML-1M cho phần so sánh |
| Mô hình đề xuất không vượt baseline mạnh | Kết quả kém hấp dẫn | Báo cáo trung thực; nhấn vào ablation, cold-start và độ đa dạng |
| Ít người dùng thật | Không chứng minh được lợi ích học lại | Thí nghiệm phát lại luồng MovieLens |
| Fine-tune làm mô hình quên dữ liệu cũ | Chất lượng giảm dần | Replay dữ liệu cũ; cổng đánh giá; huấn luyện lại toàn bộ định kỳ |
| Gợi ý tự củng cố phim phổ biến | Danh sách đơn điệu | Ghi impression; xếp hạng lại theo đa dạng; theo dõi độ phủ |
| Thiếu metadata TMDB cho một số phim | Thiếu poster, thiếu đặc trưng nội dung | Ảnh thay thế; chỉ dùng thể loại cho các phim đó |

## 11. Giả định và câu hỏi cần xác nhận

Tám điểm dưới đây chưa rõ trong yêu cầu; tài liệu đang dùng giả định ghi kèm và sẽ đổi theo câu trả lời.

- [ ] “Xem phim” nghĩa là gì? Giả định: phát trailer và nút “Đã xem”, không phát phim đầy đủ.
- [ ] Tài nguyên tính toán là gì (Colab, Kaggle, GPU riêng)? Giả định: GPU miễn phí, nên ML-32M có thể phải cắt bớt.
- [ ] Hạn nộp, ngày trình bày và số thành viên nhóm? Cần để gắn ngày cho lộ trình và chốt số baseline.
- [ ] Đồng ý hướng tuần tự dựa trên SASRec, hay muốn hướng đồ thị (LightGCN + học tương phản)? Giả định: tuần tự.
- [ ] Dùng rating từ 4.0 làm tương tác dương, hay coi mọi phim đã chấm là đã xem? Giả định: từ 4.0.
- [ ] Stack ưu tiên? Giả định: Next.js, FastAPI, PostgreSQL, PyTorch.
- [ ] Dự kiến có bao nhiêu người dùng thật thử app? Giả định: vài chục, nên cần thí nghiệm phát lại luồng.
- [ ] Giảng viên có yêu cầu riêng về thuật toán hoặc đã duyệt đề xuất đề tài nào chưa?

## Nguồn

- [MovieLens 32M, GroupLens](https://grouplens.org/datasets/movielens/32m/) và [README](https://files.grouplens.org/datasets/movielens/ml-32m-README.html)
- [eSASRec: Enhancing Transformer-based Recommendations in a Modular Fashion, RecSys 2025](https://arxiv.org/pdf/2508.06450)
- [gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling, 2023](https://arxiv.org/pdf/2308.07192)
- [A Reproducible Analysis of Sequential Recommender Systems, 2025](https://arxiv.org/pdf/2408.03873)
- [Nghiên cứu về các mặc định ẩn khi tune hyperparameter trong RecBole, 2025](https://arxiv.org/pdf/2508.21180v1)
- [RecBole](https://github.com/RUCAIBox/RecBole)
- Slide học phần IT4613 (L0–L10) và slide báo cáo mẫu “Scientific Paper Recommender”
