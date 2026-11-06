# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603649
- **Họ và tên:** Chu Thân Nhất 
- **SEED** (in ở ô đầu notebook): 1808

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước |  |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
                                F1 test thường (SUB)  F1 50 case khó  \
1. BoW + LogReg                                0.889           0.333   
2. BoW + MLP                                   0.974           0.378   
3a. Emb + kNN                                  1.000           0.724   
3b. Emb + LogReg                               0.974           0.612   
3c. Emb + MLP head                             0.974           0.692   
4a. LLM nhỏ zero-shot                          1.000           0.889   
4d. LLM nhỏ + prompt phòng thủ                 1.000           0.918   

                                acc 50 case khó  
1. BoW + LogReg                            0.52  
2. BoW + MLP                               0.54  
3a. Emb + kNN                              0.68  
3b. Emb + LogReg                           0.62  
3c. Emb + MLP head                         0.68  
4a. LLM nhỏ zero-shot                      0.86  
4d. LLM nhỏ + prompt phòng thủ             0.90  

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
BoW + LogReg
Đoán: Sẽ giữ được độ chính xác tương đối ổn định (Accuracy ~ 0.70 - 0.80) ngay cả trên các case khó.
Thực tế: Accuracy chỉ đạt 0.52 (F1 = 0.333)
vì mô hình chỉ đếm tần suất từ đơn giản (Bag-of-Words), hoàn toàn "mù" trước các chiêu trò lừa đảo tinh vi như đảo trật tự từ
## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
mô hình BoW + LogReg và BoW + MLP tụt phong độ thảm hại nhất khi chuyển từ tập test thường sang 50 case khó (Accuracy từ ~97% rơi xuống chỉ còn 52% - 54%; F1 sụt từ ~0.90 xuống còn 0.33 - 0.37).
Nhóm case cụ thể bị ảnh hưởng nặng nhất: Nhóm né từ khóa / đánh tráo ký tự (obfuscation) và natural_spam (tin lừa đảo dùng văn phong gần gũi, đời thường).

Case 1 (Obfuscation): Tin nhắn chứa từ cố tình viết lệch như "FR33 Vouchers..." hoặc dùng ký tự đặc biệt lồng vào giữa từ.

Cơ chế sai: Từ điển vocab của BoW được tạo từ các từ chuẩn. Khi gặp FR33, token này không khớp với từ free trong từ điển (hoặc bị coi là từ lạ), khiến mô hình bỏ sót đặc trưng spam quan trọng và đoán nhầm là ham.

Case 2 (Natural Spam / Contextual): Tin nhắn rác trá hình dưới dạng chào hỏi bạn bè thân thiết hoặc thông báo trúng thưởng thông thường không chứa các từ khóa spam kinh điển ("win", "free", "cash").

Cơ chế sai: BoW hoạt động dựa trên phương pháp "đếm từ đơn lẻ" (bag of words) và không quan tâm đến thứ tự hay ngữ cảnh câu. Do đó, các tin nhắn dùng từ ngữ bình thường nhưng mang hàm ý lừa đảo sẽ dễ dàng qua mặt mô hình vì thiếu các trọng số từ khóa tiêu cực.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
Đề theo SEED: Dữ liệu sau khi bỏ trùng lặp hoàn toàn (drop_duplicates) còn lại đúng 5.169 dòng (gồm 653 spam và 4.516 ham). Việc chia tập train/val/test theo tỷ lệ 70/15/15 dựa trên SEED.
Mô hình lười "luôn đoán ham" sẽ đoán đúng toàn bộ các tin ham ($4.516$ tin) trên tổng số $5.169$ tin. Acc = 0.87 
Giải thích: Vì tập dữ liệu mất cân bằng nặng (hơn 87% là tin ham), một mô hình không cần thông minh cứ đoán bừa là "ham" cũng đạt ngay độ chính xác 87%. Đây là mức nền (baseline) tối thiểu mà các mô hình thông minh phải vượt qua.

Phần 1 và 2 — Machine Learning (BoW + LogReg) & Deep Learning (BoW + MLP)
  Cách làm:
  
  Phần 1: Tự code hàm vectorize theo dạng Bag-of-Words nhị phân dựa trên 2000 từ xuất hiện nhiều nhất ở tập train. Sau đó tự huấn luyện mô hình Logistic Regression bằng Numpy (hồi quy logistic qua hàm sigmoid và tối ưu gradient descent).
  
  Phần 2: Xây dựng mạng MLP bằng PyTorch với các tầng ẩn để tự học đặc trưng phi tuyến tính từ ma trận BoW.
  
  Kết quả trên tập Test (SUB):
  
  BoW + LogReg: F1 đạt 0.889 | acc 50 case khó: 0.52 | p95 latency: ~0.1 ms.
  
  BoW + MLP: F1 đạt 0.974 | acc 50 case khó: 0.54 | p95 latency: ~0.5 ms.
  
  Giải thích: Các mô hình dựa trên từ đơn (BoW) chạy cực kỳ nhanh (chưa tới 1 miligiây) và đạt điểm số rất cao trên tập test thông thường. Tuy nhiên, khi sang 50 case khó (chứa các từ bị cố tình viết sai chính tả, ký tự dị biệt như FR33, hoặc văn phong lừa đảo tinh vi), điểm số sụt thê thảm xuống ~52–54% vì BoW chỉ đếm từ rời rạc và hoàn toàn "mù" về mặt ngữ nghĩa, không hiểu nổi từ đồng nghĩa hay biến thể câu.
Phần 3 — Embedding (MiniLM + kNN / LogReg / MLP head)
  Cách làm: Mượn mô hình sentence-transformers (MiniLM) có sẵn để chuyển mỗi câu thành một vector ngữ nghĩa nhiều chiều, sau đó gắn các bộ phân loại (kNN, LogReg, MLP head) lên trên không gian vector này.
  
  Kết quả:
  
  Emb + kNN: F1 test = 1.000 | acc 50 case khó = 0.68 | p95 latency: ~2.5 ms.
  
  Emb + MLP head: F1 test = 0.974 | acc 50 case khó = 0.68 | p95 latency: ~1.5 ms.
  
  Giải thích: Việc nhúng câu vào không gian vector nghĩa giúp mô hình hiểu được ngữ cảnh và từ đồng nghĩa tốt hơn rất nhiều so với việc chỉ đếm từ của BoW. Do đó, khả năng chống chọi với các case khó đã tăng lên rõ rệt (accuracy vọt lên gần 68%), trong khi tốc độ vẫn rất nhanh (dưới vài mili-giây, hoàn toàn vượt chuẩn SLA 50 ms).

Phần 4 — LLM (Zero-shot & Prompt phòng thủ qua Groq API)
  Cách làm: Sử dụng API của Groq (với model llama-3 hoặc tương đương) để phân loại tin nhắn trực tiếp bằng prompt zero-shot mà không cần huấn luyện (không train). Đồng thời bổ sung prompt phòng thủ (4d) để chống lại các câu lệnh tấn công dạng Prompt Injection.
  
  Kết quả:
  
  LLM nhỏ zero-shot (4a): F1 test = 1.000 | acc 50 case khó = 0.86 | p95 latency: >200 ms.
  
  LLM nhỏ + prompt phòng thủ (4d): F1 test = 1.000 | acc 50 case khó = 0.90 | p95 latency: >200 ms.
  
  Giải thích: LLM đọc hiểu văn bản thông minh như con người nên đạt độ chính xác cao nhất (đặc biệt là xử lý mượt các case khó và chặn đứng Prompt Injection khi có prompt phòng thủ). Tuy nhiên, cái giá phải trả là độ trễ rất cao (hơn 200 ms, vi phạm vạch SLA 50 ms) và tốn chi phí gọi API theo số lượng token, hoàn toàn không phù hợp để lọc thời gian thực cho hàng triệu tin nhắn mỗi giây nếu không kết hợp hệ thống lai (hybrid system).

Phần 5 — Tổng hợp & Thử thách 50 case khó
  Bảng Pareto & So sánh tổng quan: Các mô hình ML/Embedding nằm bên trái vạch SLA 50ms nhờ tốc độ siêu tốc; trong khi LLM nằm bên phải vạch SLA (quá chậm so với chuẩn 50ms) nhưng đổi lại có khả năng suy luận và xử lý case khó tốt nhất.
  
  Về 50 case khó tự nghĩ & kết quả thực tế:
  
  Em đã tự bổ sung các case lừa đảo tinh vi (né từ khóa, lồng ký tự, prompt injection).
  
  Model tụt phong độ nhiều nhất: Nhóm BoW + LogReg / MLP (từ ~97% rơi cắm đầu xuống còn ~52%).
  
  Nguyên nhân: BoW nhìn văn bản như một cái túi chứa các từ độc lập, không hiểu thứ tự từ, nên khi gặp các chiêu trò đánh tráo ký tự (như FR33) hoặc văn phong gần gũi đời thường giả danh bạn bè (natural_spam), mô hình lập tức đoán sai hoàn toàn.
## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Chọn phương án: Phương án 3c. Emb + MLP head (hoặc nhóm Embedding + Classifier).

Dẫn số liệu thực tế từ bài lab:

Độ trễ p95 đạt khoảng 1.5 ms/tin (hoặc Emb + kNN là 2.5 ms/tin), đều thấp hơn nhiều so với giới hạn SLA yêu cầu là 50 ms/tin.

Điểm F1 trên tập test đạt 0.974 và trên 50 case khó đạt 0.692 (độ chính xác cao hơn hẳn so với các mô hình BoW truyền thống).

Chi phí tính toán cực thấp (~$0.00 tiền API, chạy mượt mà trên phần cứng cục bộ thông thường).

Chấp nhận đánh đổi:

Chấp nhận hy sinh một phần độ chính xác ở các tình huống lừa đảo cực kỳ tinh vi (tụt điểm ở các case khó so với LLM như 4a/4d vì Embedding đôi khi vẫn bị đánh lừa bởi văn phong giả mạo thân thiện).
## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
Lần 1 (Khi gặp lỗi 404/429 ở LLM):

  Hỏi gì: Em hỏi cách sửa lỗi khi gọi API Gemini bị vướng lỗi 404 Not Found và 429 RateLimitError do hết quota trên bản Free.
  Nhận gì: AI hướng dẫn đổi tên model (bỏ tiền tố models/), thêm lệnh time.sleep() để giãn cách request, hoặc chuyển sang dùng dịch vụ Groq.
  Sửa gì: Em thực hiện tạo API Key bên Groq, cập nhật biến LLM_PROVIDER = "groq" và xóa bỏ các vòng chờ không cần thiết để chạy mượt mà.

Lần 2 (Khi cần giải thích khái niệm & tính tay):

  Hỏi gì: Em nhờ giải thích ngắn gọn về đường Pareto, các câu hỏi lý thuyết, cách tính tay 5 bài toán và cách viết báo cáo theo văn phong sinh viên.
  Nhận gì: Nhận được các công thức tính toán cụ thể (Accuracy, Sigmoid, F1, Cosine, Chi phí token) cùng hệ thống luận điểm phân tích so sánh các thế hệ AI.
  Sửa gì: Em đối chiếu số liệu từ SEED của mình (1808), tự làm tròn đến 2 chữ số thập phân và dùng các ý chính để điền vào bài Lab trên Colab.

Hôm nay em vỡ lẽ ra rằng độ chính xác cao trên tập test tiêu chuẩn (như F1 = 1.0 của LLM hay 0.97 của Deep Learning) đôi khi chỉ là sự "ảo tưởng" an toàn, vì khi gặp các tình huống oái oăm thực tế (50 case khó chứa lỗi đánh tráo ký tự hay Prompt Injection), các mô hình truyền thống sụt giảm phong độ nặng nề. Qua đó em hiểu sâu sắc lý do vì sao một hệ thống AI thực chiến trong sản xuất luôn phải đánh đổi giữa độ trễ (SLA < 50ms), chi phí vận hành, và khả năng chống chịu trước dữ liệu nhiễu.
