# Phân tích nguyên nhân thất bại (Failure Analysis) - 5 Whys

Tài liệu này phân tích các trường hợp thất bại và các vấn đề phát hiện được trong quá trình chạy Benchmark cho hệ thống RAG (Agent_V2_Optimized), dựa trên báo cáo từ `reports/benchmark_results.json` và `reports/summary.json`.

**Thực hiện bởi:** Đội ngũ dự án (Lead: Đào Phước Thinh)


## Tổng quan kết quả Release Gate
- **Điểm V1 (Baseline):** 3.3051
- **Điểm V2 (Có Reranker):** 3.2171
- **Quyết định (Release Gate):** `BLOCK RELEASE`
- **Lý do:** Điểm trung bình của V2 bị giảm nhẹ (Delta: -0.088) so với V1, mặc dù tỷ lệ tìm kiếm trúng đích (Hit Rate) đã tăng từ 33.3% lên 66.7%. Điều này cho thấy Reranker giúp tìm tài liệu tốt hơn nhưng cấu hình LLM (Prompt/Temperature) hoặc chất lượng nội dung đoạn text (chunk) chưa đủ để trả lời chính xác các câu hỏi phức tạp.

---

## Case 1: Lỗi truy xuất thông số kỹ thuật chi tiết (Hit Rate = 0.0)
**Câu hỏi:** "What type of display does the Phaco Vitrectomy Equipment feature?" (Thiết bị Phaco Vitrectomy trang bị loại màn hình nào?)
**Kết quả V2:** Điểm 1.72, Hit Rate 0.0

### Phân tích 5 Whys
1. **Tại sao Agent không trả lời được câu hỏi?**
   * Do Agent trả lời rằng trong phần CONTEXT không có thông tin cụ thể về loại màn hình.
2. **Tại sao thông tin lại bị thiếu trong phần CONTEXT cung cấp cho LLM?**
   * Do hệ thống Retrieval không tìm thấy đoạn văn bản chuẩn (ground truth chunk: `bvi__phaco-vitrectomy-equipment__en__014`) chứa thông số về màn hình.
3. **Tại sao hệ thống Retrieval lại thất bại trong việc tìm đúng chunk này?**
   * Top 5 chunks được lấy lên đều thuộc về cùng một sản phẩm ("Phaco Vitrectomy Equipment") nhưng tập trung vào mô tả chung (chunk 000, 013, 010) thay vì thành phần kỹ thuật chi tiết (màn hình).
4. **Tại sao các chunk mô tả chung lại có điểm số (score) cao hơn chunk chứa thông số chi tiết?**
   * Câu hỏi chứa cụm từ khoá "Phaco Vitrectomy Equipment" xuất hiện với tần suất dày đặc ở nhiều chunk. Cụm từ quan trọng "display" (màn hình) không tạo đủ sức nặng trong thuật toán Hybrid Search (Dense + BM25) so với tên sản phẩm.
5. **Tại sao từ khoá "display" không tạo được sự khác biệt?**
   * Mô hình nhúng (Embedding model - Jina) thiên về việc bắt ngữ cảnh tổng thể của sản phẩm hơn là các thuộc tính kỹ thuật nhỏ lẻ. Bên cạnh đó, chiến lược phân mảnh dữ liệu (chunking) hiện tại khiến tên sản phẩm lặp lại quá nhiều, làm lu mờ các từ khóa đặc tả khi sử dụng thuật toán Sparse (BM25).

### Đề xuất khắc phục
- **Ngắn hạn:** Tăng số lượng tài liệu trả về ban đầu `top_k` (ví dụ từ 5 lên 15) trước khi đưa qua Jina Reranker để tăng cơ hội lọt vào danh sách lọc.
- **Dài hạn:** Áp dụng kỹ thuật **Query Expansion** (HyDE) - yêu cầu LLM sinh ra câu trả lời giả định chứa các từ khóa kỹ thuật (như "LCD", "Touchscreen", "Resolution") rồi dùng nó để tìm kiếm.

---

## Case 2: Trả lời thiếu thông tin dù tìm đúng tài liệu (Hit Rate = 1.0)
**Câu hỏi:** "Kích thước và số lượng của sản phẩm Melafol Plus [VI] là bao nhiêu?"
**Kết quả V2:** Điểm 2.64, Hit Rate 1.0

### Phân tích 5 Whys
1. **Tại sao điểm số của câu này bị thấp (dưới 3.0)?**
   * LLM Judge đánh giá: *"Trả lời không chính xác và không đầy đủ so với ground truth..."*, Agent đã phản hồi là không có thông tin trong tài liệu.
2. **Tại sao Agent báo không có thông tin dù Hit Rate = 1.0?**
   * Mặc dù chunk chuẩn (`melag__melafol-plus__vi__002`) đã nằm trong top 5 tài liệu truy xuất được, Agent vẫn không thể trích xuất (extract) được kích thước và số lượng từ đó.
3. **Tại sao LLM không trích xuất được dữ liệu từ đoạn chunk chuẩn?**
   * Có thể do đoạn chunk chứa quá nhiều thông tin nhiễu, định dạng bảng (table) bị phá vỡ khi chuyển sang dạng plain text, hoặc LLM bị "ảo giác ngược" (bỏ qua thông tin có thật vì sợ trả lời sai do thiết lập temperature = 0.1).
4. **Tại sao định dạng thông tin trong chunk lại làm khó LLM?**
   * Chiến lược phân mảnh văn bản (Semantic Chunking) có thể chưa xử lý tốt các dữ liệu có cấu trúc (như bảng thông số kỹ thuật, danh sách mã sản phẩm).
5. **Tại sao chiến lược chunking chưa xử lý tốt bảng dữ liệu?**
   * Khâu tiền xử lý (Preprocessing) đang gom tất cả text lại mà không duy trì được cấu trúc phân cấp (header) hoặc cấu trúc cột/hàng của các bảng thông số kỹ thuật từ tài liệu gốc.

### Đề xuất khắc phục
- Rà soát lại nội dung của chunk `melag__melafol-plus__vi__002` trong Qdrant để xem dữ liệu thô có dễ đọc hay không.
- Bổ sung cấu trúc Markdown hoặc mô tả siêu dữ liệu (metadata) trực tiếp vào nội dung text của chunk (ví dụ: `[Thông số kỹ thuật] Kích thước: ...`) để LLM dễ nhận diện.
- Nới lỏng một chút System Prompt hoặc Temperature (ví dụ: 0.2) để Agent linh hoạt hơn trong việc kết hợp thông tin rải rác.
