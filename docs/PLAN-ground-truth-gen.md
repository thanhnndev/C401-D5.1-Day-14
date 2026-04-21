# PLAN-ground-truth-gen.md

## Mục tiêu

Dựa trên yêu cầu, chúng ta cần:

1. Tạo script chunking xử lý file markdown và trích xuất đúng metadata từ cấu trúc thư mục.
2. Nâng cấp script tạo Ground Truth (`synthetic_gen.py`) để map `ground_truth_ids` chính xác với các chunks được tạo.

## Phân tích Cấu trúc Dữ liệu

Cấu trúc thư mục hiện tại: `data/products_cleaned/{branch}/{category}/{language}/{product}.md`
VD: `data/products_cleaned/bvi/devices/en/cryo-line.md`

- **branch**: `bvi`
- **category**: `devices`
- **language**: `en`
- **product**: `cryo-line`

## Task Breakdown

### 1. Tạo Module Chunking (`data/chunking.py`)

- **Đầu vào:** Thư mục `data/products_cleaned`.
- **Trích xuất Metadata từ Path:**
  - Lấy thông tin `branch`, `category`, `language`, `product` từ các cấp thư mục tương ứng.
- **Metadata Bổ sung (Gợi ý):**
  - `chunk_id`: ID định danh duy nhất cho chunk (VD: `{branch}_{product}_{lang}_{idx}`). Rất quan trọng để đối chiếu Ground Truth.
  - `title`: Lấy từ thẻ `#` lớn nhất hoặc tự generate từ tên file.
  - `heading`: Tiêu đề của đoạn văn (`##` hoặc `###`).
  - `char_count`: Số ký tự của chunk (giúp lọc bỏ các chunk quá ngắn không đủ ngữ cảnh).
  - `token_estimate`: Ước lượng token (chia chiều dài cho ~4) để tránh vượt giới hạn token của model embedding.
- **Logic Chunking:**
  - Có thể tự viết hàm dùng regex hoặc duyệt từng dòng theo thẻ header (`## `).
  - Gộp nội dung cho đến khi gặp header mới.
- **Đầu ra:** Một List các dictionary chứa văn bản (chunk text) và metadata đính kèm.

### 2. Nâng cấp Synthetic Gen (`data/synthetic_gen.py`)

- **Đầu vào:** Gọi trực tiếp module chunking từ `data/chunking.py` để lấy danh sách chunks (thay vì tự đọc file như hiện tại).
- **Tạo QA Pairs:**
  - Lặp qua các chunks.
  - Dùng nội dung chunk để đặt câu hỏi (`question`) và sinh ra câu trả lời (`expected_answer`).
  - **Mapping Ground Truth:** Đặt `ground_truth_ids = [chunk['metadata']['chunk_id']]`. Điều này đảm bảo đánh giá Retrieval chính xác từng chunk chứ không chỉ dừng ở cấp độ file.
- **Bổ sung Edge Cases (Red Teaming):**
  - Duy trì hoặc bổ sung thêm các case _Out-of-scope_ hoặc _Adversarial_ (với `ground_truth_ids = []`) để đảm bảo yêu cầu của `GRADING_RUBRIC.md`.
- **Đầu ra:** Ghi ra `data/golden_set.jsonl` chứa hơn 50 test cases chất lượng.

## Agent Assignments

- **`backend-specialist`** (Python/Data Engineering): Phụ trách viết `chunking.py` và cập nhật `synthetic_gen.py`.

## Verification Checklist

- [ ] Khởi chạy `chunking.py` thành công và kiểm tra đúng cấu trúc metadata trích xuất.
- [ ] `synthetic_gen.py` import thành công hàm chunking.
- [ ] File `golden_set.jsonl` được tạo ra chứa các `ground_truth_ids` map chính xác với `chunk_id`.
- [ ] Số lượng test cases đạt chuẩn (>= 50 cases) theo `GRADING_RUBRIC.md`.
- [ ] Các trường metadata như `branch`, `product`, `category`, `language` xuất hiện rõ ràng ở trong data dict.

## Open Questions

- Thư viện chia văn bản: Bạn muốn viết script chunking thủ công (đọc từng dòng để tìm `##`) như `synthetic_gen.py` cũ hay sử dụng thư viện `langchain_text_splitters` (`MarkdownHeaderTextSplitter`) để chuyên nghiệp hơn?
- Xuất file chunks: Bạn có muốn `chunking.py` cũng xuất ra một file riêng (VD: `data/chunks.jsonl`) để phục vụ thẳng cho bước đưa dữ liệu vào Qdrant (Ingestion) luôn không?
