# PLAN: RAG Agent Completion & Evaluation

Mục tiêu: Hoàn thiện `agent/main_agent.py` kết nối Qdrant và Qwen LLM, thực hiện đánh giá toàn diện (Retrieval & Generation) và phân tích lỗi (Failure Analysis) để đạt điểm tối đa theo Rubric.

## User Review Required

> [!IMPORTANT]
> - **LLM Agent:** Sử dụng `qwen2.5-vl-72b-instruct` (QWEN_MODEL_NAME_AGENT).
> - **Retrieval:** Qdrant Vector Search + Optional Rerank (Jina-v3).
> - **Tone:** Trợ lý tư vấn sản phẩm y tế chuyên nghiệp.
> - **Failure Analysis:** Yêu cầu phân tích sâu "5 Whys" cho các case thất bại.

## Proposed Changes

### 1. Agent Implementation (`agent/`)

#### [MODIFY] [main_agent.py](file:///home/thinh/projects/vinuni/assiment/C401-D5.1-Day-14/agent/main_agent.py)
- Triển khai class `MainAgent` thực tế:
    - **Init:** Kết nối Qdrant Cloud, Jina Embedding, và Qwen LLM.
    - **Retrieval Phase:** 
        - Embedding câu hỏi bằng `jina-embeddings-v5-text-small`.
        - Search Qdrant (top_k=5 hoặc 10).
        - [Optional] Rerank kết quả bằng `jina-reranker-v3` nếu được bật.
        - Trích xuất metadata (product name, category) để làm giàu context.
    - **Generation Phase:**
        - Prompt Engineering: Thiết lập vai trò trợ lý y tế, hướng dẫn sử dụng context để trả lời, xử lý các câu hỏi ngoài phạm vi (Out-of-scope).
        - Gọi Qwen API để sinh câu trả lời.
    - **Return:** Trả về dict chuẩn bao gồm `answer`, `contexts`, và `metadata` (source_ids, tokens, model).

### 2. Evaluation & Optimization (`engine/` & `main.py`)

#### [MODIFY] [main.py](file:///home/thinh/projects/vinuni/assiment/C401-D5.1-Day-14/main.py)
- Script chính để điều phối:
    - Chạy **Baseline (V1)**: Retrieval đơn giản.
    - Chạy **Optimized (V2)**: Thêm Reranking hoặc Prompt tuning.
    - So sánh kết quả và chạy **Release Gate**.
    - Xuất báo cáo `reports/summary.json` và `reports/benchmark_results.json`.

### 3. Failure Analysis (`analysis/`)

#### [NEW] [failure_analysis.md](file:///home/thinh/projects/vinuni/assiment/C401-D5.1-Day-14/analysis/failure_analysis.md)
- Phân tích các trường hợp `score < 3` hoặc `hit_rate = 0`.
- Áp dụng kỹ thuật **5 Whys** để tìm nguyên nhân gốc rễ (do Chunking, Embedding, hay Prompting).
- Đề xuất giải pháp cải thiện cụ thể.

## Verification Plan

### Automated Tests
- Chạy `python check_lab.py` để verify cấu trúc project và các module.
- Chạy `main.py` để thực thi toàn bộ pipeline đánh giá.
- Kiểm tra file `reports/summary.json` có đủ các metrics: Hit Rate, MRR, Agreement Rate, Avg Score.

### Manual Verification
- Kiểm tra log của Agent để đảm bảo Metadata Filtering hoạt động đúng (ví dụ: chỉ lấy context của hãng Melag khi hỏi về Melag).
- Đọc nội dung `failure_analysis.md` để đảm bảo độ sâu về kỹ thuật.
