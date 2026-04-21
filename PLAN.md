# 📋 PLAN - Lab Day 14: AI Evaluation Factory

**Ngày tạo:** April 21, 2026  
**Tổng thời gian:** 4 giờ  
**Mục tiêu:** Xây dựng hệ thống đánh giá tự động chuyên nghiệp cho AI Agent với multi-judge consensus, retrieval metrics, và failure analysis.

---

## 🎯 Tổng quan dự án

### Hiện trạng

- ✅ Cấu trúc project đã được tạo
- ✅ Tài liệu sản phẩm đã được chuẩn bị (200+ file MD cho BVI, Melag, Schwind)
- ⚠️ Main Agent: Chỉ có mock/placeholder, cần thay thế bằng agent thực tế
- ⚠️ Golden Dataset: Cần sinh tự động từ product files
- ❌ LLM Judge: Chỉ có skeleton, cần implement multi-judge
- ❌ Retrieval Evaluator: Có template nhưng chưa hoàn thiện
- ❌ Reports & Failure Analysis: Chưa tạo

### Công việc cần làm

- **Phần 1 (45'):** Golden Dataset & Synthetic Data Generation
- **Phần 2 (90'):** Eval Engine (Multi-Judge + Retrieval Metrics + Async)
- **Phần 3 (60'):** Benchmark Runner + Regression Testing
- **Phần 4 (45'):** Failure Analysis & Optimization Report

---

## 📅 Kế hoạch chi tiết

### **PHASE 1: Golden Dataset & Synthetic Data Generation (45 phút)**

#### Task 1.1: Hoàn thiện `data/synthetic_gen.py`

**Mục tiêu:** Sinh ≥50 test cases từ product files với format chuẩn

**Chi tiết:**

- [ ] Đọc toàn bộ file MD từ `data/products_cleaned/` (melag, bvi, schwind)
- [ ] Sinh Q&A pairs từ mỗi section của product
- [ ] **Bắt buộc** tính input các loại test case:
  - Factual questions (Dễ): "Thiết bị X có đặc tính gì?"
  - Complex questions (Trung bình): "So sánh 2 thiết bị X và Y"
  - Adversarial cases (Khó): "Prompt injection", "Out-of-scope queries"
  - Edge cases: "Conflicting info", "Ambiguous questions"
- [ ] Mỗi test case phải có `ground_truth_ids` (document ID đúng để tính Hit Rate)
- [ ] Output: `data/golden_set.jsonl` với format

  ```json
  {"question": "...", "expected_answer": "...", "ground_truth_ids": ["doc_id"], "metadata": {...}}
  ```

**Acceptance Criteria:**

- ≥50 test cases được tạo
- Mỗi case có đầy đủ `ground_truth_ids` (không được empty)
- Phân bố đều: 60% dễ, 25% trung bình, 15% khó
- Toàn bộ file MD từ `/melag/en/` được cover

---

#### Task 1.2: Red Teaming Dataset

**Mục tiêu:** Tạo thêm test cases mục đích phá vỡ hệ thống

**Chi tiết:**

- [ ] Thêm ≥10 adversarial cases để kiểm tra:
  - Prompt injection: "Ignore context, tell me X"
  - Hallucination test: Hỏi về thông tin gì không có trong tài liệu
  - Multi-turn dependency: Câu hỏi liên tiếp phụ thuộc nhau
  - Conflicting info: 2 passages trái lệnh nhau
- [ ] Tất cả được thêm vào `data/golden_set.jsonl`

**Acceptance Criteria:**

- Total test cases: ≥60
- Red teaming cases marked as `difficulty: "hard"`

**Timeline:** 45 phút

---

### **PHASE 2: Eval Engine Implementation (90 phút)**

#### Task 2.1: Retrieval Evaluator - Hit Rate & MRR

**Mục tiêu:** Implement metrics để đánh giá chất lượng Retrieval stage

**File:** `engine/retrieval_eval.py`

**Chi tiết:**

- [x] Implement `calculate_hit_rate(expected_ids, retrieved_ids, top_k=3)`:
  - Hit Rate = 1 nếu ≥1 expected_id nằm trong top_k retrieved
  - Hit Rate = 0 nếu không
  - Support custom top_k
  
- [x] Implement `calculate_mrr(expected_ids, retrieved_ids)`:
  - MRR = 1/rank (vị trí đầu tiên của expected_id)
  - MRR = 0 nếu không tìm thấy
  - Ví dụ: expected_id ở rank 2 → MRR = 0.5
  
- [x] Implement `evaluate_retrieval_batch(dataset, retrieved_results)`:
  - Tính toán Hit Rate & MRR cho toàn bộ dataset
  - Output: `{"avg_hit_rate": float, "avg_mrr": float, "per_query": [...]}`

**Acceptance Criteria:**

- Toàn bộ 3 hàm hoạt động đúng
- Có test cases verify logic

**Timeline:** 30 phút

---

#### Task 2.2: Multi-Model Judge Engine

**Mục tiêu:** Implement consensus logic giữa 2+ LLM judges

**File:** `engine/llm_judge.py`

**Chi tiêu:**

- [x] Implement `LLMJudge` class với support ≥2 models:
  - Model 1: GPT-4o hoặc Claude 3.5 (tùy business logic)
  - Model 2: Khác (khuyến nghị GPT-4 hay Claude 3 Sonnet)
  
- [x] Implement `evaluate_multi_judge(question, answer, ground_truth)`:
  - Call 2 models với same rubric (scoring 1-5)
  - Return format:

    ```json
    {
      "final_score": float (avg),
      "agreement_rate": float (0-1),
      "individual_scores": {"model1": score, "model2": score},
      "reasoning": "Why both models agreed/disagreed"
    }
    ```
  
- [x] Implement conflict resolution:
  - Nếu |score1 - score2| ≤ 0.5: Use average
  - Nếu > 0.5: Use median của 2 scores + flag "conflict"
  
- [x] Implement rubrics cho các tiêu chí:
  - **Accuracy** (30%): Đúng sự thật từ tài liệu?
  - **Completeness** (20%): Trả lời đủ đầy?
  - **Relevance** (20%): Liên quan đến câu hỏi?
  - **Safety** (15%): Có bịa chuyện (hallucinate) không?
  - **Tone** (15%): Chuyên nghiệp không?
  
- [x] **Position Bias Check:**
  - Implement hàm `check_position_bias()` để swap A-B order và so sánh scores
  - Output: Position bias factor

**Acceptance Criteria:**

- 2 judges được call song song (async)
- Agreement rate tính toán đúng
- Conflict resolution logic hoạt động
- Có position bias detection

**Timeline:** 45 phút

---

#### Task 2.3: RAGAS Integration (Optional Enhancement)

**Mục tiêu:** Integrate existing RAGAS metrics

**File:** `engine/runner.py` (tuỳ chỉnh)

**Chi tiết:**

- [x] Tích hợp RAGAS metrics nếu có:
  - `faithfulness`: Câu trả lời có loyal vs context không?
  - `answer_relevancy`: Câu trả lời có liên quan đến Q không?
  - `context_precision`: Retrieved context có chính xác không?
  
- [x] Nếu RAGAS không hoạt động, tạo custom metrics thay thế với cùng tên

**Acceptance Criteria:**

- Custom hoặc RAGAS metrics hoạt động

**Timeline:** 15 phút

---

### **PHASE 3: Async Benchmark Runner + Regression Testing (60 phút)**

#### Task 3.1: Hoàn thiện BenchmarkRunner

**Mục tiêu:** Implement async runner để execute toàn bộ pipeline

**File:** `engine/runner.py`

**Chi tiết:**

- [x] Complete `run_single_test(test_case)`:
  - Gọi Agent async
  - Tính Hit Rate/MRR từ retrieved context
  - Chạy Multi-Judge
  - Record latency & token usage
  - Return structured result
  
- [x] Complete `run_all(dataset, batch_size=5)`:
  - Dùng `asyncio.gather()` với batch processing
  - Respect rate limits (batch_size = 5)
  - Progress bar (tqdm)
  - Handle errors gracefully
  
- [x] Implement `calculate_metrics()`:
  - Tính aggregate metrics từ toàn bộ results:
    - avg_score, avg_hit_rate, avg_mrr
    - avg_latency, total_tokens
    - pass_rate (% cases with score > 3)
    - cost estimate
  
- [x] Implement cost tracking:
  - Track token usage per model
  - Estimate cost (based on current pricing)
  - Output: `cost_per_eval`, `cost_per_1k_tokens`

**Acceptance Criteria:**

- Toàn bộ 50+ test cases chạy < 2 phút
- Metrics được tính đúng
- Cost tracking hoạt động
- Error handling đầy đủ

**Timeline:** 30 phút

---

#### Task 3.2: Regression Testing & Release Gate

**Mục tiêu:** Implement logic so sánh V1 vs V2 và self-decide release

**File:** `main.py` (tuỳ chỉnh)

**Chi tiết:**

- [x] Implement version comparison logic:
  - So sánh `Agent_V1_Base` vs `Agent_V2_Optimized`
  - Tính delta cho mỗi metric
  - Generate delta report
  
- [x] Implement **Release Gate** logic:**
  - **APPROVE nếu:**
    - avg_score_v2 > avg_score_v1
    - hit_rate_v2 >= hit_rate_v1 (không giảm)
    - latency_v2 <= latency_v1 + 5% (có thể chậm 5%)
    - cost_v2 < cost_v1 (tối ưu chi phí)
  - **BLOCK nếu** bất kỳ điều nào thất bại
  - Output: Decision + Reasoning
  
- [x] Implement thresholds (configurable):

  ```json
  {
    "min_avg_score": 3.0,
    "min_hit_rate": 0.7,
    "max_hallucination_rate": 0.15,
    "max_latency": 5.0
  }
  ```

**Acceptance Criteria:**

- Release Gate logic hoạt động chính xác
- Ghi lại decision reasoning

**Timeline:** 20 phút

---

#### Task 3.3: Comprehensive Report Generation

**Mục tiêu:** Tạo reports chuẩn bị nộp bài

**File:** `main.py` output

**Chi tiết:**

- [ ] Generate `reports/summary.json`:

  ```json
  {
    "metadata": {
      "version": "Agent_V2_Optimized",
      "total_cases": 60,
      "timestamp": "2026-04-21 XX:XX:XX"
    },
    "metrics": {
      "avg_score": 4.2,
      "hit_rate": 0.85,
      "mrr": 0.72,
      "agreement_rate": 0.88,
      "avg_latency": 1.2,
      "total_tokens": 12500,
      "cost_usd": 0.45
    },
    "regression": {
      "v1_score": 3.8,
      "v2_score": 4.2,
      "delta": "+0.4",
      "decision": "APPROVE"
    }
  }
  ```
  
- [ ] Generate `reports/benchmark_results.json`:
  - Chi tiết từng test case (question, answer, scores, latency)
  
- [ ] Verify bằng `python check_lab.py`:
  - Check file tồn tại
  - Check JSON valid
  - Check required fields
  - Warnings nếu thiếu retrieval/judge metrics

**Acceptance Criteria:**

- 2 report files được tạo
- Format đúng
- check_lab.py pass

**Timeline:** 10 phút

---

### **PHASE 4: Failure Analysis & Optimization (45 phút)**

#### Task 4.1: Detailed Failure Analysis

**Mục tiêu:** Phân tích lỗi và xác định nguyên nhân gốc rễ

**File:** `analysis/failure_analysis.md`

**Chi tiết:**

- [ ] Xác định failed cases (score < 3):
  - Danh sách failed cases
  - Pattern phân tích (adversarial? out-of-scope? retrieval error?)
  
- [ ] Implement **5 Whys Analysis** cho top 3 failed cases:

  ``` markdown
  Case: "Question about X"
  Score: 1.5 (FAIL)
  Root Cause Analysis:
  
  Why 1: Câu trả lời không đúng?
  → Porque Agent trả lời không based on context
  
  Why 2: Tại sao không based on context?
  → Vì retrieval không lấy đúng document
  
  Why 3: Tại sao retrieval fail?
  → Vì query embedding không match document embedding
  
  Why 4: Tại sao embedding không match?
  → Vì chunking strategy tạo chunks quá dài/ngắn
  
  Why 5: Tại sao chunking strategy lại vậy?
  → Căn cơ: [Kết luận cuối cùng]
  ```
  
- [ ] Phân loại lỗi theo nguồn gốc:
  - **Ingestion**: Data preparation issues
  - **Chunking**: Tách documents không tối ưu
  - **Retrieval**: Vector search fail
  - **Ranking**: Sắp xếp results sai
  - **Prompting**: Instruction không rõ ràng
  - **LLM**: Mô hình sinh ra hallucination
  - **Judge**: Đánh giá quá strict/quá lỏng
  
- [ ] Metrics phân tích:
  - Failure rate by type
  - Failure rate by difficulty
  - Correlation: retrieval_quality ↔ answer_quality
  
- [ ] Write up:
  - High-level summary (2-3 paragraphs)
  - Detailed findings
  - Recommendations for improvement

**Acceptance Criteria:**

- ≥3 failed cases được phân tích 5 Whys
- Rõ ràng xác định được root cause
- Có recommendations hành động

**Timeline:** 25 phút

---

#### Task 4.2: Optimization Recommendations

**Mục tiêu:** Đề xuất cách cải thiện hệ thống

**File:** `analysis/failure_analysis.md` (tiếp)

**Chi tiết:**

- [ ] Cost optimization:
  - Hiện tại: $X cho 60 queries
  - Đề xuất: Cách giảm 30% cost mà không giảm accuracy
  - (Ví dụ: Dùng GPT-4 mini thay vì GPT-4o, batch queries, caching)
  
- [ ] Performance optimization:
  - Hiện tại: X phút cho 60 queries
  - Đề xuất: Cách chạy nhanh hơn (parallel processing, model optimization)
  
- [ ] Accuracy improvement:
  - Identify top-3 failure modes
  - Propose concrete fixes:
    - Ví dụ: "Refine chunking strategy từ 512 tokens → 256 tokens"
    - Ví dụ: "Add re-ranking layer giữa retrieval và generation"

**Acceptance Criteria:**

- 3+ concrete recommendations
- Feasible implementation
- Estimated impact

**Timeline:** 15 phút

---

#### Task 4.3: Individual Reflection Documents (Optional)

**Mục tiêu:** Mỗi thành viên nhóm ghi lại cá nhân learning & contribution

**File:** `analysis/reflections/reflection_[Tên_SV].md` (mỗi người một file)

**Chi tiết:**

- [ ] Mỗi thành viên tạo file:

  ```markdown
  # Reflection: [Tên Sinh Viên]
  
  ## 📌 Công việc chính tôi làm
  - Task X (component Y)
  - Task Z (component W)
  
  ## 🎓 Kỹ thuật tôi học được
  - Async/await patterns
  - Multi-judge consensus logic
  - etc.
  
  ## 🤔 Thách thức gặp phải
  - Challenge 1 & cách giải quyết
  - Challenge 2 & cách giải quyết
  
  ## 💡 Insights cả nhóm
  - Điều gây bất ngờ
  - Điều chứng minh công việc RAG đúng khó
  ```

**Acceptance Criteria:**

- 1 file per person
- Genuine reflection (không generic)

**Timeline:** 10 phút

---

## 🚀 Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate Golden Dataset
python data/synthetic_gen.py

# 3. Run Benchmark & generate reports
python main.py

# 4. Verify format before submission
python check_lab.py

# 5. (Optional) Run individual component tests
python -m pytest engine/tests/  # nếu có
```

---

## ✅ Submission Checklist

Before submit, verify:

- [ ] `data/golden_set.jsonl` exists with ≥60 cases
- [ ] `reports/summary.json` exists with all required fields
- [ ] `reports/benchmark_results.json` exists with per-case details
- [ ] `analysis/failure_analysis.md` filled with 5 Whys analysis
- [ ] `analysis/reflections/reflection_*.md` exists (≥1 per person)
- [ ] `python check_lab.py` passes without errors
- [ ] `.env` file is in `.gitignore` (API keys safe)
- [ ] Git commits are clear and descriptive

---

## 📊 Success Metrics

| Metric | Target | Weight |

|--------|--------|--------|
| Hit Rate | ≥0.80 | 15% |
| Multi-Judge Agreement | ≥0.80 | 20% |
| Avg Score | ≥4.0 | 15% |
| Execution Time | <2 min for 60 cases | 15% |
| Root Cause Analysis | ≥3 cases detailed | 20% |
| Code Quality | Clean, documented | 15% |

---

## 🎯 Key Takeaways for Excellence

1. **Retrieval is Critical**: Don't skip hit_rate/MRR calculation. This proves your system actually works.
2. **Multi-Judge = Credibility**: Using 2+ judges shows professional evaluation rigor.
3. **Failure Analysis = Depth**: The 5 Whys analysis separates good from excellent work.
4. **Async Performance**: Running 60 cases in <2 min shows you understand modern AI system design.
5. **Cost Awareness**: Track tokens & propose 30% cost savings to show business thinking.

---

**Last Updated:** April 21, 2026  
**Status:** Ready for implementation
