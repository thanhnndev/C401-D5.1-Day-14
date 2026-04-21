# Individual Reflection - Nguyen Tri Nhan

## 1) Engineering Contribution (15đ)

### Đóng góp chính đã thực hiện

- Hoàn thiện Retrieval Evaluator trong [engine/retrieval_eval.py](engine/retrieval_eval.py):
  - Implement calculate_hit_rate(expected_ids, retrieved_ids, top_k)
  - Implement calculate_mrr(expected_ids, retrieved_ids)
  - Implement evaluate_batch cho toàn bộ dataset với thống kê evaluated/skipped/per_query
- Hoàn thiện Multi-Judge Engine trong [engine/llm_judge.py](engine/llm_judge.py):
  - Hỗ trợ 2 model judge và chấm song song async
  - Thêm consensus logic, conflict resolution (ngưỡng lệch điểm), agreement rate
  - Thêm position bias check bằng cách đảo thứ tự A/B
- Tích hợp RAGAS + fallback custom metrics trong [engine/runner.py](engine/runner.py):
  - faithfulness, answer_relevancy, context_precision
  - Nếu RAGAS lỗi/không khả dụng thì fallback custom metric để pipeline không dừng
- Hoàn thiện Async Benchmark Runner trong [engine/runner.py](engine/runner.py):
  - run_single_test đầy đủ (ragas + retrieval + judge + token/cost)
  - run_all theo batch, có progress bar, có error handling từng case
  - calculate_metrics tổng hợp score, hit_rate, mrr, latency, cost, pass_rate
- Hoàn thiện Regression Release Gate trong [main.py](main.py):
  - So sánh V1/V2 theo score, hit_rate, latency, cost
  - Quyết định APPROVE/BLOCK RELEASE có reasoning và threshold cấu hình
  - Xuất reports/summary.json + reports/benchmark_results.json

### Bằng chứng commit

- Branch làm việc: main (hiện tại)
- Hash commit cụ thể (with link):
  - [b2201265349cd3c89deda381681e77d6287ba0e3](https://github.com/thanhnndev/C401-D5.1-Day-14/pull/1/commits/b2201265349cd3c89deda381681e77d6287ba0e3)
  - [79cb4b835630ebcbefc6adb534006c9040a2cd38](https://github.com/thanhnndev/C401-D5.1-Day-14/pull/1/commits/79cb4b835630ebcbefc6adb534006c9040a2cd38)
  - [c6d005045f6f6c7847e8bb5ed988ee67587aa1d5](https://github.com/thanhnndev/C401-D5.1-Day-14/pull/1/commits/c6d005045f6f6c7847e8bb5ed988ee67587aa1d5)
  
## 2) Technical Depth (15đ)

### MRR (Mean Reciprocal Rank)

- Đã áp dụng đúng định nghĩa: với mỗi query, lấy vị trí xuất hiện đầu tiên của tài liệu đúng, tính 1/rank.
- Nếu không có tài liệu đúng trong danh sách retrieved thì MRR = 0.
- Đã hiện thực trong [engine/retrieval_eval.py](engine/retrieval_eval.py).

### Position Bias

- Đã triển khai kiểm tra thiên vị vị trí trong [engine/llm_judge.py](engine/llm_judge.py):
  - Chạy so sánh Answer A/B ở thứ tự gốc
  - Đảo thứ tự B/A và so lại
  - So consistency giữa hai lần để phát hiện bias

### Trade-off Chi phí vs Chất lượng

- Đã thêm token usage + cost estimation trong [engine/runner.py](engine/runner.py) để theo dõi cost theo từng test case và tổng hợp toàn batch.
- Đã tích hợp release gate trong [main.py](main.py) có điều kiện cost_v2 < cost_v1 đồng thời vẫn phải giữ/improve score và hit_rate.
- Nhận định thực tế:
  - Dùng judge mạnh hơn có thể tăng quality nhưng tăng chi phí.
  - Batch async giúp giảm thời gian chạy nhưng cần kiểm soát rate limit và độ ổn định API.

## 3) Problem Solving (10đ)

### Vấn đề phát sinh và cách xử lý

- Vấn đề: lỗi typing khi truy cập to_pandas từ kết quả RAGAS.
  - Cách xử lý: truy xuất an toàn bằng getattr, fallback dict-like, cast type phù hợp để hết lỗi static analysis.
- Vấn đề: runtime có thể thiếu/không ổn định RAGAS.
  - Cách xử lý: thiết kế custom-fallback metric cùng tên để pipeline luôn chạy.
- Vấn đề: benchmark có thể fail từng case làm hỏng cả batch.
  - Cách xử lý: dùng asyncio.gather(return_exceptions=True) và chuẩn hóa error result theo từng case.
- Vấn đề: khó tự động quyết định release khi metric thay đổi nhiều chiều.
  - Cách xử lý: xây gate rõ điều kiện pass/fail + threshold cấu hình + reasoning chi tiết.

### Bài học rút ra

- Khi build evaluation pipeline, tính ổn định quan trọng không kém độ chính xác.
- Cần luôn có fallback path cho các thành phần phụ thuộc dịch vụ ngoài.
- Chuẩn hóa output schema từ sớm giúp downstream report/regression làm việc nhất quán.

## 4) Tự đánh giá theo rubric

| Hạng mục | Tự đánh giá | Ghi chú |
| :--- | :---: | :--- |
| Engineering Contribution | 14/15 | Đã hoàn thiện nhiều module chính; thiếu bằng chứng commit cụ thể trong report hiện tại |
| Technical Depth | 12/15 | MRR và Position Bias rõ; Cohen's Kappa chưa có triển khai/thống kê |
| Problem Solving | 9/10 | Có nêu vấn đề thực tế và cách xử lý cụ thể |
| **Tổng** | **35/40** | Có thể tăng điểm nếu bổ sung commit hash, PR, và phân tích Kappa |
