# Báo cáo cá nhân - Lab Day 14

## Thông tin cá nhân
- **Họ và tên:** Nông Nguyễn Thành
- **Mã sinh viên:** 2A202600250
- **Vai trò trong nhóm:** Integration Lead, Reviewer & Release Coordinator

---

## 1. Đóng góp kỹ thuật (Engineering Contributions)

Trong dự án Lab Day 14, tôi tập trung vào vai trò điều phối tích hợp các nhánh chức năng, kiểm soát chất lượng đầu ra và hoàn thiện tài liệu nộp bài theo đúng rubric.

### 🧩 Tích hợp nhánh và ổn định pipeline
- Thực hiện merge theo từng mốc chức năng của dự án, đảm bảo các module từ các thành viên hoạt động đồng bộ trên nhánh chính:
  - Merge nhánh eval engine + fallback metrics.
  - Merge nhánh indexing/chunking/runner.
  - Merge nhánh agent retrieval và cập nhật tài liệu báo cáo.
- Kiểm tra tính tương thích sau merge giữa các phần Retrieval, Judge, Runner và cơ chế Release Gate trong luồng chạy `main.py`.

### ✅ Kiểm soát chất lượng theo checklist nộp bài
- Đối chiếu tiến độ với checklist trong `README.md`, bảo đảm repo có đủ thành phần yêu cầu: mã nguồn, báo cáo benchmark, failure analysis và reflection cá nhân.
- Hỗ trợ rà soát luồng chạy chuẩn (`synthetic_gen.py` -> `main.py` -> `check_lab.py`) để giảm rủi ro thiếu artifact khi nộp.

### 📝 Chuẩn hóa tài liệu báo cáo
- Điều phối và hợp nhất báo cáo cá nhân của các thành viên vào repo.
- Đảm bảo nội dung báo cáo bám rubric: nêu rõ đóng góp kỹ thuật, chiều sâu kiến thức, vấn đề đã xử lý và minh chứng commit.

---

## 2. Chiều sâu kỹ thuật (Technical Depth)

Qua quá trình tích hợp và review, tôi củng cố các điểm kỹ thuật cốt lõi của Evaluation Factory:

- **Regression Gate theo đa chỉ số:** Việc quyết định release không dựa vào một metric đơn lẻ mà cần đồng thời theo dõi chất lượng trả lời (score), hiệu quả retrieval (hit rate/mrr), độ trễ và chi phí. Cách tiếp cận này giảm nguy cơ "tăng chỉ số A nhưng làm giảm chất lượng tổng thể".
- **Ý nghĩa của pipeline bền vững:** Trong hệ thống eval thực tế, tính ổn định của pipeline (khả năng chạy hết batch, có fallback khi lỗi) quan trọng tương đương độ chính xác.
- **Tư duy tích hợp theo giai đoạn:** Tách thành các mốc merge nhỏ giúp cô lập rủi ro, dễ xác định nguyên nhân khi regression xuất hiện sau mỗi lần tích hợp.

---

## 3. Giải quyết vấn đề (Problem Solving)

**Vấn đề:** Dự án có nhiều nhánh phát triển song song (data, retrieval, judge, runner, docs), nếu hợp nhất không theo mốc rõ ràng sẽ dễ phát sinh xung đột hoặc khó truy vết khi chất lượng benchmark giảm.

**Giải pháp triển khai:**
- Tích hợp theo từng đợt chức năng, ưu tiên các nhánh nền tảng trước (engine/retrieval/runner), sau đó mới đến nhánh docs/report.
- Giữ lịch sử merge rõ ràng để có thể truy xuất ngược nguyên nhân thay đổi khi đánh giá release.
- Đối chiếu kết quả benchmark/failure analysis sau các đợt merge để đảm bảo quyết định release có cơ sở.

**Bài học rút ra:** Với bài toán nhiều thành phần liên kết, kỹ năng điều phối tích hợp và chuẩn hóa quy trình nộp bài là yếu tố quyết định để biến các module riêng lẻ thành một hệ thống đánh giá hoàn chỉnh, có thể vận hành và báo cáo được.

---

## 4. Bằng chứng đóng góp (Commit Evidence)

Theo toàn bộ lịch sử git, các commit mang author `thanhnndev` gồm:

- `ca9aa12`: docs: Add personal reflection report for Nông Nguyễn Thành on Lab Day 14.
- `7599cad`: feat: Implement automated QA pair generation from product documentation.
- `1af8a5c`: fix: Refine SCHWIND product documentation for clarity and accuracy.
- `74296a9`: fix: Enhance clarity and precision in SCHWIND product documentation.
- `2e9f09f`: fix: Update Cliniclave 45 and Melacontrol Type 5 documentation in English and Vietnamese.
- `68e1fee`: fix: Update phaco equipment documentation in English and Vietnamese.
- `d0504f7`: feat: Add new product documentation for BVI and Melag devices.

Ngoài ra, các commit merge do tài khoản hiển thị tên `Nông Nguyễn Thành` (cùng vai trò điều phối tích hợp) gồm:

- `d33ccde`: Merge pull request #1 từ nhánh `feat/eval-engine-ragas-fallback`.
- `76269a3`: Merge pull request #2 từ nhánh `feat/index`.
- `317959e`: Merge pull request #3 từ nhánh `feat/runner`.
- `e81f3a8`: Merge pull request #4 từ nhánh `feat/agent_retrieval`.
- `86d4d50`: Merge pull request #5 từ nhánh `feat/agent_retrieval`.
- `7df5ab4`: Merge pull request #6 từ nhánh `docs/NguyenTriNhan`.

Các commit trên thể hiện đầy đủ cả hai nhóm đóng góp: phát triển dữ liệu/tài liệu sản phẩm và điều phối tích hợp các nhánh chức năng vào nhánh chính.
