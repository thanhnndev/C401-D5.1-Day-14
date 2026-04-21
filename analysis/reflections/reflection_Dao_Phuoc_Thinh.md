# Báo cáo Cá nhân - Lab Day 14
## Thông tin cá nhân
- **Họ và tên:** Đào Phước Thinh
- **Mã sinh viên:** 2A202600029
- **Vai trò trong nhóm:** Data Engineer & RAG Developer

---

## 1. Đóng góp kỹ thuật (Engineering Contributions)

Trong dự án này, tôi chịu trách nhiệm xây dựng phần lớn pipeline dữ liệu và logic cốt lõi của hệ thống RAG, ngoại trừ phần Engine đánh giá (Judge logic). Các đóng góp cụ thể bao gồm:

### 🛠️ Data Ingestion & Chunking
- Triển khai **Semantic Chunking**: Thay vì chia nhỏ văn bản theo độ dài cố định, tôi đã xây dựng logic phân mảnh dựa trên ngữ nghĩa của văn bản, đảm bảo các đoạn văn bản (chunks) giữ được trọn vẹn ngữ cảnh của thông tin sản phẩm.
- **Metadata Extraction**: Tự động trích xuất các thông tin từ YAML frontmatter của tài liệu (tên sản phẩm, ngôn ngữ, mã tài liệu) để bổ sung vào metadata trong Vector DB, phục vụ cho việc lọc (filtering) và tính toán Hit Rate chính xác.

### 🏗️ Indexing & Vector Database
- Cấu hình và quản lý **Qdrant Vector Database**: Thiết lập collection với vector size tương ứng với Jina Embeddings (1024 dims).
- Tối ưu hóa quá trình indexing bằng cách kết hợp **Dense Vectors** (Jina) và **Sparse Vectors** (BM25) để hỗ trợ Hybrid Search, giúp cải thiện khả năng tìm kiếm cả về ngữ nghĩa lẫn từ khóa chính xác.

### 🧪 Synthetic Data Generation (SDG)
- Xây dựng script `data/synthetic_gen.py` sử dụng LLM để tạo ra bộ **Golden Dataset** gồm hơn 50 test cases.
- Mỗi test case bao gồm: Câu hỏi, Câu trả lời mong muốn (Ground Truth), và danh sách ID các chunk chứa thông tin (Ground Truth Context IDs) để phục vụ việc đánh giá Retrieval stage.

### 🔍 Retrieval & Generation Pipeline
- Thiết kế logic **Hybrid Retrieval**: Kết hợp kết quả từ Vector Search và Full-text Search, sau đó đưa qua **Jina Reranker** để tái sắp xếp thứ tự tài liệu, đảm bảo tài liệu phù hợp nhất nằm ở Top 1.
- Tích hợp pipeline Generation: Kết nối kết quả truy xuất vào Prompt Template để LLM (Qwen/GPT) sinh câu trả lời, đảm bảo tính nhất quán và giảm thiểu Hallucination.

---

## 2. Chiều sâu kỹ thuật (Technical Depth)

Tôi đã áp dụng và nắm vững các khái niệm then chốt trong đánh giá hệ thống RAG:

- **MRR (Mean Reciprocal Rank):** Chỉ số này giúp tôi đánh giá hiệu quả của bước Retrieval. Nó không chỉ quan tâm đến việc có tìm thấy tài liệu hay không (Hit Rate) mà còn quan tâm tài liệu đó nằm ở vị trí thứ mấy. MRR cao chứng tỏ hệ thống Reranker đang hoạt động hiệu quả khi đưa tài liệu đúng lên đầu trang.
- **Cohen's Kappa:** Khi sử dụng Multi-Judge (nhiều mô hình đánh giá khác nhau), tôi hiểu rằng cần phải đo lường độ đồng thuận giữa chúng. Cohen's Kappa giúp loại bỏ yếu tố "đoán mò" và khẳng định liệu điểm số các Judge đưa ra có thực sự khách quan hay không.
- **Position Bias:** Trong quá trình thực hiện, tôi nhận thấy LLM thường bị ảnh hưởng bởi vị trí của thông tin trong Context (hiện tượng "Lost in the Middle"). Việc sắp xếp các chunks quan trọng nhất lên đầu thông qua Reranker là cực kỳ quan trọng để tránh bias này.

---

## 3. Giải quyết vấn đề (Problem Solving)

**Vấn đề:** Trong quá trình chạy thử nghiệm ban đầu (V1), chúng tôi nhận thấy Hit Rate cho các câu hỏi về thông số kỹ thuật (như loại màn hình, kích thước) rất thấp do các từ khóa chuyên môn bị lu mờ bởi tên sản phẩm xuất hiện quá nhiều.

**Giải pháp:** Tôi đã đề xuất và triển khai **Hybrid Search**. Bằng cách bổ sung BM25, các từ khóa hiếm nhưng quan trọng (ví dụ: "display", "resolution") được đẩy trọng số cao hơn. Kết quả là Hit Rate ở phiên bản V2 đã tăng đáng kể (từ 33% lên 66% như trong báo cáo phân tích thất bại).

**Bài học rút ra:** Không có một phương pháp retrieval nào là hoàn hảo cho mọi loại dữ liệu. Việc kết hợp đa phương thức (Hybrid) và có một bộ Metrics đánh giá chi tiết là cách duy nhất để tối ưu hóa hệ thống một cách khoa học.

---

## 4. Bằng chứng đóng góp (Commit Evidence)

Để minh chứng cho các đóng góp kỹ thuật nêu trên theo yêu cầu của rubric, dưới đây là các commit tiêu biểu của tôi trong repository:

- `f0150a4`: Triển khai các module cốt lõi: `chunking.py`, `index.py`, và `synthetic_gen.py`.
- `80d8521`: Hoàn thiện quá trình chạy benchmark và thực hiện phân tích kết quả (analysis).

Các đóng góp này khẳng định vai trò của tôi trong việc xây dựng nền tảng dữ liệu và hạ tầng đánh giá cho toàn đội.

