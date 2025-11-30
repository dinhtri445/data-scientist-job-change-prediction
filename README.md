# HR Analytics: Job Change of Data Scientists

## 1. Giới thiệu (Introduction)

### Mô tả bài toán
Trong bối cảnh thị trường nhân sự ngành công nghệ thông tin, đặc biệt là lĩnh vực Khoa học Dữ liệu (Data Science), đang cạnh tranh khốc liệt, một công ty hoạt động trong lĩnh vực Big Data và Data Science muốn tuyển dụng các Kỹ sư Khoa học dữ liệu. Để tối ưu hóa chi phí và thời gian tuyển dụng, công ty tổ chức các khóa đào tạo ngắn hạn và muốn chọn lọc ra những ứng viên cam kết làm việc lâu dài sau khóa học.

### Động lực và Ứng dụng thực tế
* **Động lực:** Việc tuyển dụng, đào tạo một Data Scientist tốn kém nhiều chi phí và thời gian. Nếu ứng viên rời đi ngay sau khi được đào tạo, doanh nghiệp sẽ chịu "tổn thất kép" (mất phí đào tạo và mất nhân lực).
* **Ứng dụng:** 
    * Sàng lọc thông minh: Giúp bộ phận tuyển dụng ưu tiên các ứng viên có độ gắn kết cao
    * Tối ưu hóa ngân sách: Tập trung nguồn lực đào tạo vào nhóm ứng viên tiềm năng nhất.
    * Chiến lược giữ chân nhân tài: Đối với nhân viên hiện tại được dự báo có nguy cơ rời đi (target=1), HR có thể chủ động đề xuất lộ trình thăng tiến hoặc phúc lợi để giữ chân họ.

### Mục tiêu cụ thể
1.  Thực hiện phân tích khám phá để tìm ra các yếu tố tác động chính.
2.  Xây dựng quy trình xử lý dữ liệu từ thô đến sạch.
3.  **Xây dựng thuật toán Logistic Regression từ con số 0** chỉ sử dụng `NumPy` (không dùng Scikit-learn cho lõi thuật toán) để hiểu sâu bản chất toán học.
4.  Áp dụng SMOTE để xử lý khi mất cân bằng
4.  Đạt độ chính xác (Accuracy) và Recall (khả năng phát hiện khách rời bỏ) ở mức chấp nhận được.

## 2. Mục lục (Table of Contents)
- [Giới thiệu](#1-giới-thiệu-introduction)
- [Dataset](#3-dataset)
- [Method (Phương pháp)](#4-method-phương-pháp)
- [Installation & Setup](#5-installation--setup)
- [Usage](#6-usage)
- [Results](#7-results-kết-quả)
- [Project Structure](#8-project-structure)
- [Challenges & Solutions](#9-challenges--solutions)
- [Future Improvements](#10-future-improvements)
- [Contributors & Author](#11-contributors--author-info)
- [License](#13-license)

## 3. Dataset

**Dataset**: HR Analytics: Job Change of Data Scientists  
**Source**: https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists

**Columns**:
- `enrollee_id` : Mã định danh ứng viên
- `city`: Mã thành phố
- `city_ development _index`: Chỉ số phát triển thành phố
- `gender`: Giới tính
- `relevent_experience`: Kinh nghiệm liên quan
- `enrolled_university`: Hình thức đào tạo đại học
- `education_level`: Trình độ học vấn
- `major_discipline`: Chuyên ngành đào tạo
- `experience`: Tổng số năm kinh nghiệm
- `company_size`: Quy mô công ty
- `company_type`: Loại hình công ty
- `last_new_job`: Thời gian từ lần chuyển việc cuối
- `training_hours`: Số giờ đào tạo
- `target`: 0 – Không có ý định chuyển việc, 1 – Đang tìm kiếm/muốn chuyển đổi công việc.

## 4. Method (Phương pháp)
Dự án tuân thủ quy trình Data Science tiêu chuẩn, với điểm nhấn là việc triển khai thủ công các thuật toán toán học.

### Quy trình xử lý (Pipeline)
1.  **Data Cleaning:** Xử lý lỗi logic (`last_new_job > experience`), điền giá trị thiếu (Median/Unknown), xử lý ngoại lai (IQR Capping).
2.  **Feature Engineering:** Tạo các đặc trưng mới mang tính nghiệp vụ:
    * `is_stem_major`: Ứng viên có nền tảng Kỹ thuật không?
    * `experience_level`: Phân nhóm Junior/Mid/Senior.
    * `is_active_learner`: Đang đi học hay không?
    * `stability_ratio`: Tỷ lệ ổn định công việc.
3.  **Feature Selection:** Sử dụng **Chi-Square Test** (cho biến phân loại) và **T-Test** (cho biến số) tự viết bằng NumPy để đánh giá độ quan trọng.
4.  **Data Preparation:** Label Encoding, Train/Test Split, và Standardization (Z-score).

### Thuật toán Logistic Regression (NumPy Implementation)

Mô hình được xây dựng dựa trên công thức toán học gốc:

**1. Hàm kích hoạt (Sigmoid Function):**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**2. Hàm mất mát (Log Loss / Binary Cross Entropy):**
$$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})]$$

**3. Tối ưu hóa (Gradient Descent):**
Cập nhật trọng số $w$ và bias $b$ qua mỗi vòng lặp:
$$dw = \frac{1}{m} X^T (\hat{y} - y)$$
$$db = \frac{1}{m} \sum (\hat{y} - y)$$
$$w := w - \alpha \cdot dw$$
$$b := b - \alpha \cdot db$$

### Xử lý mất cân bằng dữ liệu (SMOTE)
Tự triển khai thuật toán **SMOTE (Synthetic Minority Over-sampling Technique)** bằng NumPy:
* Sử dụng hình học Euclid để tìm $k$ lân cận gần nhất (KNN logic).
* Tạo điểm dữ liệu giả lập bằng nội suy tuyến tính: $New = Old + (Neighbor - Old) \times \delta$.

## 5. Installation & Setup

Yêu cầu: Python 3.8+
```bash
git clone https://github.com/dinhtri445/data-scientist-job-change-prediction
cd project-name
# Cài đặt thư viện cần thiết
pip install -r requirements.txt
```

## 6. Usage
**Chạy từng phần**
- `01_data_exploration.ipynb`: Để có cái nhìn tổng quan về dữ liệu và các vấn đề cần xử lý.
- `02_preprocessing.ipynb`: Thực hiện làm sạch và tạo feature mới. Kết quả sẽ lưu ra file
- `03_modeling.ipynb` để huấn luyện và đánh giá mô hình 

## 7. Results (Kết quả)
**Metrics đạt được trên tập test**
- `Recall (Class 1)`	~67%	Phát hiện được 2/3 số nhân viên có ý định nghỉ việc.
- `Precision`	~40%	Chấp nhận tỷ lệ báo động giả để không bỏ sót nhân tài.
- `F1-Score`	~50%	Sự cân bằng giữa độ chính xác và độ phủ.
- `Accuracy`	~67%	Độ chính xác tổng thể.


## 8. Project Structure

```text
project-name/
├── data/
│   ├── raw/                   # Dữ liệu gốc (aug_train.csv, aug_test.csv)
│   └── processed/             # Dữ liệu sau khi làm sạch (aug_train_processed.csv)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA: Khám phá dữ liệu, tìm lỗi, visualize
│   ├── 02_preprocessing.ipynb     # Cleaning & Feature Engineering
│   └── 03_modeling.ipynb          # Encoding, Scaling, Train Model (NumPy), Evaluation
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Chứa hàm: clean, convert, SMOTE, PCA, T-test, Chi2...
│   ├── visualization.py       # Chứa hàm vẽ biểu đồ: Heatmap, Barplot...
│   └── models.py              # Chứa Class LogisticRegressionNumPy
├── README.md                  # Tài liệu dự án
└── requirements.txt           # Thư viện cần thiết
```

## 9. Challenges & Solutions
Trong quá trình xây dựng dự án hoàn toàn bằng NumPy, nhóm thực hiện đã đối mặt và giải quyết các vấn đề kỹ thuật sau:

| Thách thức (Challenges) | Giải pháp (Solutions) |
| :--- | :--- |
| **1. Dữ liệu mất cân bằng nghiêm trọng:** <br> Mô hình ban đầu có độ nhạy (Recall) cực thấp (~17%), xu hướng dự đoán toàn bộ là nhóm đa số (Class 0). | **Tự triển khai thuật toán SMOTE:** <br> Viết thuật toán *Synthetic Minority Over-sampling Technique* từ con số 0 bằng NumPy để sinh dữ liệu giả lập cho nhóm thiểu số, giúp tăng Recall lên ~67%. |
| **2. Hiệu năng tính toán:** <br> Việc tính toán khoảng cách (trong KNN/SMOTE) hay thống kê Chi-square bằng vòng lặp `for` thuần túy rất chậm. | **Tối ưu hóa Vectorization:** <br> Thay thế hoàn toàn vòng lặp bằng kỹ thuật Vector hóa của NumPy (Broadcasting, `np.dot`, `np.outer`, `np.sum` theo axis), giúp tăng tốc độ xử lý gấp nhiều lần. |
| **3. Lỗi Logic trong dữ liệu gốc:** <br> Xuất hiện các mẫu dữ liệu phi logic (VD: Tổng kinh nghiệm ít hơn khoảng cách lần nhảy việc cuối cùng). | **Data Sanity Check:** <br> Thiết lập quy trình kiểm tra và sửa lỗi logic (Correction Logic) ngay tại bước tiền xử lý để đảm bảo tính nhất quán của dữ liệu đầu vào. |

## 10. Future Improvements
- Mở rộng thuật toán: Thử nghiệm xây dựng Neural Network (Multi-layer Perceptron) từ đầu bằng NumPy để bắt các mối quan hệ phi tuyến tính.

- Tối ưu Hyperparameter: Viết thêm hàm Grid Search thủ công để tìm Learning Rate và Số vòng lặp tối ưu nhất.

 Triển khai: Đóng gói model thành API sử dụng Flask/FastAPI.

## 11. Contributors & Author Info
- Họ và tên : Mai Đình Trí
- MSSV: 23120377
- Lớp/Môn Học: Lập trình cho khoa học dữ liệu 

## 12. Contact
Nếu có thắc mắc , vui lòng liên hệ qua email: maidinhtri8620@gmail.com

## 13. LICENSE
Dự án được cấp phép
