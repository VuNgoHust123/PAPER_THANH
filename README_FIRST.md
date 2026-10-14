# PAPER_THANH — Hướng Dẫn Sử Dụng Repo

Repo này chứa toàn bộ tài liệu, code và dữ liệu cho đề tài nghiên cứu về **dự đoán hành vi mua hàng sau khi thêm vào giỏ hàng trên nền tảng thương mại điện tử**, sử dụng Machine Learning và Explainable AI (XAI).

---

## Mục Lục

1. [Tổng Quan Repo](#1-tổng-quan-repo)
2. [Cài Đặt Môi Trường](#2-cài-đặt-môi-trường)
3. [Hướng Dẫn Chạy Notebook](#3-hướng-dẫn-chạy-notebook)
4. [Giải Thích Từng File](#4-giải-thích-từng-file)
5. [Bộ Dữ Liệu](#5-bộ-dữ-liệu)
6. [Kết Quả Đầu Ra](#6-kết-quả-đầu-ra)

---

## 1. Tổng Quan Repo

### Về bài báo gốc
Repo này tái hiện framework của bài báo:

> Esmeli, R. & Gokce, A. (2025). *An Analysis of Consumer Purchase Behavior Following Cart Addition in E-Commerce Utilizing Explainable Artificial Intelligence.* Journal of Theoretical and Applied Electronic Commerce Research, 20, 28.

**Vấn đề:** Tỷ lệ bỏ giỏ hàng (cart abandonment) trên các sàn thương mại điện tử rất cao (67–81%). Bài báo xây dựng một framework ML 3 giai đoạn để dự đoán liệu khách hàng có mua hàng sau khi đã thêm sản phẩm vào giỏ hay không, sử dụng SHAP để giải thích kết quả.

**Kết quả chính:** Gradient Boosting (GB) đạt F1 Score **89%**, với các yếu tố quan trọng nhất là lịch sử mua hàng, vị trí địa lý, và biến động giá trong phiên.

### Cấu trúc repo

```
PAPER_THANH/
├── README_FIRST.md                        ← File này (đọc trước tiên)
├── Thanh_Paper2.pdf                       ← Bài báo gốc (PDF)
├── Purchase_Prediction_ECommerce.ipynb    ← Notebook chính
├── Purchase_Prediction_VI.md              ← Hướng dẫn chi tiết (Tiếng Việt)
├── Purchase_Prediction_EN.md              ← Hướng dẫn chi tiết (Tiếng Anh)
└── data_ecommerce/
    └── online_shoppers_intention.csv      ← Bộ dữ liệu UCI (12,330 phiên)
```

---

## 2. Cài Đặt Môi Trường

### Yêu cầu hệ thống
- Python **3.8+**
- GPU (CUDA) — khuyến nghị nhưng không bắt buộc (code tự phát hiện)

### Cài đặt thư viện

```bash
pip install torch scikit-learn pandas numpy matplotlib imbalanced-learn openpyxl
```

| Thư viện | Mục đích |
|----------|---------|
| `torch` | Xây dựng và huấn luyện MLP, CNN, LSTM |
| `scikit-learn` | LR, DT, KNN, RF, GB; StratifiedKFold; metrics |
| `pandas` | Đọc và xử lý dữ liệu |
| `numpy` | Tính toán mảng số học |
| `matplotlib` | Vẽ biểu đồ |
| `imbalanced-learn` | SMOTE để cân bằng dữ liệu |
| `openpyxl` | Đọc/ghi file Excel (nếu cần) |

### Kiểm tra cài đặt

```python
import torch, sklearn, pandas, numpy, matplotlib, imblearn
print("torch:", torch.__version__)
print("sklearn:", sklearn.__version__)
print("GPU available:", torch.cuda.is_available())
```

---

## 3. Hướng Dẫn Chạy Notebook

### Bước 1 — Mở notebook

Mở file `Purchase_Prediction_ECommerce.ipynb` trong một trong các công cụ sau:
- **VS Code** (khuyến nghị — cài extension Python + Jupyter)
- **Jupyter Notebook** (`jupyter notebook` trong terminal)
- **JupyterLab** (`jupyter lab` trong terminal)

### Bước 2 — Chạy toàn bộ

Chạy tất cả các cell **theo thứ tự từ trên xuống dưới**:
- VS Code: nhấn nút **Run All** (▶▶) trên thanh công cụ notebook
- Jupyter: menu **Kernel → Restart & Run All**

> **Quan trọng:** Không thay đổi thứ tự cell. Các cell sau phụ thuộc vào biến (`X`, `y`, `results`, v.v.) được định nghĩa ở cell trước.

### Bước 3 — Chờ kết quả

| Phần | Thời gian (GPU) | Thời gian (CPU) |
|------|----------------|----------------|
| Setup + Load data | < 1 phút | < 1 phút |
| 5 Sklearn models (LR/DT/KNN/RF/GB) | ~1–3 phút | ~2–5 phút |
| 3 Neural models (MLP/CNN/LSTM) | ~5–15 phút | ~30–60 phút |
| Vẽ biểu đồ | < 1 phút | < 1 phút |

**Tổng cộng:** ~10–20 phút trên GPU / ~35–70 phút trên CPU

### Mẹo tăng tốc (nếu máy chậm)

Trong cell **Setup (cell thứ 2)**, sửa:
```python
n_splits = 3   # Giảm từ 5 → 3
```

Trong cell **Run All Models**, sửa:
```python
results[name] = neural_kfold(..., epochs=15, ...)   # Giảm epochs từ 30 → 15
```

---

## 4. Giải Thích Từng File

### `Thanh_Paper2.pdf`
Bài báo gốc được tái hiện trong notebook này. Đọc để hiểu:
- Bối cảnh và động lực nghiên cứu (tỷ lệ bỏ giỏ hàng cao)
- Framework 3 giai đoạn đề xuất
- Kết quả và phân tích SHAP trên bộ dữ liệu của tác giả

### `Purchase_Prediction_ECommerce.ipynb`
Notebook Jupyter chính. Gồm các phần:

| Phần | Cell | Nội dung |
|------|------|---------|
| Setup | 1–2 | Import thư viện, cài đặt seed/device |
| Stage 1 | 3–5 | Tải và khám phá dữ liệu UCI |
| Stage 2 | 6–7 | Feature engineering (6 đặc trưng mới) |
| Stage 3 | 8–13 | Tiền xử lý + định nghĩa 8 model |
| CV & Training | 14–15 | Hàm K-fold CV cho sklearn và neural nets |
| Run | 16 | **Chạy tất cả 8 model với 5-fold CV** |
| Results | 17–21 | Bảng kết quả + 5 loại biểu đồ |

Sau khi chạy, toàn bộ kết quả (biểu đồ và bảng số liệu) được nhúng trực tiếp vào các cell output của notebook.

### `Purchase_Prediction_VI.md`
Hướng dẫn chi tiết **bằng Tiếng Việt** bao gồm:
- Tóm tắt bài báo và framework
- Giải thích từng cell trong notebook
- Hướng dẫn đọc hiểu từng loại biểu đồ kết quả
- So sánh với kết quả bài báo gốc

### `Purchase_Prediction_EN.md`
Hướng dẫn chi tiết **bằng Tiếng Anh** — nội dung tương tự bản Tiếng Việt.

---

## 5. Bộ Dữ Liệu

### Nguồn gốc

| Thông tin | Chi tiết |
|-----------|---------|
| Tên | UCI Online Shoppers Purchasing Intention Dataset |
| Nguồn | UCI Machine Learning Repository |
| ID | 468 |
| URL | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Cấp phép | CC BY 4.0 (mở, miễn phí sử dụng) |

### Cấu trúc dữ liệu

File: `data_ecommerce/online_shoppers_intention.csv`

- **Số dòng:** 12,330 phiên mua sắm
- **Số cột:** 18 (17 đặc trưng + 1 nhãn)
- **Nhãn:** `Revenue` — `True` (đã mua) / `False` (không mua)
- **Định dạng:** CSV, mã hóa UTF-8, dấu phẩy là dấu tách

**17 đặc trưng gốc:**

| Nhóm | Đặc trưng | Kiểu | Mô tả |
|------|-----------|------|-------|
| **Hành vi trang** | Administrative | int | Số trang quản trị đã xem |
| | Administrative_Duration | float | Thời gian trên trang quản trị (giây) |
| | Informational | int | Số trang thông tin đã xem |
| | Informational_Duration | float | Thời gian trên trang thông tin |
| | ProductRelated | int | Số trang sản phẩm đã xem |
| | ProductRelated_Duration | float | Thời gian trên trang sản phẩm |
| **Chỉ số chất lượng** | BounceRates | float | Tỷ lệ thoát trung bình (0–1) |
| | ExitRates | float | Tỷ lệ rời trang trung bình (0–1) |
| | PageValues | float | Giá trị trang trước giao dịch |
| | SpecialDay | float | Gần ngày đặc biệt (0–1) |
| **Bối cảnh** | Month | str | Tháng của phiên (Jan–Dec) |
| | OperatingSystems | int | Loại HĐH (1–8) |
| | Browser | int | Loại trình duyệt (1–13) |
| | Region | int | Vùng địa lý (1–9) |
| | TrafficType | int | Nguồn traffic (1–20) |
| | VisitorType | str | Returning / New / Other |
| | Weekend | bool | Cuối tuần hay không |

**Mất cân bằng lớp:**
- Không mua: **10,422 phiên (84.5%)**
- Có mua: **1,908 phiên (15.5%)**
- → Lý do phải dùng **SMOTE** để cân bằng trước khi huấn luyện

### Cách dữ liệu được sử dụng

Notebook thực hiện 3 bước xử lý chính:

1. **Feature Engineering** — Tính thêm 6 đặc trưng mới từ dữ liệu gốc (tương ứng các đặc trưng trong bài báo)
2. **Mã hóa** — Chuyển Month (chuỗi → số), VisitorType (chuỗi → số), Weekend (bool → int)
3. **Chuẩn hóa** — `StandardScaler` đưa tất cả đặc trưng về phân phối N(0,1)

---

## 6. Kết Quả Đầu Ra

Sau khi chạy notebook, kết quả được nhúng trực tiếp trong các cell output của file `.ipynb`. Bao gồm:

### Biểu đồ 1 — Phân tích bộ dữ liệu (Stage 1)
3 biểu đồ cạnh nhau:
- Phân bố lớp (mua vs không mua) — xác nhận mất cân bằng ~5:1
- Tỷ lệ mua theo tháng — tháng 11 và 5 thường cao nhất
- Tỷ lệ mua theo loại visitor — Returning Visitor mua nhiều hơn

### Biểu đồ 2 — Feature Engineering (Stage 2)
3 box plots so sánh phân bố đặc trưng theo kết quả mua:
- `pages_viewed` — người mua duyệt nhiều trang hơn
- `price_sensitivity` (PageValues) — người mua xem trang giá trị cao hơn
- `cat_transitions` — người mua chuyển nhiều danh mục hơn

### Biểu đồ 3 — So sánh 8 model (Bar Chart)
4 subplot, mỗi subplot là một metric (F1, Recall, Precision, ROC-AUC) cho cả 8 model:
- **Model tốt nhất** = thanh cao nhất, nhất quán ở cả 4 metric
- **Đường đỏ đứt** = ngưỡng F1=0.89 của bài báo gốc (benchmark)
- Thứ tự kỳ vọng: GB > RF > MLP > LR > CNN > LSTM > KNN > DT

### Biểu đồ 4 — ROC Curves
8 đường ROC trên cùng 1 đồ thị:
- Đường càng gần góc trên-trái = AUC càng cao = model càng tốt
- GB và RF thường tạo đường bao phía trên
- Đường chéo đứt = baseline ngẫu nhiên (AUC = 0.5)

### Biểu đồ 5 — Confusion Matrix
4 ma trận nhầm lẫn cho top 4 model theo F1:
- **TP** (góc dưới-phải): dự đoán mua đúng → gửi khuyến mãi hiệu quả
- **FN** (góc dưới-trái): bỏ sót người mua → mất cơ hội retarget
- **FP** (góc trên-phải): dự đoán nhầm không mua → lãng phí ngân sách
- **TN** (góc trên-trái): dự đoán không mua đúng → không lãng phí

### Biểu đồ 6 — Feature Importance
2 biểu đồ ngang song song cho GB và RF:
- Top 15 đặc trưng quan trọng nhất
- Kỳ vọng: `PageValues`, `ProductRelated_Duration`, `ExitRates`, `BounceRates` dẫn đầu
- Xác nhận kết luận bài báo: biến động giá và địa lý là dự báo quan trọng

### Bảng kết quả tổng hợp

Sau khi chạy, bảng `result_df` hiển thị:

| Model | F1 Score | Recall | Precision | ROC-AUC |
|-------|----------|--------|-----------|---------|
| GB | ~0.65–0.72 | ... | ... | ~0.92–0.94 |
| RF | ~0.58–0.68 | ... | ... | ~0.90–0.92 |
| MLP | ~0.55–0.65 | ... | ... | ~0.88–0.91 |
| CNN | ~0.50–0.62 | ... | ... | ~0.85–0.90 |
| LSTM | ~0.50–0.62 | ... | ... | ~0.85–0.90 |
| LR | ~0.48–0.58 | ... | ... | ~0.88–0.91 |
| KNN | ~0.45–0.58 | ... | ... | ~0.82–0.87 |
| DT | ~0.45–0.55 | ... | ... | ~0.72–0.80 |

> Kết quả thấp hơn bài báo (F1=89%) vì UCI dataset khác với bộ dữ liệu nội bộ của bài báo — thiếu lat/lon chính xác và lịch sử giá chi tiết trong phiên.

---

## Đọc Thêm

- **Hướng dẫn chi tiết Tiếng Việt:** [`Purchase_Prediction_VI.md`](Purchase_Prediction_VI.md)
- **Hướng dẫn chi tiết Tiếng Anh:** [`Purchase_Prediction_EN.md`](Purchase_Prediction_EN.md)
- **Bài báo gốc:** [`Thanh_Paper2.pdf`](Thanh_Paper2.pdf)
