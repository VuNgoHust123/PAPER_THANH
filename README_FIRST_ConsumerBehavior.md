# README_FIRST — Dự Đoán Hành Vi Mua Hàng và Marketing Chính Xác

File này dành riêng cho dự án tái hiện bài báo thứ 3 trong repo (`Thanh_Paper3.pdf`). Repo cũng chứa dự án trước đó (`Thanh_Paper2.pdf` + `Purchase_Prediction_ECommerce.ipynb`) — xem [README_FIRST.md](README_FIRST.md) cho dự án đó.

---

## Mục Lục

1. [Tổng Quan Nội Dung](#1-tổng-quan-nội-dung)
2. [Cài Đặt Môi Trường](#2-cài-đặt-môi-trường)
3. [Hướng Dẫn Chạy Notebook](#3-hướng-dẫn-chạy-notebook)
4. [Giải Thích File Kết Quả Đầu Ra](#4-giải-thích-file-kết-quả-đầu-ra)
5. [Mô Tả Bộ Dữ Liệu](#5-mô-tả-bộ-dữ-liệu)

---

## 1. Tổng Quan Nội Dung

### Bài báo gốc
> Lin, J. (2025). *Application of machine learning in predicting consumer behavior and precision marketing.* PLOS One 20(5): e0321854.

**Vấn đề:** Doanh nghiệp e-commerce cần dự đoán chính xác khách hàng nào sẽ mua hàng để tối ưu marketing cá nhân hóa. Bài báo so sánh 4 model: SVM, XGBoost, CatBoost, BPANN trên bộ dữ liệu UCI Online Shoppers, CatBoost đạt kết quả tốt nhất (ROC AUC = 0.985).

### Các file liên quan đến dự án này

```
PAPER_THANH/
├── README_FIRST_ConsumerBehavior.md         ← File này
├── Thanh_Paper3.pdf                          ← Bài báo gốc (PDF)
├── Purchase_Prediction_ConsumerBehavior.ipynb ← Notebook chính (đã chạy, có kết quả thật)
├── ConsumerBehavior_VI.md                    ← Hướng dẫn chi tiết (Tiếng Việt)
├── ConsumerBehavior_EN.md                    ← Hướng dẫn chi tiết (Tiếng Anh)
├── data_ecommerce/
│   └── online_shoppers_intention.csv         ← Bộ dữ liệu UCI (dùng chung với dự án trước)
└── outputs/                                   ← Biểu đồ và bảng kết quả sinh ra khi chạy notebook
    ├── class_distribution.png
    ├── pearson_correlation.png
    ├── metrics_comparison.png
    ├── roc_curves.png
    ├── confusion_matrices.png
    ├── feature_importance.png
    └── results_table.csv
```

---

## 2. Cài Đặt Môi Trường

### Yêu cầu hệ thống
- Python 3.8+
- GPU (CUDA) khuyến nghị cho model BPANN (PyTorch), không bắt buộc.

### Cài đặt thư viện

```bash
pip install torch scikit-learn pandas numpy matplotlib xgboost catboost imbalanced-learn
```

| Thư viện | Mục đích |
|----------|---------|
| `torch` | Xây dựng và huấn luyện BPANN |
| `scikit-learn` | SVM, train/test split, StandardScaler, LabelEncoder, IsolationForest, metrics |
| `xgboost` | Model XGBoost |
| `catboost` | Model CatBoost |
| `imbalanced-learn` | SMOTE cân bằng lớp |
| `pandas`, `numpy`, `matplotlib` | Xử lý dữ liệu và vẽ biểu đồ |

---

## 3. Hướng Dẫn Chạy Notebook

1. Mở `Purchase_Prediction_ConsumerBehavior.ipynb` trong Jupyter Notebook, JupyterLab hoặc VS Code.
2. Chạy toàn bộ cell theo thứ tự từ trên xuống (Run All / Kernel → Restart & Run All). Các cell sau phụ thuộc biến (`X`, `y`, `results`) định nghĩa ở cell trước.
3. Cell cấu hình gần đầu notebook có `n_repeats = 10` (bài báo dùng 50 lần lặp). Có thể tăng lên 50 để khớp sát bài báo hơn, đổi lại thời gian chạy lâu hơn.
4. Thời gian chạy đo thực tế trên máy có GPU với `n_repeats=10`: dưới 5 phút cho cả 4 model (SVM, XGBoost, CatBoost, BPANN).
5. Dữ liệu đã có sẵn trong `data_ecommerce/`, không cần tải lại.

---

## 4. Giải Thích File Kết Quả Đầu Ra

Notebook này đã được chạy thực tế (không chỉ viết code) — các file dưới đây là số liệu đo được, nằm trong thư mục `outputs/`:

| File | Nội dung | Ý nghĩa |
|------|---------|---------|
| `class_distribution.png` | Biểu đồ cột phân bố `Revenue` | Xác nhận mất cân bằng lớp ~5,5:1 (10.422 không mua / 1.908 có mua) |
| `pearson_correlation.png` | Ma trận tương quan Pearson giữa các đặc trưng | Dùng để loại bớt đặc trưng dư thừa (tương quan > 0.7) |
| `metrics_comparison.png` | 3 bar chart (Precision, Recall, ROC AUC) so sánh "notebook này" vs "bài báo" cho 4 model | Thấy rõ ROC AUC gần với bài báo, nhưng Precision thấp hơn đáng kể — lý do giải thích trong `ConsumerBehavior_VI.md` mục 6 |
| `roc_curves.png` | 4 đường ROC của SVM, XGBoost, CatBoost, BPANN | XGBoost và CatBoost có đường cong gần góc trên-trái nhất (AUC cao nhất, ~0.93) |
| `confusion_matrices.png` | Ma trận nhầm lẫn của CatBoost và SVM (giống Fig 3 bài báo) | So sánh trực quan số dự đoán đúng/sai từng lớp |
| `feature_importance.png` | Feature importance của CatBoost (giống Fig 5 bài báo) | Trong lần chạy này: `Month`, `OperatingSystems`, `PageValues` là 3 đặc trưng ảnh hưởng mạnh nhất — khác với bài báo (Adm, PV) do khác bộ đặc trưng cuối cùng sau lọc |
| `results_table.csv` | Bảng số: Precision/Recall/ROC AUC/Accuracy của notebook này và số liệu tương ứng trong bài báo | Dùng để đối chiếu nhanh bằng Excel/pandas |

Toàn bộ biểu đồ cũng được nhúng trực tiếp trong output của từng cell trong file `.ipynb`.

---

## 5. Mô Tả Bộ Dữ Liệu

### Nguồn gốc
- **Tên:** Online Shoppers Purchasing Intention Dataset (phiên bản 2016)
- **Nguồn:** UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
- Đây là **đúng bộ dữ liệu bài báo sử dụng** (không cần thay bằng dữ liệu proxy).

### Cấu trúc

File: `data_ecommerce/online_shoppers_intention.csv` — 12.330 dòng (session), 18 cột (17 đặc trưng + nhãn `Revenue`).

| Nhóm | Đặc trưng | Mô tả |
|------|-----------|-------|
| Hành vi trang | Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration | Số trang xem và thời gian ở lại theo từng loại trang |
| Chỉ số chất lượng | BounceRates, ExitRates, PageValues, SpecialDay | Tỷ lệ thoát, tỷ lệ rời trang, giá trị trang, độ gần ngày đặc biệt |
| Bối cảnh | Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend | Thời điểm và môi trường truy cập |
| Nhãn | Revenue | True (mua hàng) / False (không mua) |

### Mất cân bằng lớp
- Không mua: 10.422 session (84,5%)
- Có mua: 1.908 session (15,5%)
- → Notebook dùng `SMOTE` (chỉ trên tập train của mỗi lần lặp) để cân bằng khi huấn luyện, giữ nguyên tỷ lệ gốc khi đánh giá trên test.

### Cách dữ liệu được sử dụng trong notebook
1. Mã hóa `Month`, `VisitorType`, `Weekend` bằng LabelEncoder.
2. Chọn đặc trưng theo 3 bước (skewness, mutual information, tương quan Pearson) — có điều chỉnh ngưỡng so với bài báo, giải thích chi tiết trong `ConsumerBehavior_VI.md` mục 7.
3. Với mỗi lần lặp đánh giá: chia train/test 80/20 → loại outlier bằng IsolationForest trên train → cân bằng bằng SMOTE trên train → chuẩn hóa bằng StandardScaler fit trên train → huấn luyện và đánh giá trên test.

---

## Đọc thêm

- **Hướng dẫn chi tiết Tiếng Việt:** [ConsumerBehavior_VI.md](ConsumerBehavior_VI.md)
- **Hướng dẫn chi tiết Tiếng Anh:** [ConsumerBehavior_EN.md](ConsumerBehavior_EN.md)
- **Bài báo gốc:** [Thanh_Paper3.pdf](Thanh_Paper3.pdf)
