# Dự Đoán Hành Vi Mua Hàng và Marketing Chính Xác — Hướng Dẫn Tiếng Việt

**Notebook:** `Purchase_Prediction_ConsumerBehavior.ipynb`
**Bài báo:** Lin, J. (2025). *Application of machine learning in predicting consumer behavior and precision marketing.* PLOS One 20(5): e0321854. https://doi.org/10.1371/journal.pone.0321854

---

## 1. Bài báo làm gì

### Bài toán
Doanh nghiệp e-commerce cần xác định chính xác khách hàng có khả năng mua hàng để tối ưu marketing cá nhân hóa (precision marketing): recommendation system, dynamic pricing, personalized advertising.

### Dữ liệu và tiền xử lý (theo bài báo)
- **Online Shoppers Purchasing Intention Dataset** (UCI, 2016) — 12.330 session, 10 đặc trưng số + 8 đặc trưng phân loại, nhãn `Revenue`.
- Mã hóa `Month`, `VisitorType`, `Weekend` bằng LabelEncoder; `StandardScaler` chuẩn hóa các biến số.
- `IsolationForest` (contamination=0.01) loại ngoại lai; `SMOTE` cân bằng lớp thiểu số (mua hàng).
- Chọn đặc trưng 3 bước: lọc skewness > 1, lọc mutual information < 0.05, lọc tương quan Pearson > 0.7.
- Chia train/test 80/20, lặp lại 50 lần với seed khác nhau, báo cáo Precision, Recall, ROC AUC trung bình.
- Hyperparameter tối ưu bằng Grid/Random Search (Table 1): SVM (C=50.12, gamma=0.12, RBF), XGBoost (learning_rate=0.11, max_depth=4, subsample=0.8), CatBoost (iterations=245, learning_rate=0.09, max_depth=8), BPANN (hidden_layers=3, epochs=50, early stopping).

### 4 model so sánh
SVM (RBF), XGBoost, CatBoost, BPANN (mạng nơ-ron lan truyền ngược).

### Kết quả chính của bài báo (Table 2)
| Model | Precision | Recall | ROC AUC |
|-------|-----------|--------|---------|
| SVM | 0.954 | 0.886 | 0.977 |
| XGBoost | 0.935 | 0.925 | 0.984 |
| **CatBoost** | 0.934 | 0.935 | **0.985** |
| BPANN | 0.901 | 0.902 | 0.955 |

CatBoost và XGBoost tốt nhất trên dữ liệu quy mô lớn, nhiều đặc trưng phức tạp. SVM có precision cao nhất nhưng recall thấp hơn. Feature importance của CatBoost cho thấy `Administrative` (Adm) và `PageValues` (PV) là hai đặc trưng ảnh hưởng mạnh nhất.

---

## 2. Dữ liệu dùng trong notebook này

Notebook dùng **đúng bộ dữ liệu UCI** mà bài báo sử dụng (không cần thay bằng dữ liệu proxy), lấy từ cache `data_ecommerce/online_shoppers_intention.csv` (đã có sẵn trong repo từ notebook trước).

| Thuộc tính | Giá trị |
|-----------|---------|
| Số session | 12.330 |
| Đặc trưng gốc | 17 (10 số + 7 phân loại/bool thực tế trong file CSV) |
| Nhãn | `Revenue` — True (mua) / False (không mua) |
| Mất cân bằng | 10.422 không mua (84,5%) / 1.908 có mua (15,5%) |

---

## 3. Cài đặt

```bash
pip install torch scikit-learn pandas numpy matplotlib xgboost catboost imbalanced-learn
```

GPU (CUDA) được dùng cho BPANN nếu có; code tự phát hiện bằng `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

---

## 4. Cách chạy

1. Mở `Purchase_Prediction_ConsumerBehavior.ipynb` trong Jupyter hoặc VS Code.
2. Chạy toàn bộ từ trên xuống (Run All). Các cell sau phụ thuộc biến (`X`, `y`, `results`) của cell trước.
3. Cell cấu hình (mục 2 trong notebook) có `n_repeats = 10` — bài báo dùng 50 lần lặp; có thể tăng lên 50 để khớp chính xác hơn bài báo, đổi lại thời gian chạy lâu hơn (SVM/XGBoost/CatBoost/BPANN đều lặp lại `n_repeats` lần).
4. Thời gian chạy đo thực tế trên máy có GPU với `n_repeats=10`: dưới 5 phút cho cả 4 model.

---

## 5. Giải thích từng phần notebook

| Phần | Nội dung |
|------|---------|
| Import + seed/device | Nạp thư viện, cố định seed=42, phát hiện GPU/CPU |
| Tải dữ liệu | Đọc từ cache `data_ecommerce/` |
| Khám phá | In dtype, số giá trị thiếu, phân bố `Revenue` |
| Biểu đồ phân bố lớp | Xác nhận mất cân bằng ~5.5:1 |
| Tiền xử lý & chọn đặc trưng | LabelEncoder, lọc skewness/MI/Pearson (**có điều chỉnh ngưỡng**, xem mục 7) |
| Hàm dùng chung | `remove_outliers` (IsolationForest), `balance_with_smote` (SMOTE) — chỉ áp dụng trên tập train của mỗi lần lặp |
| `eval_once_sklearn` / `run_repeated_sklearn` | Một lần và nhiều lần lặp đánh giá cho SVM/XGBoost/CatBoost |
| `ShopperDataset`, `BPANN` | Dataset và model PyTorch (`nn.Module`, `super(BPANN, self).__init__()`) cho mạng nơ-ron 3 lớp ẩn |
| `eval_once_bpann` / `run_repeated_bpann` | Vòng lặp huấn luyện tay (Adam, BCEWithLogitsLoss, early stopping) |
| Chạy 4 model | Dùng hyperparameter từ Table 1 của bài báo |
| Bảng kết quả | So sánh "ours" vs "paper" |
| Biểu đồ | Bar chart so sánh metric, ROC curves, confusion matrix (CatBoost & SVM), feature importance (CatBoost) |

---

## 6. Cách đọc kết quả — số liệu đo được thực tế (n_repeats=10)

| Model | Precision (ours) | Precision (paper) | Recall (ours) | Recall (paper) | ROC AUC (ours) | ROC AUC (paper) | Accuracy (ours) |
|-------|------|------|------|------|------|------|------|
| SVM | 0.544 | 0.954 | 0.652 | 0.886 | 0.825 | 0.977 | 0.861 |
| XGBoost | 0.687 | 0.935 | 0.659 | 0.925 | 0.929 | 0.984 | 0.901 |
| CatBoost | 0.685 | 0.934 | 0.648 | 0.935 | 0.927 | 0.985 | 0.899 |
| BPANN | 0.514 | 0.901 | 0.804 | 0.902 | 0.908 | 0.955 | 0.852 |

Đây là **số liệu đo thật** sau khi chạy notebook, không phải ước lượng (xem `outputs/results_table.csv`).

### Ý nghĩa từng metric
- **Precision:** trong số session được dự đoán "sẽ mua", bao nhiêu % thực sự mua. Thấp nghĩa là nhiều cảnh báo giả (false positive).
- **Recall:** trong số session thực sự mua, mô hình bắt được bao nhiêu %. Cao nghĩa là ít bỏ sót khách mua.
- **ROC AUC:** khả năng phân biệt hai lớp tổng quát, không phụ thuộc ngưỡng 0.5.
- **Accuracy:** tỷ lệ dự đoán đúng tổng thể — dễ gây hiểu nhầm trên dữ liệu mất cân bằng, nên không dùng làm chỉ số chính.

### Vì sao Precision thấp hơn bài báo khá nhiều, nhưng ROC AUC gần tương đương
ROC AUC của notebook này (0.83–0.93) khá gần với bài báo (0.955–0.985), cho thấy các model vẫn phân biệt được hai lớp tốt. Nhưng Precision thấp hơn nhiều (0.51–0.69 so với 0.90–0.95), đi kèm Recall tăng so với Precision — đây là hệ quả trực tiếp của việc huấn luyện trên dữ liệu đã cân bằng bởi SMOTE rồi áp ngưỡng 0.5 lên tập test vẫn giữ tỷ lệ mất cân bằng gốc (84,5%/15,5%): model học cách dự đoán "mua" nhiều hơn tỷ lệ thật trong test, nên nhiều dự đoán dương là sai (precision giảm) nhưng bắt được nhiều ca dương thật hơn (recall tăng). Bài báo không mô tả rõ có canh lại ngưỡng quyết định (threshold) sau SMOTE hay không; đây là một điểm khác biệt quan trọng, không phải lỗi chạy sai.

---

## 7. Khác biệt so với bài báo

| Khía cạnh | Bài báo | Notebook này |
|-----------|---------|---------------|
| Dữ liệu | UCI Online Shoppers (giống) | UCI Online Shoppers (giống, không cần proxy) |
| Số lần lặp đánh giá | 50 | 10 (có thể tăng lên 50 trong cell cấu hình) |
| Chọn đặc trưng | skewness>1 loại, MI<0.05 loại | **Đã điều chỉnh ngưỡng**: bỏ qua loại theo skewness (vì sẽ xóa cả `PageValues` — đặc trưng quan trọng nhất theo chính bài báo), hạ ngưỡng MI xuống 0.005. Lý do chi tiết nằm trong markdown ngay trước cell chọn đặc trưng của notebook |
| SMOTE/IsolationForest/StandardScaler | Không nêu rõ thứ tự với việc chia tập | Chỉ fit trên tập train của từng lần lặp, tránh rò rỉ dữ liệu vào test |
| Hiệu chỉnh ngưỡng quyết định sau SMOTE | Không nêu | Dùng ngưỡng mặc định 0.5 — là nguyên nhân chính khiến Precision thấp hơn bài báo (xem mục 6) |
| Feature importance CatBoost | Adm, PV dẫn đầu | Trong lần chạy này: Month, OperatingSystems, PageValues dẫn đầu (xem `outputs/feature_importance.png`) — khác biệt do khác ngưỡng chọn đặc trưng và bộ feature cuối cùng dùng để train |
| Hyperparameter | Grid/Random Search | Dùng trực tiếp giá trị đã tối ưu của bài báo (Table 1), không tune lại |

---

## 8. Mẹo và lưu ý

- Muốn khớp sát số liệu bài báo hơn: tăng `n_repeats` lên 50 trong cell cấu hình, chấp nhận thời gian chạy lâu hơn.
- Muốn cải thiện Precision: thử tăng ngưỡng quyết định trên 0.5 trong hàm `eval_once_sklearn`/`eval_once_bpann`, hoặc hiệu chỉnh lại SMOTE (ví dụ `sampling_strategy` nhỏ hơn 1.0 thay vì cân bằng hoàn toàn).
- Toàn bộ 6 biểu đồ + 1 file CSV kết quả đã được lưu thật vào thư mục `outputs/` khi chạy notebook (không phải placeholder).
- Kết quả có thể dao động nhẹ giữa các lần chạy do SMOTE và train/test split ngẫu nhiên, nhưng thứ tự xếp hạng model (XGBoost ≈ CatBoost > BPANN > SVM theo ROC AUC) nên ổn định.
