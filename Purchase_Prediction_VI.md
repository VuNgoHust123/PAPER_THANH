# Dự Đoán Hành Vi Mua Hàng Thương Mại Điện Tử — Hướng Dẫn Tiếng Việt

**Notebook:** `Purchase_Prediction_ECommerce.ipynb`  
**Bài báo:** Esmeli, R. & Gokce, A. (2025). *An Analysis of Consumer Purchase Behavior Following Cart Addition in E-Commerce Utilizing Explainable Artificial Intelligence.* J. Theor. Appl. Electron. Commer. Res., 20, 28.

---

## 1. Bài Báo Làm Gì?

### Vấn đề nghiên cứu
Các nền tảng thương mại điện tử chịu tỷ lệ bỏ giỏ hàng rất cao (67–81%). Phần lớn nghiên cứu trước đây chỉ dự đoán ý định mua hàng tổng quát, mà bỏ qua **điều gì xảy ra sau khi khách hàng đã thêm sản phẩm vào giỏ** — đây là thời điểm quan trọng nhất để giữ chân họ.

Bài báo đặt ra 3 câu hỏi nghiên cứu:
1. **Chuyển đổi danh mục sản phẩm** trong phiên mua sắm ảnh hưởng như thế nào đến mô hình dự đoán?
2. **Biến động giá** trong phiên (giá thấp nhất/cao nhất, chênh lệch) có tác động gì?
3. **Yếu tố địa lý** (vị trí người dùng) ảnh hưởng như thế nào đến quyết định mua?

### Framework 3 Giai Đoạn

| Giai đoạn | Mô tả |
|-----------|-------|
| **Stage 1 — Thu thập dữ liệu** | Log phiên mua sắm từ nền tảng SaaS tại Anh (2 bộ dữ liệu, tháng 7–12/2019) |
| **Stage 2 — Feature Engineering** | Trích xuất 16 đặc trưng cấp phiên: pages viewed, giá min/max, vị trí địa lý (lat/lon), số lần đổi danh mục, loại thiết bị, giờ/ngày, lịch sử mua |
| **Stage 3 — Huấn luyện & Đánh giá** | 8 mô hình ML đánh giá bằng 10-fold CV + SMOTE; giải thích bằng SHAP |

### 8 Mô Hình So Sánh

| Mô hình | Loại |
|---------|------|
| Logistic Regression (LR) | Tuyến tính |
| Decision Tree (DT) | Cây quyết định |
| K-Nearest Neighbor (KNN) | Dựa trên khoảng cách |
| Random Forest (RF) | Ensemble |
| Gradient Boosting (GB) | Ensemble |
| Multi-Layer Perceptron (MLP) | Mạng nơ-ron |
| Convolutional Neural Network (CNN) | Mạng nơ-ron |
| Long Short-Term Memory (LSTM) | Mạng nơ-ron |

### Kết Quả Chính của Bài Báo
- **Gradient Boosting (GB)** đạt F1 Score tốt nhất: **89%**
- Các đặc trưng quan trọng nhất (qua SHAP): lịch sử mua hàng trước, tần suất phiên, số trang đã xem, vị trí địa lý (lat/lon), giá min/max
- Chuyển đổi danh mục có ảnh hưởng vừa phải (SHAP: 0.21–0.51 tùy bộ dữ liệu)
- Đặc trưng địa lý có ảnh hưởng đáng ngạc nhiên (SHAP: 0.56–0.76)

---

## 2. Bộ Dữ Liệu Dùng Trong Notebook

Vì dữ liệu gốc của bài báo là dữ liệu nội bộ (công ty SaaS tại Anh, không công khai), notebook này dùng bộ dữ liệu công khai **UCI Online Shoppers Purchasing Intention Dataset**.

| Thuộc tính | Giá trị |
|-----------|---------|
| Nguồn | UCI Machine Learning Repository (ID 468) |
| URL | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Số mẫu | 12,330 phiên mua sắm |
| Số đặc trưng | 17 gốc + 6 được tính toán = 23 tổng cộng |
| Nhãn (target) | `Revenue` — True (mua) / False (không mua) |
| Mất cân bằng lớp | ~84% không mua / ~16% mua |
| Thời gian thu thập | 1 năm, từ nền tảng bán lẻ thực tế |

### Các Đặc Trưng Gốc

| Đặc trưng | Ý nghĩa |
|-----------|---------|
| Administrative | Số trang dạng quản trị đã xem |
| Administrative_Duration | Tổng thời gian trên trang quản trị (giây) |
| Informational | Số trang thông tin đã xem |
| Informational_Duration | Tổng thời gian trên trang thông tin |
| ProductRelated | Số trang sản phẩm đã xem |
| ProductRelated_Duration | Tổng thời gian trên trang sản phẩm |
| BounceRates | Tỷ lệ thoát trung bình của các trang đã xem |
| ExitRates | Tỷ lệ rời trang trung bình |
| PageValues | Giá trị trung bình của trang trước khi giao dịch |
| SpecialDay | Mức độ gần với ngày đặc biệt (0–1) |
| Month | Tháng của phiên |
| OperatingSystems | Loại hệ điều hành (1–8) |
| Browser | Loại trình duyệt (1–13) |
| Region | Vùng địa lý (1–9) |
| TrafficType | Nguồn traffic (1–20) |
| VisitorType | Returning / New / Other |
| Weekend | Boolean — phiên có vào cuối tuần không? |

### Đặc Trưng Tính Toán Thêm (Stage 2 — tương ứng bài báo)

| Đặc trưng mới | Tương ứng trong bài báo | Công thức |
|--------------|------------------------|-----------|
| `pages_viewed` | `pagesviewed` | Administrative + Informational + ProductRelated |
| `total_duration` | thời gian phiên | Tổng tất cả cột Duration |
| `cat_transitions` | `#catchanges` | Số loại trang khác nhau đã xem (0–3) |
| `engagement_quality` | — | 1 − BounceRates |
| `price_sensitivity` | `pricedifference` (proxy) | PageValues (cao = phiên quan tâm đến giá trị) |
| `is_special_day` | `pricechange` (proxy) | 1 nếu SpecialDay > 0, ngược lại 0 |

---

## 3. Cài Đặt

```bash
pip install torch scikit-learn pandas numpy matplotlib imbalanced-learn openpyxl
```

GPU (CUDA) không bắt buộc nhưng khuyến nghị cho 3 mô hình nơ-ron (MLP, CNN, LSTM). Code tự phát hiện qua:
```python
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Bộ dữ liệu được tải tự động khi chạy lần đầu — không cần tải thủ công.

---

## 4. Hướng Dẫn Chạy

1. Mở `Purchase_Prediction_ECommerce.ipynb` trong **Jupyter Notebook** hoặc **VS Code**
2. Chạy tất cả cell **theo thứ tự từ trên xuống dưới** (dùng *Run All* hoặc Shift+Enter từng cell)
3. **Không** bỏ qua hoặc thay đổi thứ tự cell — các cell sau phụ thuộc vào biến từ cell trước
4. Thời gian chạy ước tính:
   - Sklearn models (LR, DT, KNN, RF, GB): **~1–3 phút** tổng cộng
   - Neural models (MLP, CNN, LSTM): **~5–15 phút** trên GPU / **~30–60 phút** trên CPU

> **Mẹo tăng tốc:** Giảm `n_splits` từ 5 xuống 3 trong cell setup, hoặc giảm `epochs` từ 30 xuống 15 trong cell "Run All Models".

---

## 5. Giải Thích Từng Cell

### Cell Khởi Tạo

| Cell | Nội dung |
|------|---------|
| **Imports** | Load tất cả thư viện; tự cài `imbalanced-learn` nếu thiếu |
| **Seed + Device** | Đặt `seed=42` để tái hiện kết quả; phát hiện GPU/CPU; đặt `n_splits=5` |

### Stage 1 — Tải Dữ Liệu

| Cell | Nội dung |
|------|---------|
| **Download & Load** | Tải file CSV từ UCI vào `data_ecommerce/` (một lần duy nhất); đọc vào DataFrame |
| **Explore** | In kiểu dữ liệu, số giá trị thiếu, phân bố nhãn |
| **Plot distributions** | 3 biểu đồ: phân bố lớp, tỷ lệ mua theo tháng, tỷ lệ mua theo loại visitor |

**Cần chú ý trong các biểu đồ:**
- Biểu đồ phân bố lớp: xác nhận mất cân bằng nghiêm trọng (~5:1) — lý do phải dùng SMOTE
- Theo tháng: tháng 11 (Black Friday/Cyber Monday) và tháng 5 thường có tỷ lệ mua cao hơn
- Returning Visitor mua nhiều hơn New Visitor — lòng trung thành quan trọng

### Stage 2 — Feature Engineering

| Cell | Nội dung |
|------|---------|
| **Engineer features** | Tính 6 đặc trưng mới từ công thức (xem bảng trên) |
| **Box plots** | So sánh `pages_viewed`, `price_sensitivity`, `cat_transitions` theo kết quả mua |

**Cần chú ý trong box plots:**
- `price_sensitivity` (PageValues): người mua thường xem các trang có giá trị cao hơn — đây là dự báo mạnh
- `pages_viewed`: người mua thường duyệt nhiều trang hơn trước khi quyết định
- `cat_transitions`: người mua thường xem 2–3 loại trang (sản phẩm + thông tin + so sánh) thay vì chỉ 1

### Tiền Xử Lý

| Cell | Nội dung |
|------|---------|
| **Encode + Scale** | Map Month (chuỗi→số), VisitorType (chuỗi→số), Weekend (bool→int); StandardScaler cho tất cả feature |
| **ShopperDataset** | Class PyTorch Dataset bọc numpy array thành tensor float32 |

### Định Nghĩa Model

| Cell | Kiến trúc |
|------|----------|
| **Sklearn models** | LR, DT, KNN, RF, GB định nghĩa bằng sklearn |
| **MLP_Classifier** | 3 lớp fully-connected với BatchNorm + Dropout: input→128→64→2 |
| **CNN1D_Classifier** | Coi feature vector là tín hiệu 1D: (batch, 1, features) → Conv1D×2 → MaxPool → FC |
| **LSTM_Classifier** | Coi mỗi feature là một time step: (batch, features, 1) → 2-layer LSTM → FC |

> **Tại sao dùng CNN và LSTM cho dữ liệu bảng?**  
> Cả hai mô hình được áp dụng bằng cách reshape vector đặc trưng thành "chuỗi". Đây là cách tiếp cận thực nghiệm trong bài báo để kiểm tra xem có mẫu cục bộ/tuần tự nào giữa các feature liền kề không. Thông thường hiệu suất sẽ kém hơn ensemble methods trên dữ liệu bảng — và kết quả này là bình thường, nhất quán với bài báo.

### Cross-Validation

Cả `sklearn_kfold()` và `neural_kfold()` đều tuân theo quy trình giống nhau:
1. **StratifiedKFold** — giữ tỷ lệ lớp trong mỗi fold
2. **SMOTE chỉ áp dụng cho training fold** — tránh data leakage vào test fold
3. Tất cả dự đoán từ các fold được **ghép lại** để tính metric và vẽ ROC curve

---

## 6. Hiểu Kết Quả

### Các Metric Đánh Giá

| Metric | Công thức | Ý nghĩa |
|--------|----------|---------|
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Trung bình điều hòa — tốt khi cả FP lẫn FN đều quan trọng |
| **Recall** | TP / (TP + FN) | Trong tổng số người mua thực sự, model bắt được bao nhiêu? (quan trọng để retarget) |
| **Precision** | TP / (TP + FP) | Trong số người được dự đoán là mua, bao nhiêu người thực sự mua? (quan trọng khi gửi khuyến mãi) |
| **ROC-AUC** | Diện tích dưới đường ROC | Khả năng phân biệt tổng thể; 0.5 = ngẫu nhiên, 1.0 = hoàn hảo |

### Kết Quả Kỳ Vọng trên UCI Dataset

Bộ dữ liệu UCI khác với dữ liệu gốc trong bài báo, nên con số sẽ không khớp hoàn toàn. Dải kỳ vọng:

| Mô hình | F1 Score | ROC-AUC |
|---------|----------|---------|
| GB | 0.62–0.72 | 0.92–0.94 |
| RF | 0.58–0.68 | 0.90–0.92 |
| MLP | 0.55–0.65 | 0.88–0.91 |
| CNN | 0.50–0.62 | 0.85–0.90 |
| LSTM | 0.50–0.62 | 0.85–0.90 |
| LR | 0.48–0.58 | 0.88–0.91 |
| KNN | 0.45–0.58 | 0.82–0.87 |
| DT | 0.45–0.55 | 0.72–0.80 |

> **Tại sao F1 thấp hơn 89% trong bài báo?** Vì (1) bộ dữ liệu UCI là domain khác với chất lượng feature khác; (2) bài báo có các tín hiệu giàu hơn như lat/lon chính xác và lịch sử giá chi tiết trong phiên.

### Biểu Đồ Bar Chart — 4 Metrics

- Mỗi subplot hiển thị một metric cho tất cả 8 model, sắp xếp theo F1
- **Đường đỏ đứt** ở mức 0.89 là ngưỡng F1 tốt nhất của bài báo — dùng làm chuẩn so sánh
- Tìm mô hình có **hiệu suất nhất quán ở cả 4 metric** — mô hình tốt ở 1 tiêu chí nhưng kém ở tiêu chí khác là không đáng tin cậy

### Biểu Đồ ROC Curves

- Đường ROC càng gần **góc trên-trái** thì model càng tốt
- **Đường chéo đứt** là baseline ngẫu nhiên (AUC = 0.5)
- AUC cao nhưng F1 thấp: model phân hạng tốt nhưng ngưỡng 0.5 không phù hợp — có thể cải thiện bằng cách điều chỉnh threshold
- **GB và RF** thường tạo đường bao phía trên (AUC cao nhất)

### Confusion Matrix (Top 4 Mô Hình)

```
                    Dự đoán Không Mua   Dự đoán Mua
Thực tế Không Mua       TN                  FP
Thực tế Mua             FN                  TP
```

- **TN cao + TP cao** = độ chính xác tổng thể tốt
- Trong ngữ cảnh marketing: **Recall cao** (FN thấp) quan trọng hơn — không muốn bỏ sót người mua tiềm năng
- **Precision cao** (FP thấp) tiết kiệm ngân sách marketing — chỉ target đúng người có khả năng mua

**Cách đọc từng ô:**
- **TP (Đúng Positive):** Dự đoán mua, thực tế mua → đúng, gửi khuyến mãi hiệu quả
- **TN (Đúng Negative):** Dự đoán không mua, thực tế không mua → đúng, tiết kiệm chi phí
- **FP (Sai Positive):** Dự đoán mua, thực tế không mua → lãng phí ngân sách khuyến mãi
- **FN (Sai Negative):** Dự đoán không mua, thực tế mua → mất cơ hội retarget

### Biểu Đồ Feature Importance (GB & RF)

- Thanh dài hơn = feature đó ảnh hưởng nhiều hơn đến quyết định của model
- Feature quan trọng kỳ vọng: `PageValues`/`price_sensitivity`, `ProductRelated_Duration`, `ExitRates`, `BounceRates`, `Region`
- Điều này phản ánh kết quả bài báo: **price dynamics** (biến động giá) và **geographic context** (vị trí địa lý) là hai nhóm dự báo quan trọng
- `cat_transitions` (feature tự tính) nên xuất hiện ở mức trung bình

---

## 7. So Sánh với Bài Báo Gốc

| Khía cạnh | Bài báo gốc | Notebook này |
|-----------|-------------|-------------|
| Dữ liệu | Dữ liệu nội bộ của SaaS Anh (2 bộ) | UCI công khai (1 bộ) |
| Mất cân bằng | ds1: 81% bỏ giỏ / ds2: 67% bỏ giỏ | ~84% không mua |
| Feature địa lý | Lat/lon chính xác | Mã vùng (Region 1–9) |
| Feature giá | Lịch sử giá trong phiên | PageValues làm proxy |
| CV folds | 10-fold | 5-fold (có thể chỉnh) |
| Giải thích model | SHAP | Feature importance (GB, RF) |
| F1 tốt nhất | 89% (GB trên ds1) | ~65–72% kỳ vọng (GB) |

---

## 8. Lưu Ý Quan Trọng

- **Cảnh báo data leakage với SMOTE:** SMOTE được áp dụng *bên trong* mỗi fold, chỉ trên phần training. Tuyệt đối không áp dụng SMOTE trước khi chia tập — sẽ gây data leakage và làm phồng metric.
- **Neural models trên dữ liệu bảng:** CNN và LSTM thường kém hơn ensemble (GB, RF) trên dữ liệu dạng bảng. Đây là kết quả bình thường và nhất quán với bài báo.
- **Điều chỉnh ngưỡng quyết định:** Ngưỡng mặc định là 0.5. Để tăng Recall (bắt nhiều người mua hơn), giảm ngưỡng xuống. Để tăng Precision, tăng ngưỡng lên.
- **Tái hiện kết quả:** Kết quả được cố định bởi `seed=42` cho sklearn và `torch.manual_seed(seed)` trong từng fold cho neural models. Chạy lại sẽ cho kết quả giống nhau.
- **Thứ tự chạy cell:** Không thay đổi thứ tự vì các biến như `X`, `y`, `feature_cols`, `results` được truyền giữa các cell.
