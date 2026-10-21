# Cart Abandonment / Purchase Prediction — Tái hiện Thanh_Paper2

Notebook: [`Cart_Abandonment_REES46.ipynb`](Cart_Abandonment_REES46.ipynb)
Paper gốc: [`Thanh_Paper2.pdf`](Thanh_Paper2.pdf) — Esmeli, R.; Gokce, A. *An Analysis of Consumer Purchase Behavior Following Cart Addition in E-Commerce Utilizing Explainable Artificial Intelligence*. J. Theor. Appl. Electron. Commer. Res. 2025, 20, 28.

## 1. Bài toán

Dự đoán **purchase vs. cart abandonment**: sau khi khách hàng thêm sản phẩm vào giỏ trong một phiên (session), liệu phiên đó có kết thúc bằng giao dịch mua hay bị bỏ dở. Paper gốc trả lời 3 câu hỏi nghiên cứu:

- **RQ1** — Chuyển đổi danh mục sản phẩm (category transitions) trong phiên ảnh hưởng thế nào đến dự đoán?
- **RQ2** — Biến động giá trong phiên (min/max price, price difference) ảnh hưởng thế nào?
- **RQ3** — Yếu tố địa lý (vị trí người dùng) ảnh hưởng thế nào?

## 2. Dữ liệu

Dữ liệu gốc của paper là **private** (một công ty SaaS e-commerce tại UK), chỉ có thể xin qua liên hệ tác giả ("available on request due to privacy concerns"). Notebook này dùng dữ liệu công khai thay thế:

**REES46 — "eCommerce behavior data from multi category store"** (Kaggle, `mkechinov/ecommerce-behavior-data-from-multi-category-store`), file **`2019-Oct.csv`** (~5.67 GB, 42,448,764 events).

| Cột | Mô tả |
|---|---|
| `event_time` | Thời điểm sự kiện (UTC) |
| `event_type` | `view` / `cart` / `remove_from_cart` / `purchase` |
| `product_id`, `category_id`, `category_code`, `brand` | Định danh sản phẩm |
| `price` | Giá sản phẩm tại thời điểm xem |
| `user_id`, `user_session` | Định danh người dùng / phiên |

**Giới hạn quan trọng:** REES46 **không có** cột vị trí địa lý (latitude/longitude) hay loại thiết bị (pc/tablet) như dữ liệu gốc của paper. Vì vậy notebook này **bỏ RQ3**, chỉ tái hiện **RQ1** và **RQ2**.

### Tiền xử lý
- Lọc chỉ giữ lại các session có ít nhất 1 sự kiện `cart` (đúng ngữ cảnh "sau khi thêm giỏ" của paper) → còn 573,097 / ~9 triệu session.
- Lấy mẫu ngẫu nhiên 80,000 session (random_state=42) để notebook chạy được trên máy cá nhân, quy mô tương đương ds1/ds2 trong paper (72k/26k session).
- Nhãn: `purchase=1` nếu session có sự kiện `purchase`, ngược lại `purchase=0`.

### Feature engineering (per session)
`pages_viewed`, `products_viewed`, `max_price`, `min_price`, `unique_prices`, `price_difference`, `price_change`, `cat_changes`, `hod`, `dow`, `dom`, `weekend`, `number_of_previous_sessions`, `number_of_previous_purchases`.

## 3. Mô hình

8 model theo paper gốc + 1 model bổ sung:

| Nhóm | Model |
|---|---|
| ML truyền thống (GridSearchCV) | Logistic Regression, Random Forest, Gradient Boosting, KNN, Decision Tree |
| Deep learning (PyTorch) | MLP, CNN1D, LSTM |
| **Bổ sung** | **CNN_LSTM_Maxout** — 2 nhánh song song (Conv1D stack + stacked LSTM), mỗi nhánh cho ra vector đặc trưng 64 chiều, kết hợp bằng **maxout** (lấy max theo từng phần tử), qua Dense+Sigmoid để dự đoán |

Dữ liệu mất cân bằng được xử lý bằng **SMOTE** (chỉ áp dụng trên tập train, tránh leakage). Train/test split 80/20, stratify theo nhãn. Với 4 model deep learning, tập train sau SMOTE được chia tiếp 85/15 thành train/val (chỉ để theo dõi learning curve, test set luôn giữ nguyên, không dùng để chọn model).

## 4. Kết quả

### Bảng so sánh model (test set, 16,000 session)

| Model | Accuracy | ROC AUC | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **CNN_LSTM_Maxout** | 0.695 | 0.778 | 0.683 | **0.746** | **0.713** |
| GB | 0.700 | 0.780 | 0.706 | 0.703 | 0.705 |
| LSTM | 0.699 | 0.777 | 0.704 | 0.705 | 0.704 |
| DT | 0.689 | 0.764 | 0.697 | 0.687 | 0.692 |
| RF | 0.681 | 0.763 | 0.687 | 0.684 | 0.686 |
| MLP | 0.661 | 0.746 | 0.723 | 0.540 | 0.618 |
| LR | 0.643 | 0.695 | 0.683 | 0.553 | 0.611 |
| KNN | 0.576 | 0.607 | 0.585 | 0.572 | 0.578 |
| CNN | 0.631 | 0.688 | 0.699 | 0.481 | 0.570 |

**`CNN_LSTM_Maxout` đạt F1 và Recall cao nhất** trong toàn bộ 9 model, nhỉnh hơn Gradient Boosting (model tốt nhất trong paper gốc). Maxout fusion tận dụng được tín hiệu local pattern từ nhánh CNN và tín hiệu tuần tự từ nhánh LSTM tốt hơn từng nhánh riêng lẻ (CNN đơn lẻ là model DL yếu nhất, F1 0.570).

So với paper gốc (F1 tốt nhất 0.89-0.893 trên GB), F1 trên REES46 thấp hơn đáng kể — hợp lý vì thiếu feature địa lý/thiết bị và vì REES46 không có ngữ cảnh lịch sử người dùng xuyên nhiều phiên phong phú như dataset nội bộ của paper.

### Permutation importance (model GB)

| Feature | Importance |
|---|---|
| pages_viewed | 0.1169 |
| number_of_previous_purchases | 0.0948 |
| products_viewed | 0.0857 |
| number_of_previous_sessions | 0.0192 |
| unique_prices | 0.0088 |
| price_difference | 0.0086 |
| dom | 0.0053 |
| hod | 0.0036 |
| **cat_changes** | **0.0027** |
| min_price | 0.0017 |
| max_price | 0.0003 |
| dow | 0.0002 |
| weekend | 0.0001 |
| price_change | 0.0000 |

### Trả lời RQ1 & RQ2

- **RQ1 (category transitions):** `cat_changes` xếp hạng 9/14, importance thấp (0.0027) — xác nhận kết luận của paper rằng chuyển đổi danh mục là yếu tố **phụ**, nhưng ảnh hưởng yếu hơn so với ds2 của paper (SHAP 0.51) — có thể do session REES46 ngắn hơn và category_id nhiễu hơn so với taxonomy đã được paper chọn lọc.
- **RQ2 (price dynamics):** `unique_prices` và `price_difference` có importance cao hơn `min_price`/`max_price`, trong khi `price_change` gần như bằng 0. Một phần khớp với paper (biến động giá > cờ nhị phân price_change), nhưng trên REES46 **số lượng mức giá khác nhau đã xem** ảnh hưởng mạnh hơn giá trị tuyệt đối min/max — gợi ý độ rộng duyệt web (browsing breadth) chi phối hiệu ứng giá trên dataset này.
- **Nhận xét chung:** các feature hành vi/lịch sử (`pages_viewed`, `products_viewed`, `number_of_previous_purchases`) áp đảo toàn bộ, nhất quán với kết luận của paper rằng lịch sử mua hàng là tín hiệu mạnh nhất trên cả 2 dataset gốc.

### Biểu đồ

- [`outputs/rees46_outcome_distribution.png`](outputs/rees46_outcome_distribution.png) — tỷ lệ purchase/abandon trên tập mẫu.
- [`outputs/rees46_price_vs_outcome.png`](outputs/rees46_price_vs_outcome.png) — phân phối giá và giá theo outcome (tương ứng Figure 5/6 của paper).
- [`outputs/rees46_shap_feature_importance.png`](outputs/rees46_shap_feature_importance.png) — SHAP feature importance cho model GB (tương ứng Figure 8 của paper).
- [`outputs/rees46_curves_MLP.png`](outputs/rees46_curves_MLP.png), [`outputs/rees46_curves_CNN.png`](outputs/rees46_curves_CNN.png), [`outputs/rees46_curves_LSTM.png`](outputs/rees46_curves_LSTM.png), [`outputs/rees46_curves_CNN_LSTM_Maxout.png`](outputs/rees46_curves_CNN_LSTM_Maxout.png) — train/val loss và accuracy theo epoch cho từng model deep learning.
- [`outputs/rees46_confusion_matrices.png`](outputs/rees46_confusion_matrices.png) — confusion matrix của cả 9 model trên test set.
- [`outputs/rees46_roc_curves.png`](outputs/rees46_roc_curves.png) — ROC curve chồng của cả 9 model (kèm AUC) trên test set.
- [`outputs/rees46_correlation_matrix.png`](outputs/rees46_correlation_matrix.png) — ma trận tương quan Pearson giữa 14 feature cấp session.

### Checkpoint

Toàn bộ model đã train được lưu trong `checkpoints/`:
- Model sklearn (LR, RF, GB, KNN, DT): file `.joblib` (best estimator sau GridSearchCV).
- Model PyTorch (MLP, CNN, LSTM, CNN_LSTM_Maxout): file `.pt` (state_dict).

**Lưu ý:** `checkpoints/RF.joblib` (~104MB, do GridSearchCV chọn cây không giới hạn độ sâu) vượt giới hạn 100MB của GitHub nên **không được push lên repo** (đã thêm vào `.gitignore`) — chỉ tồn tại ở máy đã chạy notebook. Muốn có lại file này, chạy lại cell GridSearchCV của RF trong notebook.

## 5. Hạn chế

- Không có RQ3 (geographic) do REES46 thiếu cột vị trí/thiết bị.
- Chỉ dùng 1 tháng dữ liệu (2019-Oct), lấy mẫu 80,000/573,097 session đủ điều kiện — kết quả có thể thay đổi nhẹ với seed/sample khác.
- CNN1D và LSTM nhận feature vector per-session như một pseudo-sequence (mỗi feature = 1 time step), khác với paper gốc vốn áp dụng trên sequence sự kiện chi tiết hơn — đây là một xấp xỉ hợp lý nhưng không hoàn toàn giống kiến trúc gốc.
- SMOTE chỉ áp dụng trên tập train (paper mô tả áp dụng trước khi split) — thay đổi này nhằm tránh data leakage, là thực hành tốt hơn.
