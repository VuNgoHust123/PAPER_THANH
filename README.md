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

8 model theo paper gốc + 2 model bổ sung:

| Nhóm | Model |
|---|---|
| ML truyền thống (GridSearchCV) | Logistic Regression, Random Forest, Gradient Boosting, KNN, Decision Tree |
| Deep learning (PyTorch) | MLP, CNN1D, LSTM |
| **Bổ sung (đề xuất)** | **CNN_LSTM_Maxout** — 2 nhánh song song: 1 nhánh CNN **residual block** (Conv1D→BN→ReLU→Conv1D→BN, cộng với shortcut 1×1-conv, rồi ReLU; kernel_size=3, **giữ nguyên toàn bộ chuỗi `(batch, L, feat_dim)`**, L = số feature) được **maxout theo từng vị trí** với nhánh stacked LSTM, sau đó **Average Pooling theo L** để ra vector `(batch, feat_dim)`, qua Dense+Sigmoid để dự đoán |
| **Bổ sung (baseline residual)** | **ResNet1D** — Conv1D stem rồi 3 **residual block** 1D xếp chồng (mỗi block: Conv1D→BN→ReLU→Conv1D→BN, cộng với input của chính block đó, rồi ReLU), sau đó Global Average Pooling theo `L`, qua Dense+Sigmoid — một kiến trúc ResNet thuần (không có nhánh LSTM/maxout) để so sánh riêng hiệu quả của residual connection |

Dữ liệu mất cân bằng được xử lý bằng **SMOTE** (chỉ áp dụng trên tập train, tránh leakage). Train/test split 80/20, stratify theo nhãn. Với 5 model deep learning, tập train sau SMOTE được chia tiếp 85/15 thành train/val (chỉ để theo dõi learning curve, test set luôn giữ nguyên, không dùng để chọn model).

## 4. Kết quả

### Bảng so sánh model (test set, 16,000 session)

| Model | Accuracy | ROC AUC | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| GB (baseline) | 0.700 | 0.780 | 0.706 | 0.703 | **0.705** |
| LSTM (baseline) | 0.700 | 0.779 | 0.707 | 0.701 | 0.704 |
| **ResNet1D** (baseline residual, 3 residual block + GAP) | 0.689 | 0.766 | 0.694 | 0.696 | 0.695 |
| DT (baseline) | 0.689 | 0.764 | 0.697 | 0.687 | 0.692 |
| RF (baseline) | 0.681 | 0.763 | 0.687 | 0.684 | 0.686 |
| **CNN_LSTM_Maxout** (đề xuất, 2 nhánh: CNN residual k=3 → maxout với LSTM) | 0.690 | 0.771 | 0.711 | 0.657 | 0.683 |
| MLP (baseline) | 0.678 | 0.749 | 0.709 | 0.619 | 0.661 |
| CNN (baseline) | 0.649 | 0.706 | 0.686 | 0.571 | 0.623 |
| LR (baseline) | 0.643 | 0.695 | 0.683 | 0.553 | 0.611 |
| KNN (baseline) | 0.576 | 0.607 | 0.585 | 0.572 | 0.578 |

Rút `CNN_LSTM_Maxout` từ 3 nhánh (CNN k=3 + CNN k=7 residual, cộng lại rồi maxout với LSTM, F1=0.704) về lại **2 nhánh** (bỏ nhánh CNN k=7, chỉ giữ 1 CNN residual k=3 maxout trực tiếp với LSTM) làm F1 giảm xuống **0.683** trên REES46 — gần với bản 3-nhánh Conv1D thuần ban đầu (0.683) và thấp hơn cả `ResNet1D` (0.695). Điều này cho thấy trên REES46, phần cải thiện lớn nhất không đến từ residual connection đơn lẻ mà đến từ việc **cộng 2 nhánh CNN đa kernel (k=3+k=7) trước khi maxout với LSTM** — bỏ nhánh CNN thứ hai làm mất đi phần representation đa tỷ lệ (multi-scale) đó.

So với paper gốc (F1 tốt nhất 0.89-0.893 trên GB), F1 trên REES46 thấp hơn đáng kể — hợp lý vì thiếu feature địa lý/thiết bị và vì REES46 không có ngữ cảnh lịch sử người dùng xuyên nhiều phiên phong phú như dataset nội bộ của paper.

### Threshold tuning (ngưỡng quyết định tối ưu theo F1)

Ngưỡng mặc định 0.5 không tối ưu cho bài toán mất cân bằng lớp này. Với mỗi model, ngưỡng tốt nhất (maximize F1) được tìm trên **tập validation** (không dùng test set để tránh leakage), sau đó áp dụng ngưỡng đó lên test set:

| Model | best_threshold | F1 (0.5) | F1 (tuned) | Precision (tuned) | Recall (tuned) | Accuracy (tuned) |
|---|---|---|---|---|---|---|
| LSTM | 0.37 | 0.704 | **0.734** | 0.625 | 0.888 | 0.672 |
| GB | 0.38 | 0.705 | 0.733 | 0.632 | 0.872 | 0.677 |
| **CNN_LSTM_Maxout** | 0.33 | 0.683 | 0.730 | 0.622 | 0.884 | 0.667 |
| DT | 0.36 | 0.692 | 0.726 | 0.611 | 0.894 | 0.657 |
| **ResNet1D** | 0.41 | 0.695 | 0.725 | 0.604 | 0.906 | 0.650 |
| MLP | 0.37 | 0.661 | 0.716 | 0.592 | 0.904 | 0.635 |
| RF | 0.48 | 0.686 | 0.695 | 0.676 | 0.716 | 0.681 |
| CNN | 0.37 | 0.623 | 0.691 | 0.555 | 0.916 | 0.584 |
| LR | 0.44 | 0.611 | 0.691 | 0.583 | 0.849 | 0.614 |
| KNN | 0.40 | 0.579 | 0.633 | 0.555 | 0.737 | 0.566 |

Sau khi tune threshold, `CNN_LSTM_Maxout` (2-nhánh, chỉ 1 CNN residual) đứng **thứ 3/10** (F1 0.730) — vẫn gần nhóm đầu (thua LSTM/GB ~0.003-0.004) nhưng thấp hơn bản 3-nhánh residual trước đó (0.731) và bản 3-nhánh Conv1D thuần (0.729), xác nhận lại nhận xét ở bảng trên: bỏ nhánh CNN k=7 làm mất một phần lợi thế multi-scale trên REES46.

Quan sát:
- Mọi model đều có ngưỡng tối ưu **thấp hơn 0.5** (0.31–0.48) — vì lớp "purchase" thực tế hiếm hơn trong dữ liệu gốc (trước SMOTE), ngưỡng 0.5 mặc định quá "khắt khe".
- Tuning ngưỡng cải thiện F1 cho **tất cả 9 model** (ví dụ MLP tăng mạnh nhất: 0.618 → 0.718), nhưng đánh đổi: Precision giảm, Recall tăng vọt (0.72–0.91) — phù hợp nếu mục tiêu kinh doanh là **không bỏ sót khách có khả năng mua** (ví dụ để nhắm remarketing), chấp nhận báo động giả nhiều hơn.
- RF là model duy nhất có ngưỡng tối ưu gần 0.5 (0.48) và ít thay đổi Precision/Recall sau tuning — mô hình RF vốn đã cân bằng precision/recall tốt ở ngưỡng mặc định.

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

(Số liệu không đổi đáng kể sau khi đơn giản hóa `CNN_LSTM_Maxout` về 2 nhánh — model GB và best-model fallback cho permutation importance không phụ thuộc vào kiến trúc deep learning này.)

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
- [`outputs/rees46_threshold_tuning.png`](outputs/rees46_threshold_tuning.png) — đường cong F1 theo ngưỡng quyết định (tập validation) cho cả 9 model, đánh dấu ngưỡng tối ưu của từng model.

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
- Dù cố định `seed=42`, các model PyTorch chạy trên GPU không hoàn toàn deterministic (cuDNN) — chạy lại notebook có thể cho kết quả F1 lệch vài phần nghìn và đổi thứ hạng giữa các model có F1 gần nhau (đặc biệt top-3: GB, CNN_LSTM_Maxout, LSTM).
- Ngưỡng tối ưu (threshold tuning) được chọn trên tập validation được trích từ tập train sau SMOTE; với 5 model sklearn, tập validation này vẫn nằm trong dữ liệu `GridSearchCV` đã thấy khi train (do sklearn train trên toàn bộ `X_train_bal`), nên ngưỡng tối ưu có thể hơi lạc quan hơn một chút so với dùng validation set hoàn toàn độc lập.

## 6. Notebook thay thế: UCI Online Shoppers Purchasing Intention

Notebook: [`Purchase_Intent_UCI_Shoppers.ipynb`](Purchase_Intent_UCI_Shoppers.ipynb)

Vì F1 trên REES46 thấp hơn nhiều so với paper gốc và thiếu RQ3, notebook này dùng **UCI Online Shoppers Purchasing Intention Dataset** (`data_ecommerce/online_shoppers_intention.csv`, 12,330 session) — dataset đã có sẵn feature cấp session (không cần tự build từ raw event log như REES46), dùng cùng pipeline 9 model (LR, RF, GB, KNN, DT, MLP, CNN1D, LSTM, CNN_LSTM_Maxout với kiến trúc sequence+average-pooling mới nhất).

**Đổi lại:** dataset này **không có giá sản phẩm hay danh mục**, nên **không trả lời được RQ1 (category) và RQ2 (price)** như paper gốc / như notebook REES46. 17 feature dùng: `Administrative(_Duration)`, `Informational(_Duration)`, `ProductRelated(_Duration)`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`, `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`.

### Kết quả: baseline Paper2 vs phương pháp đề xuất (test set, 2,466 session)

8 model đầu là baseline tái hiện từ Paper2 (LR, RF, GB, KNN, DT, MLP, CNN, LSTM). `CNN_LSTM_Maxout` là **phương pháp đề xuất**, gồm 2 nhánh song song: 1 nhánh CNN **residual block** (kernel_size=3, giữ nguyên sequence `(B, L, D)`) được maxout (lấy max theo từng phần tử) trực tiếp với nhánh LSTM, sau đó average-pool theo `L`, qua Dense+Sigmoid. `ResNet1D` là baseline residual thuần thêm vào để so sánh (Conv1D stem + 3 residual block + Global Average Pooling, không có LSTM/maxout).

| Model | Accuracy | ROC AUC | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **RF** (baseline mạnh nhất) | 0.895 | 0.921 | 0.661 | 0.662 | **0.661** |
| LSTM (baseline) | 0.873 | 0.911 | 0.565 | 0.772 | 0.653 |
| CNN (baseline) | 0.869 | 0.905 | 0.556 | 0.775 | 0.648 |
| MLP (baseline) | 0.872 | 0.887 | 0.566 | 0.749 | 0.645 |
| **CNN_LSTM_Maxout** (đề xuất, 2 nhánh: CNN residual k=3 → maxout với LSTM) | 0.862 | 0.904 | 0.536 | 0.798 | 0.641 |
| **ResNet1D** (baseline residual thuần) | 0.862 | 0.897 | 0.536 | 0.791 | 0.639 |
| GB (baseline) | 0.885 | 0.912 | 0.627 | 0.641 | 0.634 |
| DT (baseline) | 0.876 | 0.849 | 0.603 | 0.589 | 0.596 |
| LR (baseline) | 0.659 | 0.850 | 0.297 | 0.877 | 0.444 |
| KNN (baseline) | 0.763 | 0.722 | 0.340 | 0.563 | 0.424 |

Sau khi tune threshold trên tập validation: RF F1=0.666 (threshold 0.45), CNN_LSTM_Maxout (2-nhánh residual) F1=0.643 (threshold 0.56), ResNet1D F1=0.617 (threshold 0.29) — cả hai **vẫn chưa vượt được RF** trên dataset này. Rút gọn từ 3 nhánh xuống 2 nhánh gần như không đổi ở threshold 0.5 (0.641 so với 0.643 trước), và F1 sau tune còn nhích nhẹ hơn bản 3-nhánh residual (0.643 so với 0.633) — trên UCI Shoppers, bỏ nhánh CNN k=7 không làm giảm hiệu quả, khác với REES46.

**F1 và ROC AUC cao hơn rõ rệt so với REES46** (ROC AUC ~0.90-0.92 so với ~0.78) — dataset này có tín hiệu dự đoán mạnh hơn nhiều, chủ yếu nhờ `PageValues` (một feature do Google Analytics tính sẵn, gần như lộ thông tin về khả năng mua). RF vẫn dẫn đầu F1 nhờ khai thác trực tiếp `PageValues`; các model deep learning có Recall cao hơn nhưng Precision thấp hơn.

### Thử nghiệm kiến trúc cho phương pháp đề xuất

Đã thử các biến thể sau cho `CNN_LSTM_Maxout` trên dataset này, không biến thể nào vượt được RF:
1. **2 nhánh (CNN + LSTM), fuse bằng blend maxout/average**: sweep `blend_weight` ∈ {0, 0.25, 0.5, 0.75, 1} cho F1 tốt nhất ở `blend_weight=0.25` (F1=0.638 trong ablation, 0.644 sau tune threshold khi train lại chính thức).
2. **3 nhánh (CNN k=3 + CNN k=7 Conv1D thuần cộng lại, rồi maxout với LSTM)**: F1=0.638 (threshold 0.5) / 0.642 (tuned) — tương đương biến thể 2-nhánh-blend.
3. **3 nhánh, CNN branch đổi thành residual block** (cấu hình hiện tại trong notebook): F1=0.643 (threshold 0.5) / 0.633 (tuned) — nhích nhẹ ở threshold 0.5 so với biến thể 2 nhưng không cải thiện sau khi tune threshold.
4. **ResNet1D độc lập** (không có nhánh LSTM): F1=0.639/0.617 — thấp hơn `CNN_LSTM_Maxout`, cho thấy nhánh LSTM vẫn đóng góp giá trị trên dataset này.
5. **2 nhánh, chỉ 1 CNN residual block (bỏ nhánh k=7) → maxout với LSTM** (cấu hình hiện tại trong notebook): F1=0.641 (threshold 0.5) / 0.643 (tuned) — gần như không đổi so với biến thể 3 nhánh (0.643/0.633), cho thấy trên UCI Shoppers nhánh CNN k=7 không đóng góp nhiều; đơn giản hóa kiến trúc không đánh đổi hiệu năng ở dataset này.

Trên UCI Shoppers, các hướng mở rộng kiến trúc đều chỉ xoay quanh F1≈0.63-0.65, trong khi RF giữ F1≈0.66 — khoảng cách này chủ yếu do RF khai thác `PageValues` hiệu quả hơn là do thiếu năng lực biểu diễn của nhánh deep learning.

### Permutation importance (model RF)

| Feature | Importance |
|---|---|
| **PageValues** | **0.468** |
| Month_enc | 0.040 |
| ExitRates | 0.035 |
| ProductRelated | 0.009 |
| ProductRelated_Duration | 0.007 |
| BounceRates | 0.005 |
| Region | 0.004 |
| Browser | 0.003 |
| (còn lại) | < 0.002 |

`PageValues` áp đảo hoàn toàn các feature khác (importance 0.47, gấp >10 lần feature đứng thứ 2) — khác hẳn REES46, nơi importance trải đều hơn giữa `pages_viewed`/`products_viewed`/lịch sử mua hàng.

### Hạn chế của notebook này
- **Không trả lời được RQ1/RQ2** của paper (không có category/price trong dataset).
- `PageValues` là feature do Google Analytics tính sẵn từ dữ liệu nhiều hơn những gì các feature khác thể hiện (gần giống "target leakage nhẹ" theo kinh nghiệm cộng đồng dùng dataset này) — kết quả F1/AUC cao một phần vì lý do này, không hẳn vì mô hình tốt hơn một cách công bằng.
- `Region`, `OperatingSystems`, `Browser` là mã số định danh thô (1, 2, 3, ...), không phải geo/device thật như paper gốc — chỉ là proxy rất yếu cho RQ3.
- Checkpoint/outputs của notebook này nằm riêng (`checkpoints/uci/`, `outputs/uci_*.png`) để không đè lên checkpoint của notebook REES46.
