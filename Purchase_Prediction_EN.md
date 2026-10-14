# Consumer Purchase Prediction in E-Commerce — English Guide

**Notebook:** `Purchase_Prediction_ECommerce.ipynb`  
**Paper:** Esmeli, R. & Gokce, A. (2025). *An Analysis of Consumer Purchase Behavior Following Cart Addition in E-Commerce Utilizing Explainable Artificial Intelligence.* J. Theor. Appl. Electron. Commer. Res., 20, 28.

---

## 1. What the Paper Does

### Problem
E-commerce platforms suffer from high cart abandonment rates (67–81%). Most prior research predicts general purchase intent but ignores **what happens after a product is added to the cart** — the moment where abandonment is most costly. The paper asks:

1. How do **product category transitions** during a session affect ML purchase predictions?
2. How do **session-level price variations** (volatility, relative pricing) impact predictions?
3. How do **geographic characteristics** influence predictive models?

### Framework (3 Stages)
The paper proposes a 3-stage ML framework:

| Stage | Description |
|-------|-------------|
| **1 — Data Collection** | Session logs from a UK SaaS e-commerce platform (2 datasets, July–Dec 2019) |
| **2 — Feature Engineering** | Extract 16 session-level features: pages viewed, price min/max/difference, location (lat/lon), category transitions, device type, day/time, prior purchase history |
| **3 — Model Training & Evaluation** | 8 ML models evaluated with 10-fold CV + SMOTE; interpreted with SHAP |

### 8 Models Compared

| Model | Type |
|-------|------|
| Logistic Regression (LR) | Linear |
| Decision Tree (DT) | Tree |
| K-Nearest Neighbor (KNN) | Instance-based |
| Random Forest (RF) | Ensemble |
| Gradient Boosting (GB) | Ensemble |
| Multi-Layer Perceptron (MLP) | Neural Network |
| Convolutional Neural Network (CNN) | Neural Network |
| Long Short-Term Memory (LSTM) | Neural Network |

### Key Findings
- **Gradient Boosting (GB)** achieved the best F1 score of **89%**
- Most important features (via SHAP): prior purchase history, session frequency, pages viewed, geographic location (lat/lon), price min/max
- Category transitions had a moderate effect (SHAP: 0.21–0.51 depending on dataset)
- Geographic features were surprisingly influential (SHAP: 0.56–0.76)

---

## 2. Dataset Used in This Notebook

Since the paper's original data is proprietary (UK SaaS company, not public), this notebook uses the publicly available **UCI Online Shoppers Purchasing Intention Dataset**.

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository (ID 468) |
| URL | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Samples | 12,330 sessions |
| Features | 17 original + 6 engineered = 23 total |
| Target | `Revenue` — True (purchase) / False (no purchase) |
| Class imbalance | ~84% non-purchase / ~16% purchase |
| Collection period | 1 year, from a real retail platform |

### Original Features

| Feature | Description |
|---------|-------------|
| Administrative | Number of admin-type pages visited |
| Administrative_Duration | Time spent on admin pages (seconds) |
| Informational | Number of informational pages visited |
| Informational_Duration | Time spent on informational pages |
| ProductRelated | Number of product pages visited |
| ProductRelated_Duration | Time spent on product pages |
| BounceRates | Avg bounce rate of visited pages |
| ExitRates | Avg exit rate of visited pages |
| PageValues | Avg value of pages visited before a transaction |
| SpecialDay | Closeness to a special day (e.g., Valentine's Day), 0–1 |
| Month | Month of the session |
| OperatingSystems | OS type (1–8) |
| Browser | Browser type (1–13) |
| Region | Geographic region (1–9) |
| TrafficType | Traffic source (1–20) |
| VisitorType | Returning / New / Other |
| Weekend | Boolean — was the session on a weekend? |

### Engineered Features (Stage 2 mapping to paper)

| Engineered Feature | Paper Equivalent | Formula |
|--------------------|-----------------|---------|
| `pages_viewed` | `pagesviewed` | Administrative + Informational + ProductRelated |
| `total_duration` | session duration | Sum of all duration columns |
| `cat_transitions` | `#catchanges` | Count of distinct page-type categories visited (0–3) |
| `engagement_quality` | — | 1 − BounceRates |
| `price_sensitivity` | `pricedifference` proxy | PageValues (higher = more price-aware session) |
| `is_special_day` | `pricechange` proxy | 1 if SpecialDay > 0, else 0 |

---

## 3. Installation

```bash
pip install torch scikit-learn pandas numpy matplotlib imbalanced-learn openpyxl
```

GPU (CUDA) is optional but recommended for the neural network models (MLP, CNN, LSTM). The code auto-detects via:
```python
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The dataset is downloaded automatically on first run — no manual download needed.

---

## 4. How to Run

1. Open `Purchase_Prediction_ECommerce.ipynb` in **Jupyter Notebook** or **VS Code**.
2. Run all cells **in order from top to bottom** (use *Run All* or Shift+Enter each cell).
3. Do **not** skip or reorder cells — later cells depend on variables from earlier cells.
4. Estimated runtime:
   - Sklearn models (LR, DT, KNN, RF, GB): **~1–3 min** total
   - Neural models (MLP, CNN, LSTM): **~5–15 min** on GPU / **~30–60 min** on CPU

> **Tip:** To speed things up, reduce `n_splits` from 5 to 3 in the setup cell, or reduce `epochs` from 30 to 15 in the `Run All Models` cell.

---

## 5. Cell-by-Cell Walkthrough

### Setup Cells

| Cell | What it does |
|------|-------------|
| **Imports** | Loads all libraries including `imbalanced-learn` (auto-installs if missing) |
| **Seed + Device** | Sets `seed=42` for reproducibility; detects GPU/CPU; sets `n_splits=5` |

### Stage 1 — Data Loading

| Cell | What it does |
|------|-------------|
| **Download & Load** | Downloads UCI CSV to `data_ecommerce/` (once); loads into pandas DataFrame |
| **Explore** | Prints dtypes, missing value count, and class distribution |
| **Plot distributions** | 3 charts: class balance bar, purchase rate by month, purchase rate by visitor type |

**What to look for in the plots:**
- The class bar chart confirms strong imbalance (~10:1 ratio in the original paper's data, ~5:1 here) — this is why SMOTE is needed
- Monthly pattern: certain months (e.g., November, May) have higher purchase rates — likely driven by sales events
- Returning visitors have a higher purchase rate than new visitors — loyalty matters

### Stage 2 — Feature Engineering

| Cell | What it does |
|------|-------------|
| **Engineer features** | Derives 6 new session-level features (see table above) |
| **Box plots** | Shows `pages_viewed`, `price_sensitivity`, `cat_transitions` distribution by outcome |

**What to look for in the box plots:**
- `price_sensitivity` (PageValues): purchasers tend to visit significantly higher-value pages — strong predictor
- `pages_viewed`: purchasers typically browse more pages before deciding
- `cat_transitions`: buyers tend to visit 2–3 category types vs. non-buyers staying in 1

### Preprocessing

| Cell | What it does |
|------|-------------|
| **Encode + Scale** | Maps Month (string→int), VisitorType (string→int), Weekend (bool→int); StandardScaler on all features |
| **ShopperDataset** | PyTorch Dataset class wrapping numpy arrays as float32 tensors |

### Models

| Cell | Architecture |
|------|-------------|
| **Sklearn models** | LR, DT, KNN, RF, GB defined as sklearn estimators |
| **MLP_Classifier** | 3-layer MLP with BatchNorm + Dropout: input→128→64→2 |
| **CNN1D_Classifier** | Treats features as 1D signal: (batch, 1, features) → Conv1D×2 → MaxPool → FC |
| **LSTM_Classifier** | Treats each feature as a time step: (batch, features, 1) → 2-layer LSTM → FC |

> **Why CNN and LSTM on tabular data?**  
> Both models are applied by reshaping the feature vector into a "sequence". This is an experimental approach tested in the paper to see whether temporal/local patterns among ordered features can be captured. Results are usually inferior to ensemble methods on tabular data.

### Cross-Validation

Both `sklearn_kfold()` and `neural_kfold()` follow the same protocol:
1. **StratifiedKFold** — preserves class ratio in each fold
2. **SMOTE applied only to the training fold** — prevents leakage into the test fold
3. All fold predictions are **concatenated** for metric computation and ROC curve plotting

---

## 6. Understanding the Results

### Metrics

| Metric | Formula | What it means |
|--------|---------|---------------|
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean — good when both false positives and false negatives matter |
| **Recall** | TP / (TP + FN) | Of all actual purchasers, how many did we catch? (important for retargeting) |
| **Precision** | TP / (TP + FP) | Of predicted purchasers, how many actually bought? (important for discount targeting) |
| **ROC-AUC** | Area under the ROC curve | Overall discrimination ability; 0.5 = random, 1.0 = perfect |

### Expected Results on UCI Dataset

The UCI dataset differs from the paper's proprietary data, so exact numbers will differ. Typical expected ranges:

| Model | F1 Score | ROC-AUC |
|-------|----------|---------|
| GB | 0.62–0.72 | 0.92–0.94 |
| RF | 0.58–0.68 | 0.90–0.92 |
| MLP | 0.55–0.65 | 0.88–0.91 |
| CNN | 0.50–0.62 | 0.85–0.90 |
| LSTM | 0.50–0.62 | 0.85–0.90 |
| LR | 0.48–0.58 | 0.88–0.91 |
| KNN | 0.45–0.58 | 0.82–0.87 |
| DT | 0.45–0.55 | 0.72–0.80 |

> F1 scores are lower than the paper's 89% because: (1) the UCI dataset is a different domain with different feature quality, (2) the paper's data includes richer session-level signals like exact lat/lon and price history within a session.

### Bar Chart — 4 Metrics

- Each of the 4 subplots shows one metric across all 8 models, sorted by F1
- The **red dashed line** at 0.89 marks the paper's best reported F1 — a reference benchmark
- Models above the line on the paper's dataset would match or beat the paper's result
- Look for **consistent performance across all 4 metrics** — a model that excels in one but fails in another is unreliable

### ROC Curves

- The closer a curve is to the **top-left corner**, the better
- The **diagonal dashed line** is the random baseline (AUC = 0.5)
- A high AUC with low F1 indicates the model ranks well but has trouble at the default 0.5 threshold — threshold tuning could help
- **GB and RF** typically form the upper envelope (best AUC)

### Confusion Matrices (Top 4 Models)

```
                Predicted Non-Purchase   Predicted Purchase
Actual Non-Purchase    TN                     FP
Actual Purchase        FN                     TP
```

- **High TN + High TP** = good overall accuracy
- For marketing purposes, **high Recall** (low FN) is preferred — you don't want to miss potential buyers
- **High Precision** (low FP) saves marketing budget — you target only likely buyers

### Feature Importance (GB & RF)

- Features with longer bars have more influence on the prediction
- Expected top features: `PageValues` / `price_sensitivity`, `ProductRelated_Duration`, `ExitRates`, `BounceRates`, `Region`
- This mirrors the paper's finding that **price dynamics** and **geographic context** are important predictors
- `cat_transitions` (engineered) should appear somewhere in the mid-importance range

---

## 7. Key Differences from the Paper

| Aspect | Paper | This Notebook |
|--------|-------|---------------|
| Dataset | Proprietary UK SaaS data (2 datasets) | Public UCI dataset (1 dataset) |
| Class imbalance | ds1: 81% abandon / ds2: 67% abandon | ~84% no-purchase |
| Features | Exact lat/lon, per-session price history | Region code, PageValues as proxy |
| CV folds | 10-fold | 5-fold (configurable) |
| Interpretability | SHAP | Feature importance (GB, RF) |
| Best F1 | 89% (GB on ds1) | ~65–72% expected (GB) |

---

## 8. Notes & Tips

- **SMOTE leakage warning:** SMOTE is applied *inside* each CV fold to the training split only. Never apply SMOTE before splitting — it would cause data leakage and inflate metrics artificially.
- **Neural models on tabular data:** CNN and LSTM typically underperform ensemble methods (GB, RF) on tabular data. This is expected and consistent with the paper's results.
- **Threshold tuning:** The default prediction threshold is 0.5. For maximizing Recall (catching more buyers), lower the threshold; for maximizing Precision, raise it.
- **Reproducibility:** Results are fixed by `seed=42` for sklearn and `torch.manual_seed(seed)` per fold for neural models.
