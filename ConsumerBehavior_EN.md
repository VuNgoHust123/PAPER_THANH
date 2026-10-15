# Consumer Behavior Prediction and Precision Marketing — English Guide

**Notebook:** `Purchase_Prediction_ConsumerBehavior.ipynb`
**Paper:** Lin, J. (2025). *Application of machine learning in predicting consumer behavior and precision marketing.* PLOS One 20(5): e0321854. https://doi.org/10.1371/journal.pone.0321854

---

## 1. What the Paper Does

### Problem
E-commerce businesses need to accurately identify customers likely to purchase, so they can target precision marketing tactics (recommendation systems, dynamic pricing, personalized advertising).

### Data and preprocessing (per the paper)
- **Online Shoppers Purchasing Intention Dataset** (UCI, 2016) — 12,330 sessions, 10 numerical + 8 categorical features, target `Revenue`.
- Categorical columns `Month`, `VisitorType`, `Weekend` encoded with LabelEncoder; `StandardScaler` for numerical columns.
- `IsolationForest` (contamination=0.01) removes extreme outliers; `SMOTE` upsamples the minority (purchase) class.
- Three-step feature selection: drop skewness > 1, drop mutual information < 0.05, drop one of each Pearson-correlated pair (> 0.7).
- Train/test split 80/20, repeated 50 times with different random seeds; Precision, Recall, ROC AUC reported as averages.
- Hyperparameters tuned via Grid/Random Search (Table 1): SVM (C=50.12, gamma=0.12, RBF kernel), XGBoost (learning_rate=0.11, max_depth=4, subsample=0.8), CatBoost (iterations=245, learning_rate=0.09, max_depth=8), BPANN (hidden_layers=3, epochs=50, early stopping).

### 4 models compared
SVM (RBF kernel), XGBoost, CatBoost, BPANN (backpropagation artificial neural network).

### Paper's headline results (Table 2)
| Model | Precision | Recall | ROC AUC |
|-------|-----------|--------|---------|
| SVM | 0.954 | 0.886 | 0.977 |
| XGBoost | 0.935 | 0.925 | 0.984 |
| **CatBoost** | 0.934 | 0.935 | **0.985** |
| BPANN | 0.901 | 0.902 | 0.955 |

CatBoost and XGBoost perform best on large-scale, complex-feature data. SVM has the highest precision but lower recall. CatBoost's feature importance ranks `Administrative` (Adm) and `PageValues` (PV) as the two most influential features.

---

## 2. Dataset Used in This Notebook

This notebook uses the **exact same UCI dataset** the paper uses (no proxy substitution needed), loaded from the existing cache `data_ecommerce/online_shoppers_intention.csv`.

| Property | Value |
|----------|-------|
| Sessions | 12,330 |
| Original features | 17 (10 numeric + 7 categorical/boolean in the actual CSV) |
| Target | `Revenue` — True (purchase) / False (no purchase) |
| Class imbalance | 10,422 no-purchase (84.5%) / 1,908 purchase (15.5%) |

---

## 3. Installation

```bash
pip install torch scikit-learn pandas numpy matplotlib xgboost catboost imbalanced-learn
```

GPU (CUDA) is used for BPANN when available; auto-detected via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

---

## 4. How to Run

1. Open `Purchase_Prediction_ConsumerBehavior.ipynb` in Jupyter or VS Code.
2. Run all cells top to bottom (Run All). Later cells depend on variables (`X`, `y`, `results`) defined earlier.
3. The config cell (section 2 of the notebook) sets `n_repeats = 10` — the paper uses 50 repeats. Increase it to 50 for a closer match to the paper, at the cost of longer runtime (SVM/XGBoost/CatBoost/BPANN are each repeated `n_repeats` times).
4. Measured runtime on a GPU machine with `n_repeats=10`: under 5 minutes for all 4 models combined.

---

## 5. Cell-by-Cell Walkthrough

| Section | Content |
|---------|---------|
| Imports + seed/device | Load libraries, fix seed=42, detect GPU/CPU |
| Load data | Read from the `data_ecommerce/` cache |
| Explore | Print dtypes, missing-value count, `Revenue` distribution |
| Class distribution plot | Confirms ~5.5:1 imbalance |
| Preprocessing & feature selection | LabelEncoder, skewness/MI/Pearson filtering (**thresholds adjusted**, see section 7) |
| Shared helper functions | `remove_outliers` (IsolationForest), `balance_with_smote` (SMOTE) — applied only to the training fold of each repeat |
| `eval_once_sklearn` / `run_repeated_sklearn` | Single-repeat and repeated evaluation for SVM/XGBoost/CatBoost |
| `ShopperDataset`, `BPANN` | PyTorch Dataset and model (`nn.Module`, `super(BPANN, self).__init__()`) — a 3-hidden-layer feed-forward network |
| `eval_once_bpann` / `run_repeated_bpann` | Hand-written training loop (Adam, BCEWithLogitsLoss, early stopping) |
| Run 4 models | Uses the paper's Table 1 hyperparameters |
| Results table | "ours" vs "paper" comparison |
| Plots | Metric bar chart, ROC curves, confusion matrices (CatBoost & SVM), CatBoost feature importance |

---

## 6. Understanding the Results — actually measured numbers (n_repeats=10)

| Model | Precision (ours) | Precision (paper) | Recall (ours) | Recall (paper) | ROC AUC (ours) | ROC AUC (paper) | Accuracy (ours) |
|-------|------|------|------|------|------|------|------|
| SVM | 0.544 | 0.954 | 0.652 | 0.886 | 0.825 | 0.977 | 0.861 |
| XGBoost | 0.687 | 0.935 | 0.659 | 0.925 | 0.929 | 0.984 | 0.901 |
| CatBoost | 0.685 | 0.934 | 0.648 | 0.935 | 0.927 | 0.985 | 0.899 |
| BPANN | 0.514 | 0.901 | 0.804 | 0.902 | 0.908 | 0.955 | 0.852 |

These are **real measured numbers** from running this notebook, not estimates (see `outputs/results_table.csv`).

### What each metric means
- **Precision:** of sessions predicted "will purchase", what fraction actually did. Low precision means many false alarms.
- **Recall:** of sessions that actually purchased, what fraction the model caught. High recall means few missed buyers.
- **ROC AUC:** overall ability to separate the two classes, independent of the 0.5 threshold.
- **Accuracy:** overall fraction correct — misleading on imbalanced data, not used as the primary metric here.

### Why precision is noticeably lower than the paper's, while ROC AUC is close
ROC AUC in this notebook (0.83–0.93) is reasonably close to the paper (0.955–0.985), meaning the models still separate the two classes well. But precision is much lower (0.51–0.69 vs. 0.90–0.95), with recall rising relative to precision — this is a direct consequence of training on SMOTE-balanced data while the test set keeps the original imbalance (84.5%/15.5%): the model learns to predict "purchase" far more often than the true test-set rate, so more positive predictions are wrong (lower precision) but more true positives are caught (higher recall). The paper does not specify whether the decision threshold was recalibrated after SMOTE; this is an important, documented difference — not a bug.

---

## 7. Key Differences from the Paper

| Aspect | Paper | This Notebook |
|--------|-------|---------------|
| Dataset | UCI Online Shoppers | Same dataset (no proxy needed) |
| Evaluation repeats | 50 | 10 (raise to 50 in the config cell) |
| Feature selection | drop skewness > 1, drop MI < 0.05 | **Thresholds adjusted**: skewness-based elimination skipped (it would remove `PageValues`, the paper's own top feature); MI threshold lowered to 0.005. Full reasoning is in the markdown cell right before the feature-selection cell |
| SMOTE/IsolationForest/StandardScaler order relative to splitting | Not specified | Fit only on each repeat's training fold, to avoid test-set leakage |
| Decision-threshold recalibration after SMOTE | Not mentioned | Default 0.5 threshold used — the main driver of the lower precision vs. the paper (see section 6) |
| CatBoost feature importance | Adm, PV top | In this run: Month, OperatingSystems, PageValues top (see `outputs/feature_importance.png`) — differs due to the different feature-selection thresholds and final feature set used for training |
| Hyperparameters | Grid/Random Search | Uses the paper's already-tuned Table 1 values directly, no re-tuning |

---

## 8. Tips & Notes

- To match the paper's numbers more closely, raise `n_repeats` to 50 in the config cell (longer runtime).
- To improve precision, try raising the decision threshold above 0.5 in `eval_once_sklearn`/`eval_once_bpann`, or soften SMOTE (e.g., `sampling_strategy` < 1.0 instead of full balancing).
- All 6 figures and the results CSV were actually saved to `outputs/` when this notebook was run — not placeholders.
- Results may vary slightly across runs due to SMOTE and the random train/test splits, but the model ranking (XGBoost ≈ CatBoost > BPANN > SVM by ROC AUC) should remain stable.
