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

## 8. Improved Configuration — same 4 models, retuned parameters/pipeline (sections 15–17)

Sections 1–14 faithfully reproduce the paper's setup (as described above). After seeing precision come in well below the paper's (section 6), the notebook adds **sections 15–17** that try a different configuration for the **exact same 4 models** (SVM/XGBoost/CatBoost/BPANN — no new model added), aiming for better results without "cheating" (nothing is fit on the test set, SMOTE is never applied before the split):

- Replace SMOTE with **class-weighting** (`class_weight="balanced"` for SVM, `scale_pos_weight` for XGBoost, `auto_class_weights="Balanced"` for CatBoost, `pos_weight` in `BCEWithLogitsLoss` for BPANN) — avoids the train/test distribution mismatch SMOTE introduces.
- **Decision-threshold tuning** via `precision_recall_curve`, maximizing F1 on a validation slice carved out of the training fold (the test set is never touched), instead of a fixed 0.5 threshold.
- **Light hyperparameter re-search** via `RandomizedSearchCV` (scoring="f1", cv=3, run once on a single split rather than re-run every repeat) instead of reusing the paper's Table 1 values as-is.
- **Dropping the skewness/mutual-information feature elimination**, keeping only Pearson-redundancy removal (> 0.7) — this keeps 15 of the 17 original features (see `selected_features_v2` in the notebook), since tree-based/neural models don't need near-Gaussian features the way the skewness rule assumes.

### Actually measured numbers — faithful reproduction (section 6) vs. improved configuration (n_repeats=10)

| Model | Precision orig→improved | Recall orig→improved | ROC AUC orig→improved | F1 orig→improved |
|-------|--------------------------|-----------------------|------------------------|--------------------|
| SVM | 0.544 → **0.580** | 0.652 → 0.712 | 0.825 → **0.894** | 0.593 → **0.639** |
| XGBoost | 0.687 → 0.642 | 0.659 → 0.702 | 0.929 → 0.927 | 0.673 → 0.671 |
| CatBoost | 0.685 → 0.634 | 0.648 → 0.702 | 0.927 → 0.923 | 0.666 → 0.666 |
| BPANN | 0.514 → **0.628** | 0.804 → 0.690 | 0.908 → 0.910 | 0.627 → **0.658** |

(Real measured numbers from `outputs/results_table_improved.csv` and `outputs/metrics_comparison_improved.png` — not estimates.)

### Honest takeaway — not every model improved equally

- **SVM improved the most**: ROC AUC 0.825→0.894, F1 0.593→0.639. The paper's hyperparameters (C=50.12) likely overfit once SMOTE is removed; `RandomizedSearchCV` found C=1, gamma=0.01 — much smaller and a better fit for `class_weight="balanced"`.
- **BPANN improved clearly**: F1 0.627→0.658, with a large precision gain (0.514→0.628) at the cost of a modest recall drop (0.804→0.690, still reasonable) — driven by a larger network (`hidden_dim=128`, 2 hidden layers) and class-weighted loss replacing SMOTE.
- **XGBoost and CatBoost stayed essentially flat** (F1 ~0.67 in both versions): these tree-boosting models already handle class imbalance reasonably well through their structure, so class-weighting plus re-tuning made little net difference — precision dropped slightly while recall rose correspondingly, leaving F1 unchanged.
- **None of the configurations get close to the paper's reported numbers** (precision 0.90–0.95). The most likely explanation is that the paper applied SMOTE/IsolationForest before splitting train/test (data leakage) — something this notebook deliberately avoids for trustworthy results, so matching the paper's numbers through parameter tuning alone should not be expected.

---

## 9. Feature Engineering + Stacking Ensemble (sections 18–21)

Beyond the improved configuration in section 8, the notebook tries two more techniques on the **same 4 models** (no new model added):

- **Feature engineering**: adds 4 interaction/ratio features on top of `selected_features_v2`: `PageValues_per_ProductRelated` (normalizes PageValues by pages viewed), `ExitRate_BounceRate_interaction` (the product of two features dropped for linear Pearson redundancy — but their nonlinear product can carry extra signal), `Admin_Info_duration_ratio` (reflects purchase intent), `IsHolidaySeason` (a binary flag for peak shopping months, replacing the orderless numeric `Month` encoding).
- **Stacking ensemble**: combines the 4 models via a Logistic Regression meta-learner trained on **out-of-fold** predictions (StratifiedKFold, K=3) from the 4 base models — ensuring no base model ever predicts on data it was trained on when generating meta-features, which would otherwise leak information. The decision threshold is tuned on a slice carved out of those OOF predictions. Because the cost per repeat jumps (4 models × (3 folds + 1 full fit) = 16 fits), the repeat count is reduced to `n_repeats_stack=5`.

### Actually measured numbers (see `outputs/results_table_stacking.csv`, `outputs/metrics_comparison_stacking.png`)

| Model | Precision | Recall | ROC AUC | F1 |
|-------|-----------|--------|---------|-----|
| SVM | 0.580 | 0.712 | 0.894 | 0.639 |
| XGBoost | 0.642 | 0.702 | 0.927 | 0.671 |
| CatBoost | 0.634 | 0.702 | 0.923 | 0.666 |
| BPANN | 0.628 | 0.690 | 0.910 | 0.657 |
| **Stacking (4 models)** | 0.591 | **0.779** | **0.930** | **0.672** |

### Honest takeaway

- Stacking achieves the **highest ROC AUC** (0.930, edging out XGBoost's 0.927) and the **highest F1**, but only marginally (0.672 vs. XGBoost's 0.671) — a real but very small improvement, not a breakthrough.
- The more notable effect is a **trade-off shift**: stacking has a much **higher recall** (0.779, vs. 0.712 for the best individual model, SVM) but **lower precision** (0.591, lower than even XGBoost's 0.642). The Logistic Regression meta-learner is leaning toward catching more buyers (fewer missed) at the cost of more false alarms — useful if the business goal is "don't miss potential buyers" rather than "minimize wasted marketing spend".
- Why stacking doesn't break out further: XGBoost and CatBoost are both already strong and fairly similar (both gradient-boosted trees), so they don't offer each other much complementary signal; SVM and BPANN are noticeably weaker and may pull some predictions off, limiting the ensemble's upside.
- The 4 new features from section 18 are used consistently across all 4 models inside the stacking run; this notebook does not isolate the effect of feature engineering alone (with vs. without the new features) — to measure that separately, rerun section 15's individual models with `X_v3` instead of `X_v2`.

---

## 10. Tips & Notes

- To match the paper's numbers more closely, raise `n_repeats` to 50 in the config cell (longer runtime).
- To improve precision, try raising the decision threshold above 0.5 in `eval_once_sklearn`/`eval_once_bpann`, or soften SMOTE (e.g., `sampling_strategy` < 1.0 instead of full balancing) — this is exactly the direction section 8 already explores.
- To improve stacking's precision: try a meta-learner with more nonlinear capacity than Logistic Regression (keep it simple though, to avoid overfitting on just 4 input features), or tune the threshold toward an F-beta score that favors precision over recall.
- All 6 figures and the results CSV were actually saved to `outputs/` when this notebook was run — not placeholders.
- Results may vary slightly across runs due to SMOTE and the random train/test splits, but the model ranking (XGBoost ≈ CatBoost > BPANN > SVM by ROC AUC) should remain stable.
