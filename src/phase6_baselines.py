"""PHASE 6 — Baseline models, established BEFORE any deep learning, on the
Phase-1-5-corrected data (real filing dates, fixed purge gap, leave-one-out
sector feature).

Phase 6.5 rework: the feature scaling was switched to
winsorise + log1p + RobustScaler (see prepare_dataset.py), and three
outlier-robust / better-separated target families were added
(see feature_engineering.add_advanced_labels). This script now runs the
same model set against SEVERAL target sets so the old vs new labels can be
compared side by side on identical splits:

  raw      : reg=fwd_return_{h}d          clf=beats_median_{h}d        (original)
  vol_adj  : reg=fwd_return_vol_adj_{h}d  clf=top_tercile_{h}d         (tails only)
  rank     : reg=fwd_return_rank_{h}d     clf=beats_sector_median_{h}d (sector-neutral)

Models (minimum required set):
1. Historical mean baseline (regression) / majority class (classification)
2. Naive momentum baseline (raw target only - predicts trailing return)
3. Linear/Ridge Regression / Logistic Regression
4. Random Forest
5. Gradient Boosting (HistGradientBoosting - XGBoost/LightGBM not installed)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import FEATURE_COLS, HORIZONS

warnings.filterwarnings("ignore")  # sklearn convergence warnings on this noisy target are expected, not actionable

SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "splits"
MODEL_HORIZON = 30  # report full detail for this horizon; summary table covers all 4

# (tag, regression target template, classification target template, decision
# boundary for the regression DirAcc metric). rank returns live in [0, 1] so
# "up" there means "above the median rank" (0.5), not "above zero".
TARGET_SETS = [
    ("raw",     "fwd_return_{h}d",         "beats_median_{h}d",        0.0),
    ("vol_adj", "fwd_return_vol_adj_{h}d", "top_tercile_{h}d",         0.0),
    ("rank",    "fwd_return_rank_{h}d",    "beats_sector_median_{h}d", 0.5),
]


def load_splits():
    train = pd.read_csv(SPLIT_DIR / "train.csv", parse_dates=["Date"])
    val = pd.read_csv(SPLIT_DIR / "val.csv", parse_dates=["Date"])
    test = pd.read_csv(SPLIT_DIR / "test.csv", parse_dates=["Date"])
    return train, val, test


with open(SPLIT_DIR / "scaler.json") as _f:
    _SCALER = json.load(_f)
_SCALER_INDEX = {name: i for i, name in enumerate(_SCALER["feature_cols"])}


def unscale_feature(values: np.ndarray, col: str) -> np.ndarray:
    """FEATURE_COLS are winsorised + (some) log1p'd + robust-scaled in
    prepare_dataset.py - using one directly as a raw-return prediction (as
    the momentum baseline needs) requires inverting that transform first,
    or the units don't match (using the scaled value directly once
    produced a nonsensical R2 of -86.8). roc_* columns are not in
    log1p_cols, so the affine inverse alone is exact up to the 1%/99%
    winsor clip."""
    i = _SCALER_INDEX[col]
    out = values * _SCALER["scale"][i] + _SCALER["mean"][i]
    if col in _SCALER.get("log1p_cols", []):
        out = np.expm1(out)
    return out


def naive_momentum_pred(test_df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Predicts the future h-day return as EQUAL to the trailing h-day
    return (roc_{h} where available, else the closest available ROC
    window) - the classic "trend continues" naive baseline."""
    roc_col = f"roc_{horizon}" if f"roc_{horizon}" in test_df.columns else "roc_20"
    return unscale_feature(test_df[roc_col].to_numpy(), roc_col)


def evaluate_regression(y_true, y_pred, name: str, boundary: float) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    pearson_r = pearsonr(y_true, y_pred)[0] if np.std(y_pred) > 0 else 0.0
    spearman_r = spearmanr(y_true, y_pred)[0] if np.std(y_pred) > 0 else 0.0
    dir_acc = np.mean((y_true > boundary) == (y_pred > boundary))
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2,
            "Pearson_r": pearson_r, "Spearman_r": spearman_r, "DirAcc": dir_acc}


def evaluate_classification(y_true, y_pred, name: str) -> dict:
    return {"model": name, "Accuracy": (y_true == y_pred).mean()}


def run_target_set(train, test, horizon, reg_col, clf_col, boundary):
    # classification target may carry deliberate NaN (top_tercile middle
    # third) - drop those rows from both fit and score.
    clf_tr = train.dropna(subset=[clf_col])
    clf_te = test.dropna(subset=[clf_col])

    X_train, X_test = train[FEATURE_COLS], test[FEATURE_COLS]
    Xc_train, Xc_test = clf_tr[FEATURE_COLS], clf_te[FEATURE_COLS]
    y_train_reg, y_test_reg = train[reg_col].to_numpy(), test[reg_col].to_numpy()
    y_train_dir, y_test_dir = clf_tr[clf_col].to_numpy(), clf_te[clf_col].to_numpy()

    reg_results, clf_results = [], []

    # 1. Historical mean / majority class
    reg_results.append(evaluate_regression(
        y_test_reg, np.full_like(y_test_reg, y_train_reg.mean()), "1_historical_mean", boundary))
    clf_results.append(evaluate_classification(
        y_test_dir, np.full_like(y_test_dir, int(y_train_dir.mean() > 0.5)), "1_majority_class"))

    # 2. Naive momentum - only meaningful against the raw-return target
    if reg_col == f"fwd_return_{horizon}d":
        reg_results.append(evaluate_regression(
            y_test_reg, naive_momentum_pred(test, horizon), "2_naive_momentum", boundary))

    # 3. Ridge / Logistic Regression
    ridge = Ridge(alpha=10.0).fit(X_train, y_train_reg)
    reg_results.append(evaluate_regression(y_test_reg, ridge.predict(X_test), "3_ridge_regression", boundary))
    logit = LogisticRegression(max_iter=2000, C=0.1).fit(Xc_train, y_train_dir)
    clf_results.append(evaluate_classification(y_test_dir, logit.predict(Xc_test), "3_logistic_regression"))

    # 4. Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train_reg)
    reg_results.append(evaluate_regression(y_test_reg, rf.predict(X_test), "4_random_forest", boundary))
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=42)
    rf_clf.fit(Xc_train, y_train_dir)
    clf_results.append(evaluate_classification(y_test_dir, rf_clf.predict(Xc_test), "4_random_forest"))

    # 5. Gradient Boosting (HistGradientBoosting - documented XGBoost/LightGBM substitute)
    gbm = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=42)
    gbm.fit(X_train, y_train_reg)
    reg_results.append(evaluate_regression(y_test_reg, gbm.predict(X_test), "5_gradient_boosting", boundary))
    gbm_clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=42)
    gbm_clf.fit(Xc_train, y_train_dir)
    clf_results.append(evaluate_classification(y_test_dir, gbm_clf.predict(Xc_test), "5_gradient_boosting"))

    return pd.DataFrame(reg_results), pd.DataFrame(clf_results)


def main() -> None:
    train, val, test = load_splits()
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"Scaler: {_SCALER.get('method', 'legacy standard')}\n")

    print("=" * 90)
    print(f"DETAILED RESULTS — {MODEL_HORIZON}d horizon, target set 'raw'")
    print("=" * 90)
    reg_df, clf_df = run_target_set(
        train, test, MODEL_HORIZON, f"fwd_return_{MODEL_HORIZON}d", f"beats_median_{MODEL_HORIZON}d", 0.0)
    print(f"\nRegression (target: fwd_return_{MODEL_HORIZON}d)")
    print(reg_df.to_string(index=False))
    base = max(test[f"beats_median_{MODEL_HORIZON}d"].mean(), 1 - test[f"beats_median_{MODEL_HORIZON}d"].mean())
    print(f"\nClassification (target: beats_median_{MODEL_HORIZON}d, naive base rate = {base:.4f})")
    print(clf_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("SUMMARY — all horizons x all target sets (DirAcc for regression, Accuracy for classification)")
    print("=" * 90)
    summary_rows = []
    for tag, reg_tpl, clf_tpl, boundary in TARGET_SETS:
        for h in HORIZONS:
            reg_h, clf_h = run_target_set(
                train, test, h, reg_tpl.format(h=h), clf_tpl.format(h=h), boundary)
            for _, row in reg_h.iterrows():
                summary_rows.append({"target_set": tag, "horizon": f"{h}d", "task": "regression",
                                     "model": row["model"], "MAE": row["MAE"],
                                     "Spearman_r": row["Spearman_r"], "DirAcc": row["DirAcc"]})
            for _, row in clf_h.iterrows():
                summary_rows.append({"target_set": tag, "horizon": f"{h}d", "task": "classification",
                                     "model": row["model"], "Accuracy": row["Accuracy"]})
    summary = pd.DataFrame(summary_rows)
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(summary.to_string(index=False))
    out = Path(__file__).resolve().parent.parent / "data" / "phase6_baseline_results.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
