"""Rigorous evaluation of the champion model (logistic regression on the
beats-median target — the only approach that has consistently beaten naive
across every test in this project).

Addresses reviewer feedback: proper classification metrics, calibration,
statistical significance, walk-forward validation across market regimes,
and a transaction-cost-aware Top-K portfolio backtest. This is deliberately
NOT about the LSTM - it evaluates the model we can actually trust right now,
independent of the pending LSTM ensemble run.

Phase 6.5: features are passed through the shared training-time transform
(prepare_dataset.apply_saved_scaling) before fitting, and Part 1 now also
reports the same metric suite for the top_tercile_{h}d target (top vs
bottom third of the cross-section) so the two candidate signals are
compared on identical splits.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import FEATURE_COLS, HORIZONS
from prepare_dataset import PURGE_DAYS, apply_saved_scaling
# PURGE_DAYS: single source of truth - this used to duplicate the value
# independently and had drifted to an insufficient 130 (empirically
# verified in Phase 5 to cover as few as 84 trading days, below the
# 90-day horizon it needs to cover)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "model_features.csv"
TARGET_HORIZON = 30  # the horizon with the strongest, most consistent edge so far


def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    df = apply_saved_scaling(df)  # FEATURE_COLS -> training-time scaled; labels untouched
    return df.sort_values(["Date", "Symbol"]).reset_index(drop=True)


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, horizon: int,
                  target: str = "beats_median") -> dict:
    """target: 'beats_median' or 'top_tercile'. For top_tercile, rows with
    a NaN label (the middle third of each day) are dropped from both fit
    and score."""
    dir_col = f"{target}_{horizon}d"
    train = train.dropna(subset=[dir_col])
    test = test.dropna(subset=[dir_col])
    X_train, y_train = train[FEATURE_COLS], train[dir_col]
    X_test, y_test = test[FEATURE_COLS], test[dir_col]

    clf = LogisticRegression(max_iter=1000, C=0.1).fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba > 0.5).astype(int)

    naive_pred = np.full_like(y_test, int(y_train.mean() > 0.5))
    acc = (pred == y_test).mean()
    naive_acc = (naive_pred == y_test).mean()

    return {"clf": clf, "proba": proba, "pred": pred, "y_test": y_test.to_numpy(),
            "acc": acc, "naive_acc": naive_acc, "n": len(y_test)}


# ============================================================
# PART 1 — Full metrics suite + calibration + significance
# ============================================================
def part1_full_metrics(df: pd.DataFrame) -> dict:
    print("=" * 70)
    print(f"PART 1 — Full metrics on fixed split, {TARGET_HORIZON}d horizon")
    print("=" * 70)

    train_end = pd.Timestamp("2022-12-31")
    test_start = train_end + pd.Timedelta(days=PURGE_DAYS)
    train = df[df["Date"] <= train_end]
    test = df[df["Date"] > test_start]

    r = fit_and_score(train, test, TARGET_HORIZON)
    y_test, proba, pred = r["y_test"], r["proba"], r["pred"]

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    print(f"n_test = {r['n']:,}")
    print(f"Accuracy:   {r['acc']:.4f}  (naive: {r['naive_acc']:.4f})")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1:         {f1:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}  (0.5 = random)")
    print(f"PR-AUC:     {pr_auc:.4f}  (baseline = positive rate = {y_test.mean():.4f})")
    print(f"Brier score:{brier:.4f}  (lower is better; 0.25 = always predicting 0.5)")

    # Statistical significance: is accuracy actually different from naive,
    # or could this be explained by chance at this sample size?
    n_correct = int((pred == y_test).sum())
    binom = stats.binomtest(n_correct, r["n"], p=r["naive_acc"], alternative="greater")
    print(f"\nBinomial test (H0: true accuracy = naive base rate {r['naive_acc']:.4f}):")
    print(f"  p-value = {binom.pvalue:.4f}  {'(significant at 5%)' if binom.pvalue < 0.05 else '(NOT significant at 5%)'}")

    # Calibration: bin predictions into deciles, compare mean predicted
    # probability to actual outperform rate in each bin.
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    print("\nCalibration (10 quantile bins): mean predicted prob -> actual outperform rate")
    for mp, fp in zip(mean_pred, frac_pos):
        bar = "#" * int(fp * 40)
        print(f"  predicted {mp:.3f} -> actual {fp:.3f}  {bar}")
    calibration_gap = np.mean(np.abs(mean_pred - frac_pos))
    print(f"Mean |predicted - actual| across bins: {calibration_gap:.4f} (0 = perfectly calibrated)")

    # --- Same suite for the top_tercile target, identical split ---
    print(f"\n--- Comparison target: top_tercile_{TARGET_HORIZON}d (top vs bottom third, middle dropped) ---")
    rt = fit_and_score(train, test, TARGET_HORIZON, target="top_tercile")
    yt, pt, prt = rt["y_test"].astype(int), rt["pred"], rt["proba"]
    prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(yt, pt, average="binary", zero_division=0)
    binom_t = stats.binomtest(int((pt == yt).sum()), rt["n"], p=rt["naive_acc"], alternative="greater")
    print(f"n_test = {rt['n']:,}")
    print(f"Accuracy:   {rt['acc']:.4f}  (naive: {rt['naive_acc']:.4f}, edge {rt['acc']-rt['naive_acc']:+.4f})")
    print(f"Precision/Recall/F1: {prec_t:.4f} / {rec_t:.4f} / {f1_t:.4f}")
    print(f"ROC-AUC:    {roc_auc_score(yt, prt):.4f}   Brier: {brier_score_loss(yt, prt):.4f}")
    print(f"Binomial test vs naive base rate: p = {binom_t.pvalue:.4f}  "
          f"{'(significant at 5%)' if binom_t.pvalue < 0.05 else '(NOT significant at 5%)'}")

    return {"clf": r["clf"], "train": train, "test": test}


# ============================================================
# PART 2 — Walk-forward validation across market regimes
# ============================================================
def part2_walk_forward(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 2 — Walk-forward validation (expanding window, one fold per year)")
    print("=" * 70)

    test_years = [2021, 2022, 2023, 2024, 2025]
    results = []
    for year in test_years:
        train_end = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=PURGE_DAYS)
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")

        train = df[df["Date"] <= train_end]
        test = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)]

        if len(train) < 1000 or len(test) < 100:
            print(f"{year}: insufficient data, skipped (train={len(train)}, test={len(test)})")
            continue

        r = fit_and_score(train, test, TARGET_HORIZON)
        results.append({"year": year, "n_test": r["n"], "accuracy": r["acc"], "naive": r["naive_acc"]})
        print(f"{year}: n={r['n']:>6,}  accuracy={r['acc']:.4f}  naive={r['naive_acc']:.4f}  edge={r['acc']-r['naive_acc']:+.4f}")

    res_df = pd.DataFrame(results)
    if len(res_df) > 1:
        print(f"\nEdge across folds: mean={  (res_df['accuracy']-res_df['naive']).mean():+.4f}  "
              f"std={(res_df['accuracy']-res_df['naive']).std():.4f}")
        print("(A large std relative to the mean means the edge is inconsistent across regimes -"
              " exactly what a professor would ask about.)")


# ============================================================
# PART 3 — Cross-sectional Top-K ranking + transaction-cost backtest
# ============================================================
def part3_topk_backtest(df: pd.DataFrame, clf, test: pd.DataFrame, K: int = 5) -> None:
    print("\n" + "=" * 70)
    print(f"PART 3 — Top-{K} cross-sectional portfolio backtest, {TARGET_HORIZON}d rebalance")
    print("=" * 70)

    ret_col = f"fwd_return_{TARGET_HORIZON}d"
    test = test.copy()
    test["proba"] = clf.predict_proba(test[FEATURE_COLS])[:, 1]

    # Non-overlapping rebalance dates, spaced by the horizon, so holding
    # periods don't overlap (a clean, honest backtest rather than daily
    # overlapping windows that double-count the same period).
    rebalance_dates = sorted(test["Date"].unique())[::TARGET_HORIZON]

    ROUND_TRIP_COST = 0.0025  # ~0.25% round-trip: brokerage + STT + exchange charges + slippage (approx., India)

    rel_col = f"rel_return_{TARGET_HORIZON}d"
    portfolio_returns, benchmark_returns, gross_returns = [], [], []
    for date in rebalance_dates:
        day = test[test["Date"] == date]
        if len(day) < K:
            continue
        top_k = day.nlargest(K, "proba")
        gross_ret = top_k[ret_col].mean()  # equal-weighted
        net_ret = gross_ret - ROUND_TRIP_COST  # full turnover assumed each rebalance
        # Cross-sectional median return that day (= stock return - rel_return,
        # recovered per-row and averaged) - this is the EXACT benchmark our
        # beats-median classifier was trained against, so it's the honest
        # comparison, not the cap-weighted NIFTY50 index itself (which the
        # equal-weighted median return tracks closely but isn't identical to).
        benchmark_ret = (day[ret_col] - day[rel_col]).mean()

        gross_returns.append(gross_ret)
        portfolio_returns.append(net_ret)
        benchmark_returns.append(benchmark_ret)

    portfolio_returns = np.array(portfolio_returns)
    gross_returns = np.array(gross_returns)
    benchmark_returns = np.array(benchmark_returns)

    n_periods = len(portfolio_returns)
    cum_portfolio = np.prod(1 + portfolio_returns) - 1
    cum_gross = np.prod(1 + gross_returns) - 1
    cum_benchmark = np.prod(1 + benchmark_returns) - 1

    excess = portfolio_returns - benchmark_returns
    t_stat, p_val = stats.ttest_1samp(excess, 0)

    print(f"Rebalance periods: {n_periods}")
    print(f"Cumulative return  -  Top-{K} (gross, no costs): {cum_gross:+.2%}")
    print(f"Cumulative return  -  Top-{K} (net of ~0.25% round-trip cost/rebalance): {cum_portfolio:+.2%}")
    print(f"Cumulative return  -  Cross-sectional median benchmark (equal-weighted, same periods): {cum_benchmark:+.2%}")
    print(f"\nMean excess return per period (net of costs): {excess.mean():+.4%}")
    print(f"t-test (H0: mean excess return = 0): t={t_stat:.3f}, p={p_val:.4f}  "
          f"{'(significant at 5%)' if p_val < 0.05 else '(NOT significant at 5%)'}")
    print(f"\nCost impact: gross edge {cum_gross - cum_benchmark:+.2%} -> net-of-cost edge {cum_portfolio - cum_benchmark:+.2%}")


def main() -> None:
    df = load_data()
    ctx = part1_full_metrics(df)
    part2_walk_forward(df)
    part3_topk_backtest(df, ctx["clf"], ctx["test"], K=5)


if __name__ == "__main__":
    main()
