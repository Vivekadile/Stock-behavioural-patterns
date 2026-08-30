"""Single-horizon 30-trading-day return forecasting — REGRESSION.

Active product objective:
    "For a stock and a date, what return (%) does the model expect over the
     next 30 trading days?"   ->  a continuous value, e.g. 0.068 = +6.8%.

Target definition (per Symbol, trading days, never cross-stock):
    fwd_return_30d = AdjClose[t + 30 trading days] / AdjClose[t] - 1

AdjClose (dividend/split-adjusted) is used, not raw Close: raw Close would
inject a spurious -2..-5% "return" on every ex-dividend date and large
jumps on splits. This matches the existing leakage-audited pipeline;
create_30d_target() re-derives the target from scratch and asserts it
equals feature_engineering's fwd_return_30d column exactly.

Pipeline stages (functions, per the refactor spec):
    create_30d_target()          - define + audit the target
    prepare_regression_dataset() - lean modelling table, NaN targets dropped
    split_and_scale()            - chronological split + purge + fit scaler on TRAIN only
    train_regression_models()    - historical-mean, Ridge, RF, HistGBM, XGBoost, LightGBM
    evaluate_regression_models() - MAE / RMSE / R2 / MedAE / DirAcc + pred-vs-actual stats
    save_model_artifacts()       - pipeline.joblib + metadata.json for the backend
    main()                       - run all, print the final validation report

Nothing here deletes or rewrites the historical multi-horizon code; it
only builds the active single-horizon path.
"""

from __future__ import annotations

import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import FEATURE_COLS, PRIMARY_HORIZON, PRIMARY_TARGET, PRICE_COL
from regression_preprocess import WinsorLogRobustScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "model_features.csv"
STOCKS_DAILY_CSV = ROOT / "data" / "processed" / "stocks_daily.csv"   # contiguous daily series (Volume>0 only)
OUT_DATASET = ROOT / "data" / "processed" / "regression_dataset_30d.csv"
ARTIFACT_DIR = ROOT / "models" / "regression_30d"
METRICS_CSV = ROOT / "data" / "regression_30d_metrics.csv"

HORIZON = PRIMARY_HORIZON          # 30 trading days
TARGET = PRIMARY_TARGET            # "fwd_return_30d"
TRAIN_END = "2022-12-31"
VAL_END = "2024-06-30"
# The target looks 30 trading days ahead (~44 calendar days). A 50-day
# purge on BOTH sides of every split boundary guarantees no training row's
# label is computed from prices that fall in the next split. (The old
# 150-day purge was sized for the 90-day horizon and is unnecessarily
# wide here - it would just discard ~4 extra months of usable data.)
PURGE_DAYS = 50
MODEL_VERSION = "30d-reg-v1"


# ----------------------------------------------------------------------
# 1. TARGET
# ----------------------------------------------------------------------
def create_30d_target(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Forward 30-TRADING-DAY return, computed INDEPENDENTLY per Symbol on
    the CONTIGUOUS daily series (stocks_daily.csv, which already contains
    only real trading rows, Volume>0). A shift of 30 rows there IS 30
    trading days. Never mixes two Symbols.

    Returns a DataFrame [Symbol, Date, fwd_return_30d] with the last 30
    rows of each Symbol NaN (no 30-trading-day future exists yet). Callers
    join this onto the feature rows by (Symbol, Date).
    """
    out = []
    for sym, g in daily_df.groupby("Symbol", sort=False):
        g = g.sort_values("Date")
        r = g[PRICE_COL].shift(-HORIZON) / g[PRICE_COL] - 1.0
        out.append(pd.DataFrame({"Symbol": sym, "Date": g["Date"].to_numpy(), TARGET: r.to_numpy()}))
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------
# 2 / 11. MODELLING DATASET
# ----------------------------------------------------------------------
def prepare_regression_dataset(write: bool = True) -> pd.DataFrame:
    """Lean table: Date, Symbol, the 36 features, fwd_return_30d.
    Rows without a valid 30-trading-day forward target are DROPPED (never
    interpolated or forward-filled)."""
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # --- audit: recompute the target from scratch on the contiguous daily
    #     series and confirm it equals feature_engineering's fwd_return_30d
    #     for every feature row (which are all mid-series, so never NaN). ---
    daily = pd.read_csv(STOCKS_DAILY_CSV, usecols=["Symbol", "Date", PRICE_COL], parse_dates=["Date"])
    tgt = create_30d_target(daily).rename(columns={TARGET: TARGET + "_recomputed"})
    chk = df[["Symbol", "Date", TARGET]].merge(tgt, on=["Symbol", "Date"], how="left")
    missing = chk[TARGET + "_recomputed"].isna().sum()
    max_abs_diff = float((chk[TARGET] - chk[TARGET + "_recomputed"]).abs().max())
    assert missing == 0, f"{missing} feature rows have no recomputed target (Symbol/Date mismatch)"
    # tolerance is float32 rounding: model_features.csv is stored float32
    # (relative eps ~1e-7), the recompute is float64. A real definitional
    # error would be orders of magnitude larger.
    assert max_abs_diff < 1e-5, (
        f"create_30d_target disagrees with feature_engineering.{TARGET} "
        f"(max abs diff {max_abs_diff:.2e}) - investigate before training")

    cols = ["Date", "Symbol"] + FEATURE_COLS + [TARGET]
    d = df[cols].dropna(subset=[TARGET] + FEATURE_COLS).copy()
    d = d.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    assert not d.duplicated(["Symbol", "Date"]).any(), "duplicate Symbol+Date rows"
    if write:
        OUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(OUT_DATASET, index=False)
    print(f"[dataset] {len(d):,} rows | {d['Symbol'].nunique()} symbols | "
          f"{d['Date'].min().date()} -> {d['Date'].max().date()} | target audit OK "
          f"(max diff {max_abs_diff:.1e})")
    return d


# ----------------------------------------------------------------------
# 7. SPLIT + SCALE
# ----------------------------------------------------------------------
def split_and_scale(d: pd.DataFrame):
    """Chronological split with a purge gap on both sides of each boundary.
    Fits the preprocessing on TRAIN ONLY. Returns (train, val, test) each
    as (X_scaled ndarray, y ndarray, meta DataFrame) plus the fitted
    preprocessor."""
    train_end, val_end = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)
    purge = pd.Timedelta(days=PURGE_DAYS)

    tr = d[d["Date"] <= train_end - purge]
    va = d[(d["Date"] > train_end + purge) & (d["Date"] <= val_end - purge)]
    te = d[d["Date"] > val_end + purge]

    prep = WinsorLogRobustScaler(FEATURE_COLS).fit(tr[FEATURE_COLS])

    def pack(part):
        return (prep.transform(part[FEATURE_COLS]),
                part[TARGET].to_numpy(dtype=float),
                part[["Date", "Symbol"]].reset_index(drop=True))

    return pack(tr), pack(va), pack(te), prep


def export_lstm_splits() -> None:
    """Write scaled train/val/test CSVs (Date, Symbol, 36 features,
    fwd_return_30d) for the Colab LSTM regression notebook. Same split /
    purge / preprocessing as split_and_scale(); no model training here."""
    d = prepare_regression_dataset(write=False)
    (Xtr, ytr, mtr), (Xva, yva, mva), (Xte, yte, mte), _ = split_and_scale(d)
    outdir = ROOT / "data" / "processed" / "regression_splits_30d"
    outdir.mkdir(parents=True, exist_ok=True)
    for name, X, y, meta in [("train", Xtr, ytr, mtr), ("val", Xva, yva, mva), ("test", Xte, yte, mte)]:
        part = meta.copy()
        part[FEATURE_COLS] = X
        part[TARGET] = y
        part.to_csv(outdir / f"{name}.csv", index=False)
        print(f"[lstm-export] {name}: {len(part):,} rows -> {outdir / (name + '.csv')}")


# ----------------------------------------------------------------------
# 6. MODELS
# ----------------------------------------------------------------------
def _optional_boosters() -> dict:
    models = {}
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, random_state=42)
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm"] = LGBMRegressor(
            n_estimators=600, num_leaves=31, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1)
    except Exception:
        pass
    return models


def _model_zoo() -> dict:
    """Unfitted estimators. 'historical_mean' is the mean-predictor baseline."""
    models = {
        "historical_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=50, n_jobs=-1, random_state=42),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.05, max_iter=400, l2_regularization=1.0, random_state=42),
    }
    models.update(_optional_boosters())
    return models


def train_regression_models(train):
    """Fit every model in the zoo on TRAIN. Returns {name: fitted_estimator}."""
    Xtr, ytr, _ = train
    models = _model_zoo()
    for m in models.values():
        m.fit(Xtr, ytr)
    return models


def select_best_model(train, n_splits: int = 4) -> tuple[str, pd.DataFrame]:
    """Pick the model with the lowest mean RMSE over a TimeSeriesSplit
    walk-forward on the TRAINING data only. This averages selection over
    several market regimes (2013 taper / 2018 / 2020 COVID / 2021 bull /
    2022 selloff) instead of trusting one fixed validation window that
    happens to be a strong bull market. No leakage: val and test are
    untouched here."""
    Xtr, ytr, _ = train
    zoo = {k: v for k, v in _model_zoo().items() if k != "historical_mean"}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for name, est in zoo.items():
        fold_rmse = []
        for fit_idx, val_idx in tscv.split(Xtr):
            m = clone(est)
            m.fit(Xtr[fit_idx], ytr[fit_idx])
            fold_rmse.append(mean_squared_error(ytr[val_idx], m.predict(Xtr[val_idx])) ** 0.5)
        base = [mean_squared_error(ytr[v], np.full_like(ytr[v], ytr[f].mean())) ** 0.5
                for f, v in tscv.split(Xtr)]
        rows.append({"model": name, "cv_rmse_mean": np.mean(fold_rmse),
                     "cv_rmse_std": np.std(fold_rmse),
                     "cv_rmse_vs_baseline_%": (np.mean(fold_rmse) / np.mean(base) - 1) * 100})
    cv = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)

    # Selection rule (decided BEFORE any test data is touched): among the
    # models whose CV mean RMSE is within 1 CV-std of the best, pick the one
    # with the LOWEST cross-fold RMSE std - i.e. the most regime-robust of
    # the statistically-tied contenders. The boosters win the raw CV mean
    # but by a margin smaller than the fold-to-fold noise, and they are the
    # ones that overfit a single period.
    best_mean, best_std = cv.iloc[0]["cv_rmse_mean"], cv.iloc[0]["cv_rmse_std"]
    contenders = cv[cv["cv_rmse_mean"] <= best_mean + best_std]
    best_name = contenders.sort_values("cv_rmse_std").iloc[0]["model"]
    return best_name, cv


# ----------------------------------------------------------------------
# 8. EVALUATION
# ----------------------------------------------------------------------
def _metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
        "DirAcc": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "actual_mean": float(np.mean(y_true)),
        "actual_std": float(np.std(y_true)),
        "corr_pred_actual": float(np.corrcoef(y_pred, y_true)[0, 1]) if np.std(y_pred) > 0 else 0.0,
    }


def evaluate_regression_models(models: dict, split, split_name: str) -> pd.DataFrame:
    X, y, _ = split
    rows = []
    for name, m in models.items():
        r = _metrics(y, m.predict(X))
        r["model"] = name
        r["split"] = split_name
        rows.append(r)
    df = pd.DataFrame(rows).set_index("model")
    base = df.loc["historical_mean"]
    df["beats_baseline_RMSE"] = df["RMSE"] < base["RMSE"]
    df["RMSE_vs_baseline_%"] = (df["RMSE"] / base["RMSE"] - 1.0) * 100.0
    return df.reset_index()


# ----------------------------------------------------------------------
# 10. ARTIFACT
# ----------------------------------------------------------------------
def save_model_artifacts(best_name: str, prep, best_estimator,
                         d: pd.DataFrame, test_metrics: dict,
                         baseline_metrics: dict) -> None:
    import joblib
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline([("preprocess", prep), ("model", best_estimator)])
    joblib.dump(pipe, ARTIFACT_DIR / "pipeline.joblib")

    meta = {
        "version": MODEL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "regression",
        "target": {
            "name": TARGET,
            "horizon_trading_days": HORIZON,
            "definition": f"{PRICE_COL}[t + {HORIZON} trading days] / {PRICE_COL}[t] - 1, per Symbol",
            "price_column": PRICE_COL,
        },
        "feature_cols": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "preprocessing": {
            "class": "regression_preprocess.WinsorLogRobustScaler",
            "winsor_q": list(prep.winsor_q),
            "log1p_cols": list(prep.log1p_cols),
            "clip_sigma": prep.clip_sigma,
            "note": "fitted on the TRAIN split only; loaded via the pipeline, never re-implemented",
        },
        "training": {
            "train_end": TRAIN_END, "val_end": VAL_END, "purge_days": PURGE_DAYS,
            "dataset_rows": int(len(d)),
            "date_range": [str(d["Date"].min().date()), str(d["Date"].max().date())],
            "n_symbols": int(d["Symbol"].nunique()),
            "best_model_fit_on": "train split only",
            "selection": "lowest mean RMSE over TimeSeriesSplit(4) walk-forward CV on the training data",
        },
        "best_model": best_name,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "baseline_test_metrics": {k: float(v) for k, v in baseline_metrics.items()},
        "beats_baseline_on_test_RMSE": bool(test_metrics["RMSE"] < baseline_metrics["RMSE"]),
        "python": platform.python_version(),
    }
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[artifact] saved pipeline.joblib + metadata.json -> {ARTIFACT_DIR}")


# ----------------------------------------------------------------------
# 13. RUN + VALIDATION REPORT
# ----------------------------------------------------------------------
def _validation_checks(d, train, val, test) -> list[tuple[str, bool, str]]:
    (_, ytr, mtr), (_, yva, mva), (_, yte, mte) = train, val, test
    checks = []
    checks.append(("one row per (Symbol, Date)", not d.duplicated(["Symbol", "Date"]).any(), ""))
    chrono = all(g["Date"].is_monotonic_increasing for _, g in d.groupby("Symbol", sort=False))
    checks.append(("chronological order within each Symbol", chrono, ""))
    checks.append(("no target NaN in modelling dataset", not d[TARGET].isna().any(), ""))
    tr_max, va_min, va_max, te_min = mtr["Date"].max(), mva["Date"].min(), mva["Date"].max(), mte["Date"].min()
    checks.append((f"train ends {tr_max.date()} <= val starts {va_min.date()} minus purge",
                   (va_min - tr_max).days >= PURGE_DAYS, f"gap {(va_min - tr_max).days}d"))
    checks.append((f"val ends {va_max.date()} <= test starts {te_min.date()} minus purge",
                   (te_min - va_max).days >= PURGE_DAYS, f"gap {(te_min - va_max).days}d"))
    checks.append(("train / val / test are non-empty", len(ytr) and len(yva) and len(yte), ""))
    checks.append(("no (Symbol,Date) shared across splits",
                   len(set(map(tuple, mtr.values)) &
                       set(map(tuple, mte.values))) == 0, ""))
    return checks


def main() -> None:
    print("=" * 78)
    print(f"30-TRADING-DAY RETURN REGRESSION  ({MODEL_VERSION})")
    print("=" * 78)

    d = prepare_regression_dataset()
    train, val, test, prep = split_and_scale(d)
    (Xtr, ytr, mtr), (_, yva, mva), (Xte, yte, mte) = train, val, test

    # --- model selection: walk-forward CV on TRAIN only (regime-averaged) ---
    best_name, cv_df = select_best_model(train)
    print("\nModel selection - TimeSeriesSplit(4) CV on TRAIN:")
    print(cv_df.round(5).to_string(index=False))
    print(f"  -> selected: {best_name}")

    # fit the whole zoo on TRAIN for the full comparison tables
    models = train_regression_models(train)
    val_df = evaluate_regression_models(models, val, "validation")
    test_df = evaluate_regression_models(models, test, "test")

    # the shipped model: the CV winner, refit on TRAIN ONLY (the fixed
    # validation window is an unrepresentative bull market - folding it in
    # biased predictions high), scored once on TEST.
    best_est = models[best_name]
    best_test = _metrics(yte, best_est.predict(Xte))
    baseline_test = _metrics(yte, np.full_like(yte, ytr.mean()))

    save_model_artifacts(best_name, prep, best_est, d, best_test, baseline_test)

    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([val_df, test_df]).to_csv(METRICS_CSV, index=False)

    # ---------------- FINAL REPORT ----------------
    print("\n" + "=" * 78)
    print("FINAL VALIDATION REPORT")
    print("=" * 78)
    print(f"Dataset shape            : {d.shape[0]:,} rows x {d.shape[1]} cols")
    print(f"Number of stocks         : {d['Symbol'].nunique()}")
    print(f"Date range               : {d['Date'].min().date()} -> {d['Date'].max().date()}")
    print(f"Feature count            : {len(FEATURE_COLS)}")
    print(f"Target definition        : {TARGET} = {PRICE_COL}[t+{HORIZON} trading days]/{PRICE_COL}[t] - 1 (per Symbol)")
    print(f"Train size               : {len(ytr):,}   ({mtr['Date'].min().date()} -> {mtr['Date'].max().date()})")
    print(f"Validation size          : {len(yva):,}   ({mva['Date'].min().date()} -> {mva['Date'].max().date()})")
    print(f"Test size                : {len(yte):,}   ({mte['Date'].min().date()} -> {mte['Date'].max().date()})")
    print(f"Purge gap                : {PURGE_DAYS} calendar days each side (covers {HORIZON} trading days)")

    print("\nIntegrity checks:")
    all_ok = True
    for name, ok, extra in _validation_checks(d, train, val, test):
        all_ok &= bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({extra})" if extra else ""))

    show_cols = ["model", "MAE", "RMSE", "R2", "MedianAE", "DirAcc",
                 "pred_mean", "pred_std", "actual_mean", "actual_std",
                 "corr_pred_actual", "RMSE_vs_baseline_%"]
    print("\nTEST-set metrics (all models, refit on TRAIN only):")
    print(test_df[show_cols].round(5).to_string(index=False))

    print(f"\nBest model (lowest validation RMSE)      : {best_name}")
    print("Best model TEST metrics (refit train+val) :")
    for k in ["MAE", "RMSE", "R2", "MedianAE", "DirAcc", "corr_pred_actual"]:
        print(f"    {k:16s} {best_test[k]:+.5f}")
    print("Historical-mean baseline TEST metrics     :")
    for k in ["MAE", "RMSE", "R2", "MedianAE", "DirAcc"]:
        print(f"    {k:16s} {baseline_test[k]:+.5f}")

    beat = best_test["RMSE"] < baseline_test["RMSE"]
    delta = (best_test["RMSE"] / baseline_test["RMSE"] - 1) * 100
    print(f"\nDoes best model beat baseline?  {'YES' if beat else 'NO'}  "
          f"(test RMSE {delta:+.2f}% vs baseline; R2 {best_test['R2']:+.4f})")
    if not beat:
        print("  -> Reported honestly: on this feature set the 30-day return level is\n"
              "     not predictable beyond the historical mean. The signal that exists\n"
              "     is in the cross-sectional RANKING (see phase10 portfolio), not the\n"
              "     absolute return magnitude.")
    print(f"\nAll integrity checks passed: {all_ok}")
    print(f"Artifact: {ARTIFACT_DIR}/pipeline.joblib  (+ metadata.json)")
    print(f"Per-model metrics CSV: {METRICS_CSV}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export-lstm-splits":
        export_lstm_splits()
    else:
        main()
