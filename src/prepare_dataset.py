"""Time-based train/val/test split + feature scaling for model_features.csv.

Split is by DATE (not per-stock/random) so no future information ever
leaks into training, and every stock is represented consistently in each
split. Scaler is fit on the train split only, then applied to val/test.

Outputs:
  data/processed/splits/train.csv
  data/processed/splits/val.csv
  data/processed/splits/test.csv
  data/processed/splits/scaler.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import FEATURE_COLS, LABEL_COLS

# Scaling strategy (Phase 6.5 rework)
# ----------------------------------
# The previous StandardScaler(mean/std) left pathological inputs in the
# TRAIN set after scaling: fund_pe_ratio +13 sigma (raw P/E up to 1019),
# volume_ratio +22 sigma, bb_width +19 sigma, single-day log_return spikes
# +/-16 sigma (COVID). A single +22 sigma row dominates a linear model's
# coefficient fit and saturates the LSTM. Fix, in order:
#   1. Winsorise every feature to a TRAIN-derived [1%, 99%] quantile band.
#   2. log1p the strictly-positive, right-skewed ratio columns.
#   3. RobustScaler (median / IQR) instead of mean / std.
WINSOR_Q = (0.01, 0.99)
LOG1P_COLS = ["atr_pct", "volume_ratio", "bb_width", "fund_pe_ratio"]
# Final hard clip in scaled (IQR) space. The 1%/99% winsor is not enough for
# the annual-fundamental growth columns: they are piecewise-constant with
# rare catastrophic years (pct_change across a near-zero base year gives
# e.g. -1260%), so the 99th-percentile row itself still sits ~20 IQRs out.
# Clip every scaled feature to +/- this many IQRs so no single row can
# dominate a linear fit or saturate the LSTM.
CLIP_SIGMA = 5.0

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
IN_PATH = PROCESSED_DIR / "model_features.csv"
SPLIT_DIR = PROCESSED_DIR / "splits"
SCALER_PATH = SPLIT_DIR / "scaler.json"

TRAIN_END = "2022-12-31"
VAL_END = "2024-06-30"
# everything after VAL_END is test

# A row's label looks up to 90 trading days into the future, so a row dated
# just before a split boundary has a label computed from prices that fall
# on the OTHER side of that boundary - i.e. it leaks. Purge a gap wide
# enough to cover the longest horizon (90 trading days) from the start of
# the next split so no label ever reaches across.
#
# 130 calendar days was the original estimate but was verified EMPIRICALLY
# (Phase 5 audit) to contain only 84-92 real NSE trading days in this
# dataset (holidays reduce the trading-day density below a simple
# weekends-only assumption) - i.e. it did NOT reliably cover 90 trading
# days and had a real leakage risk. 150 calendar days was verified to
# contain a minimum of 97 trading days across the dataset (see
# src/phase1_audit.py-style verification in conversation), a safe margin.
PURGE_DAYS = 150


def _winsor_log1p(feat: pd.DataFrame, lo, hi) -> pd.DataFrame:
    """Shared step 1+2: clip each column to [lo, hi] (a Series indexed by
    FEATURE_COLS) then log1p the LOG1P_COLS. `feat` is FEATURE_COLS only."""
    f = feat.astype(float).clip(lower=lo, upper=hi, axis=1)
    for c in LOG1P_COLS:
        f[c] = np.log1p(f[c].clip(lower=0))
    return f


def apply_saved_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the EXACT training-time feature transform (winsor -> log1p ->
    RobustScaler -> +/-clip) recorded in splits/scaler.json to an arbitrary
    frame that has the raw FEATURE_COLS (e.g. model_features.csv rows used
    by predict.py / rigorous_evaluation.py / the backend). Returns a copy
    with FEATURE_COLS replaced by their scaled values; all other columns
    untouched. Run prepare_dataset.py first to (re)generate scaler.json."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"{SCALER_PATH} not found - run `python src/prepare_dataset.py` first."
        )
    with open(SCALER_PATH) as f:
        sc = json.load(f)
    cols = sc["feature_cols"]
    lo = pd.Series(sc["winsor_lower"], index=cols)
    hi = pd.Series(sc["winsor_upper"], index=cols)
    center = np.asarray(sc["center"], dtype=float)
    iqr = np.asarray(sc["iqr"], dtype=float)
    clip_sigma = sc.get("clip_sigma", CLIP_SIGMA)

    out = df.copy()
    feat = _winsor_log1p(out[cols], lo, hi)
    scaled = (feat.to_numpy() - center) / iqr
    out[cols] = np.clip(scaled, -clip_sigma, clip_sigma)
    return out


def main() -> None:
    df = pd.read_csv(IN_PATH, parse_dates=["Date"])
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    train_end = pd.Timestamp(TRAIN_END)
    val_end = pd.Timestamp(VAL_END)
    purge = pd.Timedelta(days=PURGE_DAYS)

    # Trim the tail of the outgoing split (its labels would reach into the
    # next split) AND delay the start of the incoming split (so its first
    # windows don't near-duplicate the outgoing split's last windows).
    train = df[df["Date"] <= train_end - purge].copy()
    val = df[(df["Date"] > train_end + purge) & (df["Date"] <= val_end - purge)].copy()
    test = df[df["Date"] > val_end + purge].copy()

    # 1. Winsor bounds from TRAIN ONLY (per feature, in original units).
    lo = train[FEATURE_COLS].astype(float).quantile(WINSOR_Q[0])
    hi = train[FEATURE_COLS].astype(float).quantile(WINSOR_Q[1])

    scaler = RobustScaler()
    scaler.fit(_winsor_log1p(train[FEATURE_COLS], lo, hi))

    for split in (train, val, test):
        scaled = scaler.transform(_winsor_log1p(split[FEATURE_COLS], lo, hi))
        split[FEATURE_COLS] = np.clip(scaled, -CLIP_SIGMA, CLIP_SIGMA)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(SPLIT_DIR / "train.csv", index=False)
    val.to_csv(SPLIT_DIR / "val.csv", index=False)
    test.to_csv(SPLIT_DIR / "test.csv", index=False)

    scaler_params = {
        "feature_cols": FEATURE_COLS,
        "method": "winsor(0.01,0.99)+log1p+robust(median,iqr)",
        "winsor_lower": lo.tolist(),
        "winsor_upper": hi.tolist(),
        "log1p_cols": LOG1P_COLS,
        "clip_sigma": CLIP_SIGMA,
        "center": scaler.center_.tolist(),
        "iqr": scaler.scale_.tolist(),
        # Back-compat aliases: phase6_baselines.unscale_feature() does
        # x*scale + mean. For RobustScaler that recovers the winsorised
        # (log-space, for LOG1P_COLS) value - fine for the momentum
        # baseline, which only unscales non-log1p roc_* columns.
        "mean": scaler.center_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(SPLIT_DIR / "scaler.json", "w") as f:
        json.dump(scaler_params, f, indent=2)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(
            f"{name:5s}: {len(split):>7,} rows | "
            f"{split['Date'].min().date()} -> {split['Date'].max().date()} | "
            f"{split['Symbol'].nunique()} symbols"
        )
    print(f"\nSaved splits + scaler to {SPLIT_DIR}")


if __name__ == "__main__":
    main()
