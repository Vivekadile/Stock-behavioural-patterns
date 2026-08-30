"""IC Decay Analysis — per-feature Spearman Rank IC (correlation with
forward return) computed independently for each year, to find features
whose relationship with returns is stable vs. regime-dependent (flips
sign or decays across years). Static feature sets that ignore this are a
direct, checkable cause of walk-forward instability.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import FEATURE_COLS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "model_features.csv"
TARGET_HORIZON = 30
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
# Spearman rank IC is invariant to any monotonic feature transform, so this
# reads model_features.csv in raw units - the Phase 6.5 winsor/log1p/robust
# scaling would not change a single number here.


def main() -> None:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    ret_col = f"fwd_return_{TARGET_HORIZON}d"

    ic_by_year = {}
    for year in YEARS:
        year_df = df[df["Date"].dt.year == year]
        if len(year_df) < 200:
            continue
        ics = {}
        for col in FEATURE_COLS:
            ic, _ = spearmanr(year_df[col], year_df[ret_col])
            ics[col] = ic
        ic_by_year[year] = ics

    ic_df = pd.DataFrame(ic_by_year).T  # rows=year, cols=feature

    print("=" * 100)
    print(f"Per-year Spearman Rank IC vs fwd_return_{TARGET_HORIZON}d")
    print("=" * 100)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print(ic_df.round(3).to_string())

    print("\n" + "=" * 100)
    print("STABILITY SUMMARY (sorted by |mean IC| / std IC - higher = more reliably useful)")
    print("=" * 100)
    summary = pd.DataFrame({
        "mean_ic": ic_df.mean(),
        "std_ic": ic_df.std(),
        "min_ic": ic_df.min(),
        "max_ic": ic_df.max(),
        "sign_flips": ic_df.apply(lambda col: (np.sign(col.dropna()).diff() != 0).sum()),
    })
    summary["stability_score"] = summary["mean_ic"].abs() / summary["std_ic"]
    summary = summary.sort_values("stability_score", ascending=False)
    print(summary.round(3).to_string())

    unstable = summary[summary["sign_flips"] >= 5]
    print(f"\n{len(unstable)} features flip sign in 5+ of {len(ic_by_year)} years (candidates to drop or treat as regime-dependent):")
    print(list(unstable.index))

    summary.to_csv(PROCESSED_DIR.parent / "phase8_ic_decay_summary.csv")
    print(f"\nSaved to data/phase8_ic_decay_summary.csv")


if __name__ == "__main__":
    main()
