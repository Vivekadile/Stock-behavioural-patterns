"""Final production inference layer.

Model choice (empirically settled, not assumed): Logistic Regression on
the beats_median_{h}d target, full 36-feature set, full expanding-history
training window. This beat every alternative tested across Phases 6-8:
Random Forest, Gradient Boosting, and a 5-model LSTM ensemble all scored
lower or less consistently; five proposed fixes for walk-forward
instability (excluding the 2020 outlier, IC-stability feature filtering,
quintile-spread re-ranking, rolling 3-year windows, volatility-scaled
targets) were tested and every one underperformed this simple baseline.

Phase 6.5: features are now passed through the SAME winsor -> log1p ->
RobustScaler -> +/-5 clip transform used for training (prepare_dataset.
apply_saved_scaling) before the logistic fit - the old path fit on raw
features that still contained +13 sigma P/E outliers. A second head on the
top_tercile_{h}d target (top vs bottom third of the cross-section, middle
dropped) is also fit and reported; on the fresh Phase-6 baselines it is
the stronger signal at 60d/90d, but beats_median stays the primary output
for continuity.

Output is a CALIBRATED PROBABILITY of beating the NIFTY50 median return
over each horizon - not a price target. This is a deliberate choice: the
project's own evidence (Phase 5-8) shows absolute return is not
predictable from this data, the edge on relative performance is real but
modest (52-55%, unstable year to year), and a portfolio-level backtest of
this edge did not survive transaction costs. Presenting a precise
"expected price" would overstate what this model can honestly support.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import FEATURE_COLS, HORIZONS
from prepare_dataset import apply_saved_scaling

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "model_features.csv"

# Test-set accuracy per horizon from the Phase 6.5 baseline rerun
# (data/phase6_baseline_results.csv, logistic regression, scaled features).
# 'median' = beats_median_{h}d target; 'tercile' = top_tercile_{h}d.
# Shown alongside every prediction so the output is never presented without
# its own honest track record attached.
HISTORICAL_ACCURACY = {
    "median":  {7: 0.512, 30: 0.523, 60: 0.527, 90: 0.485},
    "tercile": {7: 0.552, 30: 0.556, 60: 0.536, 90: 0.545},
}
CONFIDENCE_LABEL = {
    True: "Low-to-moderate (this signal has been inconsistent year to year - "
          "see Phase 8 walk-forward results; treat as a weak tilt, not a strong signal)",
}


def train_models(df: pd.DataFrame) -> dict:
    """Returns {'median': {h: clf}, 'tercile': {h: clf}}. Features are
    assumed ALREADY scaled (see load_features)."""
    models = {"median": {}, "tercile": {}}
    for h in HORIZONS:
        med = LogisticRegression(max_iter=2000, C=0.1)
        med.fit(df[FEATURE_COLS], df[f"beats_median_{h}d"])
        models["median"][h] = med

        # top_tercile_{h}d is NaN for the middle third of each day's
        # cross-section by design - drop those rows for this head only.
        ter_df = df.dropna(subset=[f"top_tercile_{h}d"])
        ter = LogisticRegression(max_iter=2000, C=0.1)
        ter.fit(ter_df[FEATURE_COLS], ter_df[f"top_tercile_{h}d"])
        models["tercile"][h] = ter
    return models


def load_features() -> pd.DataFrame:
    """model_features.csv rows with FEATURE_COLS scaled by the saved
    training transform."""
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    return apply_saved_scaling(df)


def predict_company(symbol: str, df: pd.DataFrame, models: dict) -> dict | None:
    company_rows = df[df["Symbol"] == symbol]
    if company_rows.empty:
        return None
    latest = company_rows.sort_values("Date").iloc[-1]

    result = {
        "symbol": symbol,
        "as_of_date": str(latest["Date"].date()) if hasattr(latest["Date"], "date") else str(latest["Date"]),
        "current_price": round(float(latest["Close"]), 2),
        "predictions": {},
    }
    X = latest[FEATURE_COLS].to_frame().T
    for h in HORIZONS:
        proba = models["median"][h].predict_proba(X)[0, 1]
        proba_ter = models["tercile"][h].predict_proba(X)[0, 1]
        result["predictions"][f"{h}d"] = {
            "probability_beats_nifty50_median": round(float(proba), 4),
            "probability_top_tercile": round(float(proba_ter), 4),
            "historical_accuracy_of_this_horizon": HISTORICAL_ACCURACY["median"][h],
            "historical_accuracy_top_tercile": HISTORICAL_ACCURACY["tercile"][h],
            "confidence": CONFIDENCE_LABEL[True],
        }
    return result


def main() -> None:
    df = load_features()
    print("Training final models (Logistic Regression, scaled features, full history, all 4 horizons)...")
    models = train_models(df)

    symbols = sys.argv[1:] if len(sys.argv) > 1 else sorted(df["Symbol"].unique())[:5]
    for symbol in symbols:
        result = predict_company(symbol, df, models)
        if result is None:
            print(f"\n{symbol}: not found in universe")
            continue
        print(f"\n=== {symbol} ===")
        print(f"As of: {result['as_of_date']} | Current price: Rs.{result['current_price']}")
        for h_key, pred in result["predictions"].items():
            p = pred["probability_beats_nifty50_median"]
            direction = "MORE likely than not" if p > 0.5 else "LESS likely than not"
            print(f"  {h_key}: {p:.1%} probability of beating NIFTY50 median "
                  f"({direction}) | historical accuracy at this horizon: {pred['historical_accuracy_of_this_horizon']:.1%}")


if __name__ == "__main__":
    main()
