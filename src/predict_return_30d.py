"""Inference for the 30-trading-day return regression model.

Loads the saved artifact (models/regression_30d/pipeline.joblib) - which
already contains the fitted preprocessing - and answers, for a (symbol,
date) pair, the expected forward 30-trading-day return and the implied
expected price. The backend imports predict_return() directly; it never
re-implements preprocessing.

    >>> predict_return("RELIANCE", "2024-06-14")
    {'symbol': 'RELIANCE', 'date': '2024-06-14', 'current_price': 1234.5,
     'expected_return_30d': 0.031, 'expected_return_percent': 3.1,
     'expected_price': 1272.8, 'as_of_date': '2024-06-14',
     'model_version': '30d-reg-v1',
     'disclaimer': 'Model estimate, not a guaranteed future price.'}
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import FEATURE_COLS  # noqa: E402
import regression_preprocess  # noqa: E402,F401  (registers the class for joblib unpickling)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "models" / "regression_30d"
FEATURES_CSV = ROOT / "data" / "processed" / "model_features.csv"

DISCLAIMER = "Model estimate, not a guaranteed future price."


@lru_cache(maxsize=1)
def load_artifact():
    import joblib
    pipe = joblib.load(ARTIFACT_DIR / "pipeline.joblib")
    with open(ARTIFACT_DIR / "metadata.json") as f:
        meta = json.load(f)
    return pipe, meta


@lru_cache(maxsize=1)
def _features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"],
                     usecols=["Date", "Symbol", "Close"] + FEATURE_COLS)
    return df.sort_values(["Symbol", "Date"]).reset_index(drop=True)


def _row_for(symbol: str, date) -> pd.Series | None:
    """Most recent feature row for `symbol` on or before `date`."""
    df = _features()
    d = pd.Timestamp(date)
    rows = df[(df["Symbol"] == symbol.upper()) & (df["Date"] <= d)]
    return None if rows.empty else rows.iloc[-1]


def predict_return(symbol: str, date: str | pd.Timestamp) -> dict | None:
    pipe, meta = load_artifact()
    row = _row_for(symbol, date)
    if row is None:
        return None

    X = row[FEATURE_COLS].to_frame().T
    exp_ret = float(pipe.predict(X)[0])
    current_price = float(row["Close"])

    return {
        "symbol": symbol.upper(),
        "date": str(pd.Timestamp(date).date()),
        "as_of_date": str(row["Date"].date()),          # the feature row actually used
        "current_price": round(current_price, 2),
        "expected_return_30d": round(exp_ret, 4),
        "expected_return_percent": round(exp_ret * 100, 2),
        "expected_price": round(current_price * (1 + exp_ret), 2),
        "horizon_trading_days": meta["target"]["horizon_trading_days"],
        "model_version": meta["version"],
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    pairs = [(args[0], args[1])] if len(args) == 2 else [
        ("RELIANCE", "2024-06-14"), ("TCS", "2024-06-14"), ("HDFCBANK", "2025-03-03")]
    for sym, dt in pairs:
        print(json.dumps(predict_return(sym, dt), indent=2))
