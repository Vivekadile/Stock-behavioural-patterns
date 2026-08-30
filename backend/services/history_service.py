"""Historical price + indicator series for the company detail chart.
Real stored data only - no live feed."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "model_features.csv"


@lru_cache(maxsize=1)
def _load_data() -> pd.DataFrame:
    return pd.read_csv(FEATURES_PATH, parse_dates=["Date"])


def get_history(symbol: str, days: int = 180) -> list[dict] | None:
    df = _load_data()
    rows = df[df["Symbol"] == symbol.upper()].sort_values("Date").tail(days)
    if rows.empty:
        return None
    return [
        {
            "date": str(row["Date"].date()),
            "close": round(float(row["Close"]), 2),
            "sma20": round(float(row["Close"] / (1 + row["price_to_sma20"])), 2),
            "sma50": round(float(row["Close"] / (1 + row["price_to_sma50"])), 2),
            "rsi_14": round(float(row["rsi_14"]), 1),
        }
        for _, row in rows.iterrows()
    ]
