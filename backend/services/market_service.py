"""Market-wide overview. Prices here come from the project's own stored
data (last available close), NOT a live feed - a live-price integration
(e.g. FYERS) would replace `current_price`/`last_close` fields in this
service and prediction_service.py without changing the API contract, but
no broker credentials are wired up in this project."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from services import fyers_service

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "model_features.csv"
NIFTY_INDEX_PATH = PROJECT_ROOT / "data" / "raw" / "NIFTY50_INDEX.csv"


@lru_cache(maxsize=1)
def _load_data() -> pd.DataFrame:
    return pd.read_csv(FEATURES_PATH, parse_dates=["Date"])


@lru_cache(maxsize=1)
def _load_index() -> pd.DataFrame:
    # model_features.csv only carries derived index features (return/vol/trend),
    # not the raw index level - read that separately from the raw source.
    return pd.read_csv(NIFTY_INDEX_PATH, parse_dates=["Date"])


def get_market_overview() -> dict:
    df = _load_data()
    total_universe = int(df["Symbol"].nunique())

    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date]
    # Not every stock shares this exact latest date (corporate-action masking
    # trims some stocks' tails earlier than others, e.g. VEDL) - report that
    # honestly rather than implying the smaller same-day count is the whole universe.
    n_with_data_today = int(latest["Symbol"].nunique())

    index_df = _load_index()
    stored_nifty_close = float(index_df.loc[index_df["Date"] == latest_date, "Close"].iloc[0])
    nifty_return = float(latest["nifty_return"].iloc[0])
    positive_count = int((latest["log_return"] > 0).sum())
    negative_count = int((latest["log_return"] <= 0).sum())

    live_nifty_close = fyers_service.get_nifty_ltp()
    is_live = live_nifty_close is not None
    nifty_close = live_nifty_close if is_live else stored_nifty_close

    return {
        "as_of_date": str(latest_date.date()),
        "nifty50_close": round(nifty_close, 2),
        "nifty50_daily_return_pct": round(nifty_return * 100, 2),
        "market_trend": "Positive" if nifty_return > 0 else "Negative",
        "stocks_in_universe": total_universe,
        "stocks_with_data_as_of_date": n_with_data_today,
        "stocks_positive_today": positive_count,
        "stocks_negative_today": negative_count,
        "is_live": is_live,
        "note": (
            "nifty50_close is a live FYERS quote; all other figures (returns, advance/decline "
            "counts) are computed from the stored dataset."
            if is_live else
            "Prices reflect the project's stored dataset (updated via src/fetch_data.py), not a live "
            "market feed. Not all stocks share the same latest date due to per-stock corporate-action "
            "data trimming. Run backend/fyers_login.py to enable live prices."
        ),
    }
