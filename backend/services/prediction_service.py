"""Wraps the project's model (Logistic Regression on beats_median_{h}d,
settled in Phases 6-8 as the best-performing option) as a backend service.

Design principle carried over from the whole project's findings: the ONLY
number this service treats as a validated output is the probability of
beating the NIFTY50 median. It does NOT compute or expose an "expected
return %" or "expected price" - those were tested extensively (Phases 6-8)
and found to have no signal beyond the historical average. Exposing them
here would misrepresent the model, regardless of how the frontend labels
them.

Phase 6.5: model features are passed through the shared training-time
transform (prepare_dataset.apply_saved_scaling: winsor -> log1p ->
RobustScaler -> +/-5 clip) before the logistic fit / inference. Raw
model_features.csv values are still used for the human-readable technical
and fundamental panels in get_company_detail (those must stay in natural
units). A second probability, top_tercile (top vs bottom third of the
cross-section), is now returned alongside probability_beats_nifty50_median.
"""

import sys
import warnings
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from services import fyers_service

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from feature_engineering import FEATURE_COLS, HORIZONS  # noqa: E402
from prepare_dataset import apply_saved_scaling  # noqa: E402

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "model_features.csv"

# Test-set accuracy per horizon from the Phase 6/10 NIFTY 200 rerun
# (data/phase6_baseline_results.csv, logistic regression on scaled
# features, 59k-row test set) - shown alongside every prediction so the
# API never returns a number without its own honest track record.
# The directional edge is thin (~+1 point over base rate); the project's
# actual result is the PORTFOLIO information ratio of ~1.0 after costs
# (Phase 10), not the single-name hit rate.
HORIZON_STATS = {
    7:  {"historical_accuracy": 0.502, "historical_accuracy_top_tercile": 0.503, "edge_range": (-0.010, 0.021)},
    30: {"historical_accuracy": 0.511, "historical_accuracy_top_tercile": 0.508, "edge_range": (-0.005, 0.016)},
    60: {"historical_accuracy": 0.507, "historical_accuracy_top_tercile": 0.510, "edge_range": (-0.008, 0.013)},
    90: {"historical_accuracy": 0.515, "historical_accuracy_top_tercile": 0.515, "edge_range": (-0.006, 0.015)},
}

POSITIVE_THRESHOLD = 0.54
NEGATIVE_THRESHOLD = 0.46


@lru_cache(maxsize=1)
def _load_data() -> pd.DataFrame:
    """Raw model_features.csv - natural units, used for display panels."""
    return pd.read_csv(FEATURES_PATH, parse_dates=["Date"])


def _model_input(rows: pd.DataFrame) -> pd.DataFrame:
    """Raw feature rows -> the scaled FEATURE_COLS the models expect."""
    return apply_saved_scaling(rows)[FEATURE_COLS]


@lru_cache(maxsize=1)
def _train_models() -> dict:
    """Trained once per process (Logistic Regression on ~68k rows), cached
    for the lifetime of the server. Returns
    {'median': {h: clf}, 'tercile': {h: clf}}."""
    df = _load_data()
    X = _model_input(df)
    models = {"median": {}, "tercile": {}}
    for h in HORIZONS:
        med = LogisticRegression(max_iter=2000, C=0.1)
        med.fit(X, df[f"beats_median_{h}d"])
        models["median"][h] = med

        keep = df[f"top_tercile_{h}d"].notna().to_numpy()
        ter = LogisticRegression(max_iter=2000, C=0.1)
        ter.fit(X[keep], df.loc[keep, f"top_tercile_{h}d"])
        models["tercile"][h] = ter
    return models


def _signal_label(probability: float) -> str:
    if probability >= POSITIVE_THRESHOLD:
        return "Positive"
    if probability <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def _risk_level(volatility_20: float, all_volatility: pd.Series) -> str:
    """Risk tier from the stock's OWN recent realized volatility, ranked
    against the rest of the universe on the same date - a real, computed
    quantity, not a vibe."""
    pct = (all_volatility < volatility_20).mean()
    if pct >= 0.66:
        return "High"
    if pct <= 0.33:
        return "Low"
    return "Medium"


def get_company_detail(symbol: str) -> dict | None:
    df = _load_data()
    rows = df[df["Symbol"] == symbol.upper()]
    if rows.empty:
        return None
    latest = rows.sort_values("Date").iloc[-1]

    macd_signal = "Bullish" if latest["macd_hist_norm"] > 0 else "Bearish"
    volume_trend = "Above Average" if latest["volume_ratio"] > 1.1 else (
        "Below Average" if latest["volume_ratio"] < 0.9 else "Average"
    )

    live_price = fyers_service.get_ltp(symbol.upper())

    return {
        "symbol": symbol.upper(),
        "industry": latest["Industry"],
        "last_close": round(live_price if live_price is not None else float(latest["Close"]), 2),
        "is_live": live_price is not None,
        "as_of_date": str(latest["Date"].date()),
        "fundamentals": {
            "roe_pct": round(float(latest["fund_roe"]) * 100, 2),
            "pe_ratio": round(float(latest["fund_pe_ratio"]), 2),
            "net_profit_growth_pct": round(float(latest["fund_net_profit_growth"]) * 100, 2),
            "eps_growth_pct": round(float(latest["fund_eps_growth"]) * 100, 2),
            "note": "Only fields with point-in-time-verified data are shown (see Phase 2, "
                    "src/fetch_filing_dates.py). Market Cap and P/B are not available in this dataset.",
        },
        "technicals": {
            "rsi_14": round(float(latest["rsi_14"]), 1),
            "macd_signal": macd_signal,
            "price_vs_sma50_pct": round(float(latest["price_to_sma50"]) * 100, 2),
            "price_vs_sma200_pct": round(float(latest["price_to_ema200"]) * 100, 2),
            "volatility_20d": round(float(latest["volatility_20"]), 4),
            "volume_trend": volume_trend,
        },
    }


def list_companies() -> list[dict]:
    df = _load_data()
    latest = df.sort_values("Date").groupby("Symbol").tail(1)
    live_prices = fyers_service.get_ltp_batch(latest["Symbol"].tolist())
    return [
        {
            "symbol": row["Symbol"],
            "industry": row["Industry"],
            "last_close": round(live_prices.get(row["Symbol"], float(row["Close"])), 2),
            "is_live": row["Symbol"] in live_prices,
            "as_of_date": str(row["Date"].date()),
        }
        for _, row in latest.iterrows()
    ]


def get_prediction(symbol: str, horizon: int) -> dict | None:
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")

    df = _load_data()
    company_rows = df[df["Symbol"] == symbol.upper()]
    if company_rows.empty:
        return None

    latest_date = df["Date"].max()
    latest = company_rows.sort_values("Date").iloc[-1]
    same_day = df[df["Date"] == latest["Date"]]

    models = _train_models()
    X = _model_input(latest[FEATURE_COLS].to_frame().T)
    proba = float(models["median"][horizon].predict_proba(X)[0, 1])
    proba_ter = float(models["tercile"][horizon].predict_proba(X)[0, 1])

    live_price = fyers_service.get_ltp(symbol.upper())

    stats = HORIZON_STATS[horizon]
    return {
        "symbol": symbol.upper(),
        "industry": latest["Industry"],
        "last_close": round(live_price if live_price is not None else float(latest["Close"]), 2),
        "is_live": live_price is not None,
        "as_of_date": str(latest["Date"].date()),
        "horizon_days": horizon,
        "probability_beats_nifty50_median": round(proba, 4),
        "probability_top_tercile": round(proba_ter, 4),
        "signal": _signal_label(proba),
        "risk_level": _risk_level(latest["volatility_20"], same_day["volatility_20"]),
        "historical_accuracy_at_horizon": stats["historical_accuracy"],
        "historical_accuracy_top_tercile": stats["historical_accuracy_top_tercile"],
        "confidence_note": (
            f"Single-name directional accuracy is only ~{stats['historical_accuracy']:.0%} "
            f"(barely above the ~50% base rate). The tested value of this model is at the "
            f"PORTFOLIO level: ranking the NIFTY 200 by this probability and holding the top "
            f"quintile produced a post-cost information ratio near 1.0 over 2021-2026 "
            f"(Phase 10). Treat one stock's number as a weak tilt, not a forecast."
        ),
    }


def get_top_predictions(horizon: int, limit: int = 10) -> list[dict]:
    df = _load_data()
    latest_date = df["Date"].max()
    latest_rows = df[df["Date"] == latest_date]

    models = _train_models()
    X = _model_input(latest_rows[FEATURE_COLS])
    proba = models["median"][horizon].predict_proba(X)[:, 1]
    proba_ter = models["tercile"][horizon].predict_proba(X)[:, 1]
    result = latest_rows[["Symbol", "Industry", "Close", "volatility_20"]].copy()
    result["probability"] = proba
    result["probability_top_tercile"] = proba_ter
    result = result.sort_values("probability", ascending=False).head(limit)

    vol_series = latest_rows["volatility_20"]
    stats = HORIZON_STATS[horizon]
    live_prices = fyers_service.get_ltp_batch(result["Symbol"].tolist())
    return [
        {
            "rank": i + 1,
            "symbol": row["Symbol"],
            "industry": row["Industry"],
            "last_close": round(live_prices.get(row["Symbol"], float(row["Close"])), 2),
            "is_live": row["Symbol"] in live_prices,
            "probability_beats_nifty50_median": round(float(row["probability"]), 4),
            "probability_top_tercile": round(float(row["probability_top_tercile"]), 4),
            "signal": _signal_label(row["probability"]),
            "risk_level": _risk_level(row["volatility_20"], vol_series),
            "historical_accuracy_at_horizon": stats["historical_accuracy"],
            "historical_accuracy_top_tercile": stats["historical_accuracy_top_tercile"],
        }
        for i, (_, row) in enumerate(result.iterrows())
    ]
