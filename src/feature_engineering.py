"""Build technical + market-regime features and multi-horizon return labels
from data/processed/stocks_daily.csv.

Output: data/processed/model_features.csv — one row per (Symbol, Date) with
engineered features and forward-return labels for horizons of
7/30/60/90 trading days. Rows are NOT yet windowed or scaled; that happens
in prepare_dataset.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
IN_PATH = PROCESSED_DIR / "stocks_daily.csv"
OUT_PATH = PROCESSED_DIR / "model_features.csv"

HORIZONS = [7, 30, 60, 90]  # trading days ahead
PRICE_COL = "Adj Close"  # dividend/split-adjusted, used for all return math

TECHNICAL_COLS = [
    "log_return", "price_to_sma10", "price_to_sma20", "price_to_sma50",
    "price_to_ema20", "price_to_ema50", "price_to_ema200",
    "macd_norm", "macd_hist_norm", "rsi_14", "bb_pct_b", "bb_width",
    "volatility_10", "volatility_20", "roc_5", "roc_10", "roc_20", "roc_40",
    "atr_pct", "volume_ratio", "dist_to_high20", "dist_to_low20",
    "dist_to_high50", "dist_to_low50", "obv_zscore_20",
    "nifty_return", "nifty_volatility_20", "nifty_trend",
    "relative_strength", "sector_return",
    "vix_zscore_60", "vix_change_5d",
]
LABEL_COLS = [f"fwd_return_{h}d" for h in HORIZONS]
REL_LABEL_COLS = [f"rel_return_{h}d" for h in HORIZONS]
DIR_LABEL_COLS = [f"beats_median_{h}d" for h in HORIZONS]           # direction vs. peers (cross-sectional)
ABS_DIR_LABEL_COLS = [f"positive_return_{h}d" for h in HORIZONS]     # direction vs. zero (absolute)
VOL_LABEL_COLS = [f"future_volatility_{h}d" for h in HORIZONS]       # future realized volatility

# --- Outlier-robust / better-separated label variants (added Phase 6.5) ---
# All are cross-sectional (computed within each Date) or divided by
# information known at time t, so none add look-ahead beyond what
# fwd_return_{h}d already carries. These are BEST-EFFORT columns: they are
# NOT part of the dropna gate in main(), so a NaN here (e.g. the middle
# tercile, deliberately) never drops the row for every other target.
SECTOR_REL_LABEL_COLS = [f"rel_return_sector_{h}d" for h in HORIZONS]   # market-relative, sector-neutralised
SECTOR_DIR_LABEL_COLS = [f"beats_sector_median_{h}d" for h in HORIZONS] # direction vs. same-industry peers
VOL_ADJ_LABEL_COLS = [f"fwd_return_vol_adj_{h}d" for h in HORIZONS]     # return / trailing realised vol
RANK_LABEL_COLS = [f"fwd_return_rank_{h}d" for h in HORIZONS]           # per-date percentile rank in [0, 1]
TERCILE_LABEL_COLS = [f"top_tercile_{h}d" for h in HORIZONS]            # 1=top third, 0=bottom third, NaN=middle

EXTRA_LABEL_COLS = (
    SECTOR_REL_LABEL_COLS + SECTOR_DIR_LABEL_COLS
    + VOL_ADJ_LABEL_COLS + RANK_LABEL_COLS + TERCILE_LABEL_COLS
)

# Only fundamentals available for EVERY stock (banks/NBFCs report Sales,
# Borrowings, ROCE differently on Screener.in and don't have them under
# these labels - including those here would force-drop all bank rows via
# the dropna step below).
FUNDAMENTAL_COLS = ["fund_roe", "fund_net_profit_growth", "fund_eps_growth", "fund_pe_ratio"]
FUNDAMENTALS_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "fundamentals_annual.csv"
REPORTING_LAG_DAYS = 60  # SEBI requires audited annual results within 60 days of fiscal year-end

FEATURE_COLS = TECHNICAL_COLS + FUNDAMENTAL_COLS


def add_stock_technicals(g: pd.DataFrame) -> pd.DataFrame:
    """Per-stock technical indicators. `g` must be one Symbol, sorted by Date."""
    close = g[PRICE_COL]

    g["log_return"] = np.log(close / close.shift(1))

    for w in (10, 20, 50):
        sma = close.rolling(w).mean()
        g[f"price_to_sma{w}"] = close / sma - 1

    for w in (20, 50, 200):
        ema = close.ewm(span=w, adjust=False).mean()
        ratio = close / ema - 1
        # ewm() never emits NaN (unlike rolling()), but early values are
        # unreliable since the EMA hasn't converged yet - mask them the
        # same way a rolling window's warm-up would be.
        ratio.iloc[: w - 1] = np.nan
        g[f"price_to_ema{w}"] = ratio

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    g["macd_norm"] = macd / close
    g["macd_hist_norm"] = (macd - macd_signal) / close

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    g["rsi_14"] = 100 - (100 / (1 + rs))

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    g["bb_pct_b"] = (close - lower) / (upper - lower)
    g["bb_width"] = (upper - lower) / sma20

    for w in (10, 20):
        g[f"volatility_{w}"] = g["log_return"].rolling(w).std()

    for w in (5, 10, 20, 40):
        g[f"roc_{w}"] = close / close.shift(w) - 1

    high, low, prev_close = g["High"], g["Low"], close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    g["atr_pct"] = tr.rolling(14).mean() / close

    vol_ma20 = g["Volume"].rolling(20).mean()
    g["volume_ratio"] = g["Volume"] / vol_ma20

    for w in (20, 50):
        rolling_high = high.rolling(w).max()
        rolling_low = low.rolling(w).min()
        g[f"dist_to_high{w}"] = close / rolling_high - 1  # <= 0, 0 = at the w-day high
        g[f"dist_to_low{w}"] = close / rolling_low - 1    # >= 0, 0 = at the w-day low

    # On-Balance Volume: cumulative, so unbounded/incomparable across stocks
    # and time in raw form. Use a z-score against its own recent history
    # instead of the raw running total.
    obv_direction = np.sign(g["log_return"].fillna(0))
    obv = (obv_direction * g["Volume"]).cumsum()
    obv_mean20 = obv.rolling(20).mean()
    obv_std20 = obv.rolling(20).std()
    g["obv_zscore_20"] = (obv - obv_mean20) / obv_std20

    return g


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    idx = df[["Date", "nifty_close"]].drop_duplicates("Date").sort_values("Date").copy()
    idx["nifty_return"] = np.log(idx["nifty_close"] / idx["nifty_close"].shift(1))
    idx["nifty_volatility_20"] = idx["nifty_return"].rolling(20).std()
    nifty_sma50 = idx["nifty_close"].rolling(50).mean()
    idx["nifty_trend"] = idx["nifty_close"] / nifty_sma50 - 1

    # India VIX (market-wide implied volatility / "fear gauge") - genuinely
    # new information not derived from NIFTY50 price/volume itself. Used in
    # a stationary form (z-score vs its own recent history, and short-term
    # change) rather than the raw level, since VIX's own baseline level has
    # drifted over 16 years.
    vix_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "INDIA_VIX.csv"
    vix = pd.read_csv(vix_path, parse_dates=["Date"]).sort_values("Date")
    vix_mean60 = vix["Close"].rolling(60).mean()
    vix_std60 = vix["Close"].rolling(60).std()
    vix["vix_zscore_60"] = (vix["Close"] - vix_mean60) / vix_std60
    vix["vix_change_5d"] = vix["Close"] / vix["Close"].shift(5) - 1

    df = df.merge(
        idx[["Date", "nifty_return", "nifty_volatility_20", "nifty_trend"]],
        on="Date", how="left",
    )
    df = df.merge(vix[["Date", "vix_zscore_60", "vix_change_5d"]], on="Date", how="left")
    df["relative_strength"] = df["log_return"] - df["nifty_return"]
    return df


def add_sector_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out sector average: a stock's own return is excluded from
    its sector's average, so this reflects PEERS' behavior, not a partial
    echo of the stock's own log_return (redundant especially for small
    sectors). 4 industries (SERVICES, CONSTRUCTION, FERTILISERS &
    PESTICIDES, MEDIA & ENTERTAINMENT) have exactly one member each -
    leave-one-out is undefined there (no peers exist), so those default to
    0 (neutral) rather than NaN, which would otherwise silently drop those
    4 stocks entirely via the pipeline's dropna step."""
    grp = df.groupby(["Date", "Industry"])["log_return"]
    n = grp.transform("size")
    total = grp.transform("sum")
    loo_mean = (total - df["log_return"]) / (n - 1)
    df["sector_return"] = loo_mean.where(n > 1, 0.0)
    return df


def mask_corporate_action_breaks(g: pd.DataFrame) -> pd.DataFrame:
    """Null out the return on a break date (price series is discontinuous
    there) and forbid any forward-return label from spanning across it."""
    g.loc[g["corp_action_break"], "log_return"] = np.nan

    break_positions = g.index[g["corp_action_break"]].tolist()
    if not break_positions:
        return g

    pos = np.arange(len(g))
    break_pos = np.array([g.index.get_loc(p) for p in break_positions])
    for h in HORIZONS:
        crosses = np.zeros(len(g), dtype=bool)
        for bp in break_pos:
            crosses |= (pos < bp) & (pos + h >= bp)
        for col in (f"fwd_return_{h}d", f"positive_return_{h}d", f"future_volatility_{h}d"):
            g.loc[crosses, col] = np.nan
    return g


def add_labels(g: pd.DataFrame) -> pd.DataFrame:
    close = g[PRICE_COL]
    for h in HORIZONS:
        fwd_ret = close.shift(-h) / close - 1
        g[f"fwd_return_{h}d"] = fwd_ret

        # Direction of ABSOLUTE return (return > 0) - distinct from
        # beats_median_{h}d (direction relative to peers, in
        # add_relative_labels). Both are legitimate, different questions.
        pos_ret = (fwd_ret > 0).astype(float)
        pos_ret[fwd_ret.isna()] = np.nan
        g[f"positive_return_{h}d"] = pos_ret

        # Future realized volatility: std of the h daily log returns
        # strictly AFTER t (t+1 .. t+h) - never includes today's own
        # return. Computed via reverse-rolling-reverse so the window looks
        # forward instead of pandas' default backward-looking rolling.
        shifted = g["log_return"].shift(-1)
        g[f"future_volatility_{h}d"] = shifted[::-1].rolling(h).std()[::-1]
    return g


def add_relative_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional labels: how a stock's forward return compares to the
    NIFTY50 basket's median forward return on the same date. Absolute
    return is dominated by market-wide moves (beta) that swamp any
    stock-specific signal; comparing to same-day peers strips that out and
    exposes a smaller but more learnable effect (confirmed empirically -
    a plain logistic regression beats the naive base rate on this target
    where it was exactly tied on absolute return)."""
    for h in HORIZONS:
        col = f"fwd_return_{h}d"
        median = df.groupby("Date")[col].transform("median")
        df[f"rel_return_{h}d"] = df[col] - median
        df[f"beats_median_{h}d"] = (df[col] > median).astype(float)

        # Sector-neutral variant: compare to same-day peers IN THE SAME
        # industry, not the whole basket - removes the sector tilt that
        # leaks into the plain rel_return (FINANCIAL SERVICES alone is
        # ~17% of the universe). Single-member sectors have no peers, so
        # fall back to the basket-wide median there.
        sec_grp = df.groupby(["Date", "Industry"])[col]
        sec_median = sec_grp.transform("median")
        sec_size = sec_grp.transform("size")
        sec_median = sec_median.where(sec_size > 1, median)
        df[f"rel_return_sector_{h}d"] = df[col] - sec_median
        df[f"beats_sector_median_{h}d"] = (df[col] > sec_median).astype(float)

        na = df[col].isna()
        df.loc[na, [f"rel_return_{h}d", f"beats_median_{h}d",
                    f"rel_return_sector_{h}d", f"beats_sector_median_{h}d"]] = np.nan
    return df


def add_advanced_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Outlier-robust regression targets and a better-separated direction
    target. Must run AFTER add_labels / mask_corporate_action_breaks (needs
    fwd_return_{h}d) and after add_stock_technicals (needs volatility_20)."""
    for h in HORIZONS:
        col = f"fwd_return_{h}d"

        # (a) Volatility-scaled return: raw forward return divided by the
        # stock's TRAILING 20d realised daily vol scaled to the horizon
        # (volatility_20 is known at t - no look-ahead). Turns a fat-tailed,
        # heteroskedastic target into a roughly unit-variance, near-Gaussian
        # one, so MSE stops being dominated by high-vol names and crises.
        denom = (df["volatility_20"] * np.sqrt(h)).replace(0, np.nan)
        df[f"fwd_return_vol_adj_{h}d"] = df[col] / denom

        # (b) Per-date percentile rank in [0, 1] - fully outlier-proof, only
        # the ordering of returns within a day matters, not the magnitude.
        rank = df.groupby("Date")[col].rank(pct=True)
        rank[df[col].isna()] = np.nan
        df[f"fwd_return_rank_{h}d"] = rank

        # (c) Tercile direction: 1 = top third of that day's cross-section,
        # 0 = bottom third, NaN = middle third (deliberately dropped so the
        # classifier trains on the separated tails where signal concentrates
        # instead of the ambiguous middle).
        tercile = pd.Series(np.nan, index=df.index)
        tercile[rank <= 1 / 3] = 0.0
        tercile[rank >= 2 / 3] = 1.0
        df[f"top_tercile_{h}d"] = tercile
    return df


def load_fundamentals() -> pd.DataFrame:
    """Load data/raw/fundamentals_annual.csv and compute YoY growth rates
    per stock (must be done here, at the annual-row level, before merging
    into daily data - can't recover "previous year's" value from a single
    merged daily row).

    known_date prefers the REAL filing date scraped from NSE's corporate
    announcements API (src/fetch_filing_dates.py, data/raw/filing_dates.csv)
    - 548 of 574 (95.5%) fiscal-year rows have a confirmed real date. Falls
    back to fiscal_year_end + REPORTING_LAG_DAYS (a conservative, never-early
    approximation) only for the remainder, where NSE's own disclosure text
    was too terse to identify programmatically."""
    fund = pd.read_csv(FUNDAMENTALS_PATH)
    fund["Symbol"] = fund["Symbol"].str.replace("&", "", regex=False)  # "M&M" -> "MM", matches Symbol elsewhere
    fund["fiscal_year_end"] = pd.to_datetime(fund["FiscalYearEnd"], format="%b %Y")
    # A fiscal year ending e.g. "Mar 2023" (parsed as 2023-03-01) actually
    # ends on 2023-03-31; without this the fallback +60 day lag below would
    # be computed from the wrong end of the month.
    fund["fiscal_year_end"] = fund["fiscal_year_end"] + pd.offsets.MonthEnd(0)

    filing_dates_path = FUNDAMENTALS_PATH.parent / "filing_dates.csv"
    if filing_dates_path.exists():
        filings = pd.read_csv(filing_dates_path, parse_dates=["known_date"])
        filings["Symbol"] = filings["Symbol"].str.replace("&", "", regex=False)
        fund = fund.merge(filings[["Symbol", "FiscalYearEnd", "known_date", "source"]],
                           on=["Symbol", "FiscalYearEnd"], how="left")
        missing = fund["known_date"].isna()
        fund.loc[missing, "known_date"] = fund.loc[missing, "fiscal_year_end"] + pd.Timedelta(days=REPORTING_LAG_DAYS)
    else:
        fund["known_date"] = fund["fiscal_year_end"] + pd.Timedelta(days=REPORTING_LAG_DAYS)

    fund = fund.sort_values(["Symbol", "fiscal_year_end"]).reset_index(drop=True)
    fund["fund_net_profit_growth"] = fund.groupby("Symbol")["net_profit"].pct_change()
    fund["fund_eps_growth"] = fund.groupby("Symbol")["eps"].pct_change()
    fund["fund_roe"] = fund["roe_pct"] / 100

    return fund[["Symbol", "known_date", "eps", "fund_roe", "fund_net_profit_growth", "fund_eps_growth"]]


def add_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """As-of merge: each trading day gets the most recently REPORTED
    fundamentals as of that date (known_date <= Date), never a future
    fiscal year's numbers. Stocks/dates before a company's first known
    fiscal year in our data (2015) get NaN, same as any other warm-up gap."""
    fund = load_fundamentals()

    df = df.sort_values("Date").reset_index(drop=True)
    fund = fund.sort_values("known_date").reset_index(drop=True)

    merged = pd.merge_asof(
        df, fund,
        left_on="Date", right_on="known_date",
        by="Symbol", direction="backward",
    )
    merged["fund_pe_ratio"] = merged[PRICE_COL] / merged["eps"]
    merged.loc[merged["eps"] <= 0, "fund_pe_ratio"] = np.nan  # P/E undefined for a loss-making year

    return merged.drop(columns=["known_date", "eps"])


def per_stock(df: pd.DataFrame, func) -> pd.DataFrame:
    """Apply `func` to each Symbol's rows independently and reassemble.
    Uses explicit iteration (not groupby.apply) since pandas 3.x drops the
    grouping column from what's passed to apply()."""
    parts = [func(g.copy()) for _, g in df.groupby("Symbol", sort=False)]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    df = pd.read_csv(IN_PATH, parse_dates=["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    df = per_stock(df, add_stock_technicals)
    df = per_stock(df, add_labels)
    df = add_market_regime(df)
    df = add_sector_regime(df)
    df = per_stock(df, mask_corporate_action_breaks)
    df = add_relative_labels(df)
    df = add_advanced_labels(df)
    df = add_fundamental_features(df)

    # Only the original label families gate the dropna - the EXTRA_LABEL_COLS
    # are best-effort (top_tercile_{h}d is NaN for the middle third BY
    # DESIGN and must not drag the whole row out).
    required_label_cols = LABEL_COLS + REL_LABEL_COLS + DIR_LABEL_COLS + ABS_DIR_LABEL_COLS + VOL_LABEL_COLS
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + required_label_cols).reset_index(drop=True)
    after = len(df)

    all_label_cols = required_label_cols + EXTRA_LABEL_COLS
    keep_cols = ["Date", "Symbol", "Industry", PRICE_COL, "Close"] + FEATURE_COLS + all_label_cols
    df = df[keep_cols].sort_values(["Date", "Symbol"]).reset_index(drop=True)

    # float32 is plenty for features/labels and roughly halves the file size
    # and every downstream read's memory footprint (matters at NSE-200 scale).
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    df.to_csv(OUT_PATH, index=False)

    print(f"Feature engineering done: {before:,} -> {after:,} rows (dropped {before - after:,} warm-up/tail rows)")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Required labels: {required_label_cols}")
    print(f"Extra labels:    {EXTRA_LABEL_COLS}")
    print(f"top_tercile non-NaN rows (per horizon): "
          + ", ".join(f"{h}d={int(df[f'top_tercile_{h}d'].notna().sum()):,}" for h in HORIZONS))
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
