"""Download NIFTY50 daily OHLCV data via yfinance and save to data/raw/."""

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
METADATA_PATH = RAW_DIR / "stock_metadata.csv"
START_DATE = "2010-01-01"

INDEX_TICKER = "^NSEI"
INDEX_OUT_NAME = "NIFTY50_INDEX.csv"

# Symbols whose Yahoo ticker no longer matches "<SYMBOL>.NS" after a
# corporate action (merger/demerger/rename).
TICKER_OVERRIDES = {
    "TATAMOTORS": "TMPV.NS",  # demerged; passenger-vehicle entity carries the pre-split history
}


def load_symbols() -> list[str]:
    meta = pd.read_csv(METADATA_PATH)
    return meta["Symbol"].tolist()


def to_yahoo_ticker(symbol: str) -> str:
    return TICKER_OVERRIDES.get(symbol, f"{symbol}.NS")


def to_filename(symbol: str) -> str:
    return symbol.replace("&", "")


MIN_ROWS = 300  # ~15 months; below this a stock can't clear the 200-day feature
                # warm-up + 90-day label window, so it's flagged not saved-and-forgotten
RETRIES = 3


def download_symbol(symbol: str) -> pd.DataFrame | None:
    ticker = to_yahoo_ticker(symbol)
    df = None
    for attempt in range(1, RETRIES + 1):
        try:
            df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=False)
            if not df.empty:
                break
        except Exception as exc:
            print(f"  attempt {attempt}/{RETRIES} failed: {exc}")
        time.sleep(2 * attempt)  # back off on Yahoo rate-limits
    if df is None or df.empty:
        return None
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.insert(1, "Symbol", symbol)
    return df


def download_index() -> pd.DataFrame:
    df = yf.download(INDEX_TICKER, start=START_DATE, progress=False, auto_adjust=False)
    df.columns = df.columns.get_level_values(0)
    return df.reset_index()


def main() -> None:
    symbols = load_symbols()
    failed = []
    short_history = []

    for symbol in symbols:
        out_path = RAW_DIR / f"{to_filename(symbol)}.csv"
        if out_path.exists():
            print(f"Skipping {symbol} (already downloaded)")
            continue

        print(f"Downloading {symbol}...")
        try:
            df = download_symbol(symbol)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed.append(symbol)
            continue

        if df is None:
            print("  No data returned (likely delisted/merged/renamed on Yahoo)")
            failed.append(symbol)
            continue

        df.to_csv(out_path, index=False)
        note = ""
        if len(df) < MIN_ROWS:
            short_history.append((symbol, len(df)))
            note = f"  [SHORT HISTORY - < {MIN_ROWS} rows, will contribute little/nothing]"
        print(f"  Saved {len(df)} rows -> {out_path.name}{note}")
        time.sleep(0.5)

    print("Downloading NIFTY50 index...")
    idx_df = download_index()
    idx_df.to_csv(RAW_DIR / INDEX_OUT_NAME, index=False)
    print(f"  Saved {len(idx_df)} rows -> {INDEX_OUT_NAME}")

    if failed:
        print("\nFailed / empty symbols (check Yahoo ticker, add to TICKER_OVERRIDES):")
        for s in failed:
            print(f"  - {s}  (tried {to_yahoo_ticker(s)})")

    if short_history:
        print(f"\nShort-history symbols (< {MIN_ROWS} rows - recent IPOs / demerged entities):")
        for s, n in short_history:
            print(f"  - {s}: {n} rows")

    print("\nNEXT: run process_data.py, then phase1_audit.py to catch unadjusted "
          "splits / new corporate-action breaks across the expanded universe, and "
          "add any it finds to CORPORATE_ACTION_BREAKS in process_data.py.")


if __name__ == "__main__":
    main()
