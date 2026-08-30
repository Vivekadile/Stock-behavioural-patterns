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


def download_symbol(symbol: str) -> pd.DataFrame | None:
    ticker = to_yahoo_ticker(symbol)
    df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=False)
    if df.empty:
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
            print("  No data returned (likely delisted/merged)")
            failed.append(symbol)
            continue

        df.to_csv(out_path, index=False)
        print(f"  Saved {len(df)} rows -> {out_path.name}")
        time.sleep(0.5)

    print("Downloading NIFTY50 index...")
    idx_df = download_index()
    idx_df.to_csv(RAW_DIR / INDEX_OUT_NAME, index=False)
    print(f"  Saved {len(idx_df)} rows -> {INDEX_OUT_NAME}")

    if failed:
        print("\nFailed / empty symbols (check manually):")
        for s in failed:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
