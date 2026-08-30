"""Clean and combine per-stock CSVs in data/raw/ into one tidy dataset.

Steps:
  1. Load every stock CSV + the NIFTY50 index.
  2. Drop exchange-holiday rows (zero volume, present across all stocks).
  3. Flag known non-organic price discontinuities (mergers/demergers) so
     downstream return/label calculations can exclude them instead of
     treating them as real price moves.
  4. Attach sector/industry metadata.
  5. Attach market-wide (NIFTY50 index) context columns to every row.
  6. Save the combined long-format table to data/processed/stocks_daily.csv.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "stocks_daily.csv"

PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close"]

# Known non-organic price breaks (mergers/demergers/corporate restructuring)
# where the raw return between two consecutive rows does not reflect real
# market movement. Add to this list as new cases are found during validation.
CORPORATE_ACTION_BREAKS = {
    "TATAMOTORS": ["2025-10-14"],  # demerger into passenger/commercial vehicle entities
    "VEDL": ["2026-04-30"],  # demerger into Aluminium/Oil&Gas/Power/Steel/Base Metals entities -
                             # found via src/phase1_audit.py, was NOT previously flagged (-64.9% raw)
}


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(RAW_DIR / "stock_metadata.csv")
    meta["Symbol"] = meta["Symbol"].str.replace("&", "", regex=False)
    return meta[["Symbol", "Company Name", "Industry"]]


def load_stock(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["Volume"] > 0].copy()  # drop exchange-holiday placeholder rows
    df = df.sort_values("Date").reset_index(drop=True)

    # Use the filename stem as the canonical symbol (matches stock_metadata.csv
    # and CORPORATE_ACTION_BREAKS keys) rather than the CSV's own Symbol column,
    # which for M&M was written as "M&M" instead of the sanitized "MM".
    symbol = path.stem
    df["Symbol"] = symbol
    break_dates = set(CORPORATE_ACTION_BREAKS.get(symbol, []))
    df["corp_action_break"] = df["Date"].dt.strftime("%Y-%m-%d").isin(break_dates)

    return df


def load_index() -> pd.DataFrame:
    idx = pd.read_csv(RAW_DIR / "NIFTY50_INDEX.csv", parse_dates=["Date"])
    idx = idx.sort_values("Date").reset_index(drop=True)
    rename = {c: f"nifty_{c.lower().replace(' ', '_')}" for c in PRICE_COLS + ["Volume"]}
    idx = idx.rename(columns=rename)
    return idx[["Date"] + list(rename.values())]


def main() -> None:
    metadata = load_metadata()
    index_df = load_index()

    symbols = metadata["Symbol"].tolist()
    frames = []
    skipped = []

    for symbol in symbols:
        path = RAW_DIR / f"{symbol}.csv"
        if not path.exists():
            skipped.append(symbol)
            continue
        frames.append(load_stock(path))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.merge(metadata, on="Symbol", how="left")

    # ^NSEI (Yahoo) is occasionally missing dates that individual stocks
    # traded on. Reindex the index onto every trading date seen in the
    # stock data and forward-fill, so no stock row is left without market
    # context.
    all_dates = pd.Index(sorted(combined["Date"].unique()), name="Date")
    index_df = index_df.set_index("Date").reindex(all_dates).ffill().reset_index()

    combined = combined.merge(index_df, on="Date", how="left")
    combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"Combined {len(symbols) - len(skipped)} stocks -> {len(combined):,} rows")
    print(f"Saved to {OUT_PATH}")
    if skipped:
        print(f"Skipped (no file present): {skipped}")

    n_breaks = combined["corp_action_break"].sum()
    print(f"Corporate-action break rows flagged: {n_breaks}")

    missing_nifty = combined["nifty_close"].isna().sum()
    if missing_nifty:
        print(f"WARNING: {missing_nifty} rows have no matching NIFTY50 index date")


if __name__ == "__main__":
    main()
