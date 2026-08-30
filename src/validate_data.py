"""Sanity-check the downloaded data in data/raw/."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
EXPECTED_COLS = {"Date", "Symbol", "Adj Close", "Close", "High", "Low", "Open", "Volume"}


def load_symbols() -> list[str]:
    meta = pd.read_csv(RAW_DIR / "stock_metadata.csv")
    return meta["Symbol"].str.replace("&", "", regex=False).tolist()


def validate_file(path: Path) -> list[str]:
    issues = []
    df = pd.read_csv(path, parse_dates=["Date"])

    missing_cols = EXPECTED_COLS - set(df.columns)
    if missing_cols:
        issues.append(f"missing columns: {missing_cols}")
        return issues

    if df.empty:
        issues.append("file is empty")
        return issues

    na_counts = df[["Open", "High", "Low", "Close", "Volume"]].isna().sum()
    if na_counts.sum() > 0:
        issues.append(f"NaNs found: {na_counts[na_counts > 0].to_dict()}")

    dup_dates = df["Date"].duplicated().sum()
    if dup_dates > 0:
        issues.append(f"{dup_dates} duplicate dates")

    if not df["Date"].is_monotonic_increasing:
        issues.append("dates not sorted ascending")

    bad_ohlc = df[(df["High"] < df["Low"]) | (df["Close"] > df["High"]) | (df["Close"] < df["Low"]) | (df["Open"] > df["High"]) | (df["Open"] < df["Low"])]
    if len(bad_ohlc) > 0:
        issues.append(f"{len(bad_ohlc)} rows with OHLC logic violations (High<Low or Close/Open outside High-Low range)")

    non_positive = df[(df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)]
    if len(non_positive) > 0:
        issues.append(f"{len(non_positive)} rows with zero/negative prices")

    neg_volume = df[df["Volume"] < 0]
    if len(neg_volume) > 0:
        issues.append(f"{len(neg_volume)} rows with negative volume")

    zero_volume = df[df["Volume"] == 0]
    if len(zero_volume) > 0:
        issues.append(f"{len(zero_volume)} rows with zero volume")

    daily_ret = df["Close"].pct_change().abs()
    extreme_moves = daily_ret[daily_ret > 0.25]
    if len(extreme_moves) > 0:
        issues.append(f"{len(extreme_moves)} days with >25% single-day price move (possible unadjusted split or bad tick)")

    gaps = df["Date"].diff().dt.days
    big_gaps = gaps[gaps > 10]
    if len(big_gaps) > 0:
        issues.append(f"{len(big_gaps)} gaps >10 calendar days between trading rows (max gap: {int(big_gaps.max())} days)")

    return issues


def main() -> None:
    symbols = load_symbols()
    files = sorted(RAW_DIR.glob("*.csv"))
    files = [f for f in files if f.stem not in ("stock_metadata",)]

    print(f"Found {len(files)} data files, {len(symbols)} expected symbols\n")

    missing_files = [s for s in symbols if not (RAW_DIR / f"{s}.csv").exists()]
    if missing_files:
        print(f"MISSING FILES for symbols: {missing_files}\n")

    total_issues = 0
    for f in files:
        issues = validate_file(f)
        if issues:
            total_issues += len(issues)
            print(f"[{f.stem}]")
            for issue in issues:
                print(f"  - {issue}")

    print(f"\n{'='*50}")
    if total_issues == 0:
        print("All files passed validation with no issues.")
    else:
        print(f"Total issues found: {total_issues}")


if __name__ == "__main__":
    main()
