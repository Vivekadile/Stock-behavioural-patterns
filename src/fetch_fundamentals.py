"""Scrape annual fundamentals (Sales, Net Profit, EPS, ROCE, Equity, Borrowings)
from Screener.in's free public company pages for every stock in
stock_metadata.csv.

This ONLY collects and saves the data to data/raw/fundamentals_annual.csv.
It does NOT touch feature_engineering.py, model_features.csv, or any split/
training file — that integration happens later, on explicit instruction.

Respectful scraping: only public /company/<SYMBOL>/ pages (not disallowed by
robots.txt), a real User-Agent, and a delay between requests.
"""

import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_PATH = RAW_DIR / "fundamentals_annual.csv"
REQUEST_DELAY = 2.0  # seconds between requests

# Same corporate-action ticker remapping as src/fetch_data.py's TICKER_OVERRIDES.
SCREENER_OVERRIDES = {
    "TATAMOTORS": "TMPV",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# Rows we actually need from each table, by their exact Screener label.
PL_ROWS = {"Sales": "sales", "Net Profit": "net_profit", "EPS in Rs": "eps"}
BS_ROWS = {"Equity Capital": "equity_capital", "Reserves": "reserves", "Borrowings": "borrowings"}
RATIO_ROWS = {"ROCE %": "roce_pct"}


def load_symbols() -> list[str]:
    meta = pd.read_csv(RAW_DIR / "stock_metadata.csv")
    return meta["Symbol"].tolist()  # keep "M&M" as-is; requests handles URL-encoding


def fetch_page(symbol: str) -> BeautifulSoup | None:
    screener_symbol = SCREENER_OVERRIDES.get(symbol, symbol)
    for variant in ("consolidated/", ""):
        url = f"https://www.screener.in/company/{screener_symbol}/{variant}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
        except requests.RequestException as exc:
            print(f"  request error ({variant or 'standalone'}): {exc}")
            continue
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser")
    return None


def extract_table(soup: BeautifulSoup, section_id: str, wanted_rows: dict) -> dict:
    section = soup.find("section", id=section_id)
    if section is None:
        return {}
    table = section.find("table")
    if table is None:
        return {}

    rows = table.find_all("tr")
    header_cells = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
    years = header_cells[1:]  # skip the blank first column

    data = {}
    for row in rows[1:]:
        cells = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
        if not cells:
            continue
        label = cells[0].rstrip("+")
        if label not in wanted_rows:
            continue
        key = wanted_rows[label]
        values = cells[1:]
        for year, val in zip(years, values):
            data.setdefault(year, {})[key] = val
    return data


def parse_numeric(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if val in ("", "-"):
        return None
    val = val.replace(",", "").replace("%", "").strip()
    if val in ("", "-"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def fetch_symbol(symbol: str) -> pd.DataFrame:
    soup = fetch_page(symbol)
    if soup is None:
        return pd.DataFrame()

    pl = extract_table(soup, "profit-loss", PL_ROWS)
    bs = extract_table(soup, "balance-sheet", BS_ROWS)
    ratios = extract_table(soup, "ratios", RATIO_ROWS)

    years = set(pl) | set(bs) | set(ratios)
    rows = []
    for year in years:
        if year == "TTM":
            continue  # not a fiscal year-end, skip
        row = {"Symbol": symbol, "FiscalYearEnd": year}
        row.update(pl.get(year, {}))
        row.update(bs.get(year, {}))
        row.update(ratios.get(year, {}))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["sales", "net_profit", "eps", "equity_capital", "reserves", "borrowings", "roce_pct"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_numeric)

    if {"net_profit", "equity_capital", "reserves"}.issubset(df.columns):
        equity_total = df["equity_capital"].fillna(0) + df["reserves"].fillna(0)
        df["roe_pct"] = (df["net_profit"] / equity_total * 100).round(2)
    if {"borrowings", "equity_capital", "reserves"}.issubset(df.columns):
        equity_total = df["equity_capital"].fillna(0) + df["reserves"].fillna(0)
        df["debt_to_equity"] = (df["borrowings"] / equity_total).round(3)

    return df


def main() -> None:
    symbols = load_symbols()
    all_frames = []
    failed = []

    for i, symbol in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] {symbol}...")
        try:
            df = fetch_symbol(symbol)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed.append(symbol)
            time.sleep(REQUEST_DELAY)
            continue

        if df.empty:
            print("  No data found")
            failed.append(symbol)
        else:
            print(f"  Got {len(df)} fiscal years")
            all_frames.append(df)

        time.sleep(REQUEST_DELAY)

    if not all_frames:
        print("No data collected at all.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["Symbol", "FiscalYearEnd"]).reset_index(drop=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(combined)} rows ({combined['Symbol'].nunique()} stocks) to {OUT_PATH}")
    if failed:
        print(f"Failed/empty: {failed}")


if __name__ == "__main__":
    main()
