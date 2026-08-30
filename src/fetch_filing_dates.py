"""Fetch REAL annual-results filing dates from NSE's public corporate
announcements API, to replace the fiscal_year_end + 60-day approximation
used in feature_engineering.py's load_fundamentals().

Strategy: Indian listed companies file their Q4 results together with full
audited annual figures (there is no separate "annual result" filing) - so
the correct annual filing date is the EARLIEST "Financial Result Updates"
announcement found after each fiscal year-end, within a 90-day window
(SEBI's 60-day deadline + a buffer for the rare late filer).

Falls back to the fiscal_year_end + 60-day approximation (clearly marked
as such) when NSE has no matching announcement - never invents a date.
"""

import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FUNDAMENTALS_PATH = RAW_DIR / "fundamentals_annual.csv"
OUT_PATH = RAW_DIR / "filing_dates.csv"

SEARCH_WINDOW_DAYS = 90
FALLBACK_LAG_DAYS = 60
REQUEST_DELAY = 1.2

# NSE ticker overrides, same corporate-action logic as fetch_data.py /
# fetch_fundamentals.py - keep all three in sync.
NSE_SYMBOL_OVERRIDES = {
    "TATAMOTORS": "TMPV",  # NSE also renamed the announcement-feed symbol post-demerger,
                           # same as Yahoo/Screener - confirmed via live query (TATAMOTORS
                           # returns 0 results on NSE's API for ANY date range, not just post-2025).
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get("https://www.nseindia.com", timeout=10)  # sets required cookies
    return s


def fetch_announcements(session: requests.Session, symbol: str, from_date: pd.Timestamp, to_date: pd.Timestamp):
    nse_symbol = NSE_SYMBOL_OVERRIDES.get(symbol, symbol)
    encoded = urllib.parse.quote(nse_symbol)
    url = (
        f"https://www.nseindia.com/api/corporate-announcements?index=equities"
        f"&symbol={encoded}&from_date={from_date.strftime('%d-%m-%Y')}&to_date={to_date.strftime('%d-%m-%Y')}"
    )
    r = session.get(url, timeout=15)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except ValueError:
        return None


RESULT_KEYWORDS = ("financial result", "audited financial statement", "audited financial result")


def find_annual_filing_date(announcements: list) -> pd.Timestamp | None:
    """Match on desc (older NSE filings use "Financial Result Updates" as
    the category itself) OR attchmntText (newer filings categorize the same
    disclosure under "Outcome of Board Meeting" instead, confirmed via a
    live NSE query - the desc-only filter silently missed ~100 2025/2026
    filings that NSE had simply recategorized, not omitted)."""
    if not announcements:
        return None
    candidates = [
        a for a in announcements
        if any(kw in a.get("desc", "").lower() for kw in RESULT_KEYWORDS)
        or any(kw in a.get("attchmntText", "").lower() for kw in RESULT_KEYWORDS)
    ]
    if not candidates:
        return None
    dates = [pd.to_datetime(a["an_dt"], format="%d-%b-%Y %H:%M:%S") for a in candidates]
    return min(dates)  # earliest = the Q4/annual filing, not a later re-disclosure


def main() -> None:
    fund = pd.read_csv(FUNDAMENTALS_PATH)
    pairs = fund[["Symbol", "FiscalYearEnd"]].drop_duplicates().reset_index(drop=True)

    session = make_session()
    results = []
    n_nse, n_fallback, n_error = 0, 0, 0

    for i, row in pairs.iterrows():
        symbol, fy_str = row["Symbol"], row["FiscalYearEnd"]
        fiscal_year_end = pd.to_datetime(fy_str, format="%b %Y") + pd.offsets.MonthEnd(0)
        window_end = fiscal_year_end + pd.Timedelta(days=SEARCH_WINDOW_DAYS)

        print(f"[{i+1}/{len(pairs)}] {symbol} {fy_str}...", end=" ")
        try:
            announcements = fetch_announcements(session, symbol, fiscal_year_end, window_end)
        except requests.RequestException as exc:
            print(f"request error: {exc}")
            announcements = None
            n_error += 1

        actual_date = find_annual_filing_date(announcements) if announcements is not None else None

        if actual_date is not None:
            known_date = actual_date
            source = "nse_actual"
            n_nse += 1
            print(f"-> {known_date.date()} (NSE actual)")
        else:
            known_date = fiscal_year_end + pd.Timedelta(days=FALLBACK_LAG_DAYS)
            source = "approximation_fallback"
            n_fallback += 1
            print(f"-> {known_date.date()} (fallback, no NSE match)")

        results.append({
            "Symbol": symbol, "FiscalYearEnd": fy_str,
            "known_date": known_date.strftime("%Y-%m-%d"), "source": source,
        })
        time.sleep(REQUEST_DELAY)

        # NSE sessions can go stale; refresh every ~80 requests
        if (i + 1) % 80 == 0:
            session = make_session()

    out = pd.DataFrame(results)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nDone: {n_nse} real NSE dates, {n_fallback} fallback approximations, {n_error} request errors")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
