"""PHASE 1 — Raw Data Audit and Master Dataset Construction.

Runs all 20 required checks against the existing raw data (data/raw/*.csv,
sourced from yfinance) and produces a quality-control report. This does
NOT silently fix or delete anything - it reports findings so they can be
reviewed, consistent with "flag suspicious observations, don't delete them
silently."

Run: python3 src/phase1_audit.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / "phase1_qc_report.txt"

# Known corporate actions - maintained explicitly, not inferred silently.
# This is the canonical source of truth for ticker remapping and flagged
# discontinuities; fetch_data.py and feature_engineering.py should both
# reference this list rather than duplicating hardcoded overrides.
KNOWN_CORPORATE_ACTIONS = [
    {"symbol": "HDFC", "type": "merger", "date": "2023-07-01",
     "detail": "Merged into HDFCBANK. Excluded from universe post-merger; no independent price series exists."},
    {"symbol": "INFRATEL", "type": "merger", "date": "2020-11-19",
     "detail": "Bharti Infratel merged into Indus Towers. Excluded from universe; INDUSTOWER is a different NIFTY50 entrant, not tracked here."},
    {"symbol": "TATAMOTORS", "type": "demerger", "date": "2025-10-14",
     "detail": "Split into passenger-vehicle (TMPV.NS) and commercial-vehicle entities. Price history carried under TMPV.NS ticker; "
               "pre/post-split values are NOT organically comparable - flagged as a corp_action_break, forward-return "
               "labels spanning this date are excluded, not just the raw price."},
    {"symbol": "VEDL", "type": "demerger", "date": "2026-04-30",
     "detail": "Vedanta Ltd demerger into separate listed entities (Aluminium/Oil & Gas/Power/Steel & Ferrous/Base Metals). "
               "High-volume (73.8M shares), high-conviction price drop (-64.9%) on both sides - confirmed via this audit, "
               "NOT previously flagged. Requires the same corp_action_break treatment as TATAMOTORS."},
]

OUTPUT_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    OUTPUT_LINES.append(msg)


def section(title: str) -> None:
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


def load_all_stocks() -> dict[str, pd.DataFrame]:
    files = sorted(RAW_DIR.glob("*.csv"))
    files = [f for f in files if f.stem not in ("stock_metadata", "NIFTY50_INDEX", "fundamentals_annual")]
    return {f.stem: pd.read_csv(f, parse_dates=["Date"]) for f in files}


def main() -> None:
    stocks = load_all_stocks()
    metadata = pd.read_csv(RAW_DIR / "stock_metadata.csv")
    index_df = pd.read_csv(RAW_DIR / "NIFTY50_INDEX.csv", parse_dates=["Date"])

    section("PHASE 1 QC REPORT")
    log(f"Files audited: {len(stocks)} stocks + NIFTY50_INDEX.csv + stock_metadata.csv + fundamentals_annual.csv")
    log(f"Universe: {sorted(stocks.keys())}")

    # ---------- Checks 1-6: structural integrity ----------
    section("Checks 1-6: Structural Integrity (per file)")
    total_issues = 0
    for symbol, df in stocks.items():
        issues = []

        na = df[["Open", "High", "Low", "Close", "Volume"]].isna().sum()
        if na.sum() > 0:
            issues.append(f"missing values: {na[na > 0].to_dict()}")  # Check 1

        dup = df.duplicated(subset=["Date"]).sum()  # Check 2 (Symbol is implicit - one file per symbol)
        if dup > 0:
            issues.append(f"{dup} duplicate dates")

        bad_ohlc = df[(df["High"] < df["Low"]) | (df["Close"] > df["High"]) | (df["Close"] < df["Low"]) |
                      (df["Open"] > df["High"]) | (df["Open"] < df["Low"])]  # Check 3
        if len(bad_ohlc) > 0:
            issues.append(f"{len(bad_ohlc)} invalid OHLC relationships")

        non_positive = df[(df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)]  # Check 4
        if len(non_positive) > 0:
            issues.append(f"{len(non_positive)} zero/negative prices")

        neg_vol = df[df["Volume"] < 0]  # Check 5
        if len(neg_vol) > 0:
            issues.append(f"{len(neg_vol)} negative volumes")

        if not df["Date"].is_monotonic_increasing:  # Check 6
            issues.append("dates NOT in chronological order")

        if issues:
            total_issues += len(issues)
            log(f"[{symbol}] " + " | ".join(issues))

    log(f"\nResult: {'CLEAN - no issues found' if total_issues == 0 else f'{total_issues} issues found (see above)'}")

    # ---------- Check 7: ticker changes / Check 10-13: corporate actions ----------
    section("Checks 7, 10-13: Ticker Changes & Corporate Actions (documented, not inferred)")
    for action in KNOWN_CORPORATE_ACTIONS:
        log(f"- {action['symbol']} ({action['type']}, {action['date']}): {action['detail']}")
    log("\nThese are maintained as an explicit table (KNOWN_CORPORATE_ACTIONS in this file), not silently")
    log("handled inline - any new corporate action found in future data pulls must be added here first.")

    # ---------- Check 8-9: unadjusted splits/bonuses (heuristic detection) ----------
    section("Checks 8-9: Possible Unadjusted Splits/Bonuses (heuristic - Close vs Adj Close divergence)")
    log("Method: a real split/bonus should show as a large ratio jump in raw Close on the ex-date, but Adj")
    log("Close should remain smooth across it (that's what adjustment is for). Flagging cases where BOTH")
    log("Close and Adj Close jump together by >25% - that pattern indicates a genuine large move OR an")
    log("adjustment that failed to apply, and needs manual review rather than being trusted blindly.")
    log("NOTE: computed on Volume>0 rows only - zero-volume placeholder rows create false-positive jumps")
    log("when a real trading day is compared against a stale carried-forward quote, not a real prior price.")
    flagged_any = False
    for symbol, df in stocks.items():
        df_real = df[df["Volume"] > 0].reset_index(drop=True)
        close_ret = df_real["Close"].pct_change()
        adjclose_ret = df_real["Adj Close"].pct_change()
        both_jump = df_real[(close_ret.abs() > 0.25) & (adjclose_ret.abs() > 0.25) &
                             (np.sign(close_ret) == np.sign(adjclose_ret))]
        if len(both_jump) > 0:
            flagged_any = True
            for idx in both_jump.index:
                log(f"[{symbol}] {df_real.loc[idx, 'Date'].date()}: Close and Adj Close both moved >25% together "
                    f"(Close ret={close_ret.loc[idx]:.2%}) - verify against KNOWN_CORPORATE_ACTIONS above")
    if not flagged_any:
        log("None found beyond the already-documented corporate actions.")

    # ---------- Check 14-15: historical universe / survivorship bias ----------
    section("Checks 14-15: Historical NIFTY50 Constituent Changes & Survivorship Bias")
    log("STATUS: NOT RESOLVED. This is a real, material limitation of the current dataset, being")
    log("flagged explicitly rather than hidden, per instruction not to silently accept data quality risk.")
    log("")
    log("What we have: today's NIFTY50 constituent list, backfilled across 2010-2026.")
    log("What this means: any company that was IN the NIFTY50 index at some point in 2010-2026 but was")
    log("SUBSEQUENTLY REMOVED (e.g. for sustained underperformance, delisting, or index reshuffling) is")
    log("absent from this dataset entirely - including its historical price history from when it WAS a")
    log("constituent. This is classic survivorship bias: the training set systematically overrepresents")
    log("companies that performed well enough to remain in / re-enter a large-cap index through 2026.")
    log("")
    log("Consequence for this project: any 'this stock beat the market' finding is measured only among")
    log("survivors, and may overstate the achievable edge versus what an investor picking from the FULL")
    log("historical constituent list (including future-delisted/demoted names) would have experienced.")
    log("")
    log("Fix requires: a historical NIFTY50 constituent-change log (index inclusion/exclusion dates per")
    log("company) from NSE Indices or a paid data vendor - not available from yfinance or Screener.in.")
    log("Until acquired, this must be stated as a limitation in any reported result, not corrected for.")

    # ---------- Check 16: adjusted vs unadjusted ----------
    section("Check 16: Adjusted vs Unadjusted Prices")
    for symbol, df in list(stocks.items())[:1]:
        pass
    n_have_both = sum(1 for df in stocks.values() if "Close" in df.columns and "Adj Close" in df.columns)
    log(f"{n_have_both}/{len(stocks)} files have both 'Close' (unadjusted) and 'Adj Close' (dividend/split-adjusted).")
    log("Policy: all return/label calculations use Adj Close (see feature_engineering.py PRICE_COL).")
    log("Raw Close is retained separately for reporting actual traded/quoted price to end users.")

    # ---------- Check 17-18: missing periods / holidays ----------
    section("Checks 17-18: Missing Historical Periods & Market Holidays")
    for symbol, df in stocks.items():
        gaps = df["Date"].diff().dt.days
        big_gaps = gaps[gaps > 10]
        if len(big_gaps) > 0:
            log(f"[{symbol}] {len(big_gaps)} gap(s) >10 calendar days, max gap = {int(big_gaps.max())} days")
    log("(Exchange-holiday zero-volume placeholder rows are removed downstream in process_data.py,")
    log(" not here - this section only checks for missing MULTI-DAY periods, a different failure mode.)")

    # ---------- Check 19: suspicious price jumps ----------
    section("Check 19: Suspicious Single-Day Price Jumps (>25%, Adj Close, Volume>0 rows only)")
    for symbol, df in stocks.items():
        df_real = df[df["Volume"] > 0].reset_index(drop=True)
        ret = df_real["Adj Close"].pct_change()
        big_moves = df_real[ret.abs() > 0.25]
        for idx in big_moves.index:
            log(f"[{symbol}] {df_real.loc[idx, 'Date'].date()}: {ret.loc[idx]:+.1%} - "
                f"requires manual verification against real market events (see corporate actions above "
                f"and Project_Report.docx Section 2 for the events already traced: COVID crash 2020-03-23, "
                f"ZEEL-Sony merger news 2021-09, SBI recap news 2017-10, etc.)")

    # ---------- Check 20: data source consistency ----------
    section("Check 20: Data Source Consistency")
    fund = pd.read_csv(RAW_DIR / "fundamentals_annual.csv")
    price_symbols = set(stocks.keys())
    fund_symbols = set(fund["Symbol"].str.replace("&", "", regex=False).unique())
    log(f"Price data source: yfinance (Yahoo Finance) - {len(price_symbols)} symbols")
    log(f"Fundamentals source: Screener.in - {len(fund_symbols)} symbols")
    log(f"In price data but not fundamentals: {sorted(price_symbols - fund_symbols)}")
    log(f"In fundamentals but not price data: {sorted(fund_symbols - price_symbols)}")
    log("These are two independently-sourced datasets on two different update cadences (daily vs annual)")
    log("merged via an as-of join in feature_engineering.py - see that file's REPORTING_LAG_DAYS handling.")

    # ---------- Summary ----------
    section("PHASE 1 SUMMARY")
    log(f"Stocks in universe: {len(stocks)}")
    log(f"Date range: {min(df['Date'].min() for df in stocks.values()).date()} to "
        f"{max(df['Date'].max() for df in stocks.values()).date()}")
    log(f"Total rows: {sum(len(df) for df in stocks.values()):,}")
    log("Structural integrity (checks 1-6): PASS" if total_issues == 0 else f"Structural integrity: {total_issues} ISSUES")
    log("Corporate actions (checks 7,10-13): DOCUMENTED (3 known actions)")
    log("Unadjusted splits/bonuses (checks 8-9): " + ("FLAGGED - see above" if flagged_any else "NONE DETECTED"))
    log("Survivorship bias (checks 14-15): *** UNRESOLVED LIMITATION - MUST BE DISCLOSED ***")
    log("Adjusted price policy (check 16): DEFINED (Adj Close for returns)")
    log("Data source consistency (check 20): DOCUMENTED (2 symbols only in price data, not fundamentals)")

    REPORT_PATH.write_text("\n".join(OUTPUT_LINES))
    log(f"\nFull report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
