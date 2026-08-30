"""PHASE 10 — Portfolio layer.

Stops scoring the model on "right vs wrong" and scores it on P&L. The
premise (Fundamental Law of Active Management): IR = IC * sqrt(breadth).
A ~52% directional edge (IC ~0.03-0.05) is thin, but spread across ~180
names rebalanced every 60 trading days it can still compound into a
defensible information ratio after costs.

Method (expanding-window walk-forward, no look-ahead):
  - every REBAL_STEP trading days from BACKTEST_START, retrain the model
    on every row dated <= rebalance_date - PURGE_DAYS
  - rank that day's cross-section by predicted P(beats median over 60d)
  - LONG  = top QUANTILE (equal-weighted)
  - SHORT = bottom QUANTILE  (for the market-neutral long-short leg)
  - hold 60 trading days, realise fwd_return_60d, subtract costs on the
    names actually traded (turnover-aware, not full-turnover-assumed)
  - benchmark = equal-weighted return of the ENTIRE universe that day
    (same construction as the long book minus the stock selection, so the
    excess return is pure selection alpha)

Outputs annualised return / vol / Sharpe / Max Drawdown for the long book,
the long-short book and the benchmark, plus Information Ratio and the
realised Information Coefficient, net of TRANSACTION_COST.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import FEATURE_COLS
from prepare_dataset import apply_saved_scaling, PURGE_DAYS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "model_features.csv"

HORIZON = 60                 # trading-day holding period / label horizon
REBAL_STEP = 60              # rebalance every 60 trading days -> non-overlapping holds
QUANTILE = 0.20              # top / bottom 20% (~36 names of ~180)
TRANSACTION_COST = 0.0025    # 0.25% round-trip: brokerage + STT + charges + slippage (India)
BACKTEST_START = "2021-01-01"
TARGET_COL = f"beats_median_{HORIZON}d"
RET_COL = f"fwd_return_{HORIZON}d"
PERIODS_PER_YEAR = 252 / REBAL_STEP  # ~4.2


def load() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    df = apply_saved_scaling(df).sort_values(["Date", "Symbol"]).reset_index(drop=True)
    return df


def fit_model(train: pd.DataFrame, kind: str):
    X, y = train[FEATURE_COLS], train[TARGET_COL].astype(int)
    if kind == "logit":
        return LogisticRegression(max_iter=2000, C=0.1).fit(X, y)
    return HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=42
    ).fit(X, y)


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1).min())


def annualise(period_returns: np.ndarray) -> dict:
    r = np.asarray(period_returns, dtype=float)
    ann_ret = (1 + r.mean()) ** PERIODS_PER_YEAR - 1
    ann_vol = r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    equity = np.cumprod(1 + r)
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_drawdown": max_drawdown(equity), "total_return": equity[-1] - 1}


def run(df: pd.DataFrame, kind: str) -> dict:
    all_dates = np.array(sorted(df["Date"].unique()))
    start = np.datetime64(pd.Timestamp(BACKTEST_START))
    rebal_dates = [d for d in all_dates if d >= start][::REBAL_STEP]

    purge = pd.Timedelta(days=PURGE_DAYS)
    prev_long, prev_short = set(), set()
    long_net, long_gross, ls_net, bench, ic_list, breadth, dates = [], [], [], [], [], [], []

    for t in rebal_dates:
        day = df[df["Date"] == t]
        day = day.dropna(subset=[TARGET_COL, RET_COL])
        if len(day) < 30:
            continue
        train = df[df["Date"] <= pd.Timestamp(t) - purge].dropna(subset=[TARGET_COL])
        if len(train) < 5000:
            continue

        model = fit_model(train, kind)
        p = model.predict_proba(day[FEATURE_COLS])[:, 1]
        day = day.assign(p=p)
        ranked = day.sort_values("p", ascending=False)
        k = max(1, int(len(ranked) * QUANTILE))
        longs, shorts = ranked.head(k), ranked.tail(k)

        gross = longs[RET_COL].mean()
        short_ret = shorts[RET_COL].mean()
        bench_ret = day[RET_COL].mean()

        # turnover-aware cost: fraction of the book replaced since last rebalance
        lset, sset = set(longs["Symbol"]), set(shorts["Symbol"])
        long_turnover = 1.0 if not prev_long else len(lset ^ prev_long) / (2 * len(lset))
        ls_turnover = long_turnover if not prev_short else (
            long_turnover + len(sset ^ prev_short) / (2 * len(sset))) / 2
        prev_long, prev_short = lset, sset

        long_gross.append(gross)
        long_net.append(gross - TRANSACTION_COST * long_turnover)
        ls_net.append((gross - short_ret) - TRANSACTION_COST * ls_turnover * 2)
        bench.append(bench_ret)
        ic_list.append(spearmanr(day["p"], day[RET_COL]).correlation)
        breadth.append(len(day))
        dates.append(pd.Timestamp(t))

    long_net, long_gross = np.array(long_net), np.array(long_gross)
    ls_net, bench = np.array(ls_net), np.array(bench)
    excess = long_net - bench
    ic = np.array(ic_list)

    ir = (excess.mean() / excess.std(ddof=1)) * np.sqrt(PERIODS_PER_YEAR) if excess.std() > 0 else np.nan
    t_stat, p_val = stats.ttest_1samp(excess, 0)
    ic_ir = ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic)) if ic.std() > 0 else np.nan

    curve = pd.DataFrame({
        "date": dates, "long_net": long_net, "long_short": ls_net, "benchmark": bench,
        "long_equity": np.cumprod(1 + long_net), "ls_equity": np.cumprod(1 + ls_net),
        "bench_equity": np.cumprod(1 + bench), "IC": ic,
    })

    return {
        "kind": kind, "n_periods": len(long_net), "avg_breadth": np.mean(breadth),
        "long": annualise(long_net), "long_gross": annualise(long_gross),
        "long_short": annualise(ls_net), "benchmark": annualise(bench),
        "excess_ann": excess.mean() * PERIODS_PER_YEAR, "information_ratio": ir,
        "excess_t": t_stat, "excess_p": p_val,
        "mean_IC": ic.mean(), "IC_stdev": ic.std(ddof=1), "IC_IR": ic_ir,
        "hit_rate_periods": float((long_net > bench).mean()),
        "curve": curve,
    }


def show(r: dict) -> None:
    print("=" * 78)
    print(f"MODEL: {r['kind']}   periods: {r['n_periods']}   avg universe/rebalance: {r['avg_breadth']:.0f}")
    print("=" * 78)
    hdr = f"{'':12s} {'ann.return':>12s} {'ann.vol':>10s} {'Sharpe':>8s} {'maxDD':>9s} {'totalRet':>10s}"
    print(hdr)
    for name, key in [("Long (net)", "long"), ("Long (gross)", "long_gross"),
                      ("Long-Short", "long_short"), ("Benchmark(EW)", "benchmark")]:
        m = r[key]
        print(f"{name:12s} {m['ann_return']:>11.2%} {m['ann_vol']:>10.2%} "
              f"{m['sharpe']:>8.2f} {m['max_drawdown']:>9.2%} {m['total_return']:>10.2%}")
    print("-" * 78)
    print(f"Excess return (Long - Benchmark), annualised : {r['excess_ann']:+.2%}")
    print(f"Information Ratio                            : {r['information_ratio']:.2f}")
    print(f"  t-stat {r['excess_t']:.2f}   p-value {r['excess_p']:.4f}   "
          f"{'(sig. at 5%)' if r['excess_p'] < 0.05 else '(NOT sig. at 5%)'}")
    print(f"Mean Information Coefficient (Spearman)      : {r['mean_IC']:+.4f}  (sd {r['IC_stdev']:.4f})")
    print(f"IC information ratio  (mean_IC/sd * sqrt(N)) : {r['IC_IR']:.2f}")
    print(f"Periods Long beat Benchmark                  : {r['hit_rate_periods']:.1%}")
    # Fundamental Law sanity check
    print(f"FLAM check: IC {r['mean_IC']:+.3f} x sqrt(breadth {r['avg_breadth']:.0f}) "
          f"= implied per-period IR {r['mean_IC'] * np.sqrt(r['avg_breadth']):+.3f}")


def main() -> None:
    df = load()
    print(f"Loaded {len(df):,} rows, {df['Symbol'].nunique()} symbols, "
          f"{df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"Backtest: long top {QUANTILE:.0%} by P({TARGET_COL}), "
          f"rebalance every {REBAL_STEP}d, cost {TRANSACTION_COST:.2%} round-trip\n")

    rows = []
    for kind in ("logit", "hgb"):
        r = run(df, kind)
        show(r)
        print()
        r["curve"].to_csv(PROCESSED_DIR.parent / f"phase10_portfolio_curve_{kind}.csv", index=False)
        flat = {"model": kind, "n_periods": r["n_periods"], "avg_breadth": r["avg_breadth"],
                "long_sharpe": r["long"]["sharpe"], "long_ann_return": r["long"]["ann_return"],
                "long_maxDD": r["long"]["max_drawdown"], "ls_sharpe": r["long_short"]["sharpe"],
                "bench_sharpe": r["benchmark"]["sharpe"], "information_ratio": r["information_ratio"],
                "excess_ann": r["excess_ann"], "excess_p": r["excess_p"],
                "mean_IC": r["mean_IC"], "IC_IR": r["IC_IR"]}
        rows.append(flat)

    out = PROCESSED_DIR.parent / "phase10_portfolio_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
