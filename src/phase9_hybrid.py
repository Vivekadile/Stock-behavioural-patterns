"""PHASE 9 — Hybrid / stacked ensemble.

The project's repeated finding is that a plain linear model on the current
day's features beats RF, GBM and an LSTM ensemble. This script tests
whether *combining* those base learners via stacked generalisation buys
anything the best single model doesn't already have.

Design (leakage-controlled):
  base models : LogisticRegression, RandomForest, HistGradientBoosting
                (all on the Phase-6.5 scaled features)
  meta features: out-of-fold base-model probabilities on TRAIN, produced
                by a TimeSeriesSplit with a `gap` wide enough to cover the
                label horizon (no fold ever predicts a row whose label
                overlaps its own training window)
  meta model  : LogisticRegression on the 3 base probabilities
  reported    : best single base, simple average blend, stacked meta -
                all scored on the same untouched TEST split, vs base rate

Targets: beats_median_{h}d and top_tercile_{h}d (middle third dropped) for
every horizon.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import FEATURE_COLS, HORIZONS

warnings.filterwarnings("ignore")

SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "splits"
N_SPLITS = 5
STOCKS_PER_DAY = 47  # approximate rows per trading date, to size the CV gap


def load_splits():
    train = pd.read_csv(SPLIT_DIR / "train.csv", parse_dates=["Date"])
    test = pd.read_csv(SPLIT_DIR / "test.csv", parse_dates=["Date"])
    return train, test


def base_models() -> dict:
    return {
        "logit": LogisticRegression(max_iter=2000, C=0.1),
        "rf": RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=50,
                                     n_jobs=-1, random_state=42),
        "hgb": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300,
                                              l2_regularization=1.0, random_state=42),
    }


def run_target(train: pd.DataFrame, test: pd.DataFrame, target: str, horizon: int) -> dict:
    col = f"{target}_{horizon}d"
    tr = train.dropna(subset=[col]).reset_index(drop=True)
    te = test.dropna(subset=[col]).reset_index(drop=True)

    X_tr, y_tr = tr[FEATURE_COLS].to_numpy(), tr[col].to_numpy().astype(int)
    X_te, y_te = te[FEATURE_COLS].to_numpy(), te[col].to_numpy().astype(int)

    names = list(base_models().keys())
    oof = np.zeros((len(tr), len(names)))
    gap = horizon * STOCKS_PER_DAY  # rows to skip between fold-train end and fold-val start

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=gap)
    for fit_idx, val_idx in tscv.split(X_tr):
        for j, name in enumerate(names):
            m = base_models()[name]
            m.fit(X_tr[fit_idx], y_tr[fit_idx])
            oof[val_idx, j] = m.predict_proba(X_tr[val_idx])[:, 1]
    # rows before the first fold's validation block never get an OOF pred
    scored = oof.any(axis=1)

    # base models refit on ALL of train, predict test
    test_p = np.zeros((len(te), len(names)))
    for j, name in enumerate(names):
        m = base_models()[name]
        m.fit(X_tr, y_tr)
        test_p[:, j] = m.predict_proba(X_te)[:, 1]

    # meta-learner on OOF probabilities
    meta = LogisticRegression(max_iter=2000, C=1.0)
    meta.fit(oof[scored], y_tr[scored])

    base_rate = max(y_te.mean(), 1 - y_te.mean())
    acc = {name: ((test_p[:, j] > 0.5).astype(int) == y_te).mean() for j, name in enumerate(names)}
    avg_acc = ((test_p.mean(axis=1) > 0.5).astype(int) == y_te).mean()
    stack_acc = (meta.predict(test_p) == y_te).mean()

    return {
        "target": target, "horizon": f"{horizon}d", "n_test": len(te), "base_rate": base_rate,
        **{f"base_{k}": v for k, v in acc.items()},
        "blend_avg": avg_acc, "stacked_meta": stack_acc,
        "meta_weights": dict(zip(names, meta.coef_[0].round(3))),
    }


def main() -> None:
    train, test = load_splits()
    print(f"Train {len(train):,} | Test {len(test):,} | base models: {list(base_models())}\n")

    rows = []
    for target in ("beats_median", "top_tercile"):
        for h in HORIZONS:
            r = run_target(train, test, target, h)
            rows.append(r)
            best_base = max(r[f"base_{k}"] for k in base_models())
            print(f"{target:12s} {h:>3}d  base_rate={r['base_rate']:.4f}  "
                  f"best_base={best_base:.4f}  blend={r['blend_avg']:.4f}  "
                  f"stacked={r['stacked_meta']:.4f}  (n={r['n_test']:,})  w={r['meta_weights']}")

    df = pd.DataFrame(rows).drop(columns=["meta_weights"])
    out = Path(__file__).resolve().parent.parent / "data" / "phase9_hybrid_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")
    print("\nRead: 'stacked' or 'blend' must beat both 'best_base' AND 'base_rate' by a")
    print("meaningful margin to justify the extra complexity. Ties = no value added.")


if __name__ == "__main__":
    main()
