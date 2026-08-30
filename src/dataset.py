"""Turn a flat (Date, Symbol, features..., labels...) table into LSTM-ready
windowed sequences. Windows are built per-stock so a sequence never mixes
rows from two different companies, and are built on demand rather than
pre-materialized to disk (each window overlaps lookback-1 rows with its
neighbor, so storing every window separately wastes a lot of space).
"""

from pathlib import Path

import numpy as np
import pandas as pd

LOOKBACK = 60  # trading days of history per sequence


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    lookback: int = LOOKBACK,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    stride=1 (the default) makes every window overlap lookback-1 of its
    neighbor's timesteps - with lookback=60 that's 98%+ shared content
    between adjacent samples, which massively inflates the apparent sample
    count without adding much independent information (a model can fit
    train noise very fast when consecutive "different" samples are nearly
    identical). Use stride>1 to space windows out and reduce that overlap
    at the cost of fewer total samples.

    Returns:
      X: (n_samples, lookback, n_features) float32
      y: (n_samples, n_labels) float32
      meta: DataFrame with Date/Symbol for each sample, aligned to X/y rows
            (Date/Symbol of the *last* day in each window, i.e. "today").
    """
    X_parts, y_parts, meta_parts = [], [], []

    for symbol, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        if len(g) < lookback:
            continue

        feats = g[feature_cols].to_numpy(dtype=np.float32)
        labels = g[label_cols].to_numpy(dtype=np.float32)

        # sliding_window_view avoids an explicit Python loop over rows
        windows = np.lib.stride_tricks.sliding_window_view(feats, lookback, axis=0)
        windows = windows.transpose(0, 2, 1)  # (n_windows, lookback, n_features)
        windows = windows[::stride]

        X_parts.append(windows)
        y_parts.append(labels[lookback - 1::stride])
        meta_parts.append(g.loc[lookback - 1::stride, ["Date", "Symbol"]].reset_index(drop=True))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    meta = pd.concat(meta_parts, ignore_index=True)
    return X, y, meta


if __name__ == "__main__":
    # Smoke test on the val split (smallest) to confirm shapes/correctness.
    SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "splits"
    from prepare_dataset import FEATURE_COLS, LABEL_COLS

    val = pd.read_csv(SPLIT_DIR / "val.csv", parse_dates=["Date"])
    X, y, meta = make_sequences(val, FEATURE_COLS, LABEL_COLS)

    print(f"X shape: {X.shape}  (n_samples, lookback={LOOKBACK}, n_features={len(FEATURE_COLS)})")
    print(f"y shape: {y.shape}  (n_samples, n_labels={len(LABEL_COLS)})")
    print(f"meta rows: {len(meta)}")
    assert len(X) == len(y) == len(meta)

    # spot-check: last row of a window's features should match that day's
    # row in the source table for the same (Date, Symbol)
    sample_meta = meta.iloc[100]
    src_row = val[(val["Date"] == sample_meta["Date"]) & (val["Symbol"] == sample_meta["Symbol"])]
    src_feats = src_row[FEATURE_COLS].to_numpy(dtype=np.float32)[0]
    assert np.allclose(X[100, -1, :], src_feats), "window's last row must match its labeled day"
    print("Smoke test passed: window alignment is correct.")
