"""Serializable preprocessing for the 30-day return regression model.

One fitted `WinsorLogRobustScaler` instance is saved inside the model
pipeline (models/regression_30d/pipeline.joblib) so the backend loads the
EXACT same preprocessing that was fitted on the training split - it never
re-implements any of this by hand.

Transform, in order (identical recipe to src/prepare_dataset.py, kept as a
proper sklearn transformer here so it can live inside a Pipeline):
  1. winsorize each feature to the train [1%, 99%] quantile band
  2. log1p the strictly-positive, right-skewed ratio columns
  3. RobustScaler (subtract train median, divide by train IQR)
  4. hard-clip to +/- clip_sigma
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_LOG1P_COLS = ("atr_pct", "volume_ratio", "bb_width", "fund_pe_ratio")


class WinsorLogRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols, winsor_q=(0.01, 0.99),
                 log1p_cols=DEFAULT_LOG1P_COLS, clip_sigma=5.0):
        self.feature_cols = list(feature_cols)
        self.winsor_q = tuple(winsor_q)
        self.log1p_cols = [c for c in log1p_cols if c in self.feature_cols]
        self.clip_sigma = float(clip_sigma)

    # -- helpers -----------------------------------------------------------
    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_cols].astype(float).reset_index(drop=True)
        return pd.DataFrame(np.asarray(X, dtype=float), columns=self.feature_cols)

    def _winsor_log1p(self, f: pd.DataFrame) -> pd.DataFrame:
        f = f.clip(lower=self.lo_, upper=self.hi_, axis=1)
        for c in self.log1p_cols:
            f[c] = np.log1p(f[c].clip(lower=0))
        return f

    # -- sklearn API -----------------------------------------------------
    def fit(self, X, y=None):
        f = self._as_frame(X)
        self.lo_ = f.quantile(self.winsor_q[0])
        self.hi_ = f.quantile(self.winsor_q[1])
        t = self._winsor_log1p(f)
        self.center_ = t.median()
        iqr = t.quantile(0.75) - t.quantile(0.25)
        self.iqr_ = iqr.replace(0, 1.0)          # guard constant columns
        self.n_features_in_ = len(self.feature_cols)
        return self

    def transform(self, X):
        f = self._as_frame(X)
        t = self._winsor_log1p(f)
        z = (t - self.center_) / self.iqr_
        return np.clip(z.to_numpy(), -self.clip_sigma, self.clip_sigma)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_cols, dtype=object)
