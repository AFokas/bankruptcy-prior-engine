"""Sklearn-compatible Weight-of-Evidence transformer.

Stage 3 production version of the inline ``WoeEncoder`` used in the Stage 2
exploration notebooks. Differs from the exploration version in three ways:

1. Bin edges are extended to ±inf at inference time, so out-of-range values
   in scoring data never produce NaN.
2. NaN inputs always map to a dedicated ``MISSING`` bin (never silently zeroed).
3. The transformer is a proper sklearn ``BaseEstimator + TransformerMixin``,
   so it composes inside a ``Pipeline``.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

_MISSING_BIN_INDEX = -1


class WoETransformer(BaseEstimator, TransformerMixin):
    """Quantile-bin numeric features and map each bin to its Weight-of-Evidence.

    Args:
        n_bins: Target number of quantile bins per feature. Reduced
            automatically when the feature has fewer unique values than
            ``n_bins``.
        laplace_smoothing: Pseudo-count added to per-bin event / non-event
            counts to prevent ``log(0)`` in sparse bins.

    Attributes:
        bin_edges_: Dict mapping feature name → bin edges array, with the
            left/right endpoints extended to ±inf.
        woe_maps_: Dict mapping feature name → dict[bin_index → WoE].
            ``bin_index == -1`` is the MISSING bin.
        iv_: Dict mapping feature name → Information Value (sum across bins).
        feature_names_in_: Names of the features the transformer was fit on,
            in fit order.
    """

    def __init__(self, n_bins: int = 10, laplace_smoothing: float = 0.5):
        self.n_bins = n_bins
        self.laplace_smoothing = laplace_smoothing

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoETransformer":
        """Fit per-feature bin edges and WoE maps on ``(X, y)``.

        Args:
            X: Feature matrix. Non-numeric columns are coerced via ``astype(float)``.
            y: Binary target (0/1).

        Returns:
            ``self``.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y_arr = np.asarray(y, dtype=int)
        n_positive = int(y_arr.sum())
        n_negative = int(len(y_arr) - n_positive)

        self.bin_edges_: dict[str, np.ndarray] = {}
        self.woe_maps_: dict[str, dict[int, float]] = {}
        self.iv_: dict[str, float] = {}
        self.feature_names_in_: list[str] = list(X.columns)

        for feat in self.feature_names_in_:
            x = X[feat].to_numpy(dtype=float, na_value=np.nan)
            edges, woe_map, iv = self._fit_one(x, y_arr, n_positive, n_negative)
            self.bin_edges_[feat] = edges
            self.woe_maps_[feat] = woe_map
            self.iv_[feat] = iv

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform ``X`` to WoE-encoded numeric values.

        Out-of-range values are mapped to the nearest training bin (edges are
        extended to ±inf internally). Unseen / impossible bin indices raise a
        ``UserWarning`` and map to WoE = 0.0.

        Args:
            X: Feature matrix with the same columns as seen during fit.

        Returns:
            DataFrame of the same shape, all-float, no NaN.

        Raises:
            ValueError: If ``X`` is missing a feature seen at fit time.
        """
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("WoETransformer has not been fit yet.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        missing = [c for c in self.feature_names_in_ if c not in X.columns]
        if missing:
            raise ValueError(f"transform() missing features seen at fit: {missing}")

        out = pd.DataFrame(index=X.index, columns=self.feature_names_in_, dtype=float)
        for feat in self.feature_names_in_:
            x = X[feat].to_numpy(dtype=float, na_value=np.nan)
            out[feat] = self._transform_one(feat, x)
        return out

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def get_iv_summary(self) -> pd.DataFrame:
        """Return a DataFrame of feature → IV, sorted descending."""
        if not hasattr(self, "iv_"):
            raise RuntimeError("WoETransformer has not been fit yet.")
        df = pd.DataFrame({"feature": list(self.iv_.keys()), "iv": list(self.iv_.values())})
        df = df.sort_values("iv", ascending=False, kind="mergesort").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _fit_one(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_positive: int,
        n_negative: int,
    ) -> tuple[np.ndarray, dict[int, float], float]:
        nan_mask = np.isnan(x)
        finite = x[~nan_mask]

        if len(finite) == 0:
            # all-NaN column — single MISSING bin only
            edges = np.array([-np.inf, np.inf])
            woe_map: dict[int, float] = {0: 0.0}
            woe_map[_MISSING_BIN_INDEX] = self._missing_woe(y, nan_mask, n_positive, n_negative, k_bins=1)
            iv = 0.0
            return edges, woe_map, iv

        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        raw_edges = np.unique(np.quantile(finite, quantiles))
        if len(raw_edges) < 2:
            raw_edges = np.array([finite.min(), finite.min() + 1e-9])

        # Extend to ±inf so digitize never returns the sentinel for out-of-range values.
        interior = raw_edges[1:-1]
        edges = np.concatenate([[-np.inf], interior, [np.inf]])

        n_bins_effective = len(edges) - 1
        k_total = n_bins_effective + (1 if nan_mask.any() else 0)

        bin_idx = self._assign_bins(finite, edges)
        woe_map: dict[int, float] = {}
        iv = 0.0

        y_finite = y[~nan_mask]
        for b in range(n_bins_effective):
            in_bin = bin_idx == b
            pos = int(y_finite[in_bin].sum())
            neg = int(in_bin.sum() - pos)
            w, iv_share = self._woe_and_iv_share(pos, neg, n_positive, n_negative, k_total)
            woe_map[b] = w
            iv += iv_share

        if nan_mask.any():
            woe_map[_MISSING_BIN_INDEX] = self._missing_woe(
                y, nan_mask, n_positive, n_negative, k_bins=k_total, iv_accumulator=None
            )
            # also add the IV contribution from the missing bin
            pos_m = int(y[nan_mask].sum())
            neg_m = int(nan_mask.sum() - pos_m)
            _, iv_share_m = self._woe_and_iv_share(pos_m, neg_m, n_positive, n_negative, k_total)
            iv += iv_share_m
        else:
            woe_map[_MISSING_BIN_INDEX] = 0.0

        return edges, woe_map, iv

    def _transform_one(self, feat: str, x: np.ndarray) -> np.ndarray:
        edges = self.bin_edges_[feat]
        woe_map = self.woe_maps_[feat]
        nan_mask = np.isnan(x)
        finite = x[~nan_mask]

        out = np.empty_like(x, dtype=float)
        out[nan_mask] = woe_map.get(_MISSING_BIN_INDEX, 0.0)

        if len(finite) > 0:
            bin_idx = self._assign_bins(finite, edges)
            mapped = np.empty(len(finite), dtype=float)
            unseen_mask = np.zeros(len(finite), dtype=bool)
            for i, b in enumerate(bin_idx):
                bi = int(b)
                if bi in woe_map:
                    mapped[i] = woe_map[bi]
                else:
                    mapped[i] = 0.0
                    unseen_mask[i] = True
            if unseen_mask.any():
                warnings.warn(
                    f"WoETransformer.transform: {int(unseen_mask.sum())} value(s) for "
                    f"feature '{feat}' fell into a bin not seen during fit; mapped to WoE=0.",
                    UserWarning,
                    stacklevel=3,
                )
            out[~nan_mask] = mapped
        return out

    @staticmethod
    def _assign_bins(finite: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Return bin index in [0, n_bins-1] for each finite value.

        ``edges`` has n_bins+1 entries with edges[0]=-inf, edges[-1]=+inf.
        ``np.digitize`` with ``right=True`` returns indices in [1, n_bins], so
        we subtract 1.
        """
        interior = edges[1:-1]
        if len(interior) == 0:
            return np.zeros(len(finite), dtype=int)
        # digitize maps values <= interior[i] to index i+1 (under right=True)
        bin_idx = np.digitize(finite, interior, right=True)
        return bin_idx

    def _woe_and_iv_share(
        self,
        pos: int,
        neg: int,
        n_positive: int,
        n_negative: int,
        k_bins: int,
    ) -> tuple[float, float]:
        smoothing = self.laplace_smoothing
        p_pos = (pos + smoothing) / (n_positive + smoothing * k_bins)
        p_neg = (neg + smoothing) / (n_negative + smoothing * k_bins)
        w = float(np.log(p_neg / p_pos))
        iv_share = float((p_neg - p_pos) * w)
        return w, iv_share

    def _missing_woe(
        self,
        y: np.ndarray,
        nan_mask: np.ndarray,
        n_positive: int,
        n_negative: int,
        k_bins: int,
        iv_accumulator: Any = None,
    ) -> float:
        pos = int(y[nan_mask].sum())
        neg = int(nan_mask.sum() - pos)
        w, _ = self._woe_and_iv_share(pos, neg, n_positive, n_negative, k_bins)
        return w
