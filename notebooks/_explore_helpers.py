"""Inline helpers used by the Stage 2 exploration notebooks.

These functions are deliberately quick-and-dirty exploration code. Stage 3
refactors the same logic into tested, sklearn-compatible modules under
``src/`` (``WoETransformer``, ``compute_psi``, ``bootstrap_ci``, …).

Do not import these from production code — use the ``src/`` equivalents.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# WoE / IV
# ---------------------------------------------------------------------------

def _bin_left_edge(bin_label: object) -> float:
    """Parse the left edge of a pandas-style bin label, e.g. '(1.23, 4.56]'."""
    s = str(bin_label)
    if s == "MISSING":
        return np.nan
    body = s.strip("()[]")
    parts = body.split(",")
    try:
        return float(parts[0])
    except (ValueError, IndexError):
        return np.nan


def compute_woe_table(
    x: pd.Series,
    y: pd.Series,
    n_bins: int = 10,
    laplace: float = 0.5,
) -> pd.DataFrame:
    """Quantile-bin ``x`` and return per-bin WoE + IV contribution.

    Missing values get a dedicated ``MISSING`` bin (never silently zeroed).
    Laplace smoothing prevents ``log(0)`` in sparse bins.
    """
    total_pos = int(y.sum())
    total_neg = int(len(y) - total_pos)

    df = pd.DataFrame({"x": x.values, "y": y.values})
    nan_mask = df["x"].isna()

    bins_str = pd.Series(index=df.index, dtype="object")
    bins_str[nan_mask] = "MISSING"
    finite = df.loc[~nan_mask, "x"]
    if len(finite) > 0:
        try:
            q = pd.qcut(finite, q=n_bins, duplicates="drop")
        except ValueError:
            q = pd.cut(finite, bins=n_bins)
        bins_str.loc[~nan_mask] = q.astype(str)
    df["bin"] = bins_str

    agg = df.groupby("bin", observed=True).agg(n=("y", "size"), pos=("y", "sum"))
    agg["neg"] = agg["n"] - agg["pos"]
    k = max(len(agg), 1)
    agg["p_pos"] = (agg["pos"] + laplace) / (total_pos + laplace * k)
    agg["p_neg"] = (agg["neg"] + laplace) / (total_neg + laplace * k)
    agg["woe"] = np.log(agg["p_neg"] / agg["p_pos"])
    agg["iv_contribution"] = (agg["p_neg"] - agg["p_pos"]) * agg["woe"]
    agg["bin_left"] = [_bin_left_edge(idx) for idx in agg.index]
    agg = agg.sort_values("bin_left", na_position="last")
    return agg


def compute_iv_table(X: pd.DataFrame, y: pd.Series, n_bins: int = 10, laplace: float = 0.5) -> pd.DataFrame:
    """Return DataFrame with one row per feature: feature, iv, n_bins, band."""
    rows = []
    woe_tables: dict[str, pd.DataFrame] = {}
    for feat in X.columns:
        wt = compute_woe_table(X[feat], y, n_bins=n_bins, laplace=laplace)
        iv = float(wt["iv_contribution"].sum())
        rows.append({"feature": feat, "iv": iv, "n_bins": len(wt)})
        woe_tables[feat] = wt
    out = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    out["band"] = out["iv"].apply(iv_band)
    return out, woe_tables


def iv_band(iv: float) -> str:
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    return "strong"


def is_monotonic_woe(woes: np.ndarray) -> bool:
    """True iff the WoE sequence is strictly monotonic (either direction)."""
    woes = np.asarray(woes, dtype=float)
    if len(woes) < 2 or np.any(np.isnan(woes)):
        return False
    diffs = np.diff(woes)
    return bool((diffs > 0).all() or (diffs < 0).all())


# ---------------------------------------------------------------------------
# WoE encoder for the LR pipeline in notebook 01
# ---------------------------------------------------------------------------

class WoeEncoder:
    """Minimal WoE encoder — fits per-feature quantile bins and WoE maps.

    Behaviour:
    - Missing values are mapped to the WoE of the MISSING bin (or 0 if none).
    - Out-of-range values are mapped to the nearest training bin (edges
      extended to ±inf internally).
    - Unseen categorical bins (impossible here — numeric only) would map to 0.
    """

    def __init__(self, n_bins: int = 10, laplace: float = 0.5):
        self.n_bins = n_bins
        self.laplace = laplace
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.woe_maps_: dict[str, dict[int, float]] = {}
        self.missing_woe_: dict[str, float] = {}
        self.iv_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoeEncoder":
        total_pos = int(y.sum())
        total_neg = int(len(y) - total_pos)
        for feat in X.columns:
            x = X[feat].values.astype(float)
            nan_mask = np.isnan(x)
            finite = x[~nan_mask]
            if len(finite) == 0:
                self.bin_edges_[feat] = np.array([-np.inf, np.inf])
                self.woe_maps_[feat] = {0: 0.0}
                self.missing_woe_[feat] = 0.0
                self.iv_[feat] = 0.0
                continue

            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.unique(np.quantile(finite, quantiles))
            if len(edges) < 2:
                edges = np.array([finite.min(), finite.min() + 1e-9])
            edges_ext = np.concatenate([[-np.inf], edges[1:-1], [np.inf]]) if len(edges) > 2 else np.array([-np.inf, np.inf])
            self.bin_edges_[feat] = edges_ext

            bin_idx = np.digitize(finite, edges_ext[1:-1], right=True)
            # bin_idx in [0, len(edges_ext)-2]
            k = len(edges_ext) - 1 + 1  # +1 for the MISSING bin
            woe_map: dict[int, float] = {}
            iv_total = 0.0
            for b in range(len(edges_ext) - 1):
                in_bin = bin_idx == b
                pos = int(y.values[~nan_mask][in_bin].sum())
                neg = int(in_bin.sum() - pos)
                p_pos = (pos + self.laplace) / (total_pos + self.laplace * k)
                p_neg = (neg + self.laplace) / (total_neg + self.laplace * k)
                w = float(np.log(p_neg / p_pos))
                woe_map[b] = w
                iv_total += (p_neg - p_pos) * w
            self.woe_maps_[feat] = woe_map

            if nan_mask.any():
                pos_m = int(y.values[nan_mask].sum())
                neg_m = int(nan_mask.sum() - pos_m)
                p_pos = (pos_m + self.laplace) / (total_pos + self.laplace * k)
                p_neg = (neg_m + self.laplace) / (total_neg + self.laplace * k)
                w_m = float(np.log(p_neg / p_pos))
                self.missing_woe_[feat] = w_m
                iv_total += (p_neg - p_pos) * w_m
            else:
                self.missing_woe_[feat] = 0.0
            self.iv_[feat] = iv_total
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        for feat in X.columns:
            x = X[feat].values.astype(float)
            nan_mask = np.isnan(x)
            edges = self.bin_edges_[feat]
            woe_map = self.woe_maps_[feat]

            bin_idx = np.digitize(x, edges[1:-1], right=True)
            woe_vals = np.array([woe_map.get(int(b), 0.0) for b in bin_idx], dtype=float)
            woe_vals[nan_mask] = self.missing_woe_[feat]
            out[feat] = woe_vals
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(2 * roc_auc_score(y_true, y_pred) - 1)


def brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y_true, y_pred))


class IsotonicCalibratedClassifier:
    """Wrap a fitted binary classifier with an isotonic calibrator on holdout.

    Stage 3 will replace this with the sklearn-canonical pipeline; this is the
    exploration version (sklearn 1.7+ removed ``cv='prefit'`` from
    ``CalibratedClassifierCV`` and the FrozenEstimator path adds noise we don't
    need for the exploration notebook).
    """

    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator

    @classmethod
    def fit_from_holdout(cls, base, X_cal, y_cal):
        from sklearn.isotonic import IsotonicRegression
        raw = base.predict_proba(X_cal)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y_cal)
        return cls(base, iso)

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    estimates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        estimates[i] = metric_fn(y_true[idx], y_pred[idx])
    alpha = (1 - ci_level) / 2
    return {
        "estimate": float(metric_fn(y_true, y_pred)),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1 - alpha)),
        "std": float(np.std(estimates)),
    }


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    edges: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """PSI for one feature using pre-fitted bin edges (extended to ±inf)."""
    base = baseline[~np.isnan(baseline)]
    curr = current[~np.isnan(current)]
    interior = edges[1:-1] if len(edges) > 2 else np.array([])
    base_bins = np.digitize(base, interior, right=True)
    curr_bins = np.digitize(curr, interior, right=True)
    n_bins = len(edges) - 1
    base_dist = np.array([(base_bins == b).sum() / max(len(base), 1) for b in range(n_bins)])
    curr_dist = np.array([(curr_bins == b).sum() / max(len(curr), 1) for b in range(n_bins)])
    base_dist = np.clip(base_dist, epsilon, None)
    curr_dist = np.clip(curr_dist, epsilon, None)
    return float(np.sum((curr_dist - base_dist) * np.log(curr_dist / base_dist)))


def psi_status(value: float, stable: float, monitor: float) -> str:
    if value < stable:
        return "STABLE"
    if value < monitor:
        return "MONITOR"
    return "RETRAIN"
