"""Non-parametric bootstrap confidence intervals.

Stage 3 production version of the inline helper in
``notebooks/_explore_helpers.py``. Supports both fixed-sample scalar metrics
and order-resampled convergence trajectories (used by the Part 2 LLN curve).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int,
    ci_level: float,
    seed: int,
) -> dict:
    """Non-parametric percentile bootstrap CI for any scalar metric.

    Args:
        y_true: 1-d array of ground-truth labels.
        y_pred: 1-d array of predictions aligned with ``y_true``.
        metric_fn: Callable ``(y_true, y_pred) -> float``.
        n_resamples: Number of bootstrap resamples.
        ci_level: Confidence level in ``(0, 1)``, e.g. 0.95.
        seed: Random seed for the bootstrap RNG.

    Returns:
        Dict with keys ``estimate`` (point estimate on the full sample),
        ``lower``, ``upper``, and ``std`` (of the resample distribution).

    Raises:
        ValueError: If shapes mismatch or ``ci_level`` is out of range.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    estimates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        estimates[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = (1.0 - ci_level) / 2.0
    return {
        "estimate": float(metric_fn(y_true, y_pred)),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "std": float(np.std(estimates)),
    }


def bootstrap_convergence_ci(
    responses: list[float],
    n_resamples: int,
    ci_level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap CI bands on a running-mean (Law of Large Numbers) trajectory.

    For the Part 2 LLN convergence experiment: each bootstrap resample
    reshuffles the order of ``responses`` and recomputes the running-mean
    trajectory. The CI band at position ``k`` is the percentile interval
    across resamples of the running mean after ``k`` items.

    Args:
        responses: Sequence of scalar responses to average. Order is
            resampled (not the values themselves).
        n_resamples: Number of bootstrap order resamples.
        ci_level: Confidence level in ``(0, 1)``.
        seed: Random seed.

    Returns:
        Tuple ``(lower_bound, upper_bound)``, each a 1-d array of length
        ``len(responses)``.
    """
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}")
    arr = np.asarray(responses, dtype=float)
    n = len(arr)
    if n == 0:
        raise ValueError("responses must be non-empty")

    rng = np.random.default_rng(seed)
    trajectories = np.empty((n_resamples, n), dtype=float)
    for i in range(n_resamples):
        order = rng.permutation(n)
        permuted = arr[order]
        cumulative = np.cumsum(permuted)
        trajectories[i] = cumulative / np.arange(1, n + 1)

    alpha = (1.0 - ci_level) / 2.0
    lower = np.quantile(trajectories, alpha, axis=0)
    upper = np.quantile(trajectories, 1.0 - alpha, axis=0)
    return lower, upper
