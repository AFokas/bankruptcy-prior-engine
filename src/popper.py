"""POPPER-style sequential falsification with e-values.

The Stage 4 notebook 03 has LLMs propose structured falsification
experiments (a null hypothesis, a statistical test name, the data
columns) and this module executes those experiments and accumulates
e-values for a type-I-error-controlled sequential rejection decision.

We use the Vovk-Wang p→e calibrator instead of Fisher's combined test
because e-values multiply *unconditionally* under H0 — so the running
product is a non-negative supermartingale and we can stop adaptively
without inflating type-I error.

The set of statistical tests is fixed (``SUPPORTED_TESTS``). We never
``exec()`` LLM-generated code; the LLM only chooses a test from the
allow-list and supplies its inputs.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


SUPPORTED_TESTS = frozenset({
    "mann_whitney_u",
    "fisher_exact",
    "permutation",
    "two_proportion_z",
})


# ---------------------------------------------------------------------------
# p → e calibration  (Vovk-Wang)
# ---------------------------------------------------------------------------


def p_to_e_calibrator(p_value: float, kappa: float) -> float:
    """Vovk-Wang ``e = kappa * p^(kappa - 1)`` calibrator.

    For any ``kappa in (0, 1)``, ``E[e | H0] <= 1`` unconditionally on the
    p-value distribution under H0. ``kappa = 0.5`` is a robust default.

    Args:
        p_value: A valid p-value in ``[0, 1]``. Values <= 0 are clipped to a
            small floor so the e-value is finite.
        kappa: Calibrator parameter, must be in ``(0, 1)``.

    Returns:
        Non-negative e-value.

    Raises:
        ValueError: If ``kappa`` is outside ``(0, 1)``.
    """
    if not 0.0 < kappa < 1.0:
        raise ValueError(f"kappa must be in (0, 1); got {kappa}")
    p = float(max(min(p_value, 1.0), 1e-12))
    return float(kappa * (p ** (kappa - 1.0)))


def sequential_e_accumulation(e_values: list[float], alpha: float) -> dict:
    """Multiply e-values sequentially and decide whether to reject H0.

    Reject at the first round ``k`` where the cumulative product ``E_k``
    crosses the rejection threshold ``1 / alpha``. Type-I error <= alpha by
    Ville's inequality.

    Args:
        e_values: Per-round e-values (output of :func:`p_to_e_calibrator`).
        alpha: Target type-I error rate.

    Returns:
        Dict with ``cumulative_E`` (list of running products),
        ``final_E`` (last value), ``threshold`` (= 1/alpha),
        ``rejected`` (bool), ``rejected_at_round`` (int or None).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    threshold = 1.0 / alpha
    cumulative: list[float] = []
    running = 1.0
    rejected_at: int | None = None
    for i, e in enumerate(e_values, start=1):
        running *= float(e)
        cumulative.append(running)
        if rejected_at is None and running >= threshold:
            rejected_at = i
    return {
        "cumulative_E": cumulative,
        "final_E": cumulative[-1] if cumulative else 1.0,
        "threshold": float(threshold),
        "rejected": rejected_at is not None,
        "rejected_at_round": rejected_at,
    }


# ---------------------------------------------------------------------------
# Falsification execution
# ---------------------------------------------------------------------------


def execute_falsification_experiment(
    experiment_spec: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cohort_mask: pd.Series | None = None,
) -> float:
    """Run the statistical test specified in ``experiment_spec``.

    The LLM-supplied spec must contain:

    - ``experiment_name`` (free text)
    - ``null_sub_hypothesis`` (free text)
    - ``statistical_test`` — one of :data:`SUPPORTED_TESTS`
    - ``columns_used`` — list of column names referenced by the test
    - test-specific inputs (see the per-test branches below)

    Returns 1.0 on any failure (missing test, unknown column, exception) —
    a conservative "no evidence" result that contributes nothing to the
    cumulative e-value product.

    Args:
        experiment_spec: Parsed JSON dict from the LLM.
        X_test: OOT feature matrix.
        y_test: OOT target vector.
        cohort_mask: Optional Boolean mask restricting the test to a cohort.

    Returns:
        A p-value in ``[0, 1]``.
    """
    test_name = experiment_spec.get("statistical_test")
    if test_name not in SUPPORTED_TESTS:
        logger.warning("unsupported test '%s'; returning p=1.0", test_name)
        return 1.0

    columns = experiment_spec.get("columns_used", [])
    missing = [c for c in columns if c not in X_test.columns]
    if missing:
        logger.warning("columns %s not in X_test; returning p=1.0", missing)
        return 1.0

    if cohort_mask is not None:
        X_test = X_test.loc[cohort_mask].copy()
        y_test = y_test.loc[cohort_mask].copy()
    if len(X_test) == 0:
        return 1.0

    try:
        if test_name == "mann_whitney_u":
            return _mann_whitney(X_test, y_test, experiment_spec)
        if test_name == "fisher_exact":
            return _fisher_exact(X_test, y_test, experiment_spec)
        if test_name == "two_proportion_z":
            return _two_prop_z(X_test, y_test, experiment_spec)
        if test_name == "permutation":
            return _permutation(X_test, y_test, experiment_spec)
    except Exception as exc:  # noqa: BLE001 — any failure becomes conservative
        logger.warning("test %s raised %s; returning p=1.0", test_name, exc)
        return 1.0
    return 1.0


def _mann_whitney(X: pd.DataFrame, y: pd.Series, spec: dict) -> float:
    """Two-sample test on one column, comparing y=0 vs y=1 distributions."""
    col = spec["columns_used"][0]
    pos = X.loc[y == 1, col].dropna().to_numpy()
    neg = X.loc[y == 0, col].dropna().to_numpy()
    if len(pos) < 2 or len(neg) < 2:
        return 1.0
    alt = spec.get("alternative", "two-sided")
    result = stats.mannwhitneyu(pos, neg, alternative=alt)
    return float(result.pvalue)


def _fisher_exact(X: pd.DataFrame, y: pd.Series, spec: dict) -> float:
    """2x2 Fisher test on a binarised column vs y."""
    col = spec["columns_used"][0]
    threshold = float(spec.get("threshold", X[col].median()))
    x_high = (X[col] > threshold).astype(int)
    table = pd.crosstab(x_high, y).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
    if table.shape != (2, 2) or table.sum() == 0:
        return 1.0
    _, p = stats.fisher_exact(table, alternative=spec.get("alternative", "two-sided"))
    return float(p)


def _two_prop_z(X: pd.DataFrame, y: pd.Series, spec: dict) -> float:
    """Two-proportion z-test: default rate among high vs low subset of one column."""
    col = spec["columns_used"][0]
    threshold = float(spec.get("threshold", X[col].median()))
    mask_high = X[col] > threshold
    n_high = int(mask_high.sum())
    n_low = int((~mask_high).sum())
    if n_high < 5 or n_low < 5:
        return 1.0
    p_high = float(y[mask_high].mean())
    p_low = float(y[~mask_high].mean())
    p_pool = float(y.mean())
    if p_pool in (0.0, 1.0):
        return 1.0
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_high + 1 / n_low))
    z = (p_high - p_low) / se if se > 0 else 0.0
    return float(2.0 * (1.0 - stats.norm.cdf(abs(z))))


def _permutation(X: pd.DataFrame, y: pd.Series, spec: dict) -> float:
    """Permutation test on the mean difference of one column by y."""
    col = spec["columns_used"][0]
    pos = X.loc[y == 1, col].dropna().to_numpy()
    neg = X.loc[y == 0, col].dropna().to_numpy()
    if len(pos) < 2 or len(neg) < 2:
        return 1.0
    n_perm = int(spec.get("n_permutations", 1000))
    seed = int(spec.get("seed", 42))

    def stat(a, b):
        return np.mean(a) - np.mean(b)

    result = stats.permutation_test(
        (pos, neg), stat, n_resamples=n_perm,
        alternative=spec.get("alternative", "two-sided"),
        random_state=seed,
    )
    return float(result.pvalue)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_popper_round(
    db_path: str,
    round_idx: int,
    model_name: str,
    experiment_name: str,
    null_hypothesis: str,
    statistical_test: str,
    p_value: float,
    e_value: float,
    cumulative_e: float,
    decision: str,
) -> None:
    """Write one POPPER round to the ``popper_experiments`` table."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO popper_experiments (round, model_name, experiment_name, null_hypothesis, "
            "statistical_test, p_value, e_value, cumulative_e, decision, run_timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (int(round_idx), model_name, experiment_name, null_hypothesis,
             statistical_test, float(p_value), float(e_value),
             float(cumulative_e), decision, ts),
        )
        conn.commit()
