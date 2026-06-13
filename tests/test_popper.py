"""Unit tests for the POPPER e-value framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.popper import (
    SUPPORTED_TESTS,
    execute_falsification_experiment,
    p_to_e_calibrator,
    sequential_e_accumulation,
)


# ---------------------------------------------------------------------------
# p_to_e_calibrator
# ---------------------------------------------------------------------------


def test_p_to_e_calibrator_monotonic(cfg):
    kappa = cfg["popper"]["kappa"]
    # smaller p → larger e
    assert p_to_e_calibrator(0.001, kappa) > p_to_e_calibrator(0.5, kappa)
    assert p_to_e_calibrator(0.5, kappa) > p_to_e_calibrator(0.99, kappa)


def test_p_to_e_calibrator_kappa_validated():
    with pytest.raises(ValueError):
        p_to_e_calibrator(0.5, kappa=0.0)
    with pytest.raises(ValueError):
        p_to_e_calibrator(0.5, kappa=1.0)


def test_p_to_e_calibrator_average_under_h0_at_most_one(cfg):
    """Under H0 the p-value is uniform on [0,1]; E[e] should be <= 1."""
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    uniform_ps = rng.uniform(size=10000)
    es = np.array([p_to_e_calibrator(p, cfg["popper"]["kappa"]) for p in uniform_ps])
    assert es.mean() <= 1.05  # finite-sample slack


# ---------------------------------------------------------------------------
# sequential_e_accumulation
# ---------------------------------------------------------------------------


def test_sequential_e_accumulation_rejects_when_threshold_crossed(cfg):
    alpha = cfg["popper"]["alpha"]  # 0.10 → threshold = 10
    e_values = [1.0, 5.0, 3.0]  # cumulative 1, 5, 15 → reject at round 3
    out = sequential_e_accumulation(e_values, alpha)
    assert out["rejected"] is True
    assert out["rejected_at_round"] == 3
    assert out["cumulative_E"] == [1.0, 5.0, 15.0]
    assert out["threshold"] == pytest.approx(1.0 / alpha)


def test_sequential_e_accumulation_no_rejection_when_below(cfg):
    out = sequential_e_accumulation([0.5, 1.2, 1.5], alpha=cfg["popper"]["alpha"])
    assert out["rejected"] is False
    assert out["rejected_at_round"] is None
    assert out["final_E"] == pytest.approx(0.5 * 1.2 * 1.5)


def test_sequential_e_accumulation_empty():
    out = sequential_e_accumulation([], alpha=0.1)
    assert out["final_E"] == 1.0
    assert out["rejected"] is False


# ---------------------------------------------------------------------------
# execute_falsification_experiment
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_xy():
    rng = np.random.default_rng(0)
    n = 600
    feat = rng.normal(size=n)
    # bake in a real signal so the test should reject H0
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-1.5 * feat))).astype(int)
    X = pd.DataFrame({"f": feat, "other": rng.normal(size=n)})
    return X, pd.Series(y, name="y")


def test_execute_returns_low_p_when_signal_present(fake_xy):
    X, y = fake_xy
    spec = {
        "experiment_name": "feat separates classes",
        "null_sub_hypothesis": "feat distribution is the same in default vs non-default",
        "alternative_sub_hypothesis": "feat distribution differs between classes",
        "statistical_test": "mann_whitney_u",
        "columns_used": ["f"],
        "alternative": "two-sided",
    }
    p = execute_falsification_experiment(spec, X, y)
    assert p < 0.01


def test_execute_returns_one_for_unsupported_test():
    X = pd.DataFrame({"f": [0.1, 0.2]})
    y = pd.Series([0, 1])
    spec = {"statistical_test": "not_a_real_test", "columns_used": ["f"]}
    assert execute_falsification_experiment(spec, X, y) == 1.0


def test_execute_returns_one_for_missing_column(fake_xy):
    X, y = fake_xy
    spec = {"statistical_test": "mann_whitney_u", "columns_used": ["nope"]}
    assert execute_falsification_experiment(spec, X, y) == 1.0


def test_supported_tests_contains_expected():
    assert "mann_whitney_u" in SUPPORTED_TESTS
    assert "fisher_exact" in SUPPORTED_TESTS
    assert "permutation" in SUPPORTED_TESTS
    assert "two_proportion_z" in SUPPORTED_TESTS
