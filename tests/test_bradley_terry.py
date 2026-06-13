"""Unit tests for Bradley-Terry fitting and pairwise-prompt construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bradley_terry import (
    bt_to_log_odds,
    construct_pairwise_prompts,
    fit_bradley_terry,
    parse_winner,
    validate_bt_stability,
)


# ---------------------------------------------------------------------------
# parse_winner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("A", "A"),
    ("B", "B"),
    (" a ", "A"),
    ("The answer is B.", "B"),
    ("I think A is more risky", "A"),
    ("neither", None),
    ("", None),
])
def test_parse_winner(raw, expected):
    assert parse_winner(raw) == expected


# ---------------------------------------------------------------------------
# construct_pairwise_prompts
# ---------------------------------------------------------------------------


def test_construct_pairwise_prompts_borrower_level():
    profiles = [{"id": "p1", "roa": 0.1}, {"id": "p2", "roa": -0.5}, {"id": "p3", "roa": 0.0}]
    prompts = construct_pairwise_prompts(profiles, task="borrower_level")
    assert len(prompts) == 3  # 3 choose 2
    ids = [(a, b) for (a, b, _) in prompts]
    assert ids == [("p1", "p2"), ("p1", "p3"), ("p2", "p3")]
    for (_, _, prompt) in prompts:
        assert "COMPANY A" in prompt and "COMPANY B" in prompt
        assert "default" in prompt.lower()


def test_construct_pairwise_prompts_category_level():
    profiles = [{"id": "c1", "current_ratio": 1.2}, {"id": "c2", "current_ratio": 0.5}]
    prompts = construct_pairwise_prompts(profiles, task="category_level", category="financial_health")
    assert len(prompts) == 1
    assert "financial health" in prompts[0][2].lower()


def test_construct_pairwise_prompts_validates_task():
    with pytest.raises(ValueError):
        construct_pairwise_prompts([{"id": "x"}], task="not_a_task")


def test_construct_pairwise_prompts_validates_id_present():
    with pytest.raises(ValueError):
        construct_pairwise_prompts([{"feature": 1}], task="borrower_level")


# ---------------------------------------------------------------------------
# fit_bradley_terry
# ---------------------------------------------------------------------------


def test_fit_bradley_terry_recovers_true_strengths():
    """Simulate pairwise wins from known strengths and verify recovery."""
    rng = np.random.default_rng(0)
    items = ["i1", "i2", "i3", "i4", "i5"]
    true_strength = pd.Series([2.0, 1.0, 0.0, -1.0, -2.0], index=items)
    rows = []
    n_per_pair = 200
    for ia, a in enumerate(items):
        for b in items[ia + 1:]:
            diff = true_strength[a] - true_strength[b]
            p_a_wins = 1.0 / (1.0 + np.exp(-diff))
            outcomes = rng.uniform(size=n_per_pair) < p_a_wins
            for win in outcomes:
                rows.append({"profile_a_id": a, "profile_b_id": b,
                              "winner_id": "A" if win else "B"})
    df = pd.DataFrame(rows)
    fitted = fit_bradley_terry(df)
    # both series centred to mean=0 → compare in that frame
    true_centred = true_strength - true_strength.mean()
    fitted_order = fitted.sort_values(ascending=False).index.tolist()
    true_order = true_centred.sort_values(ascending=False).index.tolist()
    assert fitted_order == true_order, fitted
    # absolute strength gap between extreme items should be close to truth
    assert abs((fitted["i1"] - fitted["i5"]) - (true_centred["i1"] - true_centred["i5"])) < 0.4


def test_fit_bradley_terry_empty_raises():
    with pytest.raises(ValueError):
        fit_bradley_terry(pd.DataFrame({"profile_a_id": [], "profile_b_id": [], "winner_id": []}))


# ---------------------------------------------------------------------------
# validate_bt_stability
# ---------------------------------------------------------------------------


def test_validate_bt_stability_flags_high_cv(cfg):
    # synthetic data where the items are nearly tied → high CV across resamples
    rng = np.random.default_rng(0)
    items = ["i1", "i2", "i3"]
    rows = []
    for _ in range(60):
        a, b = rng.choice(items, size=2, replace=False)
        rows.append({"profile_a_id": str(a), "profile_b_id": str(b),
                      "winner_id": rng.choice(["A", "B"])})
    df = pd.DataFrame(rows)
    out = validate_bt_stability(df, n_bootstrap=30, cv_threshold=0.1, seed=cfg["bootstrap"]["seed"])
    assert "is_stable" in out and "mean_cv" in out and "per_item_cv" in out


# ---------------------------------------------------------------------------
# bt_to_log_odds
# ---------------------------------------------------------------------------


def test_bt_to_log_odds_monotonic():
    bt = pd.Series([2.0, 1.0, 0.0, -1.0, -2.0], index=["a", "b", "c", "d", "e"])
    # linear calibration: percentile p → log-odds = 4*p - 2 ∈ [-2, 2]
    calibration = lambda p: 4.0 * p - 2.0
    out = bt_to_log_odds(bt, calibration)
    assert list(out.index) == list(bt.index)
    # higher BT strength → higher percentile → higher log-odds
    assert (out.sort_values(ascending=False).index.tolist() == ["a", "b", "c", "d", "e"])
