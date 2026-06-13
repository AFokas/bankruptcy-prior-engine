"""Five mandatory unit tests for ``WoETransformer``.

The Stage 3 spec requires every test below to pass before any other Stage 3
modelling code (LR / XGB / serialisation) is built or tested.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.woe_transformer import WoETransformer


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_xy(cfg: dict):
    """Synthetic two-feature dataset with controllable signal and missingness."""
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    n = 2000
    feat_a = rng.normal(size=n)
    feat_b = rng.uniform(-10, 10, size=n)
    # bake in a default rate that rises with feat_a so WoE has something to learn
    logits = -2.5 + 1.2 * feat_a
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    # sprinkle missingness
    miss_idx = rng.choice(n, size=n // 20, replace=False)
    feat_a[miss_idx] = np.nan
    X = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b})
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# 1. out-of-range values do not produce NaN
# ---------------------------------------------------------------------------

def test_no_nan_on_out_of_range(small_xy, cfg):
    X, y = small_xy
    transformer = WoETransformer(
        n_bins=cfg["woe"]["n_bins"],
        laplace_smoothing=cfg["woe"]["laplace_smoothing"],
    )
    transformer.fit(X, y)

    extreme_high = float(X["feat_a"].max()) * 1e6 + 1e9
    extreme_low = float(X["feat_a"].min()) * 1e6 - 1e9
    X_extreme = pd.DataFrame({
        "feat_a": [extreme_high, extreme_low, 0.0],
        "feat_b": [extreme_high, extreme_low, 0.0],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning here is a failure
        out = transformer.transform(X_extreme)
    assert not out.isna().any().any(), "out-of-range values produced NaN"
    assert np.isfinite(out.to_numpy()).all(), "out-of-range values produced inf"


# ---------------------------------------------------------------------------
# 2. NaN inputs produce valid WoE output
# ---------------------------------------------------------------------------

def test_missing_values_handled(small_xy, cfg):
    X, y = small_xy
    transformer = WoETransformer(
        n_bins=cfg["woe"]["n_bins"],
        laplace_smoothing=cfg["woe"]["laplace_smoothing"],
    ).fit(X, y)

    X_missing = pd.DataFrame({"feat_a": [np.nan, np.nan, 0.0], "feat_b": [np.nan, 1.0, np.nan]})
    out = transformer.transform(X_missing)
    assert not out.isna().any().any(), "NaN inputs produced NaN output"
    assert np.isfinite(out.to_numpy()).all()


# ---------------------------------------------------------------------------
# 3. Unseen / impossible bin → UserWarning + WoE=0
# ---------------------------------------------------------------------------

def test_unseen_category_warns_and_returns_zero(small_xy, cfg):
    """Force a transform path where a feature's WoE map is missing entries.

    For numeric features ``np.digitize`` cannot return a bin index outside
    [0, n_bins-1], so to exercise the "unseen bin" code path we mutate the
    fitted ``woe_maps_`` to remove one bin and confirm the transformer
    falls back to WoE=0 with a UserWarning.
    """
    X, y = small_xy
    transformer = WoETransformer(
        n_bins=cfg["woe"]["n_bins"],
        laplace_smoothing=cfg["woe"]["laplace_smoothing"],
    ).fit(X, y)

    # corrupt the WoE map: drop bin 0 from feat_b
    dropped_bin = 0
    transformer.woe_maps_["feat_b"] = {
        k: v for k, v in transformer.woe_maps_["feat_b"].items() if k != dropped_bin
    }

    # craft a sample that lands in bin 0
    edges = transformer.bin_edges_["feat_b"]
    low_value = edges[1] - 1.0  # below the first interior edge → bin 0
    X_probe = pd.DataFrame({"feat_a": [0.0], "feat_b": [low_value]})

    with pytest.warns(UserWarning, match="bin not seen during fit"):
        out = transformer.transform(X_probe)
    assert out.loc[0, "feat_b"] == 0.0, "unseen-bin sample did not map to WoE=0"


# ---------------------------------------------------------------------------
# 4. IV summary is sorted descending
# ---------------------------------------------------------------------------

def test_iv_summary_sorted_descending(small_xy, cfg):
    X, y = small_xy
    transformer = WoETransformer(
        n_bins=cfg["woe"]["n_bins"],
        laplace_smoothing=cfg["woe"]["laplace_smoothing"],
    ).fit(X, y)
    summary = transformer.get_iv_summary()
    assert list(summary.columns) == ["feature", "iv"]
    ivs = summary["iv"].to_numpy()
    assert np.all(ivs[:-1] >= ivs[1:]), "IV summary not sorted descending"
    # the engineered feat_a should clearly beat the noise feat_b
    assert summary.iloc[0]["feature"] == "feat_a"


# ---------------------------------------------------------------------------
# 5. all-null column does not crash
# ---------------------------------------------------------------------------

def test_all_null_column_does_not_crash(cfg):
    X = pd.DataFrame({
        "all_null": [np.nan] * 500,
        "good_feat": np.random.RandomState(0).normal(size=500),
    })
    y = pd.Series((np.random.RandomState(0).uniform(size=500) < 0.1).astype(int))
    transformer = WoETransformer(
        n_bins=cfg["woe"]["n_bins"],
        laplace_smoothing=cfg["woe"]["laplace_smoothing"],
    )
    transformer.fit(X, y)
    out = transformer.transform(X)
    assert not out.isna().any().any()
    assert transformer.iv_["all_null"] == 0.0
