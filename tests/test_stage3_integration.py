"""End-to-end smoke test: the Stage 3 src/ modules reproduce the nb01 numbers.

These are real-data tests against the Polish Bankruptcy dataset. They are
slow (~30s each) but only run if the raw data is present, so CI without
the dataset just skips them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARFF_FILES = [RAW_DIR / f"{i}year.arff" for i in range(1, 6)]


@pytest.fixture(scope="module")
def splits(cfg: dict):
    if not all(p.exists() for p in ARFF_FILES):
        pytest.skip("Polish Bankruptcy ARFF files not present in data/raw")
    from src.data_pipeline import load_and_combine, out_of_time_split
    df = load_and_combine(str(RAW_DIR), cfg["data"]["target_col"])
    X_tr, y_tr, X_te, y_te = out_of_time_split(
        df, cfg["data"]["date_col"],
        cfg["data"]["train_years"], cfg["data"]["test_years"],
        cfg["data"]["target_col"],
    )
    return X_tr, y_tr, X_te, y_te


# ---------------------------------------------------------------------------
# LR pipeline matches nb01 OOT Gini
# ---------------------------------------------------------------------------


def test_lr_pipeline_reproduces_nb01_oot_gini(cfg, splits):
    from src.models import build_lr_pipeline, evaluate_pipeline
    X_tr, y_tr, X_te, y_te = splits
    pipeline = build_lr_pipeline(cfg)
    result = evaluate_pipeline(pipeline, X_tr, y_tr, X_te, y_te, fit=True)
    # nb01 baseline: LR OOT Gini ≈ 0.764, Brier ≈ 0.127
    assert 0.74 < result["oot_gini"] < 0.79, result
    assert 0.10 < result["oot_brier"] < 0.15, result
    assert isinstance(result["coef_table"], pd.DataFrame)
    assert len(result["coef_table"]) > 0


# ---------------------------------------------------------------------------
# Calibrated XGB pipeline matches nb01 OOT numbers
# ---------------------------------------------------------------------------


def test_xgb_pipeline_reproduces_nb01_oot_gini(cfg, splits):
    from src.models import build_xgb_pipeline, calibrate_xgb, evaluate_pipeline
    X_tr, y_tr, X_te, y_te = splits
    n_pos = int(y_tr.sum()); n_neg = int(len(y_tr) - n_pos)
    base = build_xgb_pipeline(cfg, n_negative=n_neg, n_positive=n_pos)
    calibrated = calibrate_xgb(base, X_tr, y_tr, cfg)
    result = evaluate_pipeline(calibrated, X_tr, y_tr, X_te, y_te, fit=False)
    # nb01 tuned: XGB OOT Gini ≈ 0.910, Brier ≈ 0.026
    assert result["oot_gini"] > 0.88, result
    assert result["oot_brier"] < 0.05, result
    # best_iteration should reflect the tuned cap (n_estimators=2000), not the old 500-cap
    inner = base.named_steps["classifier"]
    assert inner.best_iteration > 1000, f"unexpected best_iteration={inner.best_iteration}"


# ---------------------------------------------------------------------------
# PSI monitoring runs end-to-end
# ---------------------------------------------------------------------------


def test_monitor_all_features_returns_sorted_psi(cfg, splits):
    from src.models import build_lr_pipeline
    from src.monitoring import monitor_all_features
    X_tr, y_tr, X_te, y_te = splits
    pipeline = build_lr_pipeline(cfg).fit(X_tr, y_tr)
    woe = pipeline.named_steps["woe"]
    psi_df = monitor_all_features(woe, X_tr, X_te, cfg, db_path=None)
    assert list(psi_df.columns) == ["feature", "psi", "status"]
    assert (psi_df["psi"].to_numpy()[:-1] >= psi_df["psi"].to_numpy()[1:]).all()
    assert set(psi_df["status"]).issubset({"STABLE", "MONITOR", "RETRAIN"})


# ---------------------------------------------------------------------------
# Bootstrap CI sanity
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_estimate(cfg):
    from sklearn.metrics import roc_auc_score
    from src.bootstrap import bootstrap_ci
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    n = 5000
    y_true = rng.integers(0, 2, size=n)
    y_pred = np.clip(0.5 + 0.4 * (y_true * 2 - 1) + rng.normal(0, 0.2, size=n), 1e-6, 1 - 1e-6)
    ci = bootstrap_ci(y_true, y_pred, roc_auc_score, n_resamples=200,
                      ci_level=cfg["bootstrap"]["ci_level"], seed=cfg["bootstrap"]["seed"])
    assert ci["lower"] <= ci["estimate"] <= ci["upper"]
    assert ci["upper"] - ci["lower"] < 0.05  # tight on n=5000


def test_bootstrap_convergence_ci_shape(cfg):
    from src.bootstrap import bootstrap_convergence_ci
    responses = [0.1, 0.2, 0.15, 0.18, 0.22, 0.19, 0.17, 0.21]
    lo, hi = bootstrap_convergence_ci(
        responses, n_resamples=200,
        ci_level=cfg["bootstrap"]["ci_level"], seed=cfg["bootstrap"]["seed"],
    )
    assert lo.shape == hi.shape == (len(responses),)
    assert (lo <= hi).all()
    # CI band narrows as k grows (more samples averaged → less variance)
    assert (hi[-1] - lo[-1]) < (hi[0] - lo[0])


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_serialize_and_score(cfg, splits, tmp_path):
    from src.models import build_lr_pipeline
    from src.serialisation import serialize_model, score_borrowers
    X_tr, y_tr, X_te, y_te = splits
    pipeline = build_lr_pipeline(cfg).fit(X_tr, y_tr)
    # redirect artefacts_dir into tmp via a shallow cfg override
    cfg_local = {**cfg, "data": {**cfg["data"], "artefacts_dir": str(tmp_path),
                                   "project_root": str(PROJECT_ROOT)}}
    # stratified test sample so both classes are present for the OOT-metric snapshot
    y_arr = np.asarray(y_te)
    neg_idx = np.where(y_arr == 0)[0][:250]
    pos_idx = np.where(y_arr == 1)[0][:250]
    sample_idx = np.sort(np.concatenate([neg_idx, pos_idx]))
    X_sample = X_te.iloc[sample_idx]
    y_sample = y_te.iloc[sample_idx]
    path = serialize_model(pipeline, X_sample, y_sample, model_type="lr",
                           cfg=cfg_local, db_path=None)
    assert path.exists()
    scores = score_borrowers(X_te.head(50), str(path))
    assert scores.shape == (50,)
    assert ((scores >= 0) & (scores <= 1)).all()
