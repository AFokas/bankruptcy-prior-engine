"""Logistic regression + calibrated XGBoost pipeline builders.

Stage 3 production refactor of the inline fitting code in
``notebooks/01_data_foundation.ipynb``. All hyperparameters live in
``config.yaml``; nothing here is hardcoded except the choice of
``IsotonicRegression`` as the calibrator (driven by config but
sklearn-version-pinned).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.woe_transformer import WoETransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public pipeline builders
# ---------------------------------------------------------------------------


def build_lr_pipeline(cfg: dict) -> Pipeline:
    """Construct an unfitted LR pipeline: WoE → StandardScaler → LogisticRegression.

    All hyperparameters read from ``cfg['woe']`` and ``cfg['models']['lr']``.

    Args:
        cfg: Loaded config dict.

    Returns:
        Unfitted ``sklearn.pipeline.Pipeline``.
    """
    lr_cfg = cfg["models"]["lr"]
    woe_cfg = cfg["woe"]
    seed = cfg["bootstrap"]["seed"]
    return Pipeline([
        ("woe", WoETransformer(
            n_bins=woe_cfg["n_bins"],
            laplace_smoothing=woe_cfg["laplace_smoothing"],
        )),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=lr_cfg["C"],
            solver=lr_cfg["solver"],
            max_iter=lr_cfg["max_iter"],
            class_weight=lr_cfg["class_weight"],
            random_state=seed,
        )),
    ])


def build_xgb_pipeline(cfg: dict, n_negative: int, n_positive: int) -> Pipeline:
    """Construct an unfitted XGBoost pipeline: StandardScaler → XGBClassifier.

    ``scale_pos_weight`` is computed from the supplied class counts.
    Early stopping is configured on the classifier but requires an
    ``eval_set`` at fit time, which the caller is responsible for supplying.

    Args:
        cfg: Loaded config dict.
        n_negative: Number of negative-class rows in the training set used to
            compute ``scale_pos_weight``.
        n_positive: Number of positive-class rows (same source).

    Returns:
        Unfitted ``sklearn.pipeline.Pipeline``.
    """
    xgb_cfg = cfg["models"]["xgb"]
    seed = cfg["bootstrap"]["seed"]
    if n_positive < 1:
        raise ValueError("n_positive must be >= 1 to compute scale_pos_weight")
    scale_pos_weight = float(n_negative / n_positive)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", xgb.XGBClassifier(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )),
    ])


# ---------------------------------------------------------------------------
# XGBoost calibration
# ---------------------------------------------------------------------------


class CalibratedXGBPipeline(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper: fitted XGB pipeline + isotonic calibrator.

    Exposes ``predict_proba`` / ``predict`` so the rest of the codebase can
    treat the calibrated XGB exactly like the LR pipeline.
    """

    def __init__(self, base_pipeline: Pipeline, calibrator: IsotonicRegression):
        self.base_pipeline = base_pipeline
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.base_pipeline.predict_proba(X)[:, 1]
        calibrated = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def calibrate_xgb(
    base_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict,
) -> CalibratedXGBPipeline:
    """Fit a calibrated XGBoost pipeline using a stratified train holdout.

    Implementation follows the spec in ``CLAUDE.md`` and the Stage 2 notebook:

    1. Hold out ``cfg.models.xgb.calibration_val_frac`` of the training set
       (stratified) for the calibrator.
    2. Inside the remaining "modelfit" rows, carve off another fraction of the
       same size for XGB's early-stopping ``eval_set``.
    3. Fit the base pipeline on the inner fit fold.
    4. Fit an isotonic calibrator on the holdout's predicted probabilities.

    Args:
        base_pipeline: Output of :func:`build_xgb_pipeline`. Will be fitted
            in place.
        X_train: Training feature matrix.
        y_train: Training target.
        cfg: Loaded config dict.

    Returns:
        ``CalibratedXGBPipeline`` ready for ``predict_proba``.
    """
    xgb_cfg = cfg["models"]["xgb"]
    seed = cfg["bootstrap"]["seed"]
    val_frac = xgb_cfg["calibration_val_frac"]

    y_arr = np.asarray(y_train)
    n_rows = np.arange(len(X_train))
    mf_idx, cal_idx = train_test_split(
        n_rows, test_size=val_frac, random_state=seed, stratify=y_arr,
    )
    X_mf, y_mf = X_train.iloc[mf_idx], y_arr[mf_idx]
    X_cal, y_cal = X_train.iloc[cal_idx], y_arr[cal_idx]

    fit_idx, es_idx = train_test_split(
        np.arange(len(X_mf)), test_size=val_frac, random_state=seed, stratify=y_mf,
    )
    X_fit, y_fit = X_mf.iloc[fit_idx], y_mf[fit_idx]
    X_es, y_es = X_mf.iloc[es_idx], y_mf[es_idx]

    # The XGBClassifier in the pipeline needs the eval_set in transformed space.
    # We pre-fit the upstream transformers manually, transform the eval set,
    # then fit the classifier with the transformed eval set.
    scaler = base_pipeline.named_steps["scaler"]
    classifier = base_pipeline.named_steps["classifier"]
    scaler.fit(X_fit)
    X_fit_s = scaler.transform(X_fit)
    X_es_s = scaler.transform(X_es)
    classifier.fit(X_fit_s, y_fit, eval_set=[(X_es_s, y_es)], verbose=False)

    logger.info(
        "XGB calibration: fit n=%d, es n=%d, calib n=%d, best_iteration=%s",
        len(X_fit), len(X_es), len(X_cal),
        getattr(classifier, "best_iteration", "?"),
    )

    raw_cal_pred = base_pipeline.predict_proba(X_cal)[:, 1]
    method = xgb_cfg["calibration_method"]
    if method != "isotonic":
        raise NotImplementedError(f"calibration_method='{method}' not supported; use 'isotonic'")
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal_pred, y_cal)
    return CalibratedXGBPipeline(base_pipeline=base_pipeline, calibrator=iso)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_pipeline(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fit: bool = True,
) -> dict:
    """Fit (optionally) and evaluate a pipeline on in-time and OOT data.

    Args:
        pipeline: Sklearn-compatible classifier. Anything with
            ``predict_proba``. If ``fit=False``, must already be fitted.
        X_train, y_train: Training data.
        X_test, y_test: OOT data.
        fit: If True, call ``pipeline.fit(X_train, y_train)`` first.

    Returns:
        Dict with keys ``in_time_gini``, ``oot_gini``, ``in_time_brier``,
        ``oot_brier``, ``calibration_curve_data`` (tuple of arrays for OOT),
        and ``coef_table`` (only populated for LR pipelines).
    """
    if fit:
        pipeline.fit(X_train, y_train)
    in_pred = pipeline.predict_proba(X_train)[:, 1]
    oot_pred = pipeline.predict_proba(X_test)[:, 1]

    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_test, oot_pred, n_bins=10, strategy="quantile")

    result: dict[str, Any] = {
        "in_time_gini": _gini(y_train, in_pred),
        "oot_gini": _gini(y_test, oot_pred),
        "in_time_brier": float(brier_score_loss(y_train, in_pred)),
        "oot_brier": float(brier_score_loss(y_test, oot_pred)),
        "calibration_curve_data": {"fraction_of_positives": frac_pos.tolist(),
                                    "mean_predicted": mean_pred.tolist()},
        "coef_table": _extract_coef_table(pipeline),
    }
    return result


def _gini(y_true: Any, y_pred: np.ndarray) -> float:
    return float(2.0 * roc_auc_score(y_true, y_pred) - 1.0)


def _extract_coef_table(pipeline: Any) -> pd.DataFrame | None:
    """Return per-feature coefficient table if ``pipeline`` is LR; else None."""
    if not isinstance(pipeline, Pipeline):
        return None
    classifier = pipeline.named_steps.get("classifier")
    if not isinstance(classifier, LogisticRegression):
        return None
    woe = pipeline.named_steps.get("woe")
    if woe is None or not hasattr(woe, "feature_names_in_"):
        return None
    return pd.DataFrame({
        "feature": woe.feature_names_in_,
        "coef": classifier.coef_.ravel(),
    }).sort_values("coef", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
