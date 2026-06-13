"""Model serialisation + scoring helpers.

Persist a fitted sklearn-style pipeline with joblib, record metadata to
``model_metadata.json`` and to the SQLite ``model_versions`` registry, and
expose a thin ``score_borrowers`` wrapper around the loaded artefact.
"""

from __future__ import annotations

import json
import logging
import platform
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.models import _gini

logger = logging.getLogger(__name__)


def serialize_model(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    cfg: dict,
    db_path: str | None = None,
) -> Path:
    """Persist a fitted pipeline and register its OOT performance.

    Args:
        pipeline: Fitted classifier exposing ``predict_proba``.
        X_test: OOT feature matrix.
        y_test: OOT target.
        model_type: Identifier for the registry (e.g. ``'lr'``, ``'xgb'``,
            ``'posterior'``).
        cfg: Loaded config dict.
        db_path: Optional SQLite path. If provided, a row is inserted into
            ``model_versions`` and marked as the current version for this
            ``model_type``.

    Returns:
        Path to the saved ``.joblib`` artefact.
    """
    from sklearn.metrics import brier_score_loss

    project_root = Path(cfg["data"].get("project_root", "."))
    artefacts_dir = (project_root / cfg["data"]["artefacts_dir"]).resolve()
    artefacts_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artefact_path = artefacts_dir / f"{model_type}_pipeline_{ts}.joblib"
    joblib.dump(pipeline, artefact_path)

    oot_pred = pipeline.predict_proba(X_test)[:, 1]
    oot_gini = _gini(y_test, oot_pred)
    oot_brier = float(brier_score_loss(y_test, oot_pred))

    metadata = {
        "timestamp_utc": ts,
        "model_type": model_type,
        "artefact": str(artefact_path.relative_to(artefacts_dir.parent)),
        "n_test": int(len(y_test)),
        "test_default_rate": float(np.mean(np.asarray(y_test))),
        "oot_gini": oot_gini,
        "oot_brier": oot_brier,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "feature_count": _feature_count(pipeline),
    }
    meta_path = artefacts_dir / f"{model_type}_metadata_{ts}.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=float))

    if db_path is not None:
        _register_in_db(db_path, model_type, str(artefact_path), oot_gini, oot_brier, ts)

    logger.info("serialised %s -> %s (OOT Gini=%.4f, Brier=%.4f)",
                model_type, artefact_path.name, oot_gini, oot_brier)
    return artefact_path


def score_borrowers(df: pd.DataFrame, model_path: str) -> pd.Series:
    """Load a serialised pipeline and score every row of ``df``.

    Args:
        df: Feature matrix. Must contain the columns the underlying pipeline
            was fitted on.
        model_path: Path to the joblib artefact.

    Returns:
        ``pd.Series`` of predicted PDs, aligned with ``df.index``.
    """
    pipeline = joblib.load(model_path)
    preds = pipeline.predict_proba(df)[:, 1]
    return pd.Series(preds, index=df.index, name="predicted_pd")


# ---------------------------------------------------------------------------
# internal
# ---------------------------------------------------------------------------


def _feature_count(pipeline: Any) -> int | None:
    from sklearn.pipeline import Pipeline
    if not isinstance(pipeline, Pipeline):
        return None
    woe = pipeline.named_steps.get("woe")
    if woe is not None and hasattr(woe, "feature_names_in_"):
        return len(woe.feature_names_in_)
    return None


def _register_in_db(
    db_path: str,
    model_type: str,
    artefact_path: str,
    oot_gini: float,
    oot_brier: float,
    ts: str,
) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        # demote any prior current version of this model_type
        conn.execute(
            "UPDATE model_versions SET is_current = 0 WHERE model_type = ?",
            (model_type,),
        )
        conn.execute(
            "INSERT INTO model_versions (model_type, artefact_path, oot_gini, oot_brier, created_at, is_current) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (model_type, artefact_path, float(oot_gini), float(oot_brier), ts),
        )
        conn.commit()
