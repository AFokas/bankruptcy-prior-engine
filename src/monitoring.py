"""Population Stability Index monitoring.

PSI per feature against a fitted ``WoETransformer`` baseline, plus a
helper that scans every feature and writes the result to the SQLite
``psi_log`` table.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.woe_transformer import WoETransformer

logger = logging.getLogger(__name__)


def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    edges: np.ndarray,
    epsilon: float,
) -> float:
    """Population Stability Index for one feature.

    NaN values are dropped from both samples before bin assignment.
    ``edges`` is the ``WoETransformer.bin_edges_`` array (with ±inf endpoints),
    so out-of-range values in ``current`` are placed in the boundary bins
    instead of being dropped.

    Args:
        baseline: 1-d array of values from the training distribution.
        current: 1-d array of values from the new cohort.
        edges: Bin edges with ±inf endpoints (as produced by
            :class:`WoETransformer`).
        epsilon: Floor for per-bin proportions, to prevent ``log(0)``.

    Returns:
        PSI value as a float. ``>= 0``.
    """
    base = baseline[~np.isnan(baseline)]
    curr = current[~np.isnan(current)]
    if len(base) == 0 or len(curr) == 0:
        return 0.0

    interior = edges[1:-1] if len(edges) > 2 else np.array([])
    n_bins = max(len(edges) - 1, 1)

    if len(interior) == 0:
        # one giant bin → distributions are trivially identical → PSI = 0
        return 0.0

    base_bins = np.digitize(base, interior, right=True)
    curr_bins = np.digitize(curr, interior, right=True)

    base_dist = np.bincount(base_bins, minlength=n_bins).astype(float) / len(base)
    curr_dist = np.bincount(curr_bins, minlength=n_bins).astype(float) / len(curr)
    base_dist = np.clip(base_dist, epsilon, None)
    curr_dist = np.clip(curr_dist, epsilon, None)
    return float(np.sum((curr_dist - base_dist) * np.log(curr_dist / base_dist)))


def monitor_all_features(
    transformer: WoETransformer,
    X_train: pd.DataFrame,
    X_current: pd.DataFrame,
    cfg: dict,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """Compute PSI for every feature and (optionally) persist to SQLite.

    Args:
        transformer: Fitted ``WoETransformer`` whose ``bin_edges_`` define
            the baseline bins.
        X_train: Training feature matrix (the baseline distribution).
        X_current: Current cohort feature matrix.
        cfg: Loaded config dict (reads ``cfg['psi']``).
        db_path: Optional SQLite database path. If provided, every row is
            inserted into the ``psi_log`` table with a UTC timestamp.

    Returns:
        DataFrame with columns ``feature``, ``psi``, ``status``, sorted by
        ``psi`` descending.

    Raises:
        ValueError: If ``X_current`` contains features not in
            ``transformer.feature_names_in_``.
    """
    if not hasattr(transformer, "feature_names_in_"):
        raise RuntimeError("transformer has not been fit")
    extra = [c for c in X_current.columns if c not in transformer.feature_names_in_]
    if extra:
        raise ValueError(f"X_current has unexpected features: {extra}")

    epsilon = cfg["psi"]["epsilon"]
    stable = cfg["psi"]["stable_threshold"]
    monitor = cfg["psi"]["monitor_threshold"]

    rows = []
    for feat in transformer.feature_names_in_:
        if feat not in X_current.columns:
            logger.warning("Skipping %s: not present in current cohort", feat)
            continue
        base = X_train[feat].to_numpy(dtype=float, na_value=np.nan)
        curr = X_current[feat].to_numpy(dtype=float, na_value=np.nan)
        edges = transformer.bin_edges_[feat]
        psi = compute_psi(base, curr, edges, epsilon)
        rows.append({"feature": feat, "psi": psi, "status": _psi_status(psi, stable, monitor)})

    df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

    if db_path is not None:
        _persist_to_sqlite(df, db_path)
    return df


def _psi_status(psi: float, stable: float, monitor: float) -> str:
    if psi < stable:
        return "STABLE"
    if psi < monitor:
        return "MONITOR"
    return "RETRAIN"


def _persist_to_sqlite(df: pd.DataFrame, db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO psi_log (feature, psi_value, status, computed_at) VALUES (?, ?, ?, ?)",
            [(row["feature"], float(row["psi"]), row["status"], ts) for _, row in df.iterrows()],
        )
        conn.commit()
    logger.info("Wrote %d PSI rows to %s", len(df), db_path)
