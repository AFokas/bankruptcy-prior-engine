"""Pairwise comparison + Bradley-Terry fitting for Stages 2 (LLM) and 3 (analyst sim).

The Bradley-Terry model assigns each item a latent strength ``s_i`` and
predicts that item ``i`` beats item ``j`` with probability
``sigmoid(s_i - s_j)``. Fitting maximises the binomial log-likelihood of
the observed comparison wins.

Two task types are supported via :func:`construct_pairwise_prompts`:

- ``'borrower_level'`` — holistic "which company is more likely to
  default?" prompts, used by the LLN convergence experiment (notebook 02).
- ``'category_level'`` — "which company has worse {category}?", used by
  the analyst simulation (notebook 04).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.ollama_client import infer_model_family, query_ollama

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior credit risk analyst. You will be shown two anonymised "
    "company profiles labelled A and B. Choose which company is MORE LIKELY "
    "to default (borrower_level) or which has the WORSE category outcome "
    "(category_level). Reply with a single character: A or B. Do not explain."
)

_RESPONSE_RE = re.compile(r"\b([AB])\b")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def construct_pairwise_prompts(
    profiles: list[dict],
    task: str,
    category: str = "",
) -> list[tuple[str, str, str]]:
    """Build pairwise prompts for every distinct ordered pair (i < j).

    Args:
        profiles: List of profile dicts. Each must contain an ``'id'`` key.
            All other keys are rendered as JSON into the prompt body.
        task: Either ``'borrower_level'`` or ``'category_level'``.
        category: Feature category for ``'category_level'`` (e.g.
            ``'financial_health'``). Ignored for borrower-level prompts.

    Returns:
        List of ``(profile_a_id, profile_b_id, prompt)`` triples.

    Raises:
        ValueError: If ``task`` is unrecognised or profiles lack ``id``.
    """
    if task not in {"borrower_level", "category_level"}:
        raise ValueError(f"unknown task '{task}'; expected borrower_level or category_level")
    if task == "category_level" and not category:
        raise ValueError("category_level prompts require a non-empty `category`")
    for p in profiles:
        if "id" not in p:
            raise ValueError(f"profile missing 'id' key: {p}")

    prompts: list[tuple[str, str, str]] = []
    for prof_a, prof_b in combinations(profiles, 2):
        body_a = json.dumps({k: v for k, v in prof_a.items() if k != "id"}, default=_to_jsonable)
        body_b = json.dumps({k: v for k, v in prof_b.items() if k != "id"}, default=_to_jsonable)
        if task == "borrower_level":
            instruction = "Which company is MORE LIKELY to default? Reply with a single character: A or B."
        else:
            instruction = (
                f"Which company has WORSE {category.replace('_', ' ')}? "
                "Reply with a single character: A or B."
            )
        prompt = f"COMPANY A:\n{body_a}\n\nCOMPANY B:\n{body_b}\n\n{instruction}"
        prompts.append((str(prof_a["id"]), str(prof_b["id"]), prompt))
    return prompts


def _to_jsonable(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return str(v)


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------


def parse_winner(raw: str) -> Optional[str]:
    """Extract 'A' or 'B' from a model response. Returns None on ambiguity."""
    if not raw:
        return None
    text = raw.strip().upper()
    if not text:
        return None
    # prefer the first standalone A/B token
    match = _RESPONSE_RE.search(text)
    if match is not None:
        return match.group(1)
    # fallback: first character
    if text[0] in {"A", "B"}:
        return text[0]
    return None


def run_pairwise_experiment(
    models: list[str],
    profiles: list[dict],
    task: str,
    category: str,
    cfg: dict,
    db_path: Optional[str] = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> pd.DataFrame:
    """Run every model against every pairwise prompt and persist results.

    Args:
        models: Ollama model names. The function aborts a single model on
            connection / response errors and records the failure rather than
            re-raising.
        profiles: Profile dicts (see :func:`construct_pairwise_prompts`).
        task: ``'borrower_level'`` or ``'category_level'``.
        category: Required iff ``task == 'category_level'``.
        cfg: Loaded config dict (uses ``cfg['llm']``).
        db_path: Optional SQLite path. Rows are written to
            ``borrower_comparisons`` or ``category_comparisons`` depending
            on ``task``.
        progress: Optional callback ``(model_name, done, total)`` invoked
            after each query — convenient for notebook progress bars.

    Returns:
        DataFrame with one row per (model, pair) with columns
        ``model_name, model_family, profile_a_id, profile_b_id, winner_id,
        raw_response, query_timestamp`` (and ``category`` for category-level).
    """
    base_url = cfg["llm"]["ollama_base_url"]
    temperature = cfg["llm"]["temperature"]

    prompts = construct_pairwise_prompts(profiles, task=task, category=category)
    rows: list[dict] = []
    total = len(models) * len(prompts)
    done = 0

    for model in models:
        family = infer_model_family(model)
        model_rows: list[dict] = []
        for (a_id, b_id, prompt) in prompts:
            try:
                raw = query_ollama(model, SYSTEM_PROMPT, prompt, temperature=temperature, base_url=base_url)
                winner = parse_winner(raw)
            except Exception as exc:  # noqa: BLE001 — we want any failure to be a missing data point
                raw = f"<error: {exc}>"
                winner = None
                logger.warning("%s failed on (%s, %s): %s", model, a_id, b_id, exc)
            model_rows.append({
                "model_name": model,
                "model_family": family,
                "profile_a_id": a_id,
                "profile_b_id": b_id,
                "winner_id": winner,
                "raw_response": raw,
                "query_timestamp": datetime.now(timezone.utc).isoformat(),
                "category": category if task == "category_level" else None,
            })
            done += 1
            if progress is not None:
                progress(model, done, total)
        # incremental persistence — flush this model's rows before moving on,
        # so a slow/failing later model can't lose what we already have
        if db_path is not None and model_rows:
            _persist_comparisons(pd.DataFrame(model_rows), task=task, db_path=db_path)
        rows.extend(model_rows)

    return pd.DataFrame(rows)


def _persist_comparisons(df: pd.DataFrame, task: str, db_path: str) -> None:
    table = "borrower_comparisons" if task == "borrower_level" else "category_comparisons"
    valid = df[df["winner_id"].isin(["A", "B"])]
    if valid.empty:
        return
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        if task == "borrower_level":
            conn.executemany(
                f"INSERT INTO {table} (model_name, model_family, profile_a_id, profile_b_id, "
                "winner_id, raw_response, query_timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (r.model_name, r.model_family, r.profile_a_id, r.profile_b_id,
                     r.winner_id, r.raw_response, r.query_timestamp)
                    for r in valid.itertuples(index=False)
                ],
            )
        else:
            conn.executemany(
                f"INSERT INTO {table} (analyst_id, company_a_id, company_b_id, category, "
                "winner_id, review_date, weight) VALUES (?, ?, ?, ?, ?, ?, 1.0)",
                [
                    (r.model_name, r.profile_a_id, r.profile_b_id, r.category,
                     r.winner_id, r.query_timestamp)
                    for r in valid.itertuples(index=False)
                ],
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Bradley-Terry MLE
# ---------------------------------------------------------------------------


def fit_bradley_terry(
    comparisons: pd.DataFrame,
    item_col_a: str = "profile_a_id",
    item_col_b: str = "profile_b_id",
    winner_col: str = "winner_id",
) -> pd.Series:
    """Fit Bradley-Terry latent strengths via MLE.

    Args:
        comparisons: DataFrame with one row per pairwise comparison.
            ``winner_col`` must contain ``'A'`` or ``'B'``.
        item_col_a: Column holding the item-A identifier.
        item_col_b: Column holding the item-B identifier.
        winner_col: Column holding the winner indicator.

    Returns:
        ``pd.Series`` indexed by item id, holding fitted latent strengths.
        Strengths are centred so the mean is 0 (the model is identifiable
        only up to an additive constant).

    Raises:
        ValueError: If no valid comparisons remain after filtering.
    """
    valid = comparisons[comparisons[winner_col].isin(["A", "B"])].copy()
    if valid.empty:
        raise ValueError("no valid A/B comparisons to fit Bradley-Terry on")

    items = sorted(set(valid[item_col_a]) | set(valid[item_col_b]))
    idx = {item: i for i, item in enumerate(items)}
    n = len(items)
    a_idx = valid[item_col_a].map(idx).to_numpy(dtype=int)
    b_idx = valid[item_col_b].map(idx).to_numpy(dtype=int)
    a_wins = (valid[winner_col] == "A").to_numpy(dtype=int)

    # MLE on identifiable parameters: fix strengths[0] = 0, optimise the rest.
    def neg_log_lik(theta_free):
        theta = np.concatenate([[0.0], theta_free])
        diff = theta[a_idx] - theta[b_idx]
        # log(sigmoid(diff)) = -log(1 + exp(-diff))
        log_p_a_wins = -np.log1p(np.exp(-diff))
        log_p_b_wins = -np.log1p(np.exp(diff))
        return -float(np.sum(a_wins * log_p_a_wins + (1 - a_wins) * log_p_b_wins))

    x0 = np.zeros(n - 1)
    res = minimize(neg_log_lik, x0, method="BFGS")
    full = np.concatenate([[0.0], res.x])
    centred = full - full.mean()
    return pd.Series(centred, index=items, name="bt_strength")


def validate_bt_stability(
    comparisons: pd.DataFrame,
    n_bootstrap: int,
    cv_threshold: float,
    item_col_a: str = "profile_a_id",
    item_col_b: str = "profile_b_id",
    winner_col: str = "winner_id",
    seed: int = 42,
) -> dict:
    """Bootstrap BT strengths and check whether items have converged.

    Resamples comparison rows with replacement ``n_bootstrap`` times, refits
    Bradley-Terry, and computes the coefficient of variation per item.

    Args:
        comparisons: DataFrame of pairwise comparisons.
        n_bootstrap: Number of resamples.
        cv_threshold: If the mean cross-item CV exceeds this, flag as
            unstable.
        item_col_a, item_col_b, winner_col: Column names.
        seed: RNG seed.

    Returns:
        Dict with ``is_stable`` (bool), ``mean_cv`` (float), and
        ``per_item_cv`` (pd.Series).
    """
    rng = np.random.default_rng(seed)
    n = len(comparisons)
    matrices: list[pd.Series] = []
    items_seen: set = set()
    for _ in range(n_bootstrap):
        sample = comparisons.iloc[rng.integers(0, n, size=n)]
        try:
            bt = fit_bradley_terry(sample, item_col_a, item_col_b, winner_col)
        except ValueError:
            continue
        matrices.append(bt)
        items_seen.update(bt.index)

    if not matrices:
        warnings.warn("Bradley-Terry validation produced no valid resamples", RuntimeWarning, stacklevel=2)
        return {"is_stable": False, "mean_cv": float("nan"), "per_item_cv": pd.Series(dtype=float)}

    stacked = pd.concat(matrices, axis=1).fillna(0.0)
    mean = stacked.mean(axis=1)
    std = stacked.std(axis=1)
    # CV with safe denominator
    cv = (std / mean.abs().clip(lower=1e-3))
    mean_cv = float(cv.mean())
    is_stable = mean_cv <= cv_threshold
    if not is_stable:
        warnings.warn(
            f"Bradley-Terry strengths unstable across bootstrap: mean CV={mean_cv:.3f} > {cv_threshold}",
            RuntimeWarning, stacklevel=2,
        )
    return {"is_stable": is_stable, "mean_cv": mean_cv, "per_item_cv": cv}


# ---------------------------------------------------------------------------
# Strength → log-odds calibration
# ---------------------------------------------------------------------------


def bt_to_log_odds(
    bt_scores: pd.Series,
    calibration_curve: Callable[[np.ndarray], np.ndarray],
) -> pd.Series:
    """Map Bradley-Terry strengths through a calibration curve to log-odds.

    Args:
        bt_scores: Series indexed by item id.
        calibration_curve: Callable mapping a 1-d array of BT percentiles in
            ``[0, 1]`` to a 1-d array of log-odds. Typically fit elsewhere
            against empirical default rates by BT band (NOT against LR
            predicted probabilities — that would create circular dependency).

    Returns:
        Series of log-odds, same index as ``bt_scores``.
    """
    ranks = bt_scores.rank(method="average").to_numpy(dtype=float)
    percentiles = (ranks - 0.5) / len(bt_scores)
    log_odds = calibration_curve(percentiles)
    return pd.Series(log_odds, index=bt_scores.index, name="log_odds")
