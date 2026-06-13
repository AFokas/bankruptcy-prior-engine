"""Configuration loader.

Single source of truth for hyperparameters. The codebase must never hardcode
values that live in ``config.yaml`` — always import via ``load_config``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REQUIRED_TOP_LEVEL_KEYS = (
    "data",
    "woe",
    "models",
    "bootstrap",
    "psi",
    "llm",
    "bradley_terry",
    "popper",
    "analyst_sim",
    "api",
    "colours",
)

REQUIRED_NESTED_KEYS = {
    "data": ("raw_dir", "processed_dir", "target_col", "date_col", "train_years", "test_years"),
    "models": ("lr", "xgb"),
    "psi": ("stable_threshold", "monitor_threshold", "epsilon"),
    "bootstrap": ("n_resamples", "ci_level", "seed"),
}


def load_config(path: str = "config.yaml") -> dict:
    """Load and validate ``config.yaml``.

    Args:
        path: Path to the YAML config file, relative to the working directory
            or absolute.

    Returns:
        Parsed config as a dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If any required top-level or nested key is missing, or the
            YAML root is not a mapping.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg).__name__}")

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required top-level keys: {missing}")

    for parent, children in REQUIRED_NESTED_KEYS.items():
        parent_block = cfg[parent]
        if not isinstance(parent_block, dict):
            raise ValueError(f"Config key '{parent}' must be a mapping, got {type(parent_block).__name__}")
        missing_nested = [k for k in children if k not in parent_block]
        if missing_nested:
            raise ValueError(f"Config['{parent}'] missing required keys: {missing_nested}")

    logger.debug("Loaded config from %s", cfg_path)
    return cfg
