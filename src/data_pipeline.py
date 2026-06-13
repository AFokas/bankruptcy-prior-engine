"""Polish Bankruptcy data loading and out-of-time splitting."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import arff
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_YEAR_FILE_RE = re.compile(r"^(\d+)year\.arff$", re.IGNORECASE)


def _read_arff(path: Path) -> pd.DataFrame:
    with path.open("r") as f:
        raw = arff.load(f)
    columns = [attr[0] for attr in raw["attributes"]]
    df = pd.DataFrame(raw["data"], columns=columns)
    # ARFF nominal class labels arrive as strings ('0' / '1'); missing values arrive as None.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_and_combine(raw_dir: str, target_col: str) -> pd.DataFrame:
    """Load all ``{n}year.arff`` files from ``raw_dir`` and concatenate.

    Args:
        raw_dir: Directory containing ``1year.arff`` … ``5year.arff``.
        target_col: Name to assign the bankruptcy class column.

    Returns:
        Combined DataFrame with one row per company-year, a ``year`` column
        (1-5), and the target column cast to int. Numeric features keep NaN
        for ARFF missing values.

    Raises:
        FileNotFoundError: If ``raw_dir`` contains no ARFF files matching the
            ``{n}year.arff`` pattern.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_path.resolve()}")

    frames: list[pd.DataFrame] = []
    for entry in sorted(raw_path.iterdir()):
        match = _YEAR_FILE_RE.match(entry.name)
        if not match:
            continue
        year = int(match.group(1))
        df = _read_arff(entry)
        # The ARFF target is the last attribute, conventionally named "class".
        original_target = df.columns[-1]
        df = df.rename(columns={original_target: target_col})
        df[target_col] = df[target_col].astype("Int64")
        df["year"] = year
        frames.append(df)
        logger.info("Loaded %s: %d rows, default rate=%.4f", entry.name, len(df), df[target_col].mean())

    if not frames:
        raise FileNotFoundError(f"No ARFF files matching '{{n}}year.arff' found in {raw_path.resolve()}")

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined[target_col] = combined[target_col].astype(int)
    logger.info(
        "Combined dataset: shape=%s, overall default rate=%.4f",
        combined.shape,
        combined[target_col].mean(),
    )
    return combined


def out_of_time_split(
    df: pd.DataFrame,
    date_col: str,
    train_years: list[int],
    test_years: list[int],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split a dataset by year into in-time train and out-of-time test sets.

    Args:
        df: Combined DataFrame produced by :func:`load_and_combine`.
        date_col: Column holding the year index (e.g. ``'year'``).
        train_years: Years to include in the training set.
        test_years: Years to include in the OOT test set.
        target_col: Name of the binary target column.

    Returns:
        ``(X_train, y_train, X_test, y_test)``. Feature matrices exclude both
        ``target_col`` and ``date_col``.

    Raises:
        AssertionError: If ``train_years`` and ``test_years`` overlap.
        ValueError: If either split is empty.
    """
    overlap = set(train_years) & set(test_years)
    assert not overlap, f"train_years and test_years must not overlap: {overlap}"

    train_mask = df[date_col].isin(train_years)
    test_mask = df[date_col].isin(test_years)

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty:
        raise ValueError(f"Train split is empty for years={train_years}")
    if test_df.empty:
        raise ValueError(f"Test split is empty for years={test_years}")

    feature_cols = [c for c in df.columns if c not in (target_col, date_col)]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)

    logger.info(
        "Train split: n=%d, default_rate=%.4f, years=%s",
        len(y_train),
        float(y_train.mean()),
        sorted(train_df[date_col].unique().tolist()),
    )
    logger.info(
        "Test  split: n=%d, default_rate=%.4f, years=%s",
        len(y_test),
        float(y_test.mean()),
        sorted(test_df[date_col].unique().tolist()),
    )

    return X_train, y_train, X_test, y_test
