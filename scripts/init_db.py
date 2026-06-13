"""Initialise the SQLite database used to log comparisons, BT ratings, PSI, and POPPER runs.

Idempotent — every CREATE statement uses ``IF NOT EXISTS``.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS borrower_comparisons (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name      TEXT NOT NULL,
        model_family    TEXT NOT NULL,
        profile_a_id    TEXT NOT NULL,
        profile_b_id    TEXT NOT NULL,
        winner_id       TEXT NOT NULL,
        raw_response    TEXT,
        query_timestamp TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS category_comparisons (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        analyst_id      TEXT NOT NULL,
        company_a_id    TEXT NOT NULL,
        company_b_id    TEXT NOT NULL,
        category        TEXT NOT NULL,
        winner_id       TEXT NOT NULL,
        review_date     TEXT NOT NULL,
        weight          REAL DEFAULT 1.0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bt_ratings (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        source          TEXT NOT NULL,
        source_id       TEXT NOT NULL,
        category        TEXT NOT NULL,
        company_id      TEXT NOT NULL,
        bt_score        REAL NOT NULL,
        fitted_at       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS popper_experiments (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        round           INTEGER NOT NULL,
        model_name      TEXT NOT NULL,
        experiment_name TEXT NOT NULL,
        null_hypothesis TEXT NOT NULL,
        statistical_test TEXT NOT NULL,
        p_value         REAL NOT NULL,
        e_value         REAL NOT NULL,
        cumulative_e    REAL NOT NULL,
        decision        TEXT NOT NULL,
        run_timestamp   TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS psi_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        feature         TEXT NOT NULL,
        psi_value       REAL NOT NULL,
        status          TEXT NOT NULL,
        computed_at     TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_versions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        model_type      TEXT NOT NULL,
        artefact_path   TEXT NOT NULL,
        oot_gini        REAL,
        oot_brier       REAL,
        created_at      TEXT NOT NULL,
        is_current      INTEGER DEFAULT 0
    )
    """,
]


def init_database(db_path: str) -> None:
    """Create all tables in the SQLite database. Idempotent."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        for stmt in SCHEMA:
            conn.execute(stmt)
        conn.commit()


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    db_path = PROJECT_ROOT / cfg["data"]["db_path"]
    init_database(str(db_path))

    with sqlite3.connect(db_path) as conn:
        tables = sorted(
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        )
    print(f"Initialised database at {db_path}")
    print("Tables:")
    for t in tables:
        print(f"  - {t}")


if __name__ == "__main__":
    main()
