"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config(str(PROJECT_ROOT / "config.yaml"))
