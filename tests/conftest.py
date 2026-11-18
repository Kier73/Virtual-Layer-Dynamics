"""Pytest configuration ensuring subprocess crash-recovery workers inherit `src/` on PYTHONPATH."""

from __future__ import annotations

import os
from pathlib import Path
import pytest


@pytest.fixture(scope="session", autouse=True)
def _propagate_src_pythonpath() -> None:
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if src_dir not in parts:
        os.environ["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing}" if existing else src_dir
