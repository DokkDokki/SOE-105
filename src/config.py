"""Configuration loading.

All tunable behaviour lives in config.yaml so the team can change assumptions
without editing code. Paths are resolved relative to the project root - there
are no absolute paths anywhere in this project.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# project root = the folder containing this src/ package
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_UPLOADS = PROJECT_ROOT / "data" / "uploads"
OUTPUTS = PROJECT_ROOT / "outputs"
DOCS = PROJECT_ROOT / "docs"

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Config(dict):
    """Dict with attribute access, so cfg.fiscal['horizon'] reads naturally."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc


def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml. Falls back to the project default when path is None."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found: {p}\n"
            "Expected config.yaml in the project root."
        )
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)


def ensure_dirs() -> None:
    """Create the output folders on first run."""
    for d in (DATA_RAW, DATA_UPLOADS, OUTPUTS, DOCS):
        d.mkdir(parents=True, exist_ok=True)
