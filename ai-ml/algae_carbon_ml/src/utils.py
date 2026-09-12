"""
Utility helpers for the Algae Carbon 360 ML pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config import PROJECT_ROOT


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
