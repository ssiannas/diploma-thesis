"""Centralized logging setup for the postprocessing pipeline."""

import logging
import sys
from typing import Optional


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure root logger: stderr + optional file handler.

    Idempotent: clears existing handlers before adding new ones, so calling
    this multiple times (e.g. in tests) does not produce duplicate output.
    """
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        logging.getLogger(__name__).info(f"Logging to {log_file}")
