from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from app.job_context import JobContextFilter
from app.paths import LOG_DIR


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        LOG_DIR / "application.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s job=%(job_id)s %(name)s: %(message)s"
        )
    )
    handler.addFilter(JobContextFilter())
    root.addHandler(handler)
