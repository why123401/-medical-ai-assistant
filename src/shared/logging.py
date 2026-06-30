"""Structured JSON logging for production-grade observability.

Replaces the old logger_handler.py with JSON-structured logs
that integrate with ELK/Loki/Grafana.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Attach exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)
        # Attach any extra fields
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
        return json.dumps(log_data, ensure_ascii=False)


def _ensure_log_dir() -> None:
    log_dir = Path(settings.log_full_path)
    log_dir.mkdir(parents=True, exist_ok=True)


def get_logger(
    name: str,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Create a structured logger with console + file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = JSONFormatter()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    _ensure_log_dir()
    log_file = Path(settings.log_full_path) / f"{name}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def log_with_data(logger: logging.Logger, level: int, msg: str, data: dict[str, Any] | None = None) -> None:
    """Helper to log a message with extra structured data."""
    record = logger.makeRecord(
        logger.name, level, "(unknown)", 0, msg, (), None
    )
    if data:
        record.extra_data = data
    logger.handle(record)
