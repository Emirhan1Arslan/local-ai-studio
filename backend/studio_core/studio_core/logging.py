"""structlog tabanlı yapılandırılmış loglama + dosya rotasyonu.

Loglar hem konsola (dev: renkli) hem ~/LocalAIStudio/logs/studio.log dosyasına
JSON-lines olarak yazılır. WS Console paneli aynı olayları `log` event'i olarak alır.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys

import structlog

from studio_core.config import get_settings


def setup_logging(level: str | None = None) -> None:
    settings = get_settings()
    settings.ensure_dirs()
    log_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)

    file_handler = logging.handlers.RotatingFileHandler(
        settings.logs_dir / "studio.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(level=log_level, handlers=[file_handler, console_handler], force=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
