import logging
import os
import sys
from pathlib import Path

# logger.py
# Lightweight, configurable logger for applications.
# Usage:
#   from logger import get_logger
#   logger = get_logger(__name__)
#   logger.info("hello")

import logging.handlers

# Environment-configurable defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_FILE = os.getenv("LOG_FILE", "app.log")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
LOG_CONSOLE = os.getenv("LOG_CONSOLE", "1") != "0"  # set to "0" to disable console logging
AUTO_CONFIGURE = os.getenv("DISABLE_AUTO_LOG_CONFIG", "") == ""  # set to any value to disable auto-config

DEFAULT_FORMAT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s %(levelname)s [%(name)s] pid=%(process)d thread=%(threadName)s %(message)s",
)
DATE_FORMAT = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%dT%H:%M:%S%z")


def _ensure_log_dir():
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If creating the directory fails, fall back to current working dir for file handler.
        return Path.cwd()
    return LOG_DIR


def _configure_root_logger():
    root = logging.getLogger()
    # Avoid double configuration: check for handler names we set
    existing_names = {getattr(h, "name", None) for h in root.handlers}

    level = getattr(logging, LOG_LEVEL, logging.DEBUG)
    root.setLevel(level)

    formatter = logging.Formatter(DEFAULT_FORMAT, DATE_FORMAT)

    # Rotating file handler
    file_handler_name = "rotating_file_handler"
    if file_handler_name not in existing_names:
        try:
            log_dir = _ensure_log_dir()
            file_path = log_dir / LOG_FILE
            fh = logging.handlers.RotatingFileHandler(
                filename=str(file_path),
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
                delay=True,
            )
            fh.setLevel(level)
            fh.setFormatter(formatter)
            fh.name = file_handler_name
            root.addHandler(fh)
        except Exception:
            # If file handler cannot be created, ensure at least console logging exists.
            pass

    # Console handler
    console_handler_name = "console_handler"
    if LOG_CONSOLE and console_handler_name not in existing_names:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        ch.name = console_handler_name
        root.addHandler(ch)


def configure_logging(force: bool = False):
    """
    Configure the root logger with a rotating file handler and optional console handler.

    If AUTO_CONFIGURE is enabled, configure_logging is automatically invoked at import.
    Set force=True to re-apply configuration even if handlers already exist.
    """
    if force:
        # remove handlers we previously added to reconfigure
        root = logging.getLogger()
        root.handlers = [h for h in root.handlers if getattr(h, "name", None) not in ("rotating_file_handler", "console_handler")]
    _configure_root_logger()


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a logger with the given name. Ensures module's configuration is applied.
    """
    if AUTO_CONFIGURE:
        _configure_root_logger()
    return logging.getLogger(name)


# Auto-configure on import unless disabled.
if AUTO_CONFIGURE:
    try:
        _configure_root_logger()
    except Exception:
        # Avoid raising on import if logging setup fails.
        pass

__all__ = ["get_logger", "configure_logging"]