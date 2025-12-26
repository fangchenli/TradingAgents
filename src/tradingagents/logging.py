"""Logging configuration for TradingAgents."""

from __future__ import annotations

import logging
import os
import sys

# Package-level logger
logger = logging.getLogger("tradingagents")


def setup_logging(
    level: int | str = logging.INFO,
    format_string: str | None = None,
) -> None:
    """Configure logging for TradingAgents.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(format_string))

    logger.setLevel(level)
    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: Module name (e.g., "dataflows", "agents.analysts")

    Returns:
        Logger instance for the module
    """
    return logger.getChild(name)


# Auto-configure based on environment variable
_env_level = os.getenv("TRADINGAGENTS_LOG_LEVEL", "WARNING")
setup_logging(level=_env_level)
