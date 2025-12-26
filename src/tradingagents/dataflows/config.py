"""Configuration management for dataflows."""

from __future__ import annotations

import os
from typing import Any

import tradingagents.default_config as default_config

# Use default config but allow it to be overridden
_config: dict[str, Any] | None = None

# Legacy DATA_DIR for local data files (used by local.py vendor)
DATA_DIR: str | None = os.getenv("TRADINGAGENTS_DATA_DIR")


def initialize_config() -> None:
    """Initialize the configuration with default values."""
    global _config
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()


def set_config(config: dict[str, Any]) -> None:
    """Update the configuration with custom values."""
    global _config
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()
    _config.update(config)


def get_config() -> dict[str, Any]:
    """Get the current configuration."""
    if _config is None:
        initialize_config()
    return _config.copy()


def reset_config() -> None:
    """Reset config to None (useful for testing)."""
    global _config
    _config = None


# Initialize with default config
initialize_config()
