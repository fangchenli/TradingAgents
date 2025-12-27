"""Backward compatibility module for default configuration.

This module re-exports the configuration from the new Pydantic-based config module.
New code should import from `tradingagents.config` directly.
"""

from __future__ import annotations

from tradingagents.config import (
    DEFAULT_CONFIG as _DEFAULT_CONFIG,
)
from tradingagents.config import (
    DataVendor,
    DataVendorsConfig,
    LLMProvider,
    TradingAgentsConfig,
)

# Export the default config as a dict for backward compatibility
DEFAULT_CONFIG = _DEFAULT_CONFIG.to_dict()

__all__ = [
    "DEFAULT_CONFIG",
    "DataVendor",
    "DataVendorsConfig",
    "LLMProvider",
    "TradingAgentsConfig",
]
