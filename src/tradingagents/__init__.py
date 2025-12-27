"""TradingAgents - Multi-Agent LLM Financial Trading Framework."""

from __future__ import annotations

from tradingagents.backtesting import (
    Backtester,
    BacktestMetrics,
    Portfolio,
    Position,
    Trade,
    calculate_metrics,
)
from tradingagents.config import (
    DEFAULT_CONFIG,
    DataVendor,
    DataVendorsConfig,
    LLMProvider,
    TradingAgentsConfig,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph

__all__ = [
    # Core
    "TradingAgentsGraph",
    "TradingAgentsConfig",
    "DEFAULT_CONFIG",
    "LLMProvider",
    "DataVendor",
    "DataVendorsConfig",
    # Backtesting
    "Backtester",
    "Portfolio",
    "Position",
    "Trade",
    "BacktestMetrics",
    "calculate_metrics",
]
__version__ = "0.2.0"
