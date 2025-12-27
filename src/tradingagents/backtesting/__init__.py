"""Backtesting module for TradingAgents.

Provides tools for historical backtesting of trading strategies:
- Portfolio: Track positions, cash, and portfolio value
- Backtester: Run strategies over historical date ranges
- Metrics: Calculate performance metrics (Sharpe, drawdown, etc.)
"""

from __future__ import annotations

from tradingagents.backtesting.backtester import Backtester
from tradingagents.backtesting.metrics import BacktestMetrics, calculate_metrics
from tradingagents.backtesting.portfolio import Portfolio, Position, Signal, Trade

__all__ = [
    "Backtester",
    "Portfolio",
    "Position",
    "Signal",
    "Trade",
    "BacktestMetrics",
    "calculate_metrics",
]
