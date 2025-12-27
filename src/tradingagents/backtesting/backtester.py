"""Backtester for running trading strategies over historical data.

Iterates over date ranges, generates trading signals using TradingAgentsGraph,
and tracks portfolio performance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from tradingagents.backtesting.metrics import BacktestMetrics, calculate_metrics
from tradingagents.backtesting.portfolio import Portfolio, Signal
from tradingagents.logging import get_logger

if TYPE_CHECKING:
    from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = get_logger("backtesting.backtester")


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""

    metrics: BacktestMetrics
    portfolio: Portfolio
    decisions: list[dict[str, Any]]
    config: dict[str, Any]

    def save(self, path: str | Path) -> None:
        """Save backtest results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metrics": {
                "total_return_pct": self.metrics.total_return_pct,
                "annualized_return_pct": self.metrics.annualized_return_pct,
                "volatility_pct": self.metrics.volatility_pct,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "sortino_ratio": self.metrics.sortino_ratio,
                "calmar_ratio": self.metrics.calmar_ratio,
                "max_drawdown_pct": self.metrics.max_drawdown_pct,
                "max_drawdown_duration_days": self.metrics.max_drawdown_duration_days,
                "num_trades": self.metrics.num_trades,
                "num_winning_trades": self.metrics.num_winning_trades,
                "num_losing_trades": self.metrics.num_losing_trades,
                "win_rate_pct": self.metrics.win_rate_pct,
                "profit_factor": self.metrics.profit_factor,
                "trading_days": self.metrics.trading_days,
                "start_date": self.metrics.start_date,
                "end_date": self.metrics.end_date,
            },
            "portfolio_summary": self.portfolio.summary(),
            "equity_curve": self.portfolio.get_equity_curve(),
            "trades": [
                {
                    "date": t.date,
                    "ticker": t.ticker,
                    "signal": t.signal.value,
                    "shares": t.shares,
                    "price": t.price,
                    "value": t.value,
                    "commission": t.commission,
                }
                for t in self.portfolio.trades
            ],
            "decisions": self.decisions,
            "config": self.config,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Saved backtest results to %s", path)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to DataFrame."""
        curve = self.portfolio.get_equity_curve()
        return pd.DataFrame(curve, columns=["date", "value"])


class Backtester:
    """Run backtests over historical date ranges.

    Example:
        from tradingagents import TradingAgentsGraph
        from tradingagents.backtesting import Backtester

        ta = TradingAgentsGraph(config=config)
        backtester = Backtester(ta, initial_cash=100000)

        result = backtester.run(
            ticker="NVDA",
            start_date="2024-01-01",
            end_date="2024-06-30",
            frequency="weekly",
        )

        print(result.metrics)
        result.save("backtest_results/nvda_2024.json")
    """

    def __init__(
        self,
        trading_graph: TradingAgentsGraph,
        initial_cash: float = 100000.0,
        commission_per_trade: float = 0.0,
        allocation_per_signal: float = 0.1,
        enable_learning: bool = False,
    ):
        """Initialize backtester.

        Args:
            trading_graph: TradingAgentsGraph instance for generating signals
            initial_cash: Starting portfolio cash
            commission_per_trade: Fixed commission per trade
            allocation_per_signal: Fraction of portfolio to allocate per BUY signal
            enable_learning: Whether to call reflect_and_remember after trades
        """
        self.trading_graph = trading_graph
        self.initial_cash = initial_cash
        self.commission_per_trade = commission_per_trade
        self.allocation_per_signal = allocation_per_signal
        self.enable_learning = enable_learning

    def run(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        frequency: str = "weekly",
    ) -> BacktestResult:
        """Run backtest over date range.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Trading frequency - "daily", "weekly", or "monthly"

        Returns:
            BacktestResult with metrics, portfolio, and decision history
        """
        # Generate trading dates
        trading_dates = self._generate_trading_dates(start_date, end_date, frequency)

        logger.info(
            "Starting backtest for %s from %s to %s (%d trading dates)",
            ticker,
            start_date,
            end_date,
            len(trading_dates),
        )

        # Initialize portfolio
        portfolio = Portfolio(
            initial_cash=self.initial_cash,
            commission_per_trade=self.commission_per_trade,
        )

        decisions = []

        # Iterate over trading dates
        for i, trade_date in enumerate(trading_dates):
            logger.info(
                "Processing %s (%d/%d)",
                trade_date,
                i + 1,
                len(trading_dates),
            )

            try:
                # Get trading decision from TradingAgentsGraph
                state, signal_str = self.trading_graph.propagate(ticker, trade_date)

                # Parse signal
                signal = self._parse_signal(signal_str)

                # Record decision
                decision = {
                    "date": trade_date,
                    "ticker": ticker,
                    "signal": signal.value,
                    "raw_signal": signal_str,
                    "final_trade_decision": state.get("final_trade_decision", ""),
                }
                decisions.append(decision)

                # Execute trade
                trade = portfolio.execute_signal(
                    ticker=ticker,
                    signal=signal,
                    date=trade_date,
                    allocation_pct=self.allocation_per_signal,
                )

                # Take portfolio snapshot
                portfolio.take_snapshot(trade_date)

                # Optional: learning from trade outcomes
                if self.enable_learning and trade is not None:
                    # For learning, we need to wait for price changes
                    # This is a simplified version - real implementation would
                    # wait for actual returns
                    pass

            except Exception as e:
                logger.error("Error processing %s: %s", trade_date, e)
                decisions.append(
                    {
                        "date": trade_date,
                        "ticker": ticker,
                        "signal": "ERROR",
                        "error": str(e),
                    }
                )
                # Still take snapshot on error
                portfolio.take_snapshot(trade_date)

        # Calculate metrics
        metrics = calculate_metrics(portfolio)

        # Build config for result
        config = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "frequency": frequency,
            "initial_cash": self.initial_cash,
            "commission_per_trade": self.commission_per_trade,
            "allocation_per_signal": self.allocation_per_signal,
            "enable_learning": self.enable_learning,
        }

        return BacktestResult(
            metrics=metrics,
            portfolio=portfolio,
            decisions=decisions,
            config=config,
        )

    async def run_async(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        frequency: str = "weekly",
    ) -> BacktestResult:
        """Async version of run().

        Uses propagate_async for potentially better performance with
        async data fetching.
        """
        trading_dates = self._generate_trading_dates(start_date, end_date, frequency)

        logger.info(
            "Starting async backtest for %s from %s to %s (%d trading dates)",
            ticker,
            start_date,
            end_date,
            len(trading_dates),
        )

        portfolio = Portfolio(
            initial_cash=self.initial_cash,
            commission_per_trade=self.commission_per_trade,
        )

        decisions = []

        for i, trade_date in enumerate(trading_dates):
            logger.info(
                "Processing %s (%d/%d)",
                trade_date,
                i + 1,
                len(trading_dates),
            )

            try:
                # Use async version
                state, signal_str = await self.trading_graph.propagate_async(ticker, trade_date)

                signal = self._parse_signal(signal_str)

                decision = {
                    "date": trade_date,
                    "ticker": ticker,
                    "signal": signal.value,
                    "raw_signal": signal_str,
                    "final_trade_decision": state.get("final_trade_decision", ""),
                }
                decisions.append(decision)

                portfolio.execute_signal(
                    ticker=ticker,
                    signal=signal,
                    date=trade_date,
                    allocation_pct=self.allocation_per_signal,
                )

                portfolio.take_snapshot(trade_date)

            except Exception as e:
                logger.error("Error processing %s: %s", trade_date, e)
                decisions.append(
                    {
                        "date": trade_date,
                        "ticker": ticker,
                        "signal": "ERROR",
                        "error": str(e),
                    }
                )
                portfolio.take_snapshot(trade_date)

        metrics = calculate_metrics(portfolio)

        config = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "frequency": frequency,
            "initial_cash": self.initial_cash,
            "commission_per_trade": self.commission_per_trade,
            "allocation_per_signal": self.allocation_per_signal,
            "enable_learning": self.enable_learning,
        }

        return BacktestResult(
            metrics=metrics,
            portfolio=portfolio,
            decisions=decisions,
            config=config,
        )

    def _generate_trading_dates(
        self,
        start_date: str,
        end_date: str,
        frequency: str,
    ) -> list[str]:
        """Generate list of trading dates based on frequency.

        Filters out weekends. Does not filter holidays (would need
        a holiday calendar for that).
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if frequency == "daily":
            delta = timedelta(days=1)
        elif frequency == "weekly":
            delta = timedelta(weeks=1)
        elif frequency == "monthly":
            delta = timedelta(days=30)  # Approximate
        else:
            raise ValueError(f"Unknown frequency: {frequency}")

        dates = []
        current = start

        while current <= end:
            # Skip weekends
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                dates.append(current.strftime("%Y-%m-%d"))
            current += delta

        return dates

    def _parse_signal(self, signal_str: str) -> Signal:
        """Parse signal string to Signal enum.

        The signal processor returns cleaned BUY/HOLD/SELL strings,
        but we handle variations just in case.
        """
        signal_upper = signal_str.upper().strip()

        if "BUY" in signal_upper:
            return Signal.BUY
        elif "SELL" in signal_upper:
            return Signal.SELL
        else:
            return Signal.HOLD


def run_multi_ticker_backtest(
    trading_graph: TradingAgentsGraph,
    tickers: list[str],
    start_date: str,
    end_date: str,
    frequency: str = "weekly",
    initial_cash: float = 100000.0,
    allocation_per_ticker: float | None = None,
) -> dict[str, BacktestResult]:
    """Run backtests for multiple tickers.

    Args:
        trading_graph: TradingAgentsGraph instance
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        frequency: Trading frequency
        initial_cash: Starting cash (split across tickers)
        allocation_per_ticker: Allocation per ticker (default: equal split)

    Returns:
        Dict mapping ticker to BacktestResult
    """
    if allocation_per_ticker is None:
        allocation_per_ticker = 1.0 / len(tickers)

    cash_per_ticker = initial_cash / len(tickers)

    results = {}

    for ticker in tickers:
        logger.info("Running backtest for %s", ticker)

        backtester = Backtester(
            trading_graph=trading_graph,
            initial_cash=cash_per_ticker,
            allocation_per_signal=allocation_per_ticker,
        )

        results[ticker] = backtester.run(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
        )

    return results


async def run_multi_ticker_backtest_async(
    trading_graph: TradingAgentsGraph,
    tickers: list[str],
    start_date: str,
    end_date: str,
    frequency: str = "weekly",
    initial_cash: float = 100000.0,
    allocation_per_ticker: float | None = None,
) -> dict[str, BacktestResult]:
    """Async version of run_multi_ticker_backtest.

    Runs backtests for each ticker sequentially using async methods.
    For parallel execution across tickers, use asyncio.gather with
    separate Backtester instances.
    """

    if allocation_per_ticker is None:
        allocation_per_ticker = 1.0 / len(tickers)

    cash_per_ticker = initial_cash / len(tickers)

    results = {}

    for ticker in tickers:
        logger.info("Running async backtest for %s", ticker)

        backtester = Backtester(
            trading_graph=trading_graph,
            initial_cash=cash_per_ticker,
            allocation_per_signal=allocation_per_ticker,
        )

        results[ticker] = await backtester.run_async(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
        )

    return results
