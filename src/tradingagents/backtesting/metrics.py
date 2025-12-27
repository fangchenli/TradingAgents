"""Performance metrics for backtesting.

Calculates standard financial metrics like Sharpe ratio, maximum drawdown,
win rate, and other performance statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradingagents.backtesting.portfolio import Portfolio


@dataclass
class BacktestMetrics:
    """Container for backtest performance metrics."""

    # Returns
    total_return_pct: float
    annualized_return_pct: float
    volatility_pct: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown_pct: float
    max_drawdown_duration_days: int

    # Trade statistics
    num_trades: int
    num_winning_trades: int
    num_losing_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float

    # Other
    trading_days: int
    start_date: str
    end_date: str

    def __str__(self) -> str:
        """Format metrics as readable string."""
        return f"""
Backtest Results ({self.start_date} to {self.end_date})
{'=' * 50}

Returns:
  Total Return:       {self.total_return_pct:>10.2f}%
  Annualized Return:  {self.annualized_return_pct:>10.2f}%
  Volatility:         {self.volatility_pct:>10.2f}%

Risk-Adjusted:
  Sharpe Ratio:       {self.sharpe_ratio:>10.2f}
  Sortino Ratio:      {self.sortino_ratio:>10.2f}
  Calmar Ratio:       {self.calmar_ratio:>10.2f}

Drawdown:
  Max Drawdown:       {self.max_drawdown_pct:>10.2f}%
  Max DD Duration:    {self.max_drawdown_duration_days:>10} days

Trade Statistics:
  Total Trades:       {self.num_trades:>10}
  Winning Trades:     {self.num_winning_trades:>10}
  Losing Trades:      {self.num_losing_trades:>10}
  Win Rate:           {self.win_rate_pct:>10.2f}%
  Avg Win:            {self.avg_win_pct:>10.2f}%
  Avg Loss:           {self.avg_loss_pct:>10.2f}%
  Profit Factor:      {self.profit_factor:>10.2f}

Trading Days:         {self.trading_days:>10}
"""


def calculate_metrics(
    portfolio: Portfolio,
    risk_free_rate: float = 0.04,
    trading_days_per_year: int = 252,
) -> BacktestMetrics:
    """Calculate comprehensive backtest metrics.

    Args:
        portfolio: Portfolio with trading history
        risk_free_rate: Annual risk-free rate (default 4%)
        trading_days_per_year: Number of trading days per year

    Returns:
        BacktestMetrics object with all calculated metrics
    """
    # Get returns and equity curve
    returns = portfolio.get_returns()
    equity_curve = portfolio.get_equity_curve()

    if not equity_curve:
        return _empty_metrics()

    # Basic info
    start_date = equity_curve[0][0]
    end_date = equity_curve[-1][0]
    trading_days = len(equity_curve)

    # Total return
    initial_value = portfolio.initial_cash
    final_value = portfolio.total_value
    total_return_pct = ((final_value - initial_value) / initial_value) * 100

    # Annualized return
    years = trading_days / trading_days_per_year
    if years > 0 and final_value > 0 and initial_value > 0:
        annualized_return_pct = ((final_value / initial_value) ** (1 / years) - 1) * 100
    else:
        annualized_return_pct = 0.0

    # Volatility (annualized)
    if returns:
        daily_std = _std(returns)
        volatility_pct = daily_std * math.sqrt(trading_days_per_year) * 100
    else:
        volatility_pct = 0.0

    # Sharpe ratio
    if returns and volatility_pct > 0:
        daily_rf = risk_free_rate / trading_days_per_year
        excess_returns = [r - daily_rf for r in returns]
        avg_excess = sum(excess_returns) / len(excess_returns)
        sharpe_ratio = (avg_excess / daily_std) * math.sqrt(trading_days_per_year)
    else:
        sharpe_ratio = 0.0

    # Sortino ratio (uses downside deviation)
    if returns:
        daily_rf = risk_free_rate / trading_days_per_year
        negative_returns = [r - daily_rf for r in returns if r < daily_rf]
        if negative_returns:
            downside_std = _std(negative_returns)
            if downside_std > 0:
                avg_return = sum(returns) / len(returns)
                sortino_ratio = ((avg_return - daily_rf) / downside_std) * math.sqrt(
                    trading_days_per_year
                )
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = float("inf") if sum(returns) > 0 else 0.0
    else:
        sortino_ratio = 0.0

    # Maximum drawdown
    max_dd_pct, max_dd_duration = _calculate_max_drawdown(equity_curve)

    # Calmar ratio
    if max_dd_pct > 0:
        calmar_ratio = annualized_return_pct / max_dd_pct
    else:
        calmar_ratio = float("inf") if annualized_return_pct > 0 else 0.0

    # Trade statistics
    trade_stats = _calculate_trade_statistics(portfolio)

    return BacktestMetrics(
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized_return_pct,
        volatility_pct=volatility_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_duration_days=max_dd_duration,
        num_trades=trade_stats["num_trades"],
        num_winning_trades=trade_stats["num_winning"],
        num_losing_trades=trade_stats["num_losing"],
        win_rate_pct=trade_stats["win_rate_pct"],
        avg_win_pct=trade_stats["avg_win_pct"],
        avg_loss_pct=trade_stats["avg_loss_pct"],
        profit_factor=trade_stats["profit_factor"],
        trading_days=trading_days,
        start_date=start_date,
        end_date=end_date,
    )


def _std(values: list[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _calculate_max_drawdown(equity_curve: list[tuple[str, float]]) -> tuple[float, int]:
    """Calculate maximum drawdown and duration.

    Returns:
        Tuple of (max_drawdown_pct, max_duration_days)
    """
    if not equity_curve:
        return 0.0, 0

    peak = equity_curve[0][1]
    max_drawdown = 0.0
    current_dd_start = 0
    max_dd_duration = 0
    in_drawdown = False

    for i, (_date, value) in enumerate(equity_curve):
        if value > peak:
            peak = value
            if in_drawdown:
                duration = i - current_dd_start
                max_dd_duration = max(max_dd_duration, duration)
                in_drawdown = False
        else:
            if not in_drawdown:
                current_dd_start = i
                in_drawdown = True

            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

    # Check if still in drawdown at end
    if in_drawdown:
        duration = len(equity_curve) - current_dd_start
        max_dd_duration = max(max_dd_duration, duration)

    return max_drawdown * 100, max_dd_duration


def _calculate_trade_statistics(portfolio: Portfolio) -> dict:
    """Calculate trade-level statistics.

    Pairs BUY and SELL trades to calculate win/loss stats.
    """
    from tradingagents.backtesting.portfolio import Signal

    trades = portfolio.trades
    if not trades:
        return {
            "num_trades": 0,
            "num_winning": 0,
            "num_losing": 0,
            "win_rate_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
        }

    # Track open positions and completed trades
    open_positions: dict[str, list[tuple[float, float]]] = {}  # ticker -> [(shares, price)]
    completed_trades: list[float] = []  # pnl percentages

    for trade in trades:
        ticker = trade.ticker

        if trade.signal == Signal.BUY:
            if ticker not in open_positions:
                open_positions[ticker] = []
            open_positions[ticker].append((trade.shares, trade.price))

        elif trade.signal == Signal.SELL:
            if ticker in open_positions and open_positions[ticker]:
                # FIFO matching
                shares_to_close = trade.shares
                total_cost = 0.0

                while shares_to_close > 0 and open_positions[ticker]:
                    open_shares, open_price = open_positions[ticker][0]

                    if open_shares <= shares_to_close:
                        total_cost += open_shares * open_price
                        shares_to_close -= open_shares
                        open_positions[ticker].pop(0)
                    else:
                        total_cost += shares_to_close * open_price
                        open_positions[ticker][0] = (
                            open_shares - shares_to_close,
                            open_price,
                        )
                        shares_to_close = 0

                # Calculate PnL
                shares_closed = trade.shares - shares_to_close
                if shares_closed > 0 and total_cost > 0:
                    proceeds = shares_closed * trade.price
                    pnl_pct = ((proceeds - total_cost) / total_cost) * 100
                    completed_trades.append(pnl_pct)

    if not completed_trades:
        return {
            "num_trades": len(trades),
            "num_winning": 0,
            "num_losing": 0,
            "win_rate_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
        }

    # Calculate stats
    winning = [t for t in completed_trades if t > 0]
    losing = [t for t in completed_trades if t < 0]

    num_winning = len(winning)
    num_losing = len(losing)
    win_rate_pct = (num_winning / len(completed_trades)) * 100

    avg_win_pct = sum(winning) / len(winning) if winning else 0.0
    avg_loss_pct = sum(losing) / len(losing) if losing else 0.0

    # Profit factor = gross profit / gross loss
    gross_profit = sum(winning) if winning else 0.0
    gross_loss = abs(sum(losing)) if losing else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "num_trades": len(trades),
        "num_winning": num_winning,
        "num_losing": num_losing,
        "win_rate_pct": win_rate_pct,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "profit_factor": profit_factor,
    }


def _empty_metrics() -> BacktestMetrics:
    """Return empty metrics when no data available."""
    return BacktestMetrics(
        total_return_pct=0.0,
        annualized_return_pct=0.0,
        volatility_pct=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        max_drawdown_pct=0.0,
        max_drawdown_duration_days=0,
        num_trades=0,
        num_winning_trades=0,
        num_losing_trades=0,
        win_rate_pct=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        profit_factor=0.0,
        trading_days=0,
        start_date="",
        end_date="",
    )
