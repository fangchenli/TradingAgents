"""Portfolio management for backtesting.

Tracks positions, cash, and calculates portfolio value over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import yfinance as yf

from tradingagents.logging import get_logger

logger = get_logger("backtesting.portfolio")


class Signal(Enum):
    """Trading signal types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Trade:
    """Record of a single trade execution."""

    date: str
    ticker: str
    signal: Signal
    shares: float
    price: float
    value: float  # shares * price
    commission: float = 0.0

    @property
    def net_value(self) -> float:
        """Value after commission."""
        return self.value - self.commission


@dataclass
class Position:
    """Current position in a single security."""

    ticker: str
    shares: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis of position."""
        return self.shares * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time."""

    date: str
    cash: float
    positions_value: float
    total_value: float
    positions: dict[str, Position]


class Portfolio:
    """Simulated portfolio for backtesting.

    Tracks cash, positions, and trade history. Supports position sizing
    and commission modeling.

    Example:
        portfolio = Portfolio(initial_cash=100000)
        portfolio.execute_signal("AAPL", Signal.BUY, "2024-01-15", allocation_pct=0.1)
        print(portfolio.total_value)
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_per_trade: float = 0.0,
        max_position_pct: float = 0.25,
    ):
        """Initialize portfolio.

        Args:
            initial_cash: Starting cash amount
            commission_per_trade: Fixed commission per trade
            max_position_pct: Maximum position size as fraction of portfolio
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_per_trade = commission_per_trade
        self.max_position_pct = max_position_pct

        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.history: list[PortfolioSnapshot] = []

        # Price cache to avoid repeated API calls
        self._price_cache: dict[tuple[str, str], float] = {}

    def get_price(self, ticker: str, date: str) -> float:
        """Get closing price for a ticker on a date.

        Uses yfinance with caching to minimize API calls.
        """
        cache_key = (ticker, date)
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        try:
            ticker_obj = yf.Ticker(ticker)
            # Fetch a small window around the date
            hist = ticker_obj.history(start=date, period="5d")
            if hist.empty:
                logger.warning("No price data for %s on %s", ticker, date)
                return 0.0

            # Get the first available price (might be next trading day)
            price = float(hist["Close"].iloc[0])
            self._price_cache[cache_key] = price
            return price

        except Exception as e:
            logger.error("Error fetching price for %s on %s: %s", ticker, date, e)
            return 0.0

    @property
    def positions_value(self) -> float:
        """Total market value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self.cash + self.positions_value

    @property
    def total_return(self) -> float:
        """Total return as percentage."""
        return ((self.total_value - self.initial_cash) / self.initial_cash) * 100

    def update_prices(self, date: str) -> None:
        """Update all position prices to current date."""
        for ticker, position in self.positions.items():
            price = self.get_price(ticker, date)
            if price > 0:
                position.current_price = price

    def execute_signal(
        self,
        ticker: str,
        signal: Signal,
        date: str,
        allocation_pct: float | None = None,
        shares: float | None = None,
    ) -> Trade | None:
        """Execute a trading signal.

        Args:
            ticker: Stock ticker symbol
            signal: BUY, SELL, or HOLD
            date: Trade date (YYYY-MM-DD)
            allocation_pct: Percentage of portfolio to allocate (for BUY)
            shares: Specific number of shares (overrides allocation_pct)

        Returns:
            Trade record if executed, None if HOLD or insufficient funds
        """
        if signal == Signal.HOLD:
            logger.debug("HOLD signal for %s on %s - no action", ticker, date)
            return None

        price = self.get_price(ticker, date)
        if price <= 0:
            logger.warning("Cannot execute %s for %s - no valid price", signal.value, ticker)
            return None

        if signal == Signal.BUY:
            return self._execute_buy(ticker, date, price, allocation_pct, shares)
        elif signal == Signal.SELL:
            return self._execute_sell(ticker, date, price, shares)

        return None

    def _execute_buy(
        self,
        ticker: str,
        date: str,
        price: float,
        allocation_pct: float | None,
        shares: float | None,
    ) -> Trade | None:
        """Execute a buy order."""
        # Determine number of shares to buy
        if shares is not None:
            shares_to_buy = shares
        elif allocation_pct is not None:
            # Calculate shares based on portfolio allocation
            target_value = self.total_value * min(allocation_pct, self.max_position_pct)
            shares_to_buy = target_value / price
        else:
            # Default: use max position size
            target_value = self.total_value * self.max_position_pct
            shares_to_buy = target_value / price

        # Calculate order value
        order_value = shares_to_buy * price
        total_cost = order_value + self.commission_per_trade

        # Check if we have enough cash
        if total_cost > self.cash:
            # Adjust to what we can afford
            available = self.cash - self.commission_per_trade
            if available <= 0:
                logger.warning("Insufficient cash for BUY %s", ticker)
                return None
            shares_to_buy = available / price
            order_value = shares_to_buy * price
            total_cost = order_value + self.commission_per_trade

        # Execute the trade
        self.cash -= total_cost

        # Update or create position
        if ticker in self.positions:
            pos = self.positions[ticker]
            # Calculate new average cost
            total_shares = pos.shares + shares_to_buy
            if total_shares > 0:
                pos.avg_cost = (pos.cost_basis + order_value) / total_shares
            pos.shares = total_shares
            pos.current_price = price
        else:
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares_to_buy,
                avg_cost=price,
                current_price=price,
            )

        # Record trade
        trade = Trade(
            date=date,
            ticker=ticker,
            signal=Signal.BUY,
            shares=shares_to_buy,
            price=price,
            value=order_value,
            commission=self.commission_per_trade,
        )
        self.trades.append(trade)

        logger.info(
            "BUY %s: %.2f shares @ $%.2f = $%.2f",
            ticker,
            shares_to_buy,
            price,
            order_value,
        )

        return trade

    def _execute_sell(
        self,
        ticker: str,
        date: str,
        price: float,
        shares: float | None,
    ) -> Trade | None:
        """Execute a sell order."""
        if ticker not in self.positions or self.positions[ticker].shares <= 0:
            logger.warning("No position to sell for %s", ticker)
            return None

        pos = self.positions[ticker]

        # Determine shares to sell (default: entire position)
        shares_to_sell = min(shares, pos.shares) if shares is not None else pos.shares

        # Calculate proceeds
        order_value = shares_to_sell * price
        net_proceeds = order_value - self.commission_per_trade

        # Update position
        pos.shares -= shares_to_sell
        pos.current_price = price

        # Remove position if fully sold
        if pos.shares <= 0:
            del self.positions[ticker]

        # Update cash
        self.cash += net_proceeds

        # Record trade
        trade = Trade(
            date=date,
            ticker=ticker,
            signal=Signal.SELL,
            shares=shares_to_sell,
            price=price,
            value=order_value,
            commission=self.commission_per_trade,
        )
        self.trades.append(trade)

        logger.info(
            "SELL %s: %.2f shares @ $%.2f = $%.2f",
            ticker,
            shares_to_sell,
            price,
            order_value,
        )

        return trade

    def take_snapshot(self, date: str) -> PortfolioSnapshot:
        """Take a snapshot of current portfolio state."""
        self.update_prices(date)

        snapshot = PortfolioSnapshot(
            date=date,
            cash=self.cash,
            positions_value=self.positions_value,
            total_value=self.total_value,
            positions={k: Position(**v.__dict__) for k, v in self.positions.items()},
        )
        self.history.append(snapshot)
        return snapshot

    def get_equity_curve(self) -> list[tuple[str, float]]:
        """Get equity curve as list of (date, value) tuples."""
        return [(snap.date, snap.total_value) for snap in self.history]

    def get_returns(self) -> list[float]:
        """Get daily returns as list of percentages."""
        if len(self.history) < 2:
            return []

        returns = []
        for i in range(1, len(self.history)):
            prev_value = self.history[i - 1].total_value
            curr_value = self.history[i].total_value
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)

        return returns

    def summary(self) -> dict:
        """Get portfolio summary."""
        return {
            "initial_cash": self.initial_cash,
            "current_cash": self.cash,
            "positions_value": self.positions_value,
            "total_value": self.total_value,
            "total_return_pct": self.total_return,
            "num_trades": len(self.trades),
            "num_positions": len(self.positions),
            "positions": {
                ticker: {
                    "shares": pos.shares,
                    "avg_cost": pos.avg_cost,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                }
                for ticker, pos in self.positions.items()
            },
        }
