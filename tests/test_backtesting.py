"""Tests for backtesting module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.backtesting.backtester import Backtester, BacktestResult
from tradingagents.backtesting.metrics import (
    BacktestMetrics,
    _calculate_max_drawdown,
    _std,
    calculate_metrics,
)
from tradingagents.backtesting.portfolio import Portfolio, Position, Signal, Trade


class TestSignal:
    """Tests for Signal enum."""

    def test_signal_values(self) -> None:
        """Verify Signal enum has correct values."""
        assert Signal.BUY.value == "BUY"
        assert Signal.SELL.value == "SELL"
        assert Signal.HOLD.value == "HOLD"


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self) -> None:
        """Verify Position can be created with values."""
        pos = Position(ticker="AAPL", shares=100, avg_cost=150.0, current_price=160.0)
        assert pos.ticker == "AAPL"
        assert pos.shares == 100
        assert pos.avg_cost == 150.0
        assert pos.current_price == 160.0

    def test_market_value(self) -> None:
        """Verify market_value calculation."""
        pos = Position(ticker="AAPL", shares=100, avg_cost=150.0, current_price=160.0)
        assert pos.market_value == 16000.0

    def test_cost_basis(self) -> None:
        """Verify cost_basis calculation."""
        pos = Position(ticker="AAPL", shares=100, avg_cost=150.0, current_price=160.0)
        assert pos.cost_basis == 15000.0

    def test_unrealized_pnl(self) -> None:
        """Verify unrealized_pnl calculation."""
        pos = Position(ticker="AAPL", shares=100, avg_cost=150.0, current_price=160.0)
        assert pos.unrealized_pnl == 1000.0

    def test_unrealized_pnl_pct(self) -> None:
        """Verify unrealized_pnl_pct calculation."""
        pos = Position(ticker="AAPL", shares=100, avg_cost=150.0, current_price=160.0)
        assert pos.unrealized_pnl_pct == pytest.approx(6.67, rel=0.01)

    def test_unrealized_pnl_pct_zero_cost(self) -> None:
        """Verify unrealized_pnl_pct handles zero cost."""
        pos = Position(ticker="AAPL", shares=0, avg_cost=0.0, current_price=160.0)
        assert pos.unrealized_pnl_pct == 0.0


class TestTrade:
    """Tests for Trade dataclass."""

    def test_trade_creation(self) -> None:
        """Verify Trade can be created."""
        trade = Trade(
            date="2024-01-15",
            ticker="AAPL",
            signal=Signal.BUY,
            shares=50,
            price=150.0,
            value=7500.0,
            commission=10.0,
        )
        assert trade.date == "2024-01-15"
        assert trade.ticker == "AAPL"
        assert trade.signal == Signal.BUY
        assert trade.shares == 50
        assert trade.price == 150.0
        assert trade.value == 7500.0
        assert trade.commission == 10.0

    def test_net_value(self) -> None:
        """Verify net_value calculation."""
        trade = Trade(
            date="2024-01-15",
            ticker="AAPL",
            signal=Signal.BUY,
            shares=50,
            price=150.0,
            value=7500.0,
            commission=10.0,
        )
        assert trade.net_value == 7490.0


class TestPortfolio:
    """Tests for Portfolio class."""

    @pytest.fixture
    def portfolio(self) -> Portfolio:
        """Create a basic portfolio for testing."""
        return Portfolio(initial_cash=100000.0, commission_per_trade=10.0)

    def test_portfolio_initialization(self, portfolio: Portfolio) -> None:
        """Verify portfolio initializes correctly."""
        assert portfolio.initial_cash == 100000.0
        assert portfolio.cash == 100000.0
        assert portfolio.commission_per_trade == 10.0
        assert portfolio.positions == {}
        assert portfolio.trades == []

    def test_total_value_no_positions(self, portfolio: Portfolio) -> None:
        """Verify total_value with no positions."""
        assert portfolio.total_value == 100000.0

    def test_total_return_no_change(self, portfolio: Portfolio) -> None:
        """Verify total_return with no changes."""
        assert portfolio.total_return == 0.0

    @patch.object(Portfolio, "get_price")
    def test_execute_buy_signal(self, mock_price: MagicMock, portfolio: Portfolio) -> None:
        """Verify BUY signal creates position."""
        mock_price.return_value = 100.0

        trade = portfolio.execute_signal(
            ticker="AAPL",
            signal=Signal.BUY,
            date="2024-01-15",
            shares=50,
        )

        assert trade is not None
        assert trade.signal == Signal.BUY
        assert trade.shares == 50
        assert trade.price == 100.0
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].shares == 50

    @patch.object(Portfolio, "get_price")
    def test_execute_sell_signal(self, mock_price: MagicMock, portfolio: Portfolio) -> None:
        """Verify SELL signal reduces position."""
        mock_price.return_value = 100.0

        # First buy
        portfolio.execute_signal(
            ticker="AAPL",
            signal=Signal.BUY,
            date="2024-01-15",
            shares=50,
        )

        # Then sell
        mock_price.return_value = 110.0
        trade = portfolio.execute_signal(
            ticker="AAPL",
            signal=Signal.SELL,
            date="2024-01-20",
            shares=25,
        )

        assert trade is not None
        assert trade.signal == Signal.SELL
        assert trade.shares == 25
        assert portfolio.positions["AAPL"].shares == 25

    def test_execute_hold_signal(self, portfolio: Portfolio) -> None:
        """Verify HOLD signal does nothing."""
        trade = portfolio.execute_signal(
            ticker="AAPL",
            signal=Signal.HOLD,
            date="2024-01-15",
        )

        assert trade is None
        assert len(portfolio.trades) == 0

    @patch.object(Portfolio, "get_price")
    def test_sell_without_position(self, mock_price: MagicMock, portfolio: Portfolio) -> None:
        """Verify SELL without position returns None."""
        mock_price.return_value = 100.0

        trade = portfolio.execute_signal(
            ticker="AAPL",
            signal=Signal.SELL,
            date="2024-01-15",
        )

        assert trade is None

    @patch.object(Portfolio, "get_price")
    def test_buy_with_allocation_pct(self, mock_price: MagicMock, portfolio: Portfolio) -> None:
        """Verify BUY with allocation percentage."""
        mock_price.return_value = 100.0

        trade = portfolio.execute_signal(
            ticker="AAPL",
            signal=Signal.BUY,
            date="2024-01-15",
            allocation_pct=0.1,  # 10% of portfolio
        )

        assert trade is not None
        # 10% of 100000 = 10000, at $100/share = 100 shares
        assert trade.shares == pytest.approx(100, rel=0.01)

    @patch.object(Portfolio, "get_price")
    def test_take_snapshot(self, mock_price: MagicMock, portfolio: Portfolio) -> None:
        """Verify take_snapshot records state."""
        mock_price.return_value = 100.0

        portfolio.execute_signal(
            ticker="AAPL",
            signal=Signal.BUY,
            date="2024-01-15",
            shares=50,
        )

        snapshot = portfolio.take_snapshot("2024-01-15")

        assert snapshot.date == "2024-01-15"
        assert len(portfolio.history) == 1
        assert snapshot.total_value == portfolio.total_value

    @patch.object(Portfolio, "get_price")
    def test_get_equity_curve(self, mock_price: MagicMock, portfolio: Portfolio) -> None:
        """Verify get_equity_curve returns dates and values."""
        mock_price.return_value = 100.0

        portfolio.take_snapshot("2024-01-15")
        portfolio.take_snapshot("2024-01-16")

        curve = portfolio.get_equity_curve()

        assert len(curve) == 2
        assert curve[0][0] == "2024-01-15"
        assert curve[1][0] == "2024-01-16"

    @patch.object(Portfolio, "get_price")
    def test_summary(self, mock_price: MagicMock, portfolio: Portfolio) -> None:
        """Verify summary returns correct structure."""
        mock_price.return_value = 100.0

        portfolio.execute_signal(
            ticker="AAPL",
            signal=Signal.BUY,
            date="2024-01-15",
            shares=50,
        )

        summary = portfolio.summary()

        assert "initial_cash" in summary
        assert "current_cash" in summary
        assert "positions_value" in summary
        assert "total_value" in summary
        assert "num_trades" in summary
        assert summary["num_trades"] == 1


class TestMetrics:
    """Tests for metrics calculations."""

    def test_std_empty_list(self) -> None:
        """Verify std handles empty list."""
        assert _std([]) == 0.0

    def test_std_single_value(self) -> None:
        """Verify std handles single value."""
        assert _std([5.0]) == 0.0

    def test_std_calculation(self) -> None:
        """Verify std calculates correctly."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        # Known std for this sample is ~2.138
        assert _std(values) == pytest.approx(2.138, rel=0.01)

    def test_max_drawdown_empty(self) -> None:
        """Verify max_drawdown handles empty curve."""
        dd, duration = _calculate_max_drawdown([])
        assert dd == 0.0
        assert duration == 0

    def test_max_drawdown_no_drawdown(self) -> None:
        """Verify max_drawdown with monotonic increase."""
        curve = [("2024-01-01", 100), ("2024-01-02", 110), ("2024-01-03", 120)]
        dd, duration = _calculate_max_drawdown(curve)
        assert dd == 0.0

    def test_max_drawdown_calculation(self) -> None:
        """Verify max_drawdown calculates correctly."""
        curve = [
            ("2024-01-01", 100),
            ("2024-01-02", 110),  # peak
            ("2024-01-03", 99),  # 10% drawdown from 110
            ("2024-01-04", 105),
            ("2024-01-05", 120),  # new peak
        ]
        dd, duration = _calculate_max_drawdown(curve)
        # Max drawdown is (110 - 99) / 110 = 10%
        assert dd == pytest.approx(10.0, rel=0.01)

    @patch.object(Portfolio, "get_price")
    def test_calculate_metrics_basic(self, mock_price: MagicMock) -> None:
        """Verify calculate_metrics returns valid metrics."""
        mock_price.return_value = 100.0

        portfolio = Portfolio(initial_cash=100000.0)

        # Create some history
        portfolio.take_snapshot("2024-01-01")
        portfolio.take_snapshot("2024-01-02")
        portfolio.take_snapshot("2024-01-03")

        metrics = calculate_metrics(portfolio)

        assert isinstance(metrics, BacktestMetrics)
        assert metrics.start_date == "2024-01-01"
        assert metrics.end_date == "2024-01-03"
        assert metrics.trading_days == 3

    def test_backtest_metrics_str(self) -> None:
        """Verify BacktestMetrics string representation."""
        metrics = BacktestMetrics(
            total_return_pct=10.5,
            annualized_return_pct=25.0,
            volatility_pct=15.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_drawdown_pct=5.0,
            max_drawdown_duration_days=10,
            num_trades=20,
            num_winning_trades=12,
            num_losing_trades=8,
            win_rate_pct=60.0,
            avg_win_pct=3.0,
            avg_loss_pct=-2.0,
            profit_factor=1.8,
            trading_days=252,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        output = str(metrics)
        assert "10.50%" in output
        assert "Sharpe Ratio" in output
        assert "Max Drawdown" in output


class TestBacktester:
    """Tests for Backtester class."""

    @pytest.fixture
    def mock_trading_graph(self) -> MagicMock:
        """Create a mock TradingAgentsGraph."""
        mock = MagicMock()
        mock.propagate.return_value = ({"final_trade_decision": "BUY"}, "BUY")
        return mock

    def test_backtester_initialization(self, mock_trading_graph: MagicMock) -> None:
        """Verify Backtester initializes correctly."""
        backtester = Backtester(
            trading_graph=mock_trading_graph,
            initial_cash=50000.0,
            commission_per_trade=5.0,
        )

        assert backtester.initial_cash == 50000.0
        assert backtester.commission_per_trade == 5.0

    def test_generate_trading_dates_weekly(self, mock_trading_graph: MagicMock) -> None:
        """Verify weekly date generation."""
        backtester = Backtester(trading_graph=mock_trading_graph)

        dates = backtester._generate_trading_dates("2024-01-01", "2024-01-31", "weekly")

        # Should have ~4-5 weekly dates
        assert len(dates) >= 4
        assert len(dates) <= 5
        # All should be weekdays
        from datetime import datetime

        for date in dates:
            dt = datetime.strptime(date, "%Y-%m-%d")
            assert dt.weekday() < 5

    def test_generate_trading_dates_daily(self, mock_trading_graph: MagicMock) -> None:
        """Verify daily date generation."""
        backtester = Backtester(trading_graph=mock_trading_graph)

        dates = backtester._generate_trading_dates("2024-01-08", "2024-01-12", "daily")

        # Mon-Fri = 5 weekdays
        assert len(dates) == 5

    def test_parse_signal_buy(self, mock_trading_graph: MagicMock) -> None:
        """Verify signal parsing for BUY."""
        backtester = Backtester(trading_graph=mock_trading_graph)
        assert backtester._parse_signal("BUY") == Signal.BUY
        assert backtester._parse_signal("buy") == Signal.BUY
        assert backtester._parse_signal("Strong BUY recommendation") == Signal.BUY

    def test_parse_signal_sell(self, mock_trading_graph: MagicMock) -> None:
        """Verify signal parsing for SELL."""
        backtester = Backtester(trading_graph=mock_trading_graph)
        assert backtester._parse_signal("SELL") == Signal.SELL
        assert backtester._parse_signal("sell") == Signal.SELL

    def test_parse_signal_hold(self, mock_trading_graph: MagicMock) -> None:
        """Verify signal parsing for HOLD."""
        backtester = Backtester(trading_graph=mock_trading_graph)
        assert backtester._parse_signal("HOLD") == Signal.HOLD
        assert backtester._parse_signal("wait") == Signal.HOLD
        assert backtester._parse_signal("unknown") == Signal.HOLD


class TestBacktestResult:
    """Tests for BacktestResult class."""

    @pytest.fixture
    def sample_result(self) -> BacktestResult:
        """Create a sample BacktestResult for testing."""
        portfolio = Portfolio(initial_cash=100000.0)
        portfolio.take_snapshot("2024-01-01")

        metrics = BacktestMetrics(
            total_return_pct=5.0,
            annualized_return_pct=10.0,
            volatility_pct=12.0,
            sharpe_ratio=0.8,
            sortino_ratio=1.0,
            calmar_ratio=0.5,
            max_drawdown_pct=8.0,
            max_drawdown_duration_days=5,
            num_trades=10,
            num_winning_trades=6,
            num_losing_trades=4,
            win_rate_pct=60.0,
            avg_win_pct=2.5,
            avg_loss_pct=-1.5,
            profit_factor=1.5,
            trading_days=20,
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

        return BacktestResult(
            metrics=metrics,
            portfolio=portfolio,
            decisions=[{"date": "2024-01-01", "signal": "BUY"}],
            config={"ticker": "AAPL"},
        )

    def test_save_and_load(self, sample_result: BacktestResult) -> None:
        """Verify save creates valid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            sample_result.save(path)

            assert path.exists()

            with open(path) as f:
                data = json.load(f)

            assert "metrics" in data
            assert "portfolio_summary" in data
            assert "equity_curve" in data
            assert "trades" in data
            assert "decisions" in data
            assert "config" in data

    def test_to_dataframe(self, sample_result: BacktestResult) -> None:
        """Verify to_dataframe returns DataFrame."""
        df = sample_result.to_dataframe()

        assert "date" in df.columns
        assert "value" in df.columns
        assert len(df) == 1  # One snapshot


class TestModuleImports:
    """Test that all backtesting exports are accessible."""

    def test_imports_from_backtesting(self) -> None:
        """Verify imports from backtesting module."""
        from tradingagents.backtesting import (
            Backtester,
            BacktestMetrics,
            Portfolio,
            Position,
            Trade,
            calculate_metrics,
        )

        assert Backtester is not None
        assert Portfolio is not None
        assert Position is not None
        assert Trade is not None
        assert BacktestMetrics is not None
        assert calculate_metrics is not None

    def test_imports_from_main_package(self) -> None:
        """Verify imports from main tradingagents package."""
        from tradingagents import (
            Backtester,
            BacktestMetrics,
            Portfolio,
            Position,
            Trade,
            calculate_metrics,
        )

        assert Backtester is not None
        assert Portfolio is not None
        assert Position is not None
        assert Trade is not None
        assert BacktestMetrics is not None
        assert calculate_metrics is not None
