"""Integration tests for TradingAgents framework.

These tests verify the integration between components without making
actual API calls. They use mocking to simulate LLM responses and
data fetching while testing the full workflow.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.backtesting import Backtester, Portfolio, Signal
from tradingagents.backtesting.metrics import calculate_metrics


class TestPortfolioIntegration:
    """Integration tests for Portfolio with real trade sequences."""

    def test_full_trading_cycle(self) -> None:
        """Test a complete buy-hold-sell trading cycle."""
        portfolio = Portfolio(initial_cash=100000.0, commission_per_trade=10.0)

        with patch.object(portfolio, "get_price") as mock_price:
            # Day 1: Buy at $100
            mock_price.return_value = 100.0
            buy_trade = portfolio.execute_signal(
                ticker="AAPL",
                signal=Signal.BUY,
                date="2024-01-15",
                shares=100,
            )
            portfolio.take_snapshot("2024-01-15")

            assert buy_trade is not None
            assert buy_trade.shares == 100
            assert portfolio.positions["AAPL"].shares == 100
            initial_value = portfolio.total_value

            # Day 2: Hold (price goes up)
            mock_price.return_value = 110.0
            hold_trade = portfolio.execute_signal(
                ticker="AAPL",
                signal=Signal.HOLD,
                date="2024-01-16",
            )
            portfolio.take_snapshot("2024-01-16")

            assert hold_trade is None  # HOLD should not execute trade
            assert portfolio.positions["AAPL"].shares == 100  # Still holding

            # Day 3: Sell at $115
            mock_price.return_value = 115.0
            sell_trade = portfolio.execute_signal(
                ticker="AAPL",
                signal=Signal.SELL,
                date="2024-01-17",
            )
            portfolio.take_snapshot("2024-01-17")

            assert sell_trade is not None
            assert sell_trade.shares == 100
            assert "AAPL" not in portfolio.positions  # Position closed
            assert portfolio.total_value > initial_value  # Made profit

        # Verify equity curve
        curve = portfolio.get_equity_curve()
        assert len(curve) == 3
        assert curve[0][0] == "2024-01-15"
        assert curve[2][0] == "2024-01-17"

        # Verify returns
        returns = portfolio.get_returns()
        assert len(returns) == 2
        assert returns[0] > 0  # Price went from 100 to 110
        assert returns[1] > 0  # Price went from 110 to 115

    def test_multiple_positions(self) -> None:
        """Test managing multiple positions simultaneously."""
        portfolio = Portfolio(initial_cash=100000.0)

        with patch.object(portfolio, "get_price") as mock_price:
            # Buy AAPL
            mock_price.return_value = 150.0
            portfolio.execute_signal("AAPL", Signal.BUY, "2024-01-15", shares=50)

            # Buy MSFT
            mock_price.return_value = 400.0
            portfolio.execute_signal("MSFT", Signal.BUY, "2024-01-15", shares=25)

            # Buy NVDA
            mock_price.return_value = 500.0
            portfolio.execute_signal("NVDA", Signal.BUY, "2024-01-15", shares=20)

            portfolio.take_snapshot("2024-01-15")

        assert len(portfolio.positions) == 3
        assert portfolio.positions["AAPL"].shares == 50
        assert portfolio.positions["MSFT"].shares == 25
        assert portfolio.positions["NVDA"].shares == 20

        summary = portfolio.summary()
        assert summary["num_positions"] == 3
        assert summary["num_trades"] == 3

    def test_partial_sell(self) -> None:
        """Test selling partial position."""
        portfolio = Portfolio(initial_cash=100000.0)

        with patch.object(portfolio, "get_price") as mock_price:
            mock_price.return_value = 100.0

            # Buy 100 shares
            portfolio.execute_signal("AAPL", Signal.BUY, "2024-01-15", shares=100)

            # Sell 40 shares
            portfolio.execute_signal("AAPL", Signal.SELL, "2024-01-16", shares=40)

            assert portfolio.positions["AAPL"].shares == 60

            # Sell remaining 60 shares
            portfolio.execute_signal("AAPL", Signal.SELL, "2024-01-17", shares=60)

            assert "AAPL" not in portfolio.positions


class TestMetricsIntegration:
    """Integration tests for metrics calculation with real portfolios."""

    def test_metrics_with_winning_trades(self) -> None:
        """Test metrics calculation with profitable trades."""
        portfolio = Portfolio(initial_cash=100000.0)

        with patch.object(portfolio, "get_price") as mock_price:
            # Simulate profitable trading
            prices = [100, 105, 110, 108, 115, 120]

            for i, price in enumerate(prices):
                mock_price.return_value = float(price)
                date = f"2024-01-{15 + i:02d}"

                if i == 0:
                    portfolio.execute_signal("AAPL", Signal.BUY, date, shares=100)
                elif i == 5:
                    portfolio.execute_signal("AAPL", Signal.SELL, date)

                portfolio.take_snapshot(date)

        metrics = calculate_metrics(portfolio)

        assert metrics.total_return_pct > 0
        assert metrics.trading_days == 6
        assert metrics.num_trades == 2  # 1 buy + 1 sell
        assert metrics.start_date == "2024-01-15"
        assert metrics.end_date == "2024-01-20"

    def test_metrics_with_drawdown(self) -> None:
        """Test metrics calculation captures drawdown correctly."""
        portfolio = Portfolio(initial_cash=100000.0)

        with patch.object(portfolio, "get_price") as mock_price:
            # Simulate trading with a drawdown period
            # Buy 500 shares to make position significant relative to cash
            prices = [100, 110, 120, 100, 90, 95, 105, 115]

            mock_price.return_value = float(prices[0])
            portfolio.execute_signal("AAPL", Signal.BUY, "2024-01-01", shares=500)

            for i, price in enumerate(prices):
                mock_price.return_value = float(price)
                date = f"2024-01-{i + 1:02d}"
                portfolio.take_snapshot(date)

        metrics = calculate_metrics(portfolio)

        # Should have a max drawdown (position value dropped significantly)
        # With 500 shares: peak at 120 = 60000, trough at 90 = 45000
        # Portfolio peak ~= 50000 cash + 60000 = 110000
        # Portfolio trough ~= 50000 cash + 45000 = 95000
        # Drawdown = (110000 - 95000) / 110000 = 13.6%
        assert metrics.max_drawdown_pct > 10
        assert metrics.max_drawdown_duration_days > 0


class TestBacktesterIntegration:
    """Integration tests for Backtester with mocked TradingAgentsGraph."""

    @pytest.fixture
    def mock_trading_graph(self) -> MagicMock:
        """Create a mock TradingAgentsGraph with realistic responses."""
        mock = MagicMock()

        # Simulate varying trading signals
        signals = ["BUY", "HOLD", "HOLD", "SELL", "HOLD", "BUY", "HOLD", "SELL"]
        call_count = [0]

        def mock_propagate(ticker: str, date: str):
            idx = call_count[0] % len(signals)
            call_count[0] += 1
            signal = signals[idx]
            state = {
                "company_of_interest": ticker,
                "trade_date": date,
                "final_trade_decision": f"Based on analysis, the recommendation is {signal}",
                "market_report": "Market analysis...",
                "news_report": "News analysis...",
            }
            return state, signal

        mock.propagate = mock_propagate
        return mock

    def test_backtester_run_with_trades(self, mock_trading_graph: MagicMock) -> None:
        """Test Backtester executes trades based on signals."""
        backtester = Backtester(
            trading_graph=mock_trading_graph,
            initial_cash=100000.0,
            allocation_per_signal=0.2,
        )

        # Mock price fetching
        with patch.object(Portfolio, "get_price", return_value=100.0):
            result = backtester.run(
                ticker="AAPL",
                start_date="2024-01-01",
                end_date="2024-02-01",
                frequency="weekly",
            )

        assert result is not None
        assert result.metrics is not None
        assert len(result.decisions) > 0
        assert result.config["ticker"] == "AAPL"

        # Verify decisions were recorded
        signals_in_decisions = [d["signal"] for d in result.decisions]
        assert "BUY" in signals_in_decisions
        assert "SELL" in signals_in_decisions

    def test_backtester_handles_errors_gracefully(self) -> None:
        """Test Backtester continues after individual date errors."""
        mock_graph = MagicMock()

        call_count = [0]

        def mock_propagate(ticker: str, date: str):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated API error")
            return {"final_trade_decision": "HOLD"}, "HOLD"

        mock_graph.propagate = mock_propagate

        backtester = Backtester(trading_graph=mock_graph, initial_cash=100000.0)

        with patch.object(Portfolio, "get_price", return_value=100.0):
            result = backtester.run(
                ticker="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-15",
                frequency="daily",
            )

        # Should have completed despite error
        assert result is not None

        # Check that error was recorded
        error_decisions = [d for d in result.decisions if d.get("signal") == "ERROR"]
        assert len(error_decisions) == 1
        assert "error" in error_decisions[0]

    def test_backtester_result_save_and_load(self, mock_trading_graph: MagicMock) -> None:
        """Test saving and loading backtest results."""
        backtester = Backtester(
            trading_graph=mock_trading_graph,
            initial_cash=100000.0,
        )

        with patch.object(Portfolio, "get_price", return_value=100.0):
            result = backtester.run(
                ticker="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-31",
                frequency="weekly",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "results.json"
            result.save(save_path)

            assert save_path.exists()

            with open(save_path) as f:
                loaded_data = json.load(f)

            assert "metrics" in loaded_data
            assert "portfolio_summary" in loaded_data
            assert "equity_curve" in loaded_data
            assert "trades" in loaded_data
            assert "decisions" in loaded_data
            assert loaded_data["config"]["ticker"] == "AAPL"


class TestTradingAgentsGraphIntegration:
    """Integration tests for TradingAgentsGraph initialization and configuration."""

    @patch("tradingagents.graph.trading_graph.ChatOpenAI")
    @patch("tradingagents.agents.utils.memory.OpenAI")
    @patch("tradingagents.agents.utils.memory.AsyncOpenAI")
    @patch("tradingagents.agents.utils.memory.chromadb.Client")
    def test_graph_initialization_with_config(
        self,
        mock_chroma: MagicMock,
        mock_async_openai: MagicMock,
        mock_openai: MagicMock,
        mock_chat_openai: MagicMock,
    ) -> None:
        """Test TradingAgentsGraph initializes with custom config."""
        from tradingagents import TradingAgentsConfig, TradingAgentsGraph

        # Setup mocks
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        config = TradingAgentsConfig(
            llm_provider="openai",
            deep_think_llm="gpt-4o",
            quick_think_llm="gpt-4o-mini",
            max_debate_rounds=2,
            max_risk_discuss_rounds=2,
        )

        graph = TradingAgentsGraph(
            selected_analysts=["market", "fundamentals"],
            config=config,
        )

        assert graph is not None
        assert graph.config["max_debate_rounds"] == 2
        assert graph.config["max_risk_discuss_rounds"] == 2

    @patch("tradingagents.graph.trading_graph.ChatOpenAI")
    @patch("tradingagents.agents.utils.memory.OpenAI")
    @patch("tradingagents.agents.utils.memory.AsyncOpenAI")
    @patch("tradingagents.agents.utils.memory.chromadb.Client")
    def test_graph_with_dict_config(
        self,
        mock_chroma: MagicMock,
        mock_async_openai: MagicMock,
        mock_openai: MagicMock,
        mock_chat_openai: MagicMock,
    ) -> None:
        """Test TradingAgentsGraph accepts dict config for backward compatibility."""
        from tradingagents import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = 3

        graph = TradingAgentsGraph(config=config)

        assert graph.config["max_debate_rounds"] == 3


class TestEndToEndWorkflow:
    """End-to-end workflow tests simulating complete trading scenarios."""

    @patch("tradingagents.graph.trading_graph.ChatOpenAI")
    @patch("tradingagents.agents.utils.memory.OpenAI")
    @patch("tradingagents.agents.utils.memory.AsyncOpenAI")
    @patch("tradingagents.agents.utils.memory.chromadb.Client")
    def test_complete_analysis_to_backtest_workflow(
        self,
        mock_chroma: MagicMock,
        mock_async_openai: MagicMock,
        mock_openai: MagicMock,
        mock_chat_openai: MagicMock,
    ) -> None:
        """Test complete workflow from graph creation to backtesting."""
        from tradingagents import Backtester, TradingAgentsGraph

        # Setup mocks
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Create graph
        graph = TradingAgentsGraph(selected_analysts=["market"])

        # Mock the propagate method to return realistic results
        def mock_propagate(ticker: str, date: str):
            state = {
                "company_of_interest": ticker,
                "trade_date": date,
                "market_report": "Technical analysis shows bullish momentum",
                "sentiment_report": "",
                "news_report": "",
                "fundamentals_report": "",
                "investment_debate_state": {"judge_decision": "BUY"},
                "trader_investment_plan": "Buy with 10% allocation",
                "risk_debate_state": {"judge_decision": "Approved"},
                "investment_plan": "Conservative buy",
                "final_trade_decision": "BUY - Strong technical signals",
            }
            return state, "BUY"

        graph.propagate = mock_propagate

        # Create backtester
        backtester = Backtester(
            trading_graph=graph,
            initial_cash=50000.0,
            allocation_per_signal=0.15,
        )

        # Run backtest
        with patch.object(Portfolio, "get_price", return_value=150.0):
            result = backtester.run(
                ticker="NVDA",
                start_date="2024-01-01",
                end_date="2024-01-31",
                frequency="weekly",
            )

        # Verify complete workflow
        assert result is not None
        assert result.metrics.trading_days > 0
        assert len(result.decisions) > 0
        assert all(d["signal"] == "BUY" for d in result.decisions)
        assert result.portfolio.total_value >= 0

    def test_portfolio_to_metrics_integration(self) -> None:
        """Test that Portfolio data flows correctly to metrics calculation."""
        portfolio = Portfolio(initial_cash=100000.0)

        with patch.object(portfolio, "get_price") as mock_price:
            # Simulate a month of trading with varying prices
            prices = [100, 102, 98, 105, 103, 110, 108, 115]
            dates = [f"2024-01-{i + 1:02d}" for i in range(len(prices))]

            # Buy on day 1
            mock_price.return_value = float(prices[0])
            portfolio.execute_signal("AAPL", Signal.BUY, dates[0], shares=200)

            # Take snapshots for each day
            for price, date in zip(prices, dates, strict=True):
                mock_price.return_value = float(price)
                portfolio.take_snapshot(date)

            # Sell on last day
            mock_price.return_value = float(prices[-1])
            portfolio.execute_signal("AAPL", Signal.SELL, dates[-1])

        # Calculate metrics
        metrics = calculate_metrics(portfolio)

        # Verify data flows correctly
        assert metrics.trading_days == len(prices)
        assert metrics.total_return_pct > 0  # Price went from 100 to 115
        assert metrics.num_trades == 2
        assert metrics.num_winning_trades == 1
        assert metrics.win_rate_pct == 100.0


class TestAsyncIntegration:
    """Integration tests for async functionality."""

    @pytest.mark.asyncio
    async def test_async_backtest_run(self) -> None:
        """Test async backtesting execution."""
        mock_graph = MagicMock()

        async def mock_propagate_async(ticker: str, date: str):
            return {"final_trade_decision": "HOLD"}, "HOLD"

        mock_graph.propagate_async = mock_propagate_async

        backtester = Backtester(
            trading_graph=mock_graph,
            initial_cash=100000.0,
        )

        with patch.object(Portfolio, "get_price", return_value=100.0):
            result = await backtester.run_async(
                ticker="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-15",
                frequency="weekly",
            )

        assert result is not None
        assert len(result.decisions) > 0


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_config_propagates_to_components(self) -> None:
        """Test that config values propagate correctly through the system."""
        from tradingagents import DataVendor, DataVendorsConfig, TradingAgentsConfig

        # Use valid vendors for each category per the validation rules
        data_vendors = DataVendorsConfig(
            core_stock_apis=DataVendor.YFINANCE,
            technical_indicators=DataVendor.YFINANCE,
            fundamental_data=DataVendor.ALPHA_VANTAGE,  # Must be openai, local, or alpha_vantage
            news_data=DataVendor.OPENAI,
        )

        config = TradingAgentsConfig(
            llm_provider="openai",
            deep_think_llm="gpt-4o",
            quick_think_llm="gpt-4o-mini",
            data_vendors=data_vendors,
            max_debate_rounds=3,
        )

        # Convert to dict and verify all values
        config_dict = config.to_dict()

        assert config_dict["llm_provider"] == "openai"
        assert config_dict["deep_think_llm"] == "gpt-4o"
        assert config_dict["max_debate_rounds"] == 3
        assert config_dict["data_vendors"]["core_stock_apis"] == "yfinance"
        assert config_dict["data_vendors"]["fundamental_data"] == "alpha_vantage"
        assert config_dict["data_vendors"]["news_data"] == "openai"

    def test_config_validation_errors(self) -> None:
        """Test that invalid configs are rejected."""
        from pydantic import ValidationError

        from tradingagents import TradingAgentsConfig

        # Invalid debate rounds
        with pytest.raises(ValidationError):
            TradingAgentsConfig(max_debate_rounds=0)

        with pytest.raises(ValidationError):
            TradingAgentsConfig(max_debate_rounds=15)

        # Empty model name
        with pytest.raises(ValidationError):
            TradingAgentsConfig(deep_think_llm="")
