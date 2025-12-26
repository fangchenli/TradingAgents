"""Tests for graph execution modules."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.graph.propagation import Propagator
from tradingagents.graph.reflection import Reflector


class TestPropagator:
    """Tests for Propagator class."""

    def test_init_default_values(self) -> None:
        """Verify default initialization values."""
        propagator = Propagator()
        assert propagator.max_recur_limit == 100

    def test_init_custom_values(self) -> None:
        """Verify custom initialization values."""
        propagator = Propagator(max_recur_limit=50)
        assert propagator.max_recur_limit == 50

    def test_create_initial_state_structure(self) -> None:
        """Verify initial state has correct structure."""
        propagator = Propagator()
        state = propagator.create_initial_state("AAPL", "2024-01-15")

        # Check required keys
        assert "messages" in state
        assert "company_of_interest" in state
        assert "trade_date" in state
        assert "investment_debate_state" in state
        assert "risk_debate_state" in state
        assert "market_report" in state
        assert "fundamentals_report" in state
        assert "sentiment_report" in state
        assert "news_report" in state

    def test_create_initial_state_values(self) -> None:
        """Verify initial state values are set correctly."""
        propagator = Propagator()
        state = propagator.create_initial_state("NVDA", "2024-05-10")

        assert state["company_of_interest"] == "NVDA"
        assert state["trade_date"] == "2024-05-10"
        assert state["market_report"] == ""
        assert state["fundamentals_report"] == ""

    def test_create_initial_state_debate_states(self) -> None:
        """Verify debate states are initialized correctly."""
        propagator = Propagator()
        state = propagator.create_initial_state("TSLA", "2024-03-01")

        invest_state = state["investment_debate_state"]
        assert invest_state["count"] == 0
        assert invest_state["history"] == ""

        risk_state = state["risk_debate_state"]
        assert risk_state["count"] == 0
        assert risk_state["history"] == ""

    def test_get_graph_args(self) -> None:
        """Verify graph arguments are correct."""
        propagator = Propagator(max_recur_limit=75)
        args = propagator.get_graph_args()

        assert args["stream_mode"] == "values"
        assert args["config"]["recursion_limit"] == 75


class TestReflector:
    """Tests for Reflector class."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Return a mock LLM."""
        mock = MagicMock()
        mock.invoke.return_value = MagicMock(
            content="Lesson learned: Should have been more cautious."
        )
        return mock

    @pytest.fixture
    def sample_state(self) -> dict[str, Any]:
        """Return a sample final state for reflection."""
        return {
            "company_of_interest": "AAPL",
            "trade_date": "2024-01-15",
            "market_report": "Market is bullish",
            "sentiment_report": "Positive sentiment",
            "news_report": "Good earnings",
            "fundamentals_report": "Strong balance sheet",
            "investment_debate_state": {
                "bull_history": "Bull argument",
                "bear_history": "Bear argument",
                "history": "Full debate",
                "current_response": "Bull wins",
                "judge_decision": "Invest",
                "count": 2,
            },
            "investment_plan": "Buy 100 shares",
            "trader_investment_plan": "Execute buy",
            "risk_debate_state": {
                "risky_history": "Risky view",
                "safe_history": "Safe view",
                "neutral_history": "Neutral view",
                "history": "Risk debate",
                "latest_speaker": "Neutral",
                "current_risky_response": "",
                "current_safe_response": "",
                "current_neutral_response": "",
                "judge_decision": "Approved",
                "count": 3,
            },
            "final_trade_decision": "BUY",
        }

    def test_reflector_init(self, mock_llm: MagicMock) -> None:
        """Verify Reflector initialization."""
        reflector = Reflector(mock_llm)
        assert reflector.quick_thinking_llm is mock_llm

    def test_reflect_bull_researcher(
        self, mock_llm: MagicMock, sample_state: dict[str, Any]
    ) -> None:
        """Verify bull researcher reflection calls LLM."""
        reflector = Reflector(mock_llm)
        mock_memory = MagicMock()

        reflector.reflect_bull_researcher(sample_state, 500, mock_memory)

        # Verify LLM was invoked
        mock_llm.invoke.assert_called_once()

        # Verify memory was updated
        mock_memory.add_situations.assert_called_once()

    def test_reflect_bear_researcher(
        self, mock_llm: MagicMock, sample_state: dict[str, Any]
    ) -> None:
        """Verify bear researcher reflection calls LLM."""
        reflector = Reflector(mock_llm)
        mock_memory = MagicMock()

        reflector.reflect_bear_researcher(sample_state, -200, mock_memory)

        mock_llm.invoke.assert_called_once()
        mock_memory.add_situations.assert_called_once()

    def test_reflect_trader(self, mock_llm: MagicMock, sample_state: dict[str, Any]) -> None:
        """Verify trader reflection calls LLM."""
        reflector = Reflector(mock_llm)
        mock_memory = MagicMock()

        reflector.reflect_trader(sample_state, 1000, mock_memory)

        mock_llm.invoke.assert_called_once()
        mock_memory.add_situations.assert_called_once()

    def test_reflect_invest_judge(self, mock_llm: MagicMock, sample_state: dict[str, Any]) -> None:
        """Verify investment judge reflection calls LLM."""
        reflector = Reflector(mock_llm)
        mock_memory = MagicMock()

        reflector.reflect_invest_judge(sample_state, 0, mock_memory)

        mock_llm.invoke.assert_called_once()
        mock_memory.add_situations.assert_called_once()

    def test_reflect_risk_manager(self, mock_llm: MagicMock, sample_state: dict[str, Any]) -> None:
        """Verify risk manager reflection calls LLM."""
        reflector = Reflector(mock_llm)
        mock_memory = MagicMock()

        reflector.reflect_risk_manager(sample_state, -500, mock_memory)

        mock_llm.invoke.assert_called_once()
        mock_memory.add_situations.assert_called_once()


class TestTradingAgentsGraphInit:
    """Tests for TradingAgentsGraph initialization (without full execution)."""

    @patch("tradingagents.graph.trading_graph.ChatOpenAI")
    @patch("tradingagents.graph.trading_graph.FinancialSituationMemory")
    @patch("tradingagents.graph.trading_graph.set_config")
    def test_init_creates_llms(
        self,
        mock_set_config: MagicMock,
        mock_memory_class: MagicMock,
        mock_chat_openai: MagicMock,
    ) -> None:
        """Verify TradingAgentsGraph creates LLMs on init."""
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = {
            "project_dir": "/tmp",
            "llm_provider": "openai",
            "deep_think_llm": "gpt-4o",
            "quick_think_llm": "gpt-4o-mini",
            "backend_url": "https://api.openai.com/v1",
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "max_recur_limit": 100,
            "data_vendors": {
                "core_stock_apis": "yfinance",
                "technical_indicators": "yfinance",
                "fundamental_data": "alpha_vantage",
                "news_data": "alpha_vantage",
            },
            "tool_vendors": {},
        }

        _ta = TradingAgentsGraph(config=config)

        # Verify ChatOpenAI was called for both LLMs
        assert mock_chat_openai.call_count == 2

        # Verify memories were created
        assert mock_memory_class.call_count == 5  # bull, bear, trader, invest_judge, risk_manager
