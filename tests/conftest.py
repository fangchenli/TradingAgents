"""Pytest configuration and fixtures."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Return a sample configuration for testing."""
    return {
        "project_dir": "/tmp/test_project",
        "results_dir": "/tmp/test_results",
        "data_cache_dir": "/tmp/test_cache",
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


@pytest.fixture
def mock_llm() -> MagicMock:
    """Return a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="BUY")
    return mock


@pytest.fixture
def sample_agent_state() -> dict[str, Any]:
    """Return a sample agent state for testing."""
    return {
        "messages": [],
        "company_of_interest": "AAPL",
        "trade_date": "2024-01-15",
        "sender": "test",
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {
            "bull_history": "",
            "bear_history": "",
            "history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": {
            "risky_history": "",
            "safe_history": "",
            "neutral_history": "",
            "history": "",
            "latest_speaker": "",
            "current_risky_response": "",
            "current_safe_response": "",
            "current_neutral_response": "",
            "judge_decision": "",
            "count": 0,
        },
        "final_trade_decision": "",
    }
