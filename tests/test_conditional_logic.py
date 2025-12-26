"""Tests for conditional logic module."""

from __future__ import annotations

from unittest.mock import MagicMock

from tradingagents.graph.conditional_logic import ConditionalLogic


class TestConditionalLogic:
    """Tests for ConditionalLogic class."""

    def test_init_default_values(self) -> None:
        """Verify default initialization values."""
        logic = ConditionalLogic()
        assert logic.max_debate_rounds == 1
        assert logic.max_risk_discuss_rounds == 1

    def test_init_custom_values(self) -> None:
        """Verify custom initialization values."""
        logic = ConditionalLogic(max_debate_rounds=3, max_risk_discuss_rounds=2)
        assert logic.max_debate_rounds == 3
        assert logic.max_risk_discuss_rounds == 2

    def test_should_continue_market_with_tool_calls(self) -> None:
        """Verify market continues when there are tool calls."""
        logic = ConditionalLogic()
        mock_message = MagicMock()
        mock_message.tool_calls = [{"name": "get_stock_data"}]
        state = {"messages": [mock_message]}

        result = logic.should_continue_market(state)
        assert result == "tools_market"

    def test_should_continue_market_without_tool_calls(self) -> None:
        """Verify market clears when there are no tool calls."""
        logic = ConditionalLogic()
        mock_message = MagicMock()
        mock_message.tool_calls = []
        state = {"messages": [mock_message]}

        result = logic.should_continue_market(state)
        assert result == "Msg Clear Market"

    def test_should_continue_social_with_tool_calls(self) -> None:
        """Verify social continues when there are tool calls."""
        logic = ConditionalLogic()
        mock_message = MagicMock()
        mock_message.tool_calls = [{"name": "get_news"}]
        state = {"messages": [mock_message]}

        result = logic.should_continue_social(state)
        assert result == "tools_social"

    def test_should_continue_social_without_tool_calls(self) -> None:
        """Verify social clears when there are no tool calls."""
        logic = ConditionalLogic()
        mock_message = MagicMock()
        mock_message.tool_calls = []
        state = {"messages": [mock_message]}

        result = logic.should_continue_social(state)
        assert result == "Msg Clear Social"

    def test_should_continue_debate_goes_to_bear(self) -> None:
        """Verify debate routes to bear after bull speaks."""
        logic = ConditionalLogic()
        state = {
            "investment_debate_state": {
                "count": 1,
                "current_response": "Bull: I think we should buy...",
            }
        }

        result = logic.should_continue_debate(state)
        assert result == "Bear Researcher"

    def test_should_continue_debate_goes_to_bull(self) -> None:
        """Verify debate routes to bull after bear speaks."""
        logic = ConditionalLogic()
        state = {
            "investment_debate_state": {
                "count": 1,
                "current_response": "Bear: I think we should sell...",
            }
        }

        result = logic.should_continue_debate(state)
        assert result == "Bull Researcher"

    def test_should_continue_debate_ends_at_max_rounds(self) -> None:
        """Verify debate ends after max rounds."""
        logic = ConditionalLogic(max_debate_rounds=2)
        state = {
            "investment_debate_state": {
                "count": 4,  # 2 * max_debate_rounds
                "current_response": "Bull: ...",
            }
        }

        result = logic.should_continue_debate(state)
        assert result == "Research Manager"

    def test_should_continue_risk_analysis_routes_correctly(self) -> None:
        """Verify risk analysis routes through all analysts."""
        logic = ConditionalLogic()

        # After risky speaks, go to safe
        state = {"risk_debate_state": {"count": 1, "latest_speaker": "Risky Analyst"}}
        assert logic.should_continue_risk_analysis(state) == "Safe Analyst"

        # After safe speaks, go to neutral
        state = {"risk_debate_state": {"count": 2, "latest_speaker": "Safe Analyst"}}
        assert logic.should_continue_risk_analysis(state) == "Neutral Analyst"

        # After neutral speaks, go to risky
        state = {"risk_debate_state": {"count": 2, "latest_speaker": "Neutral Analyst"}}
        assert logic.should_continue_risk_analysis(state) == "Risky Analyst"

    def test_should_continue_risk_analysis_ends_at_max_rounds(self) -> None:
        """Verify risk analysis ends after max rounds."""
        logic = ConditionalLogic(max_risk_discuss_rounds=2)
        state = {
            "risk_debate_state": {
                "count": 6,  # 3 * max_risk_discuss_rounds
                "latest_speaker": "Risky Analyst",
            }
        }

        result = logic.should_continue_risk_analysis(state)
        assert result == "Risk Judge"
