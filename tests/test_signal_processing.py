"""Tests for signal processing module."""

from __future__ import annotations

from unittest.mock import MagicMock

from tradingagents.graph.signal_processing import SignalProcessor


class TestSignalProcessor:
    """Tests for SignalProcessor class."""

    def test_init(self, mock_llm: MagicMock) -> None:
        """Verify initialization stores the LLM."""
        processor = SignalProcessor(mock_llm)
        assert processor.quick_thinking_llm is mock_llm

    def test_process_signal_returns_buy(self, mock_llm: MagicMock) -> None:
        """Verify processor extracts BUY decision."""
        mock_llm.invoke.return_value = MagicMock(content="BUY")
        processor = SignalProcessor(mock_llm)

        result = processor.process_signal("After careful analysis, we recommend buying AAPL...")
        assert result == "BUY"

    def test_process_signal_returns_sell(self, mock_llm: MagicMock) -> None:
        """Verify processor extracts SELL decision."""
        mock_llm.invoke.return_value = MagicMock(content="SELL")
        processor = SignalProcessor(mock_llm)

        result = processor.process_signal("Due to declining fundamentals, we recommend selling...")
        assert result == "SELL"

    def test_process_signal_returns_hold(self, mock_llm: MagicMock) -> None:
        """Verify processor extracts HOLD decision."""
        mock_llm.invoke.return_value = MagicMock(content="HOLD")
        processor = SignalProcessor(mock_llm)

        result = processor.process_signal("Market conditions are uncertain, recommend holding...")
        assert result == "HOLD"

    def test_process_signal_calls_llm_with_correct_messages(self, mock_llm: MagicMock) -> None:
        """Verify LLM is called with correct message format."""
        processor = SignalProcessor(mock_llm)
        test_signal = "Test trading signal"

        processor.process_signal(test_signal)

        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]

        # Check message structure
        assert len(call_args) == 2
        assert call_args[0][0] == "system"
        assert "SELL, BUY, or HOLD" in call_args[0][1]
        assert call_args[1][0] == "human"
        assert call_args[1][1] == test_signal
