"""Tests for dataflows configuration module."""

from __future__ import annotations

from tradingagents.dataflows.config import get_config, reset_config, set_config


class TestDataflowsConfig:
    """Tests for dataflows config functions."""

    def setup_method(self) -> None:
        """Reset config before each test."""
        reset_config()

    def teardown_method(self) -> None:
        """Reset config after each test."""
        reset_config()

    def test_get_config_returns_default(self) -> None:
        """Verify get_config returns default configuration."""
        config = get_config()
        assert "llm_provider" in config
        assert "data_vendors" in config

    def test_get_config_returns_copy(self) -> None:
        """Verify get_config returns a copy, not the original."""
        config1 = get_config()
        config1["llm_provider"] = "modified"

        config2 = get_config()
        assert config2["llm_provider"] != "modified"

    def test_set_config_updates_values(self) -> None:
        """Verify set_config updates configuration values."""
        set_config({"llm_provider": "anthropic"})

        config = get_config()
        assert config["llm_provider"] == "anthropic"

    def test_set_config_preserves_unmodified_values(self) -> None:
        """Verify set_config preserves values not being updated."""
        original = get_config()
        original_max_rounds = original["max_debate_rounds"]

        set_config({"llm_provider": "anthropic"})

        config = get_config()
        assert config["max_debate_rounds"] == original_max_rounds

    def test_reset_config_clears_state(self) -> None:
        """Verify reset_config clears the configuration state."""
        set_config({"llm_provider": "anthropic"})
        reset_config()

        # After reset, get_config should return fresh defaults
        config = get_config()
        assert config["llm_provider"] == "openai"
