"""Tests for configuration module."""

from __future__ import annotations

from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG."""

    def test_config_has_required_keys(self) -> None:
        """Verify all required configuration keys are present."""
        required_keys = [
            "project_dir",
            "results_dir",
            "data_cache_dir",
            "llm_provider",
            "deep_think_llm",
            "quick_think_llm",
            "backend_url",
            "max_debate_rounds",
            "max_risk_discuss_rounds",
            "max_recur_limit",
            "data_vendors",
            "tool_vendors",
        ]
        for key in required_keys:
            assert key in DEFAULT_CONFIG, f"Missing required config key: {key}"

    def test_project_dir_is_valid_path(self) -> None:
        """Verify project_dir points to a valid directory."""
        project_dir = Path(DEFAULT_CONFIG["project_dir"])
        assert project_dir.exists(), "project_dir should exist"
        assert project_dir.is_dir(), "project_dir should be a directory"

    def test_llm_provider_is_valid(self) -> None:
        """Verify llm_provider is a supported value."""
        valid_providers = ["openai", "anthropic", "google", "ollama", "openrouter"]
        assert DEFAULT_CONFIG["llm_provider"] in valid_providers

    def test_data_vendors_has_required_categories(self) -> None:
        """Verify data_vendors has all required categories."""
        required_categories = [
            "core_stock_apis",
            "technical_indicators",
            "fundamental_data",
            "news_data",
        ]
        for category in required_categories:
            assert category in DEFAULT_CONFIG["data_vendors"]

    def test_debate_rounds_are_positive(self) -> None:
        """Verify debate round settings are positive integers."""
        assert DEFAULT_CONFIG["max_debate_rounds"] > 0
        assert DEFAULT_CONFIG["max_risk_discuss_rounds"] > 0

    def test_config_is_copyable(self) -> None:
        """Verify config can be safely copied and modified."""
        config_copy = DEFAULT_CONFIG.copy()
        config_copy["llm_provider"] = "anthropic"
        assert DEFAULT_CONFIG["llm_provider"] == "openai"
        assert config_copy["llm_provider"] == "anthropic"
