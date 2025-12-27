"""Tests for Pydantic configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tradingagents.config import (
    DEFAULT_CONFIG,
    DataVendor,
    DataVendorsConfig,
    LLMProvider,
    TradingAgentsConfig,
)


class TestLLMProviderEnum:
    """Tests for LLMProvider enum."""

    def test_all_providers_have_values(self) -> None:
        """Verify all providers have string values."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GOOGLE.value == "google"
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.OPENROUTER.value == "openrouter"

    def test_provider_count(self) -> None:
        """Verify expected number of providers."""
        assert len(LLMProvider) == 5


class TestDataVendorEnum:
    """Tests for DataVendor enum."""

    def test_all_vendors_have_values(self) -> None:
        """Verify all vendors have string values."""
        assert DataVendor.YFINANCE.value == "yfinance"
        assert DataVendor.ALPHA_VANTAGE.value == "alpha_vantage"
        assert DataVendor.OPENAI.value == "openai"
        assert DataVendor.GOOGLE.value == "google"
        assert DataVendor.LOCAL.value == "local"


class TestDataVendorsConfig:
    """Tests for DataVendorsConfig."""

    def test_default_values(self) -> None:
        """Verify default vendor values."""
        config = DataVendorsConfig()
        assert config.core_stock_apis == "yfinance"
        assert config.technical_indicators == "yfinance"
        assert config.fundamental_data == "alpha_vantage"
        assert config.news_data == "alpha_vantage"

    def test_custom_values(self) -> None:
        """Verify custom vendor values are accepted."""
        config = DataVendorsConfig(
            core_stock_apis=DataVendor.ALPHA_VANTAGE,
            technical_indicators=DataVendor.LOCAL,
            fundamental_data=DataVendor.OPENAI,
            news_data=DataVendor.GOOGLE,
        )
        assert config.core_stock_apis == "alpha_vantage"
        assert config.news_data == "google"

    def test_invalid_stock_vendor_rejected(self) -> None:
        """Verify invalid vendors for stock APIs are rejected."""
        with pytest.raises(ValidationError):
            DataVendorsConfig(core_stock_apis=DataVendor.GOOGLE)

    def test_invalid_fundamental_vendor_rejected(self) -> None:
        """Verify invalid vendors for fundamental data are rejected."""
        with pytest.raises(ValidationError):
            DataVendorsConfig(fundamental_data=DataVendor.YFINANCE)


class TestTradingAgentsConfig:
    """Tests for TradingAgentsConfig."""

    def test_default_config_is_valid(self) -> None:
        """Verify default configuration is valid."""
        config = TradingAgentsConfig()
        assert config.llm_provider == "openai"
        assert config.deep_think_llm == "gpt-4o"
        assert config.quick_think_llm == "gpt-4o-mini"
        assert config.max_debate_rounds == 1
        assert config.memory_persistence is True

    def test_custom_llm_provider(self) -> None:
        """Verify custom LLM provider is accepted."""
        config = TradingAgentsConfig(llm_provider=LLMProvider.ANTHROPIC)
        assert config.llm_provider == "anthropic"

    def test_custom_llm_models(self) -> None:
        """Verify custom LLM model names are accepted."""
        config = TradingAgentsConfig(
            deep_think_llm="claude-3-opus",
            quick_think_llm="claude-3-haiku",
        )
        assert config.deep_think_llm == "claude-3-opus"
        assert config.quick_think_llm == "claude-3-haiku"

    def test_debate_rounds_validation(self) -> None:
        """Verify debate rounds are validated."""
        # Valid range
        config = TradingAgentsConfig(max_debate_rounds=5)
        assert config.max_debate_rounds == 5

        # Below minimum
        with pytest.raises(ValidationError):
            TradingAgentsConfig(max_debate_rounds=0)

        # Above maximum
        with pytest.raises(ValidationError):
            TradingAgentsConfig(max_debate_rounds=100)

    def test_empty_model_name_rejected(self) -> None:
        """Verify empty model names are rejected."""
        with pytest.raises(ValidationError):
            TradingAgentsConfig(deep_think_llm="")

    def test_extra_fields_rejected(self) -> None:
        """Verify unknown fields are rejected."""
        with pytest.raises(ValidationError):
            TradingAgentsConfig(unknown_field="value")

    def test_to_dict_returns_dict(self) -> None:
        """Verify to_dict returns a dictionary."""
        config = TradingAgentsConfig()
        result = config.to_dict()
        assert isinstance(result, dict)
        assert "llm_provider" in result
        assert "data_vendors" in result

    def test_to_dict_converts_paths_to_strings(self) -> None:
        """Verify to_dict converts Path objects to strings."""
        config = TradingAgentsConfig()
        result = config.to_dict()
        assert isinstance(result["project_dir"], str)
        assert isinstance(result["results_dir"], str)
        assert isinstance(result["memory_dir"], str)

    def test_from_dict_creates_config(self) -> None:
        """Verify from_dict creates a valid config."""
        data = {
            "llm_provider": "anthropic",
            "deep_think_llm": "claude-3-opus",
            "max_debate_rounds": 3,
        }
        config = TradingAgentsConfig.from_dict(data)
        assert config.llm_provider == "anthropic"
        assert config.deep_think_llm == "claude-3-opus"
        assert config.max_debate_rounds == 3

    def test_project_dir_is_valid_path(self) -> None:
        """Verify project_dir is a valid path."""
        config = TradingAgentsConfig()
        assert isinstance(config.project_dir, Path)
        assert config.project_dir.exists()


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG constant."""

    def test_default_config_is_tradingagentsconfig(self) -> None:
        """Verify DEFAULT_CONFIG is a TradingAgentsConfig instance."""
        assert isinstance(DEFAULT_CONFIG, TradingAgentsConfig)

    def test_default_config_has_expected_values(self) -> None:
        """Verify DEFAULT_CONFIG has expected default values."""
        assert DEFAULT_CONFIG.llm_provider == "openai"
        assert DEFAULT_CONFIG.max_debate_rounds == 1
        assert DEFAULT_CONFIG.memory_persistence is True


class TestBackwardCompatibility:
    """Tests for backward compatibility with dict-based config."""

    def test_dict_config_accepted_by_from_dict(self) -> None:
        """Verify old-style dict configs work with from_dict."""
        old_style_config = {
            "llm_provider": "openai",
            "deep_think_llm": "gpt-4",
            "quick_think_llm": "gpt-3.5-turbo",
            "backend_url": "https://api.openai.com/v1",
            "max_debate_rounds": 2,
            "max_risk_discuss_rounds": 2,
            "data_vendors": {
                "core_stock_apis": "yfinance",
                "technical_indicators": "yfinance",
                "fundamental_data": "alpha_vantage",
                "news_data": "alpha_vantage",
            },
        }
        config = TradingAgentsConfig.from_dict(old_style_config)
        assert config.llm_provider == "openai"
        assert config.max_debate_rounds == 2

    def test_to_dict_roundtrip(self) -> None:
        """Verify config survives to_dict -> from_dict roundtrip."""
        original = TradingAgentsConfig(
            llm_provider=LLMProvider.ANTHROPIC,
            max_debate_rounds=3,
        )
        dict_form = original.to_dict()
        restored = TradingAgentsConfig.from_dict(dict_form)

        assert restored.llm_provider == original.llm_provider
        assert restored.max_debate_rounds == original.max_debate_rounds
