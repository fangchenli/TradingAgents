"""Pydantic configuration models for TradingAgents."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Package root directory
_PACKAGE_DIR = Path(__file__).parent


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class DataVendor(str, Enum):
    """Supported data vendors."""

    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    OPENAI = "openai"
    GOOGLE = "google"
    LOCAL = "local"


class DataVendorsConfig(BaseModel):
    """Configuration for data vendor sources."""

    model_config = ConfigDict(use_enum_values=True)

    core_stock_apis: DataVendor = Field(
        default=DataVendor.YFINANCE,
        description="Vendor for core stock data (price, volume, etc.)",
    )
    technical_indicators: DataVendor = Field(
        default=DataVendor.YFINANCE,
        description="Vendor for technical indicators (MACD, RSI, etc.)",
    )
    fundamental_data: DataVendor = Field(
        default=DataVendor.ALPHA_VANTAGE,
        description="Vendor for fundamental data (financials, ratios)",
    )
    news_data: DataVendor = Field(
        default=DataVendor.ALPHA_VANTAGE,
        description="Vendor for news and sentiment data",
    )

    @field_validator("core_stock_apis", "technical_indicators")
    @classmethod
    def validate_stock_vendors(cls, v: DataVendor) -> DataVendor:
        """Validate that stock/indicator vendors are appropriate."""
        valid = {DataVendor.YFINANCE, DataVendor.ALPHA_VANTAGE, DataVendor.LOCAL}
        if v not in valid:
            raise ValueError(f"Must be one of: {[x.value for x in valid]}")
        return v

    @field_validator("fundamental_data")
    @classmethod
    def validate_fundamental_vendors(cls, v: DataVendor) -> DataVendor:
        """Validate that fundamental data vendors are appropriate."""
        valid = {DataVendor.OPENAI, DataVendor.ALPHA_VANTAGE, DataVendor.LOCAL}
        if v not in valid:
            raise ValueError(f"Must be one of: {[x.value for x in valid]}")
        return v

    @field_validator("news_data")
    @classmethod
    def validate_news_vendors(cls, v: DataVendor) -> DataVendor:
        """Validate that news data vendors are appropriate."""
        valid = {DataVendor.OPENAI, DataVendor.ALPHA_VANTAGE, DataVendor.GOOGLE, DataVendor.LOCAL}
        if v not in valid:
            raise ValueError(f"Must be one of: {[x.value for x in valid]}")
        return v


class TradingAgentsConfig(BaseModel):
    """Main configuration for TradingAgents framework.

    This configuration can be loaded from environment variables, a config file,
    or passed directly. Environment variables take precedence.

    Example:
        >>> from tradingagents.config import TradingAgentsConfig
        >>> config = TradingAgentsConfig()  # Uses defaults
        >>> config = TradingAgentsConfig(llm_provider="anthropic", deep_think_llm="claude-3-opus")
    """

    model_config = ConfigDict(
        use_enum_values=True,
        validate_default=True,
        extra="forbid",  # Raise error on unknown fields
    )

    # Directory settings
    project_dir: Path = Field(
        default=_PACKAGE_DIR,
        description="Root directory of the tradingagents package",
    )
    results_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results")),
        description="Directory for output results",
    )
    data_cache_dir: Path = Field(
        default=_PACKAGE_DIR / "dataflows" / "data_cache",
        description="Directory for caching data from vendors",
    )

    # LLM settings
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider to use",
    )
    deep_think_llm: str = Field(
        default="gpt-4o",
        description="Model for deep thinking tasks (research, analysis)",
        min_length=1,
    )
    quick_think_llm: str = Field(
        default="gpt-4o-mini",
        description="Model for quick tasks (reflection, signal processing)",
        min_length=1,
    )
    backend_url: Annotated[str, Field(description="Base URL for LLM API")] = (
        "https://api.openai.com/v1"
    )

    # Debate and discussion settings
    max_debate_rounds: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum rounds of bull/bear debate",
    )
    max_risk_discuss_rounds: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum rounds of risk analysis discussion",
    )
    max_recur_limit: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum recursion limit for graph execution",
    )

    # Memory settings
    memory_persistence: bool = Field(
        default=True,
        description="Whether to persist memories to disk between runs",
    )
    memory_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("TRADINGAGENTS_MEMORY_DIR", "./memory")),
        description="Directory for persistent memory storage",
    )

    # Data vendor configuration
    data_vendors: DataVendorsConfig = Field(
        default_factory=DataVendorsConfig,
        description="Configuration for data sources",
    )

    # Tool-level vendor overrides
    tool_vendors: dict[str, str] = Field(
        default_factory=dict,
        description="Per-tool vendor overrides (tool_name -> vendor)",
    )

    @model_validator(mode="after")
    def validate_backend_url_matches_provider(self) -> TradingAgentsConfig:
        """Validate that backend_url is appropriate for the provider."""
        # Provider-specific default URLs for reference
        # OPENAI: "https://api.openai.com/v1"
        # ANTHROPIC: "https://api.anthropic.com"
        # OLLAMA: "http://localhost:11434/v1"
        # User might have intentionally set a custom URL, so we don't validate
        return self

    def to_dict(self) -> dict:
        """Convert config to dictionary format for backward compatibility.

        Returns a dict with string paths (not Path objects) for compatibility
        with existing code that expects the old dict format.
        """
        data = self.model_dump()
        # Convert Path objects to strings for backward compatibility
        for key in ["project_dir", "results_dir", "data_cache_dir", "memory_dir"]:
            if key in data and isinstance(data[key], Path):
                data[key] = str(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict) -> TradingAgentsConfig:
        """Create config from dictionary.

        Handles the old dict format with string paths.
        """
        return cls.model_validate(data)


# Default configuration instance
DEFAULT_CONFIG = TradingAgentsConfig()
