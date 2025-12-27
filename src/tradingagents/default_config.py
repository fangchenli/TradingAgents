from __future__ import annotations

import os
from pathlib import Path

# Package root directory
_PACKAGE_DIR = Path(__file__).parent

DEFAULT_CONFIG = {
    # Directory settings
    "project_dir": str(_PACKAGE_DIR),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": str(_PACKAGE_DIR / "dataflows" / "data_cache"),
    # LLM settings
    "llm_provider": "openai",  # Options: openai, anthropic, google, ollama, openrouter
    "deep_think_llm": "gpt-4o",
    "quick_think_llm": "gpt-4o-mini",
    "backend_url": "https://api.openai.com/v1",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Memory settings
    "memory_persistence": True,  # If True, memories persist to disk between runs
    "memory_dir": os.getenv("TRADINGAGENTS_MEMORY_DIR", "./memory"),
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",  # Options: yfinance, alpha_vantage, local
        "technical_indicators": "yfinance",  # Options: yfinance, alpha_vantage, local
        "fundamental_data": "alpha_vantage",  # Options: openai, alpha_vantage, local
        "news_data": "alpha_vantage",  # Options: openai, alpha_vantage, google, local
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
        # Example: "get_news": "openai",               # Override category default
    },
}
