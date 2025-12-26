# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradingAgents is a multi-agent LLM financial trading framework that mirrors real-world trading firms. It uses LangGraph for workflow orchestration and LangChain for LLM integration, creating a system where specialized AI agents collaborate on market analysis and trading decisions.

## Setup & Development

```bash
# Install with uv (recommended)
uv sync --all-extras

# Required environment variables (see .env.example)
OPENAI_API_KEY=...
ALPHA_VANTAGE_API_KEY=...

# Run interactive CLI
uv run python -m cli.main

# Run tests
uv run pytest tests/ -v

# Run linter
uv run ruff check src/ tests/

# Run type checker
uv run mypy src/
```

## Python API Usage

```python
from tradingagents import TradingAgentsGraph, DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4o-mini"
config["quick_think_llm"] = "gpt-4o-mini"

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2024-05-10")

# After trade execution, reflect on results
# ta.reflect_and_remember(position_returns)
```

## Architecture

The framework processes trading decisions through a pipeline of agent teams:

**1. Analyst Team** (parallel execution)
- Market Analyst: Technical indicators (MACD, RSI, Bollinger Bands, ATR, etc.)
- Social Media Analyst: Reddit/social sentiment
- News Analyst: News and macroeconomic indicators
- Fundamentals Analyst: Balance sheets, P/E ratios, financials

**2. Research Team** (structured debate)
- Bull Researcher: Pro-investment arguments
- Bear Researcher: Risk/bear arguments
- Research Manager: Synthesizes debate into investment plan

**3. Trader Agent**: Converts plan to BUY/HOLD/SELL decision

**4. Risk Management Team** (final review debate)
- Conservative/Aggressive/Neutral Debators
- Risk Manager: Final approval/rejection

**Memory System**: ChromaDB-based embeddings store past decisions for agents to learn from similar situations.

## Key Configuration (default_config.py)

```python
{
    "llm_provider": "openai",  # openai, anthropic, google, ollama, openrouter
    "deep_think_llm": "gpt-4o",
    "quick_think_llm": "gpt-4o-mini",
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "data_vendors": {
        "core_stock_apis": "yfinance",       # yfinance, alpha_vantage, local
        "technical_indicators": "yfinance",  # yfinance, alpha_vantage, local
        "fundamental_data": "alpha_vantage", # openai, alpha_vantage, local
        "news_data": "alpha_vantage",        # openai, alpha_vantage, google, local
    }
}
```

## Directory Structure (src layout)

```
src/
└── tradingagents/           # Main package
    ├── agents/              # Agent implementations
    │   ├── analysts/        # Market, social, news, fundamentals analysts
    │   ├── researchers/     # Bull/bear debate agents
    │   ├── managers/        # Research and risk managers
    │   ├── trader/          # Trading decision agent
    │   ├── risk_mgmt/       # Risk debate agents
    │   └── utils/           # Agent states, tools, memory system
    ├── graph/               # LangGraph orchestration
    │   ├── trading_graph.py # Main orchestrator class
    │   ├── setup.py         # Graph construction
    │   ├── conditional_logic.py
    │   └── reflection.py    # Learning from past trades
    ├── dataflows/           # Data vendor abstraction layer
    │   ├── interface.py     # Vendor routing
    │   ├── y_finance.py
    │   ├── alpha_vantage.py
    │   └── ...
    ├── cli/                 # Interactive CLI
    │   └── main.py
    └── default_config.py
tests/                       # Unit tests
```

## Extension Points

- **New analysts**: Add to `src/tradingagents/agents/analysts/`, follow existing patterns
- **New data vendors**: Add to `src/tradingagents/dataflows/`, register in `interface.py`
- **Custom agent prompts**: Modify system prompts in agent creation functions
- **State modifications**: Update `agent_states.py` (AgentState, InvestDebateState, RiskDebateState)

## Important Notes

- Python 3.11+ required
- Uses uv for dependency management
- The framework makes many API calls; use smaller models for cost control
- yfinance is free for price data; Alpha Vantage requires API key for fundamentals
- Debate rounds are configurable to trade off deliberation depth vs. speed/cost
