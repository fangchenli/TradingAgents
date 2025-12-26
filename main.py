from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

# Load environment variables from .env file
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4o-mini"
config["quick_think_llm"] = "gpt-4o-mini"
config["max_debate_rounds"] = 1

# Configure data vendors - using yfinance (free) and openai for analysis
config["data_vendors"] = {
    "core_stock_apis": "yfinance",  # Free stock data
    "technical_indicators": "yfinance",  # Free technical indicators
    "fundamental_data": "yfinance",  # Use yfinance for fundamentals (free)
    "news_data": "openai",  # Use OpenAI for news analysis
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# Forward propagate - analyze NVDA for a recent date
_, decision = ta.propagate("NVDA", "2024-12-20")
print(f"\n{'='*50}")
print(f"Final Decision: {decision}")
print(f"{'='*50}")

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000)  # parameter is the position returns
