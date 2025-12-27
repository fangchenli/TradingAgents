# TradingAgents/graph/trading_graph.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_sentiment,
    get_insider_transactions,
    get_news,
    get_stock_data,
)
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.config import DEFAULT_CONFIG, TradingAgentsConfig
from tradingagents.dataflows.config import set_config

from .conditional_logic import ConditionalLogic
from .propagation import Propagator
from .reflection import Reflector
from .setup import GraphSetup
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts: list[str] | None = None,
        debug: bool = False,
        config: TradingAgentsConfig | dict[str, Any] | None = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration (TradingAgentsConfig, dict, or None for defaults)
        """
        if selected_analysts is None:
            selected_analysts = ["market", "social", "news", "fundamentals"]
        self.debug = debug

        # Handle config: accept both Pydantic model and dict
        if config is None:
            self._config = DEFAULT_CONFIG
        elif isinstance(config, TradingAgentsConfig):
            self._config = config
        else:
            self._config = TradingAgentsConfig.from_dict(config)

        # Convert to dict for backward compatibility with existing code
        self.config = self._config.to_dict()

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs
        if (
            self.config["llm_provider"].lower() == "openai"
            or self.config["llm_provider"] == "ollama"
            or self.config["llm_provider"] == "openrouter"
        ):
            self.deep_thinking_llm = ChatOpenAI(
                model=self.config["deep_think_llm"], base_url=self.config["backend_url"]
            )
            self.quick_thinking_llm = ChatOpenAI(
                model=self.config["quick_think_llm"], base_url=self.config["backend_url"]
            )
        elif self.config["llm_provider"].lower() == "anthropic":
            self.deep_thinking_llm = ChatAnthropic(
                model=self.config["deep_think_llm"], base_url=self.config["backend_url"]
            )
            self.quick_thinking_llm = ChatAnthropic(
                model=self.config["quick_think_llm"], base_url=self.config["backend_url"]
            )
        elif self.config["llm_provider"].lower() == "google":
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"])
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"])
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")

        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config.get("max_debate_rounds", 1),
            max_risk_discuss_rounds=self.config.get("max_risk_discuss_rounds", 1),
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _create_tool_nodes(self) -> dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_sentiment,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(company_name, trade_date)
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            # LangGraph stream yields {node_name: state_update} chunks
            final_state = None
            seen_messages = set()
            for chunk in self.graph.stream(init_agent_state, **args):
                # Each chunk is {node_name: state_update}
                for _node_name, state_update in chunk.items():
                    if "messages" in state_update and state_update["messages"]:
                        # Print new messages (avoid duplicates)
                        for msg in state_update["messages"]:
                            msg_id = getattr(msg, "id", None) or id(msg)
                            if msg_id not in seen_messages:
                                seen_messages.add(msg_id)
                                msg.pretty_print()
                    # Keep track of the latest state update
                    if final_state is None:
                        final_state = state_update
                    else:
                        final_state.update(state_update)

            # If streaming didn't work as expected, fall back to invoke
            if final_state is None:
                final_state = self.graph.invoke(init_agent_state, **args)
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"]["current_response"],
                "judge_decision": final_state["investment_debate_state"]["judge_decision"],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(self.curr_state, returns_losses, self.bull_memory)
        self.reflector.reflect_bear_researcher(self.curr_state, returns_losses, self.bear_memory)
        self.reflector.reflect_trader(self.curr_state, returns_losses, self.trader_memory)
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)

    # Async methods
    async def propagate_async(self, company_name: str, trade_date: str):
        """Async version: Run the trading agents graph for a company on a specific date.

        This is useful when you want to run multiple analyses concurrently
        or integrate with async web frameworks.
        """

        from tradingagents.async_utils import to_async

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(company_name, trade_date)
        args = self.propagator.get_graph_args()

        # LangGraph's ainvoke for async execution
        if hasattr(self.graph, "ainvoke"):
            final_state = await self.graph.ainvoke(init_agent_state, **args)
        else:
            # Fallback: run sync invoke in thread pool
            final_state = await to_async(self.graph.invoke, init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state (file I/O in thread)
        await to_async(self._log_state, trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    async def reflect_and_remember_async(self, returns_losses):
        """Async version: Reflect on decisions and update memory based on returns."""
        import asyncio

        from tradingagents.async_utils import to_async

        # Run all reflections concurrently
        await asyncio.gather(
            to_async(
                self.reflector.reflect_bull_researcher,
                self.curr_state,
                returns_losses,
                self.bull_memory,
            ),
            to_async(
                self.reflector.reflect_bear_researcher,
                self.curr_state,
                returns_losses,
                self.bear_memory,
            ),
            to_async(
                self.reflector.reflect_trader,
                self.curr_state,
                returns_losses,
                self.trader_memory,
            ),
            to_async(
                self.reflector.reflect_invest_judge,
                self.curr_state,
                returns_losses,
                self.invest_judge_memory,
            ),
            to_async(
                self.reflector.reflect_risk_manager,
                self.curr_state,
                returns_losses,
                self.risk_manager_memory,
            ),
        )
