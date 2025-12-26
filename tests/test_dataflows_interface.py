"""Tests for dataflows interface module."""

from __future__ import annotations

import pytest

from tradingagents.dataflows.config import reset_config, set_config
from tradingagents.dataflows.interface import (
    TOOLS_CATEGORIES,
    VENDOR_METHODS,
    get_category_for_method,
    get_vendor,
)


class TestToolsCategories:
    """Tests for TOOLS_CATEGORIES structure."""

    def test_has_required_categories(self) -> None:
        """Verify all required categories exist."""
        required = ["core_stock_apis", "technical_indicators", "fundamental_data", "news_data"]
        for category in required:
            assert category in TOOLS_CATEGORIES

    def test_each_category_has_tools(self) -> None:
        """Verify each category has at least one tool."""
        for category, info in TOOLS_CATEGORIES.items():
            assert "tools" in info
            assert len(info["tools"]) > 0, f"Category {category} has no tools"

    def test_each_category_has_description(self) -> None:
        """Verify each category has a description."""
        for _category, info in TOOLS_CATEGORIES.items():
            assert "description" in info
            assert len(info["description"]) > 0


class TestVendorMethods:
    """Tests for VENDOR_METHODS mapping."""

    def test_get_stock_data_has_vendors(self) -> None:
        """Verify get_stock_data has vendor implementations."""
        assert "get_stock_data" in VENDOR_METHODS
        assert "yfinance" in VENDOR_METHODS["get_stock_data"]

    def test_get_indicators_has_vendors(self) -> None:
        """Verify get_indicators has vendor implementations."""
        assert "get_indicators" in VENDOR_METHODS
        assert "yfinance" in VENDOR_METHODS["get_indicators"]

    def test_all_methods_are_callable_or_list(self) -> None:
        """Verify all vendor implementations are callable or list of callables."""
        for method, vendors in VENDOR_METHODS.items():
            for vendor, impl in vendors.items():
                if isinstance(impl, list):
                    for fn in impl:
                        assert callable(fn), f"{method}.{vendor} contains non-callable"
                else:
                    assert callable(impl), f"{method}.{vendor} is not callable"


class TestGetCategoryForMethod:
    """Tests for get_category_for_method function."""

    def test_get_stock_data_category(self) -> None:
        """Verify get_stock_data is in core_stock_apis."""
        assert get_category_for_method("get_stock_data") == "core_stock_apis"

    def test_get_indicators_category(self) -> None:
        """Verify get_indicators is in technical_indicators."""
        assert get_category_for_method("get_indicators") == "technical_indicators"

    def test_get_fundamentals_category(self) -> None:
        """Verify get_fundamentals is in fundamental_data."""
        assert get_category_for_method("get_fundamentals") == "fundamental_data"

    def test_get_news_category(self) -> None:
        """Verify get_news is in news_data."""
        assert get_category_for_method("get_news") == "news_data"

    def test_unknown_method_raises(self) -> None:
        """Verify unknown method raises ValueError."""
        with pytest.raises(ValueError, match="not found in any category"):
            get_category_for_method("unknown_method")


class TestGetVendor:
    """Tests for get_vendor function."""

    def setup_method(self) -> None:
        """Reset config before each test."""
        reset_config()

    def teardown_method(self) -> None:
        """Reset config after each test."""
        reset_config()

    def test_returns_default_vendor_for_category(self) -> None:
        """Verify default vendor is returned from config."""
        vendor = get_vendor("core_stock_apis")
        assert vendor == "yfinance"

    def test_tool_level_override_takes_precedence(self) -> None:
        """Verify tool-level config overrides category-level."""
        set_config(
            {
                "data_vendors": {"core_stock_apis": "yfinance"},
                "tool_vendors": {"get_stock_data": "alpha_vantage"},
            }
        )
        vendor = get_vendor("core_stock_apis", method="get_stock_data")
        assert vendor == "alpha_vantage"

    def test_falls_back_to_category_when_no_tool_override(self) -> None:
        """Verify category config is used when no tool override."""
        set_config(
            {
                "data_vendors": {"core_stock_apis": "alpha_vantage"},
                "tool_vendors": {},
            }
        )
        vendor = get_vendor("core_stock_apis", method="get_stock_data")
        assert vendor == "alpha_vantage"
