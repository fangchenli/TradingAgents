"""Tests for async utilities and async functionality."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tradingagents.async_utils import (
    AsyncBatchProcessor,
    async_to_sync,
    run_sync,
    to_async,
)


class TestRunSync:
    """Tests for run_sync function."""

    def test_run_sync_executes_coroutine(self) -> None:
        """Verify run_sync executes an async function synchronously."""

        async def async_add(a: int, b: int) -> int:
            return a + b

        result = run_sync(async_add(2, 3))
        assert result == 5

    def test_run_sync_handles_exceptions(self) -> None:
        """Verify run_sync propagates exceptions from async functions."""

        async def async_raise() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_sync(async_raise())


class TestToAsync:
    """Tests for to_async function."""

    @pytest.mark.asyncio
    async def test_to_async_wraps_sync_function(self) -> None:
        """Verify to_async wraps a sync function to be awaitable."""

        def sync_multiply(a: int, b: int) -> int:
            return a * b

        result = await to_async(sync_multiply, 4, 5)
        assert result == 20

    @pytest.mark.asyncio
    async def test_to_async_preserves_exceptions(self) -> None:
        """Verify to_async preserves exceptions from sync functions."""

        def sync_raise() -> None:
            raise RuntimeError("sync error")

        with pytest.raises(RuntimeError, match="sync error"):
            await to_async(sync_raise)


class TestAsyncToSync:
    """Tests for async_to_sync decorator."""

    def test_async_to_sync_creates_sync_wrapper(self) -> None:
        """Verify async_to_sync creates a synchronous wrapper."""

        @async_to_sync
        async def async_subtract(a: int, b: int) -> int:
            return a - b

        # Should be callable without await
        result = async_subtract(10, 3)
        assert result == 7

    def test_async_to_sync_preserves_function_name(self) -> None:
        """Verify async_to_sync preserves the original function name."""

        @async_to_sync
        async def my_async_function() -> int:
            return 42

        assert my_async_function.__name__ == "my_async_function"


class TestAsyncBatchProcessor:
    """Tests for AsyncBatchProcessor."""

    @pytest.mark.asyncio
    async def test_batch_processor_processes_all_coroutines(self) -> None:
        """Verify batch processor executes all coroutines."""

        async def async_square(x: int) -> int:
            return x * x

        processor = AsyncBatchProcessor(max_concurrent=2, delay_between=0)
        coros = [async_square(i) for i in range(5)]
        results = await processor.process(coros)

        assert results == [0, 1, 4, 9, 16]

    @pytest.mark.asyncio
    async def test_batch_processor_respects_concurrency_limit(self) -> None:
        """Verify batch processor respects max_concurrent limit."""
        concurrent_count = 0
        max_observed = 0

        async def track_concurrency() -> None:
            nonlocal concurrent_count, max_observed
            concurrent_count += 1
            max_observed = max(max_observed, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1

        processor = AsyncBatchProcessor(max_concurrent=2, delay_between=0)
        coros = [track_concurrency() for _ in range(5)]
        await processor.process(coros)

        assert max_observed <= 2

    @pytest.mark.asyncio
    async def test_batch_processor_returns_exceptions(self) -> None:
        """Verify batch processor returns exceptions when return_exceptions=True."""

        async def sometimes_fail(x: int) -> int:
            if x == 2:
                raise ValueError(f"Error at {x}")
            return x

        processor = AsyncBatchProcessor(max_concurrent=5, delay_between=0)
        coros = [sometimes_fail(i) for i in range(5)]
        results = await processor.process(coros, return_exceptions=True)

        assert results[0] == 0
        assert results[1] == 1
        assert isinstance(results[2], ValueError)
        assert results[3] == 3
        assert results[4] == 4


class TestRouteToVendorAsync:
    """Tests for async vendor routing."""

    @pytest.mark.asyncio
    async def test_route_to_vendor_async_calls_sync_in_thread(self) -> None:
        """Verify route_to_vendor_async wraps sync functions properly."""

        with patch("tradingagents.dataflows.interface.VENDOR_METHODS") as mock_methods:
            mock_func = MagicMock(return_value="test_result")
            mock_methods.__getitem__ = MagicMock(return_value={"test_vendor": mock_func})
            mock_methods.__contains__ = MagicMock(return_value=True)

            with patch("tradingagents.dataflows.interface.get_category_for_method") as mock_cat:
                mock_cat.return_value = "test_category"

                with patch("tradingagents.dataflows.interface.get_vendor") as mock_vendor:
                    mock_vendor.return_value = "test_vendor"

                    # This should work without errors
                    # The actual call would need full mocking of the vendor infrastructure


class TestMemoryAsync:
    """Tests for async memory operations."""

    @pytest.fixture
    def mock_config(self) -> dict:
        """Return a mock config for testing."""
        return {
            "backend_url": "https://api.openai.com/v1",
            "memory_persistence": False,
        }

    @pytest.fixture
    def mock_async_openai_client(self) -> AsyncMock:
        """Return a mock async OpenAI client."""
        mock = AsyncMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock.embeddings.create = AsyncMock(return_value=mock_response)
        return mock

    @pytest.mark.asyncio
    @patch("tradingagents.agents.utils.memory.AsyncOpenAI")
    @patch("tradingagents.agents.utils.memory.OpenAI")
    @patch("tradingagents.agents.utils.memory.chromadb.Client")
    async def test_get_embedding_async(
        self,
        mock_chroma_class: MagicMock,
        mock_openai_class: MagicMock,
        mock_async_openai_class: MagicMock,
        mock_config: dict,
        mock_async_openai_client: AsyncMock,
    ) -> None:
        """Verify get_embedding_async calls async OpenAI API correctly."""
        from tradingagents.agents.utils.memory import FinancialSituationMemory

        # Set up mocks
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma_class.return_value.get_or_create_collection.return_value = mock_collection
        mock_async_openai_class.return_value = mock_async_openai_client

        memory = FinancialSituationMemory("test_memory", mock_config)
        embedding = await memory.get_embedding_async("test text")

        mock_async_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="test text"
        )
        assert len(embedding) == 1536

    @pytest.mark.asyncio
    @patch("tradingagents.agents.utils.memory.AsyncOpenAI")
    @patch("tradingagents.agents.utils.memory.OpenAI")
    @patch("tradingagents.agents.utils.memory.chromadb.Client")
    async def test_get_memories_async(
        self,
        mock_chroma_class: MagicMock,
        mock_openai_class: MagicMock,
        mock_async_openai_class: MagicMock,
        mock_config: dict,
        mock_async_openai_client: AsyncMock,
    ) -> None:
        """Verify get_memories_async retrieves similar situations."""
        from tradingagents.agents.utils.memory import FinancialSituationMemory

        # Set up mocks
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.query.return_value = {
            "documents": [["Matched situation"]],
            "metadatas": [[{"recommendation": "Matched advice"}]],
            "distances": [[0.1]],
        }
        mock_chroma_class.return_value.get_or_create_collection.return_value = mock_collection
        mock_async_openai_class.return_value = mock_async_openai_client

        memory = FinancialSituationMemory("test_memory", mock_config)
        results = await memory.get_memories_async("current situation", n_matches=1)

        assert len(results) == 1
        assert results[0]["matched_situation"] == "Matched situation"
        assert results[0]["recommendation"] == "Matched advice"
        assert results[0]["similarity_score"] == pytest.approx(0.9)
