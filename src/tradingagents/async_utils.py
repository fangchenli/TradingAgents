"""Async utilities for TradingAgents.

Provides helper functions for async operations including:
- HTTP client management
- Sync-to-async wrappers
- Async context managers
"""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

import httpx

T = TypeVar("T")

# Global async HTTP client (reused across requests)
_async_client: httpx.AsyncClient | None = None


async def get_async_client() -> httpx.AsyncClient:
    """Get or create the shared async HTTP client."""
    global _async_client
    if _async_client is None or _async_client.is_closed:
        _async_client = httpx.AsyncClient(timeout=30.0)
    return _async_client


async def close_async_client() -> None:
    """Close the shared async HTTP client."""
    global _async_client
    if _async_client is not None and not _async_client.is_closed:
        await _async_client.aclose()
        _async_client = None


def run_sync(coro) -> Any:
    """Run an async coroutine synchronously.

    This is useful for calling async functions from sync code.
    Creates a new event loop if not running in an async context.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(coro)
    else:
        # Already in async context - this shouldn't happen often
        # but handle it gracefully
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()


async def to_async(func: Callable[..., T], *args, **kwargs) -> T:
    """Run a sync function in a thread pool to make it async.

    Useful for wrapping blocking I/O operations that don't have
    native async versions (like yfinance, pandas file I/O).
    """
    return await asyncio.to_thread(func, *args, **kwargs)


def async_to_sync(async_func: Callable) -> Callable:
    """Decorator to create a sync version of an async function.

    The sync version will run the async function using run_sync().
    """

    @wraps(async_func)
    def wrapper(*args, **kwargs):
        return run_sync(async_func(*args, **kwargs))

    return wrapper


class AsyncBatchProcessor:
    """Process multiple async operations concurrently with rate limiting."""

    def __init__(self, max_concurrent: int = 5, delay_between: float = 0.1):
        """Initialize the batch processor.

        Args:
            max_concurrent: Maximum number of concurrent operations
            delay_between: Delay in seconds between starting operations
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.delay = delay_between

    async def process(
        self, coros: list, return_exceptions: bool = True
    ) -> list[Any | BaseException]:
        """Process a list of coroutines concurrently.

        Args:
            coros: List of coroutines to process
            return_exceptions: If True, exceptions are returned instead of raised

        Returns:
            List of results (or exceptions if return_exceptions=True)
        """

        async def limited_coro(coro, index: int):
            if index > 0 and self.delay > 0:
                await asyncio.sleep(self.delay * index)
            async with self.semaphore:
                return await coro

        tasks = [limited_coro(coro, i) for i, coro in enumerate(coros)]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
