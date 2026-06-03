import asyncio
import pytest
from src.utils.async_timeout import async_timeout

@async_timeout(timeout_seconds=0.1)
async def slow_function():
    await asyncio.sleep(0.5)

@pytest.mark.asyncio
async def test_async_timeout_works():
    with pytest.raises(TimeoutError):
        await slow_function()

