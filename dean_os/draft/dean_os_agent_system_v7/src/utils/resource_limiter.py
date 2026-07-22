# Resource limiter for concurrent operations
from asyncio import Semaphore


class ResourceLimiter:
    """Limit concurrent resource usage."""

    def __init__(self, max_concurrent: int = 5):
        self.semaphore = Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent

    async def acquire(self):
        await self.semaphore.acquire()

    def release(self):
        self.semaphore.release()

    async def __aenter__(self):
        await self.acquire()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


# Global limiters
api_limiter = ResourceLimiter(max_concurrent=3)  # Limit API calls
collector_limiter = ResourceLimiter(max_concurrent=5)  # Limit collectors
processor_limiter = ResourceLimiter(max_concurrent=2)  # Limit heavy processing
