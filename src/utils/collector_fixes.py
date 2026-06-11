
# Collector fixes for timeout and resource issues
import asyncio


class CollectorFixes:
    """Collection of fixes for common collector issues."""

    @staticmethod
    async def safe_collect_with_retry(collector, max_retries: int = 3, timeout: int = 120):
        """Collect data with retry and timeout protection."""

        for attempt in range(max_retries):
            try:
                # Add timeout to collection
                result = await asyncio.wait_for(
                    collector.run(),
                    timeout=timeout
                )

                if result is not None and len(result) > 0:
                    return result

            except asyncio.TimeoutError:
                print(f"Collector {collector.__class__.__name__} timed out (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)  # Wait before retry
                continue

            except Exception as e:
                print(f"Collector {collector.__class__.__name__} failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)  # Wait before retry
                continue

        print(f"Collector {collector.__class__.__name__} failed after {max_retries} attempts")
        return None

    @staticmethod
    def chunk_ticker_processing(tickers: list[str], chunk_size: int = 5):
        """Split tickers into chunks for processing."""

        for i in range(0, len(tickers), chunk_size):
            yield tickers[i:i + chunk_size]

    @staticmethod
    async def parallel_collect(collectors: list, tickers: list[str], max_concurrent: int = 3):
        """Run collectors in parallel with resource limits."""

        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_collect(collector):
            async with semaphore:
                return await CollectorFixes.safe_collect_with_retry(collector)

        tasks = [limited_collect(collector) for collector in collectors]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return results
