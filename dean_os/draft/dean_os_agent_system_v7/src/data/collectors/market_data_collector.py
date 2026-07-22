# src/data/collectors/market_data_collector.py
import asyncio
import logging
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector

logger = ProjectLogger.get_logger("MarketDataCollector")

class MarketDataCollector(BaseCollector):
    """
    Orchestrates the collection of market data from various API clients.

    This class manages multiple API clients and can fetch data for multiple
    tickers in parallel using async/await patterns.
    """
    collector_type = "market_data"

    def __init__(self, configs: dict[str, Any], http_client_factory: HttpClientFactory,
                 db_manager: DataManager, cache_manager: CacheManager | None = None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.api_clients = configs.get('api_clients', [])
        if not self.api_clients:
            raise ValueError("MarketDataCollector requires 'api_clients' in config.")
        logger.info(f"MarketDataCollector initialized with {len(self.api_clients)} API client(s).")



    async def _fetch_data_for_ticker_async(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame | None:
        """
        Asynchronously tries to fetch data for a single ticker from available clients.
        """
        for i, client in enumerate(self.api_clients):
            client_name = client.__class__.__name__
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Attempting to fetch {ticker} using {client_name} (Client {i+1}/{len(self.api_clients)}).")
            try:
                # Wrap blocking I/O in asyncio.to_thread
                data = await asyncio.to_thread(client.get_historical_data, ticker, period, interval)
                if data is not None and not data.empty:
                    logger.info(f"Successfully fetched {ticker} using {client_name}.")
                    return data
                else:
                    logger.warning(f"{client_name} returned no data for {ticker}. Trying next client.")
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.error(f"Client {client_name} failed for {ticker}: {e}. Trying next client.", exc_info=True)
                await asyncio.sleep(0.5)  # Brief delay before retry

        logger.error(f"All {len(self.api_clients)} clients failed to fetch data for {ticker}.")
        return None

    async def run(self, tickers: list[str], **kwargs) -> dict[str, pd.DataFrame] | None:
        """
        Asynchronously collects historical data for a batch of tickers.

        Args:
            tickers: List of ticker symbols to fetch.
            **kwargs: Additional parameters (period, interval, etc.)

        Returns:
            Dictionary mapping each ticker to its historical DataFrame, or None on failure.
        """
        period = kwargs.get('period', '1y')
        interval = kwargs.get('interval', '1d')

        logger.info(f"Starting async batch data collection for {len(tickers)} tickers.")
        results: dict[str, pd.DataFrame] = {}

        # Create async tasks for all tickers
        tasks = [self._fetch_data_for_ticker_async(ticker, period, interval) for ticker in tickers]

        # Gather results with error handling
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for ticker, response in zip(tickers, responses, strict=False):
            if isinstance(response, Exception):
                logger.exception(f"Exception for {ticker}: {response}")
            elif response is not None:
                results[ticker] = response

        logger.info(f"Batch collection complete. Successfully fetched data for {len(results)}/{len(tickers)} tickers.")
        return results if results else None
