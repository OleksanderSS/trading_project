# src/data/collectors/market_data_collector.py

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd

# NOTE: BaseAPIClient not found in the codebase, this import will fail.
# from src.integrations.base import BaseAPIClient 
from src.core.logging.logger import ProjectLogger

# Initialize logger for the module
logger = ProjectLogger.get_logger("MarketDataCollector")

class MarketDataCollector:
    """
    Orchestrates the collection of market data from various API clients.

    This class manages a list of API clients and can fetch data for multiple
    tickers in parallel, handling failures gracefully by trying subsequent clients.
    """

    def __init__(self, api_clients: List):
        if not api_clients:
            raise ValueError("MarketDataCollector requires at least one API client.")
        self.clients = api_clients
        logger.info(f"MarketDataCollector initialized with {len(self.clients)} API client(s).")

    def _fetch_data_for_ticker(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Tries to fetch data for a single ticker from the list of available clients.
        It attempts to use clients in the order they are provided.
        """
        for i, client in enumerate(self.clients):
            client_name = client.__class__.__name__
            logger.debug(f"Attempting to fetch {ticker} using {client_name} (Client {i+1}/{len(self.clients)})." )
            try:
                data = client.get_historical_data(ticker, period, interval)
                if data is not None and not data.empty:
                    logger.info(f"Successfully fetched {ticker} using {client_name}.")
                    return data
                else:
                    logger.warning(f"{client_name} returned no data for {ticker}. Trying next client.")
            except Exception as e:
                logger.error(f"Client {client_name} failed for {ticker}: {e}. Trying next client.", exc_info=True)
        
        logger.error(f"All {len(self.clients)} clients failed to fetch data for {ticker}.")
        return None

    def collect_batch_data(
        self, 
        tickers: List[str], 
        period: str = "1y", 
        interval: str = "1d", 
        max_workers: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Collects historical data for a batch of tickers in parallel.

        Args:
            tickers: A list of ticker symbols to fetch.
            period: The time period for the historical data.
            interval: The data interval.
            max_workers: The maximum number of concurrent threads to use.

        Returns:
            A dictionary mapping each ticker to its historical data DataFrame.
            Tickers that could not be fetched are omitted.
        """
        logger.info(f"Starting batch data collection for {len(tickers)} tickers.")
        results: Dict[str, pd.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # A dictionary to map futures back to their tickers
            future_to_ticker = {executor.submit(self._fetch_data_for_ticker, ticker, period, interval): ticker for ticker in tickers}

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None:
                        results[ticker] = data
                except Exception as e:
                    logger.error(f"An unexpected error occurred while processing ticker {ticker}: {e}", exc_info=True)
        
        logger.info(f"Batch collection complete. Successfully fetched data for {len(results)}/{len(tickers)} tickers.")
        return results
