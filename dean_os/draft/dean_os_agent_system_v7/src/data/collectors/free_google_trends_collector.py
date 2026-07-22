import asyncio
import logging
from typing import Any

import pandas as pd
from pytrends.request import TrendReq

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class FreeGoogleTrendsCollector(BaseCollector):
    """
    Asynchronously collects Google Trends temporal logic data.
    Keywords and tickers parameters are delegated dynamically at runtime.
    """
    collector_type = 'free_google_trends'
    data_type = 'alternative'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.geo = self.configs.get('geo', 'US')
        self.timeframe = self.configs.get('timeframe', 'today 5-y')
        self.language = self.configs.get('language', 'en-US')
        self.timezone = self.configs.get('timezone', 360)
        self.batch_size = self.configs.get('batch_size', 4)
        self.request_delay = self.configs.get('request_delay_seconds', 5)
        self.cat = self.configs.get('cat', 0)
        self.gprop = self.configs.get('gprop', '')
        self.pytrends: TrendReq | None = None

    def _initialize_pytrends(self):
        """Lazy initialization mechanism for the TrendReq protocol wrapper."""
        if self.pytrends is None:
            try:
                self.pytrends = TrendReq(hl=self.language, tz=self.timezone)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                raise ConnectionError(
                    f'Failed to bootstrap TrendReq execution state: {e}') from e

    async def run(self, tickers: list[str], keywords: list[str] | None = None,
        **kwargs) ->list[dict[str, Any]]:
        """
        Asynchronously pulls Google Trends metrics bounded by given ticker and metric lists.
        """
        keywords = keywords or []
        search_terms = list(set(tickers + keywords))
        if not search_terms:
            self.logger.warning(
                'No search parameters provided to Google Trends. Aborting task logic.'
                )
            return []
        try:
            self._initialize_pytrends()
        except ConnectionError as e:
            raise RuntimeError("Failed to initialize Google Trends client") from e
        self.logger.info(
            f'Issuing Google Trends query for {len(search_terms)} queries (buffered to {self.batch_size} instances).'
            )
        keyword_batches = [search_terms[i:i + self.batch_size] for i in
            range(0, len(search_terms), self.batch_size)]
        all_trends_data: list[dict[str, Any]] = []
        for i, batch in enumerate(keyword_batches):
            try:
                if i > 0:
                    await asyncio.sleep(self.request_delay)
                batch_data = await self._fetch_trends_for_batch(batch)
                if batch_data:
                    all_trends_data.extend(batch_data)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                raise RuntimeError(
                    f"Failed to fetch Google Trends batch {batch}"
                ) from e
        self.logger.info(
            f'Retrieved {len(all_trends_data)} temporal points across Google temporal scope queries.'
            )
        return all_trends_data

    async def _fetch_trends_for_batch(self, keyword_batch: list[str]) ->list[
        dict[str, Any]]:
        """
        Wraps and isolates a blocking sync execution via a delegated thread instance.
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f'Pulling Google Trends context for instance array: {keyword_batch}'
                )
        try:
            interest_df = await asyncio.to_thread(self.
                _run_pytrends_request, keyword_batch)
            if interest_df is None or interest_df.empty:
                self.logger.warning(
                    f'Pytrends resulted in a null dataframe for requested block: {keyword_batch}'
                    )
                return []
            if 'isPartial' in interest_df.columns:
                interest_df = interest_df.drop(columns=['isPartial'])
            long_df = interest_df.reset_index().melt(id_vars='date',
                var_name='keyword', value_name='interest')
            long_df['geo'] = self.geo
            long_df['date'] = long_df['date'].astype(str)
            return long_df.to_dict('records')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(
                f'Context error during instance request execution for batch {keyword_batch}: {e}'
                , exc_info=True)
            raise

    def _run_pytrends_request(self, keyword_batch: list[str]) ->pd.DataFrame | None:
        """
        Pure synchronous request block mechanism interacting with the pytrends layer.
        """
        if not self.pytrends:
            raise RuntimeError(
                'The Pytrends handler was not correctly established.')
        self.pytrends.build_payload(kw_list=keyword_batch, cat=self.cat,
            timeframe=self.timeframe, geo=self.geo, gprop=self.gprop)
        return self.pytrends.interest_over_time()
