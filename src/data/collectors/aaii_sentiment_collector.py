import hashlib
import re
from datetime import datetime
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class AIISentimentCollector(BaseCollector):
    """Collector for AAII Investor Sentiment Survey data."""
    collector_type = 'aaii_sentiment'
    data_type = 'alternative'
    collector_name = 'aaii_sentiment'

    def __init__(self, configs: dict[str, Any], http_client_factory:
        HttpClientFactory, db_manager: DataManager, cache_manager: CacheManager | None=None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager,
            cache_manager, **kwargs)
        self.enabled = self.configs.get('enabled', True)
        self.timeout = self.configs.get('timeout', 30)
        self.table_name = self.configs.get('table_name', 'aaii_sentiment_data')
        self.hash_keys = self.configs.get('hash_keys', ['date', 'bullish',
            'bearish', 'neutral'])
        self.base_url = self.configs.get(
            'base_url', 'https://www.aaii.com/sentimentsurvey')
        # AAII answers HTTP 403 to this project's user agent. That is a
        # deliberate block on automated access, not a misconfiguration, and it
        # is not worked around here: sending a browser string to defeat it
        # would be circumventing a stated access decision by a site whose
        # survey sits behind a paid membership. The collector therefore still
        # returns nothing -- but it now says so by name in the collection
        # summary instead of counting as a success. Disable it in config, or
        # replace it with a licensed source.
        self.user_agent = self.configs.get('user_agent')
        self.logger.info(
            f'AIISentimentCollector initialized. Enabled: {self.enabled}')

    def generate_hash(self, row: pd.Series) ->str:
        """Generates a stable hash for a record."""
        hash_string = '|'.join(str(row.get(key, '')) for key in self.hash_keys)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    async def run(self, **kwargs) ->pd.DataFrame | None:
        """Fetches AAII Sentiment data and returns DataFrame."""
        if not self.enabled:
            self.logger.warning('AIISentimentCollector is disabled')
            return None
        try:
            self.logger.info('Fetching AAII Sentiment data from AAII website')
            data = await self._fetch_aaii_data()
            if not data:
                return None
            df = pd.DataFrame(data)
            if df.empty:
                self.logger.warning('No AAII Sentiment data received')
                return None
            df = self._standardize_columns(df)
            df['collector_type'] = self.collector_type
            df['collector_name'] = self.collector_name
            df['data_type'] = self.data_type
            df['collected_at'] = datetime.now()
            df['record_hash'] = df.apply(self.generate_hash, axis=1)
            self.logger.info(
                f'Successfully fetched {len(df)} AAII Sentiment records')
            return df
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN
            self.logger.error(f'Error in AIISentimentCollector: {e}')
            return None

    async def _fetch_aaii_data(self) ->list[dict[str, Any]]:
        """Fetches data from AAII website."""
        try:
            # /sentimentsurveyresults is HTTP 404 -- that path does not exist.
            # The survey figures are on the section page itself.
            url = self.base_url
            client = await self.http_client_factory.get_http_client(
                timeout=self.timeout, user_agent=self.user_agent)
            async with client as http_client:
                response = await http_client.get(url)
                if response.status_code == 404:
                    self.logger.error(
                        f'AAII sentiment survey page not found (404). URL may have changed: {url}'
                        )
                    self.logger.error(
                        'AAII may have changed their website structure or requires authentication.'
                        )
                    return []
                elif response.status_code != 200:
                    self.logger.error(f'Failed to fetch AAII data: {response}')
                    return []
                html_content = response.text
                if not html_content:
                    self.logger.warning('Empty HTML content received from AAII'
                        )
                    return []
            data = self._parse_aaii_html(html_content)
            if not data:
                self.logger.warning('No data parsed from AAII HTML')
                return []
            return data
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN
            self.logger.error(f'Error fetching AAII data: {e}')
            return []

    def _parse_aaii_html(self, html_content: str) ->list[dict[str, Any]]:
        """Parse AAII HTML content to extract sentiment data."""
        try:
            data = self._extract_raw_data(html_content)
            if not data:
                self.logger.error(
                    'CRITICAL: AAII HTML parsing failed and no data could be extracted.'
                    )
                return []
            return data
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN
            self.logger.error(f'Error parsing AAII HTML: {e}', exc_info=True)
            return []

    def _extract_raw_data(self, html_content: str) ->list[dict[str, Any]]:
        """Extract sentiment data using regex patterns."""
        data = []
        date_pattern = '(\\w{3}\\s+\\d{1,2},\\s+\\d{4})'
        dates = re.findall(date_pattern, html_content)
        bullish_pattern = 'Bullish[^0-9]*([0-9]+\\.?[0-9]*)%'
        bearish_pattern = 'Bearish[^0-9]*([0-9]+\\.?[0-9]*)%'
        neutral_pattern = 'Neutral[^0-9]*([0-9]+\\.?[0-9]*)%'
        bullish_values = re.findall(bullish_pattern, html_content)
        bearish_values = re.findall(bearish_pattern, html_content)
        neutral_values = re.findall(neutral_pattern, html_content)
        if dates and bullish_values and bearish_values and neutral_values:
            for i, date_str in enumerate(dates[:10]):
                try:
                    date_obj = datetime.strptime(date_str, '%b %d, %Y')
                    if i < len(bullish_values) and i < len(bearish_values
                        ) and i < len(neutral_values):
                        bullish = float(bullish_values[i])
                        bearish = float(bearish_values[i])
                        neutral = float(neutral_values[i])
                        # Three independent regexes over a whole page pair by
                        # position, not by row: the i-th "Bullish" and the
                        # i-th "Bearish" need not belong to the same week.
                        # Measured on the live page, the first matches were
                        # 49.5 / 52.0 / 31.4 -- three different readings that
                        # would have been stored as one survey. The survey is
                        # a partition of respondents, so its three shares sum
                        # to 100; anything else is a mispairing, and a wrong
                        # sentiment number is worse than a missing one.
                        total = bullish + bearish + neutral
                        if abs(total - 100.0) > 1.0:
                            self.logger.warning(
                                'AAII record %d rejected: bullish %.1f + '
                                'bearish %.1f + neutral %.1f = %.1f, not 100. '
                                'The page layout has changed and these three '
                                'values are not one survey.',
                                i, bullish, bearish, neutral, total,
                            )
                            continue
                        data.append({'date': date_obj.strftime('%Y-%m-%d'),
                            'bullish': bullish, 'bearish': bearish,
                            'neutral': neutral, 'spread': bullish - bearish,
                            'total_responses': bullish + bearish + neutral,
                            'timestamp': date_obj})
                except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                    self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                    self.logger.warning(f'Error parsing AAII record {i}: {e}')
                    continue
        return data

    def _standardize_columns(self, df: pd.DataFrame) ->pd.DataFrame:
        """Standardizes column names and data types."""
        try:
            if 'date' not in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.strftime(
                    '%Y-%m-%d')
            required_cols = ['bullish', 'bearish', 'neutral', 'spread']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"AAII data missing '{col}' column")
                    return pd.DataFrame()
            df['date'] = pd.to_datetime(df['date'])
            for col in (required_cols + ['total_responses']):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)
            df['sentiment_level'] = df['spread'].apply(self.
                _categorize_sentiment)
            df['sentiment_signal'] = df['spread'].apply(self._get_signal)
            df['bullish_pct'] = df['bullish'] / df['total_responses'] * 100
            df['bearish_pct'] = df['bearish'] / df['total_responses'] * 100
            df['neutral_pct'] = df['neutral'] / df['total_responses'] * 100
            return df
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error standardizing AAII columns: {e}')
            return pd.DataFrame()

    def _categorize_sentiment(self, spread: float) ->str:
        """Categorize sentiment based on bullish-bearish spread."""
        if spread > 20:
            return 'very_bullish'
        elif spread > 10:
            return 'bullish'
        elif spread > -10:
            return 'neutral'
        elif spread > -20:
            return 'bearish'
        else:
            return 'very_bearish'

    def _get_signal(self, spread: float) ->int:
        """Get trading signal based on sentiment spread."""
        if spread > 15:
            return 1
        elif spread < -15:
            return -1
        else:
            return 0

    async def collect_data(self, **kwargs) ->list[dict[str, Any]] | None:
        """
        UNIFIED data collection - retrieval only, without database storage.
        """
        df = await self.run(**kwargs)
        return df.to_dict('records') if df is not None else None
