from typing import Any

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class EconomicCalendarCollector(BaseCollector):
    """Fetches upcoming economic calendar events from ForexFactory's free
    JSON feed (this-week window), filtered by configured countries.

    Note: collectors.yaml's economic_calendar block still documents an
    Investing.com-based config (api_url, headers, days_ahead/days_back,
    filter.exclude_title_keywords, backoff_factor, max_retries, timeout)
    that this implementation does not read at all - that config predates
    the ForexFactory rewrite and is currently inert. Also: hash_keys is
    (timestamp, country, event), deliberately excluding actual/forecast/
    previous - once an event is first stored before its release (actual
    empty), a later fetch with the real actual value hashes identically
    and is filtered out by DataManager.filter_new_records as a duplicate,
    so the eventual actual print is never persisted anywhere. This is a
    real, known gap (not a design choice to preserve point-in-time
    integrity - DataManager.upsert's insert-if-absent semantics already
    handle that correctly elsewhere); fixing it needs a data-model change
    (e.g. a separate collected_at dimension distinguishing the
    pre-release and post-release snapshots as two legitimate historical
    facts), not a quick patch - left as-is pending that decision.
    """
    collector_type = 'economic_calendar'
    data_type = 'economic'

    def __init__(self, configs: dict[str, Any], http_client_factory:
        HttpClientFactory, db_manager: DataManager, cache_manager: CacheManager | None=None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager,
            cache_manager, **kwargs)

    async def run(self, tickers: list[str] | None = None, **kwargs) -> list[dict[str, Any]]:
        # Using ForexFactory free JSON calendar
        api_url = 'https://nfs.faireconomy.media/ff_calendar_thisweek.json'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        
        try:
            client = await self.http_client_factory.get_http_client()
            response = await client.get(api_url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            records = []
            
            for item in data:
                # Filter by country if configured
                countries = [c.upper() for c in self.configs.get('countries', ['US', 'EUR', 'GBP'])]
                if item.get('country') not in countries:
                    continue
                    
                records.append({
                    'timestamp': item.get('date'),
                    'country': item.get('country'),
                    'impact': item.get('impact'),
                    'event': item.get('title'),
                    'actual': item.get('actual', ''),
                    'forecast': item.get('forecast', ''),
                    'previous': item.get('previous', '')
                })
                
            self.logger.info(f'[EconCalendar] Fetched {len(records)} events from ForexFactory.')
            return records
            
        except Exception as e:
            self.logger.error(f'Failed to fetch economic calendar: {e}')
            return []

    async def collect_data(self, **kwargs) -> list[dict[str, Any]] | None:
        data = await self.run(tickers=None, **kwargs)
        return data if data else None
