from typing import Any

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector

#: ForexFactory tags each event with the CURRENCY it affects (USD, EUR, JPY...),
#: not an ISO country code. collectors.yaml still lists ISO codes
#: (us, eu, gb, cn, jp, de) left over from the Investing.com implementation
#: this collector replaced, so the filter compared 'US' against 'USD' and
#: matched nothing -- 92 live events in, 0 out, on every single run.
#: Both spellings are accepted now.
_ISO_TO_CURRENCY = {
    'US': 'USD', 'USA': 'USD',
    'EU': 'EUR', 'EA': 'EUR', 'EMU': 'EUR',
    # Eurozone members report under EUR on this feed.
    'DE': 'EUR', 'FR': 'EUR', 'IT': 'EUR', 'ES': 'EUR', 'NL': 'EUR',
    'GB': 'GBP', 'UK': 'GBP',
    'JP': 'JPY',
    'CN': 'CNY',
    'CA': 'CAD',
    'AU': 'AUD',
    'NZ': 'NZD',
    'CH': 'CHF',
}


def _to_currency_codes(values: Any) -> set[str]:
    """Normalise configured country/currency codes to ForexFactory currencies."""
    out: set[str] = set()
    for raw in values or []:
        code = str(raw).strip().upper()
        if not code:
            continue
        out.add(_ISO_TO_CURRENCY.get(code, code))
    return out


class EconomicCalendarCollector(BaseCollector):
    """Fetches upcoming economic calendar events from ForexFactory's free
    JSON feed (this-week window), filtered by configured countries.

    Note: collectors.yaml's economic_calendar block still documents an
    Investing.com-based config (api_url, headers, days_ahead/days_back,
    filter.exclude_title_keywords, backoff_factor, max_retries, timeout)
    that this implementation does not read at all - that config predates
    the ForexFactory rewrite and is currently inert.

    RESOLVED 2026-08-15 - the missing `actual`. hash_keys was (timestamp,
    country, event), so an event first stored BEFORE its release, with
    actual empty, hashed identically to the post-release fetch carrying the
    real print; filter_new_records dropped the second as a duplicate and the
    actual was never persisted. Measured: all 147 stored rows have an empty
    `actual` while 101 carry a forecast, which made the surprise -- actual
    minus forecast, the only thing this source exists for -- impossible to
    compute by construction.

    The note here said the fix needed a data-model change rather than a
    patch, and named the shape of it: keep the pre-release and post-release
    snapshots as two legitimate historical facts. That is exactly the
    decision already taken for FRED vintages, where realtime_start is part
    of the key precisely so a revision does not overwrite the first print.
    So `actual` joined hash_keys: both snapshots are stored, consumers take
    the row that has an actual, and the pre-release row remains as the
    record of what the market expected.
    """
    collector_type = 'economic_calendar'
    data_type = 'economic'

    def __init__(self, configs: dict[str, Any], http_client_factory:
        HttpClientFactory, db_manager: DataManager, cache_manager: CacheManager | None=None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager,
            cache_manager, **kwargs)

    # This is the only feed the publisher offers. Checked, because the obvious
    # repair was to fetch last week as well: ff_calendar_lastweek,
    # _nextweek, _today, _yesterday, _pastweek and _thismonth all return 404.
    # One week, from today forward.
    #
    # Which sets the ceiling on what the calendar can ever tell us. An entry is
    # published with a forecast and acquires its `actual` on release, so a
    # surprise feature needs the SAME event seen twice -- before and after. The
    # feed supports that only within a single week, and only if a run happens
    # on a weekday after the release. There is no way to recover the actual for
    # an event whose week has passed.
    #
    # State when this was written, so the next reader does not re-derive it:
    # 216 stored events, 158 with a forecast, 190 with a previous, `actual`
    # filled on ZERO. That looks like a defect and may not be one. `actual`
    # only joined hash_keys on 2026-08-15 -- before that the post-release
    # snapshot hashed identically to the pre-release one and was discarded as a
    # duplicate -- and the only runs since were 2026-08-15 and 2026-08-16, a
    # Saturday and a Sunday. The feed on 2026-08-16 covers Sun 16th to Fri
    # 21st with no actuals, which is correct for a Sunday, not evidence of
    # anything.
    #
    # The first weekday run is therefore the test. If `already released` below
    # stays at 0 on a Wednesday, the mechanism is broken; until then there is
    # nothing to fix and a fix would be guesswork.
    FEED = 'https://nfs.faireconomy.media/ff_calendar_thisweek.json'

    async def run(self, tickers: list[str] | None = None, **kwargs) -> list[dict[str, Any]]:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

        try:
            client = await self.http_client_factory.get_http_client()
            response = await client.get(self.FEED, headers=headers)
            response.raise_for_status()

            data = response.json()
            # The number that decides whether the calendar can produce a
            # surprise feature at all. It was not reported, so three rebuilds
            # went by with the enricher saying "nothing to be surprised about"
            # and nobody able to see why.
            self.logger.info(
                '[EconCalendar] feed carries %d events, %d already released.',
                len(data),
                sum(1 for item in data if str(item.get('actual') or '').strip()),
            )

            records = []
            wanted = _to_currency_codes(
                self.configs.get('countries', ['US', 'EU', 'GB'])
            )
            seen_codes: set[str] = set()

            for item in data:
                code = str(item.get('country') or '').upper()
                seen_codes.add(code)
                if wanted and code not in wanted:
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
                
            self.logger.info(
                f'[EconCalendar] Fetched {len(records)} of {len(data)} events '
                f'from ForexFactory (filter={sorted(wanted)}).'
            )
            if data and not records:
                # Silence here is how this sat broken: 92 events in, 0 out, and
                # nothing said why.
                self.logger.warning(
                    f'[EconCalendar] Every event was filtered out. Feed carries '
                    f'{sorted(seen_codes)}; configured filter resolves to '
                    f'{sorted(wanted)}. Check collectors.yaml `countries`.'
                )
            return records
            
        except Exception as e:
            self.logger.error(f'Failed to fetch economic calendar: {e}')
            return []

    async def collect_data(self, **kwargs) -> list[dict[str, Any]] | None:
        data = await self.run(tickers=None, **kwargs)
        return data if data else None
