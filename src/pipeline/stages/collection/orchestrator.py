# src/pipeline/stages/stage_1_collection.py

import asyncio
import hashlib
import json
from itertools import chain
from typing import ClassVar

import pandas as pd
from datetime import datetime

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.data.collectors.aaii_sentiment_collector import AIISentimentCollector
from src.data.collectors.bigquery_collector import BigQueryCollector
from src.data.collectors.cftc_collector import CFTCCollector
from src.data.collectors.collector_factory import CollectorFactory
from src.data.collectors.economic_calendar_collector import EconomicCalendarCollector
from src.data.collectors.fear_greed_collector import FearGreedCollector
from src.data.collectors.fred_collector import FredCollector
from src.data.collectors.free_google_trends_collector import FreeGoogleTrendsCollector
from src.data.collectors.google_news_collector import GoogleNewsCollector
from src.data.collectors.huggingface_collector import HuggingfaceCollector
from src.data.collectors.insider_collector import InsiderCollector
from src.data.collectors.newsapi_collector import NewsAPICollector
from src.data.collectors.put_call_ratio_collector import PutCallRatioCollector
from src.data.collectors.reddit_sentiment_collector import RedditSentimentCollector
from src.data.collectors.rss_collector import RSSCollector
from src.data.collectors.sec_filings_collector import SECFilingsCollector
from src.data.collectors.vix_collector import VIXCollector
from src.data.collectors.yf_collector import YFCollector
from src.data.management.data_manager import DataManager
from src.pipeline.stages.base_stage import BaseStage


# Which family a stored table belongs to decides how stage 2 hands it on:
# families are concatenated into one shared frame, anything else keeps its own
# name. Two independent things used to answer that question — the collector's
# declared type and the table's name — and they were checked in separate
# branches of one chain, so a source could be claimed by the first branch while
# the rule meant for it sat unreachable further down. That is exactly what
# happened to the calendar. Keeping the decision here, as one function with no
# I/O, is what makes it checkable.
_FAMILY_BY_COLLECTOR_TYPE = {
    'google_news': 'news',
    'rss': 'news',
    'newsapi': 'news',
    # A filing is an event, not an article. Filed under 'news' it went into
    # the news frame, where the alias list said `filing_date` and the table
    # says `filingDate`, so 24,365 dated ticker-tagged filings were dropped
    # every run over one capital letter -- counted into a lump warning about
    # 762,436 "lost news records" that hid which source they came from.
    #
    # Renaming the column would have been the wrong fix: what a filing carries
    # is `form` and `primaryDocDescription`, codes like "10-Q", not prose.
    # CorporateFilingsEnricher reads them as events instead -- when, how often,
    # what kind -- and stage 3 already forwards every collected frame, so the
    # only thing needed here was to stop calling them news.
    'sec_filings': 'corporate_filings',
    'huggingface': 'news',
    'hugging_face': 'news',
    'fred': 'macro_data',
    'custom_csv': 'macro_data',
    'yahoo_finance': 'market_data',
    'free_google_trends': 'google_trends',
    'reddit': 'reddit_sentiment',
    'reddit_sentiment': 'reddit_sentiment',
    # A calendar entry is not a macro observation. FRED publishes a value for a
    # date; the calendar publishes an actual, a forecast and the gap between
    # them, which is the only thing that source is for. Folded into the macro
    # frame it lost its identity, no separate key ever reached stage 3, and the
    # enricher that looks for one found nothing and said so on every run.
    'economic_calendar': 'economic_calendar',
}

_FAMILY_BY_TABLE_NAME = {
    'fred_data': 'macro_data',
    # `news_patterns` was listed among the macro tables, but the news branch
    # ran first and matched it on the 'news' fragment, so it has always been
    # handled as news. Left as news here: this change is about the calendar,
    # and quietly re-filing another source under cover of it is how the next
    # unexplained shift gets introduced.
    'economic_calendar': 'economic_calendar',
    'market_data_raw': 'market_data',
    'market_data': 'market_data',
}

_FAMILY_BY_NAME_FRAGMENT = (
    ('news', 'news'),
    ('trends', 'google_trends'),
    ('reddit', 'reddit_sentiment'),
)


#: Column names a news source may use for its publication time. The gate that
#: admits a table into the news frame and the rename that normalises it read
#: the same list, so a source can never pass one and fail the other.
NEWS_DATE_ALIASES = (
    'published_at', 'publishedAt', 'published_date', 'filing_date',
    'date', 'timestamp',
)


def classify_source_table(table_name: str, collector_info: dict | None = None) -> str | None:
    """Say which family a stored table belongs to, or None if nothing claims it.

    An explicit ``data_type`` in the collector's config always wins. Otherwise
    the collector's own type decides, then the table name, then — last, because
    it is the loosest rule — a fragment of the name.
    """
    info = collector_info or {}
    declared = info.get('data_type')
    if declared:
        return declared

    collector_type = info.get('type', '') or ''
    if collector_type in _FAMILY_BY_COLLECTOR_TYPE:
        return _FAMILY_BY_COLLECTOR_TYPE[collector_type]
    if table_name in _FAMILY_BY_TABLE_NAME:
        return _FAMILY_BY_TABLE_NAME[table_name]
    for fragment, family in _FAMILY_BY_NAME_FRAGMENT:
        if fragment in table_name:
            return family
    return None


class CollectionStage(BaseStage):
    """Stage for collecting data from various sources."""

    def __init__(
        self,
        config_manager: UnifiedConfigManager,
        db_manager: DataManager,
        error_handler: ErrorHandler,
        **kwargs,
    ):
        super().__init__(config_manager, error_handler, **kwargs)
        self.db_manager = db_manager
        self.logger = ProjectLogger.get_logger(__name__)

        collector_configs = self.config_manager.get_config('collectors')
        self.factory = CollectorFactory(
            configs=collector_configs,
            http_client_factory=self.http_client_factory,
            config_manager=self.config_manager,
            db_manager=self.db_manager,
        )
        self.collectors = self.factory.get_all_collectors()
        self.logger.info(f"Loaded {len(self.collectors)} collectors.")

    async def run(self, **kwargs) -> dict:
        self.logger.info("Starting data collection stage...")

        # --- Тікери ---
        assets_config = self.config_manager.get_config('assets') or {}
        active_preset = assets_config.get('active_preset')
        # ФІКС: правильно дістаємо список тікерів з пресету
        tickers = (
            assets_config
            .get('presets', {})
            .get(active_preset, {})
            .get('tickers', [])
        )
        tickers_override = kwargs.get('tickers')
        if tickers_override:
            if isinstance(tickers_override, dict):
                tickers = list(tickers_override.keys())
            else:
                tickers = list(tickers_override)
            self.logger.info(f"Using {len(tickers)} tickers from pipeline inputs override.")
        if not tickers:
            self.logger.error(f"No tickers found for preset '{active_preset}'. Aborting collection.")
            # Return the same SHAPE as the success path (fetch_all_data_from_db
            # returns a flat {data_type: DataFrame} map, which the orchestrator
            # assigns straight to stage_outputs['raw_data']). The old
            # {'raw_data': {}} nested one level deeper AND was truthy, so the
            # abort survived every emptiness check and only surfaced later as a
            # confusing failure in ProcessingStage.
            return {}
        self.logger.info(f"Loaded {len(tickers)} tickers from preset '{active_preset}'.")

        # --- Keywords: flatten словника категорій ---
        knowledge_base = self.config_manager.get_config('knowledge_base') or {}
        keywords_raw = knowledge_base.get('keywords', {})
        if isinstance(keywords_raw, dict):
            all_keywords = list(set(chain.from_iterable(keywords_raw.values())))
        else:
            all_keywords = list(keywords_raw)

        # Тікери в lowercase додаємо до keywords для новинних колекторів
        keywords = list(set(all_keywords + [t.lower() for t in tickers]))
        self.logger.info(f"Loaded {len(keywords)} unique keywords.")

        # --- Запуск колекторів ---
        # Розділяємо на важкі (новини) і легкі колектори
        # Важкі запускаємо з семафором щоб не перевантажити мережу
        tasks_to_run = []
        force_collectors = kwargs.get('force_collectors', [])

        for collector in self.collectors:
            freq = collector.configs.get('frequency', 'daily')
            collector_name = collector.collector_type

            # Пропуск On-Demand колекторів (погода, GDELT), якщо вони не викликані явно
            if freq == 'on_demand' and collector_name not in force_collectors:
                self.logger.debug(f"Skipping {collector_name} (on_demand, not forced).")
                continue

            # Пропуск Weekly колекторів, якщо сьогодні не понеділок (і не forced)
            if freq == 'weekly' and datetime.now().weekday() != 0 and collector_name not in force_collectors:
                self.logger.debug(f"Skipping {collector_name} (weekly, runs on Monday).")
                continue

            task = asyncio.create_task(
                self._run_collector(collector, tickers=tickers, keywords=keywords)
            )
            tasks_to_run.append(task)

        if tasks_to_run:
            results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
            self.process_and_save_results(results, self.collectors)
        else:
            self.logger.info("No collectors were configured to run.")

        self.logger.info("Collection stage finished.")
        return self.fetch_all_data_from_db(tickers=tickers)

    #: Seconds each collector gets before it is cancelled.
    #:
    #: One hardcoded 300 stood in for all of them, with a single ad-hoc 600
    #: for Google Trends. On 2026-08-17 the daily history window went from two
    #: years to thirty, and the consequence was not "yahoo took longer": TEN
    #: collectors died at the same instant, exactly 300s after collection
    #: began, because they share the event loop and one slow member starves
    #: the rest. yahoo_finance, cftc, fear_greed, fred, insider, newsapi,
    #: reddit_sentiment, sdmx_macro, sec_filings and wikimedia_attention all
    #: saved nothing, and the pipeline reported success.
    #:
    #: Worse, the loss is total rather than partial. YFCollector downloads
    #: every ticker, then filters, then upserts once at the end -- so a
    #: cancellation two thirds of the way through discards ~180,000 rows that
    #: were already on the machine. The log showed "Successfully downloaded
    #: 7541 rows for AAPL/1d" and the table did not gain a single row.
    #:
    #: Values are measured, not guessed: the 30-year download alone ran 456s
    #: before the dedup pass even started.
    _COLLECTOR_TIMEOUT_SECONDS: ClassVar[dict[str, int]] = {
        'yahoo_finance': 1800,       # 30y x 24 tickers, then dedup ~180k rows
        'huggingface': 900,          # ~1M rows to scan
        'free_google_trends': 600,   # was the one hand-tuned exception
        'sec_filings': 600,
        'insider': 600,              # one HTTP round trip per ticker
        'wikimedia_attention': 600,
    }

    #: Everything not named above. Raised from 300 because that number was
    #: never chosen for any collector in particular -- it was chosen once, for
    #: all of them, which is exactly the shape this table exists to remove.
    _DEFAULT_COLLECTOR_TIMEOUT_SECONDS = 900

    @classmethod
    def _collector_timeout(cls, collector, collector_name: str) -> int:
        configured = getattr(collector, 'configs', None) or {}
        override = configured.get('collector_timeout_seconds')
        if isinstance(override, (int, float)) and override > 0:
            return int(override)
        return cls._COLLECTOR_TIMEOUT_SECONDS.get(
            collector_name, cls._DEFAULT_COLLECTOR_TIMEOUT_SECONDS
        )

    async def _run_collector(self, collector, tickers: list[str], keywords: list[str]):
        """Запускає колектор з правильними аргументами залежно від його типу."""

        name = collector.__class__.__name__
        collector_name = getattr(collector, 'collector_name', name)

        try:
            timeout = self._collector_timeout(collector, collector_name)

            if isinstance(collector, YFCollector):
                return await asyncio.wait_for(collector.run(tickers=tickers), timeout=timeout)
            elif isinstance(collector, (SECFilingsCollector, InsiderCollector)):
                return await asyncio.wait_for(collector.run(tickers=tickers), timeout=timeout)
            elif isinstance(collector, (GoogleNewsCollector, NewsAPICollector)):
                return await asyncio.wait_for(collector.run(tickers=tickers, keywords=keywords), timeout=timeout)
            elif isinstance(collector, RSSCollector):
                kb = self.config_manager.get_config("knowledge_base")
                return await asyncio.wait_for(collector.run(
                    tickers=tickers,
                    keywords=keywords,
                    rss_feeds=kb.get("rss_feeds", []),
                ), timeout=timeout)
            elif isinstance(collector, FreeGoogleTrendsCollector):
                return await asyncio.wait_for(collector.run(tickers=tickers, keywords=keywords), timeout=timeout)
            elif isinstance(collector, HuggingfaceCollector):
                return await asyncio.wait_for(collector.run(), timeout=timeout)
            elif isinstance(collector, (FredCollector, EconomicCalendarCollector, BigQueryCollector,
                                        VIXCollector, PutCallRatioCollector, FearGreedCollector,
                                        AIISentimentCollector, CFTCCollector, RedditSentimentCollector)):
                return await asyncio.wait_for(collector.run(), timeout=timeout)
            else:
                self.logger.warning(f"No specific run args for {name}, trying generic run().")
                return await asyncio.wait_for(collector.run(tickers=tickers, keywords=keywords), timeout=timeout)

        except TimeoutError:
            # Say what a timeout costs, not just that one happened.
            #
            # asyncio.wait_for CANCELS the coroutine. Collectors accumulate
            # rows in a local list and upsert once at the end, so a collector
            # cancelled after fetching and before writing loses everything it
            # had. That is how a run could log "завантажено 7541 рядок"
            # thirteen times and leave the table empty -- the fetch was real,
            # the write never happened, and the only trace was a timeout line
            # that read like a stall rather than a loss.
            #
            # Per-collector timeouts removed the trigger and this has not
            # reproduced since; the shape is still there, so the message now
            # names it. The real repair is incremental persistence inside each
            # collector, which is a change to sixteen of them.
            self.logger.error(
                "Collector %s exceeded its %s-second timeout and was CANCELLED. "
                "Anything it had already fetched but not yet written is lost: "
                "collectors persist once at the end, so a cancellation between "
                "those two points leaves no rows and no error from the write.",
                name, timeout,
            )
            # Re-raise (not return None) so this reaches
            # process_and_save_results as an Exception via
            # asyncio.gather(..., return_exceptions=True) - that method
            # already has a dedicated `isinstance(res, Exception)` branch
            # for real failures, but it was dead code as long as this
            # swallowed every failure into None, which is treated
            # identically to "collector ran fine, nothing new to report"
            # and counted as a success.
            raise
        except Exception as e:
            self.logger.error(f"Collector {name} failed: {e}", exc_info=True)
            raise

    def process_and_save_results(self, results: list, collectors: list):
        """
        УНІФІКОВАНА обробка та збереження результатів колекторів.
        Всі колектори обробляються однаково - без SELF_SAVING.
        """
        # A single `successful` counter treated three different outcomes as
        # one. Measured on the 2026-08-11 run: 16 collectors enabled, 10
        # delivered rows, and aaii_sentiment (HTTP 403), put_call_ratio
        # (HTTP 403), fear_greed and wikimedia_attention delivered nothing --
        # yet the summary read "Successfully processed 16 collectors", because
        # returning None counts the same as returning data. Four dead sources
        # stayed invisible for as long as nobody read the collectors' own logs.
        delivered: list[str] = []
        silent: list[str] = []
        failed: list[str] = []

        for i, res in enumerate(results):
            collector = collectors[i]
            collector_type = collector.collector_type

            if isinstance(res, Exception):
                # `{res}` alone printed "Error in 'fear_greed': " with nothing
                # after the colon, because str() on some exceptions is empty --
                # the same signature that hid 54 drift timeouts. The type name
                # is what makes an empty message identifiable.
                self.logger.error(
                    f"Error in '{collector_type}': {type(res).__name__}: {res}",
                    exc_info=res,
                )
                failed.append(collector_type)
                continue

            if res is None:
                self.logger.info(f"Collector '{collector_type}' returned no new data.")
                silent.append(collector_type)
                continue

            # Конвертуємо в DataFrame якщо потрібно
            df = self._normalize_dataframe(res)

            if df is None or df.empty:
                self.logger.info(f"Collector '{collector_type}' returned empty data.")
                silent.append(collector_type)
                continue

            self.logger.info(f"Received {len(df)} records from '{collector_type}'.")

            # ЄДИНА логіка обробки для всіх колекторів
            try:
                # 1. Нормалізація даних
                df = self._normalize_data(df, collector)

                # 2. Збереження в базу даних
                self._save_collector_data(collector, df)

                # 3. Кешування результатів
                self._cache_collector_data(collector, df)

                delivered.append(collector_type)

            except Exception as e:
                self.logger.error(
                    f"Failed to process data from '{collector_type}': "
                    f"{type(e).__name__}: {e}",
                    exc_info=True,
                )
                failed.append(collector_type)
                continue

        total = len(delivered) + len(silent) + len(failed)
        self.logger.info(
            f"Collection: {len(delivered)}/{total} collectors delivered rows."
        )
        if silent:
            # Not an error -- a source can legitimately have nothing new, and
            # deduplication removes repeats within a day. But a source that is
            # silent every run is a source that has stopped working, and that
            # is only visible if the names are said out loud.
            self.logger.warning(
                f"Delivered nothing this run: {', '.join(sorted(silent))}. "
                f"Legitimate when there is genuinely nothing new; check the "
                f"collector's own log if a name keeps appearing here."
            )
        if failed:
            self.logger.error(f"Failed outright: {', '.join(sorted(failed))}.")

    def _normalize_dataframe(self, result) -> pd.DataFrame | None:
        """
        Нормалізує результат в DataFrame.
        """
        if isinstance(result, pd.DataFrame):
            return result
        elif isinstance(result, list) and len(result) > 0:
            return pd.DataFrame(result)
        else:
            return None

    def _normalize_data(self, df: pd.DataFrame, collector) -> pd.DataFrame:
        """
        Нормалізує дані: конвертація дат, додавання хешів.
        """
        # Конвертуємо дати
        date_col = self._find_date_column_in_df(df)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')

        # Додаємо хеші якщо потрібно.
        # collector.generate_hash() doesn't exist on BaseCollector or most
        # subclasses (some define a private _generate_hash, some define a
        # public generate_hash, some define neither) - calling it here
        # unconditionally raised AttributeError for every collector that
        # doesn't happen to have a public generate_hash, which was silently
        # swallowed by the broad except in process_and_save_results,
        # discarding real collected data before it ever reached the DB.
        # Every collector that does define a hash method uses the exact
        # same formula (see cftc/fear_greed/put_call_ratio/vix/aaii/
        # fred_collector.py), so compute it directly here instead of
        # depending on a per-collector method existing at all.
        if 'hash' not in df.columns and collector.get_hash_keys():
            hash_keys = collector.get_hash_keys()
            df['hash'] = df.apply(
                lambda row: hashlib.sha256(
                    '|'.join(str(row.get(key, '')) for key in hash_keys).encode()
                ).hexdigest(),
                axis=1,
            )

        return df

    def _save_collector_data(self, collector, df: pd.DataFrame):
        """
        Єдине збереження даних для всіх колекторів.
        """
        table_name = collector.get_table_name()
        hash_keys = collector.get_hash_keys()

        # Фільтруємо нові записи
        new_df = self.db_manager.filter_new_records(table_name, df)

        if not new_df.empty:
            self.db_manager.upsert(table_name, new_df, unique_on=hash_keys)
            self.logger.info(f"Saved {len(new_df)} new records to '{table_name}'.")
        else:
            self.logger.info(f"No new records for '{table_name}' after filtering.")

    def _cache_collector_data(self, collector, df: pd.DataFrame):
        """
        Єдине кешування для всіх колекторів.
        """
        if not collector.cache_manager:
            return

        cache_key = f"{collector.collector_type}_data"
        cache_params = {
            "tickers": collector.get_active_tickers(),
            "data_type": collector.get_data_type(),
        }

        try:
            collector.cache_manager.set(
                cache_key,
                df.to_dict("records"),
                cache_params,
                ttl=collector.get_cache_ttl(),
                namespace="collectors"
            )
            self.logger.debug(f"Cached {len(df)} records for '{collector.collector_type}'.")
        except Exception as e:
            self.logger.warning(f"Failed to cache data for '{collector.collector_type}': {e}")


    # Macro tables whose own event time IS the moment the number became
    # public, so it can honestly serve as the point-in-time availability
    # stamp. An economic-calendar entry is published when the event happens;
    # a news pattern exists from when the news appeared.
    _MACRO_SELF_TIMED_TABLES: ClassVar[dict[str, str]] = {
        'economic_calendar': 'timestamp',
        'news_patterns': 'timestamp',
    }
    _MACRO_AVAILABILITY_COLUMNS: ClassVar[tuple[str, ...]] = (
        'available_at', 'released_at', 'realtime_start',
    )

    def _ensure_macro_availability(
        self, df: pd.DataFrame, table_name: str
    ) -> pd.DataFrame:
        """Give every macro source a point-in-time availability column.

        The macro sources are concatenated into one frame, and pd.concat
        fills columns a source lacks with NaN. fred_data carries
        realtime_start; economic_calendar carries none, so its rows arrived
        with realtime_start = NaN and ProcessingStage's point-in-time check
        rejected the whole frame:

            Macro data contains missing or invalid point-in-time values in
            realtime_start.

        That check is right -- using a macro figure before it was published
        is look-ahead. It only became reachable once the collection repairs
        earlier in this audit took economic_calendar from 0 rows to 71.

        Only tables whose own timestamp genuinely IS the publication moment
        are filled. A source with a real release lag (annual World Bank
        series dated '1960', say) must not be given an invented one, and is
        left to fail the check loudly.
        """
        if df is None or df.empty:
            return df

        # Normalise EVERY source onto the same column. Leaving fred with
        # realtime_start and the calendar with available_at just moves the
        # problem: the downstream check takes the FIRST of
        # (available_at, released_at, realtime_start) that exists anywhere in
        # the concatenated frame, so whichever it picks is null for the other
        # source's rows.
        existing = next(
            (c for c in self._MACRO_AVAILABILITY_COLUMNS if c in df.columns), None
        )
        if existing:
            if existing != 'available_at':
                df = df.copy()
                df['available_at'] = pd.to_datetime(
                    df[existing], errors='coerce', utc=True
                )
            return self._defer_date_only_availability(df, table_name)

        source_column = self._MACRO_SELF_TIMED_TABLES.get(table_name)
        if source_column is None or source_column not in df.columns:
            self.logger.warning(
                "Macro table '%s' carries no availability column and no "
                "known self-timed source column; its rows will fail the "
                "point-in-time check downstream. Add the release timestamp "
                "at collection rather than inventing one here.",
                table_name,
            )
            return df

        df = df.copy()
        df['available_at'] = pd.to_datetime(
            df[source_column], errors='coerce', utc=True
        )
        self.logger.info(
            "Macro table '%s': derived available_at from '%s' (its event "
            "time is its publication time).",
            table_name, source_column,
        )
        return self._defer_date_only_availability(df, table_name)

    def _defer_date_only_availability(
        self, df: pd.DataFrame, table_name: str
    ) -> pd.DataFrame:
        """Push a date-only availability to the END of that date.

        `fred_data.realtime_start` is a DATE with no time -- '2026-06-04',
        which parses to midnight UTC. Taken literally that says a figure
        published at 08:30 ET was knowable at 00:00 the same day, so an
        intraday model gets it roughly eight hours early. On daily bars this
        is invisible; on the 60m and 15m series this project also trains,
        it is a straightforward look-ahead.

        Deferring to 23:59:59 of the stated date is deliberately the
        CONSERVATIVE repair rather than the precise one. The precise version
        is a table of official release times per indicator -- which is what
        MacroReleaseTimingGuard holds, unwired, in 501 lines: GDP 08:30 ET
        quarter-end, CPI 08:30 monthly, and so on. That approach needs every
        FRED series mapped to an indicator, drifts as schedules change, and
        fails silently toward being too EARLY, which is the direction that
        creates a leak. This one can only ever be late, and it is late by at
        most one day on series that move monthly or quarterly.

        Rows that already carry a real time of day are left alone -- the
        economic calendar publishes '2026-08-03 03:30:00+03:00' and that is
        better information than anything inferred here.
        """
        if 'available_at' not in df.columns:
            return df

        available = pd.to_datetime(df['available_at'], errors='coerce', utc=True)
        # Exactly midnight means the source gave a date, not a moment. A
        # genuine 00:00:00 publication is possible in principle and would be
        # deferred by a day; that costs freshness and cannot cause a leak.
        midnight = available.notna() & (
            (available.dt.hour == 0)
            & (available.dt.minute == 0)
            & (available.dt.second == 0)
        )
        if not bool(midnight.any()):
            return df

        df = df.copy()
        df.loc[midnight, 'available_at'] = (
            available[midnight] + pd.Timedelta(hours=23, minutes=59, seconds=59)
        )
        self.logger.info(
            "Macro table '%s': %d row(s) carried a date-only availability; "
            "deferred to end of day so an intraday model cannot read them "
            "before publication.",
            table_name, int(midnight.sum()),
        )
        return df

    def fetch_all_data_from_db(self, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
            """Завантажує всі дані з БД для наступного етапу."""
            raw_data = {}
            all_news_dfs = []

            collector_configs = self.config_manager.get_config('collectors', {}) or {}
            table_name_to_info = {}
            for name, info in collector_configs.items():
                if isinstance(info, dict):
                    table = info.get('table_name')
                    if table:
                        table_name_to_info[table] = {**info, "_name": name}

            table_names = self.db_manager.get_all_table_names()

            # Завантажуємо активні тікери та ключові слова для фільтрації
            if tickers:
                active_tickers = tickers
            else:
                assets_config = self.config_manager.get_config('assets') or {}
                active_preset = assets_config.get('active_preset')
                active_tickers = (
                    assets_config
                    .get('presets', {})
                    .get(active_preset, {})
                    .get('tickers', [])
                )
            active_tickers_lower = [t.lower() for t in active_tickers]

            knowledge_base = self.config_manager.get_config('knowledge_base') or {}
            keywords_raw = knowledge_base.get('keywords', {})
            if isinstance(keywords_raw, dict):
                all_keywords = list(set(chain.from_iterable(keywords_raw.values())))
            else:
                all_keywords = list(keywords_raw)
            all_keywords_lower = [k.lower() for k in all_keywords]

            for table_name in table_names:
                # Пропускаємо службову таблицю кешу
                if table_name == 'cache_metadata':
                    continue

                collector_info = table_name_to_info.get(table_name, {})
                data_type = classify_source_table(table_name, collector_info)

                # Classify before reading. A news table with no publication
                # time is 999,396 rows we would filter for fourteen minutes
                # and then drop whole, and until now that is what happened.
                if data_type == 'news' and not self._news_table_can_be_dated(table_name):
                    continue

                df = self.db_manager.fetch_data_from_table(table_name)
                if df is None or df.empty:
                    continue

                if data_type == 'news':
                    df_filtered = self._filter_news_by_keywords_and_tickers(
                        df, all_keywords_lower, active_tickers_lower
                    )
                    if not df_filtered.empty:
                        all_news_dfs.append(df_filtered)
                        self.logger.info(
                            f"Fetched {len(df_filtered)}/{len(df)} records from news table '{table_name}' "
                            f"(filtered by keywords/tickers)."
                        )
                    else:
                        self.logger.info(f"No matching records in news table '{table_name}' after filtering.")
                elif data_type in ('macro_data', 'macro', 'macro_context'):
                    df = self._ensure_macro_availability(df, table_name)
                    if 'macro_data' in raw_data:
                        raw_data['macro_data'] = pd.concat([raw_data['macro_data'], df], ignore_index=True)
                    else:
                        raw_data['macro_data'] = df
                    self.logger.info(f"Fetched {len(df)} records from macro table '{table_name}'.")
                elif data_type in ('market_data', 'market'):
                    df_filtered = self._filter_market_data_by_tickers(df, active_tickers_lower)
                    if not df_filtered.empty:
                        if 'market_data' in raw_data:
                            raw_data['market_data'] = pd.concat([raw_data['market_data'], df_filtered], ignore_index=True)
                        else:
                            raw_data['market_data'] = df_filtered
                        self.logger.info(
                            f"Fetched {len(df_filtered)}/{len(df)} records from market table '{table_name}' "
                            f"(filtered by tickers)."
                        )
                    else:
                        self.logger.info(f"No matching records in market table '{table_name}' after filtering.")
                elif data_type == 'google_trends':
                    df_filtered = self._filter_trends_by_keywords_and_tickers(
                        df, all_keywords_lower, active_tickers_lower
                    )
                    if not df_filtered.empty:
                        raw_data['google_trends'] = df_filtered
                        self.logger.info(
                            f"Fetched {len(df_filtered)}/{len(df)} records from trends table '{table_name}' "
                            f"(filtered by keywords/tickers)."
                        )
                elif data_type == 'reddit_sentiment':
                    raw_data['reddit_sentiment'] = df
                    self.logger.info(f"Fetched {len(df)} records from reddit table '{table_name}'.")
                else:
                    raw_data[table_name] = df
                    self.logger.info(f"Fetched {len(df)} records from '{table_name}'.")

            if all_news_dfs:
                # Спочатку нормалізуємо назви колонок з датами в усіх датафреймах
                normalized_news_dfs = []
                for df in all_news_dfs:
                    df = df.copy()
                    # Знаходимо колонку, яка містить дату публікації (різні варіанти)
                    date_col = None
                    for possible_name in NEWS_DATE_ALIASES:
                        if possible_name in df.columns:
                            date_col = possible_name
                            break
                    # Якщо знайшли і вона не є стандартною 'published_at' – перейменовуємо
                    if date_col and date_col != 'published_at':
                        df = df.rename(columns={date_col: 'published_at'})
                        self.logger.debug(f"Renamed date column '{date_col}' to 'published_at' for a news source")
                    normalized_news_dfs.append(df)

                # Об'єднуємо всі нормалізовані датафрейми
                news_df = pd.concat(normalized_news_dfs, ignore_index=True)

                # --- ПОЧАТОК ВИПРАВЛЕНОГО БЛОКУ ---
                # Примусово серіалізуємо всі колонки, які містять хоча б одне значення-словник
                self.logger.info("Checking for dictionary columns in news DataFrame...")
                for col in news_df.select_dtypes(include=['object']).columns:
                    # Перевіряємо ВСІ непорожні значення на наявність dict
                    non_null = news_df[col].dropna()
                    if non_null.empty:
                        continue
                    has_dict = non_null.apply(lambda x: isinstance(x, dict)).any()
                    if has_dict:
                        self.logger.warning(f"Column '{col}' contains dictionaries. Converting to JSON strings.")
                        # Додатково: покажемо приклад
                        example = non_null[non_null.apply(lambda x: isinstance(x, dict))].iloc[0]
                        self.logger.info(f"Example dict in column '{col}': {example}")
                        news_df[col] = news_df[col].apply(
                            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x
                        )
                # --- КІНЕЦЬ ВИПРАВЛЕНОГО БЛОКУ ---

                # Визначаємо, які з потрібних колонок реально присутні
                required_cols = ['title', 'published_at', 'source']
                existing_cols = [col for col in required_cols if col in news_df.columns]

                if existing_cols:
                    try:
                        # pandas treats NaN as equal to NaN when finding
                        # duplicates, so every row that carries NO value in any
                        # key column shares one key and collapses to a single
                        # survivor. That is not deduplication, it is deletion by
                        # a key the source does not have: on 2026-08-14 six
                        # sources totalling ~777,000 records became 15,860, and
                        # the line below reported it as an ordinary dedup.
                        #
                        # A source whose columns are named differently is a
                        # mapping problem to fix, not rows to discard silently.
                        # They are still dropped here -- a row with no title,
                        # no timestamp and no source cannot be attached to a
                        # bar causally, so keeping it would only inflate the
                        # FinBERT pass -- but the count is now stated.
                        keyed = news_df[existing_cols].notna().any(axis=1)
                        keyless = int((~keyed).sum())
                        if keyless:
                            self.logger.warning(
                                "%d news records carry no %s at all. They cannot "
                                "be deduplicated or placed in time, and are "
                                "dropped. If a source is simply naming these "
                                "columns differently, map them in the collector "
                                "-- this is where its rows are being lost.",
                                keyless, existing_cols,
                            )
                        news_df = news_df[keyed].drop_duplicates(subset=existing_cols)
                        self.logger.info(f"Deduplicated news by {existing_cols}: {len(news_df)} records")
                    except TypeError as e:
                        self.logger.error(f"TypeError in drop_duplicates: {e}")
                        # Діагностика проблемної колонки
                        for col in existing_cols:
                            try:
                                news_df.duplicated(subset=[col])
                            except TypeError as e2:
                                self.logger.error(f"Problematic column: '{col}' - {e2}")
                                sample = news_df[col].dropna().head(5).tolist()
                                self.logger.error(f"Sample values in column '{col}': {sample}")
                                # Примусово конвертуємо в рядок
                                news_df[col] = news_df[col].astype(str)
                        # Повторюємо спробу
                        news_df = news_df.drop_duplicates(subset=existing_cols)
                        self.logger.info(f"Deduplicated after forced conversion: {len(news_df)} records")
                else:
                    self.logger.warning("No common columns for deduplication found, skipping duplicate removal")

                raw_data['news'] = news_df
                self.logger.info(
                    f"Combined {len(all_news_dfs)} news sources → "
                    f"{len(raw_data['news'])} records."
                )
            # Backwards-compatible aliases for legacy entrypoints
            if 'market_data' in raw_data and 'market_data_raw' not in raw_data:
                raw_data['market_data_raw'] = raw_data['market_data']
            if 'macro_data' in raw_data and 'fred_data' not in raw_data:
                raw_data['fred_data'] = raw_data['macro_data']

            total = sum(len(df) for df in raw_data.values() if isinstance(df, pd.DataFrame))
            self.logger.info(f"Total {total} records fetched from DB for next stage.")
            return raw_data

    def _find_date_column_in_df(self, df: pd.DataFrame) -> str | None:
        for col in ['created_at', 'published_at', 'timestamp', 'date', 'updated_at']:
            if col in df.columns:
                return col
        return None

    def _news_table_can_be_dated(self, table_name: str) -> bool:
        """Can rows from this table be placed in time at all?

        `huggingface_data` holds 999,396 rows of two columns, ``text`` and
        ``hash``. Every run read all of them, ran them through the keyword and
        ticker filter for fourteen and a half minutes, contributed the 728,862
        survivors to the news frame, and dropped every one of them at the
        deduplication step for carrying no title, no timestamp and no source.
        The net contribution to the pipeline was zero, and the only visible
        trace was a warning that 762,436 news records had been discarded --
        which read like lost data rather than like a source that was never
        news.

        A row with no publication time cannot be attached to a bar without
        looking ahead, so it is not admissible however many of them there are.
        Deciding that from the schema costs one query instead of a gigabyte.

        Fails open: if the schema cannot be read, the table is admitted and
        the old behaviour applies.
        """
        try:
            columns = set(self.db_manager.get_table_schema(table_name))
        except Exception as exc:  # noqa: BLE001 - fail open, never lose a source
            self.logger.warning(
                "Could not read the schema of news table '%s' (%s); "
                "admitting it and letting the usual path decide.",
                table_name, exc,
            )
            return True

        if columns & set(NEWS_DATE_ALIASES):
            return True

        self.logger.warning(
            "News table '%s' carries no publication time (columns: %s). Its "
            "rows cannot be placed against a bar, so it is skipped here "
            "instead of being read, filtered and then discarded. Expected one "
            "of %s -- if this source names its date column something else, "
            "add it to NEWS_DATE_ALIASES.",
            table_name, sorted(columns), list(NEWS_DATE_ALIASES),
        )
        return False

    def _filter_news_by_keywords_and_tickers(
        self, df: pd.DataFrame, keywords: list[str], tickers: list[str]
    ) -> pd.DataFrame:
        """Фільтрує новини за ключовими словами та тікерами."""
        if df.empty:
            return df

        text_cols = [c for c in ['title', 'description', 'content', 'text'] if c in df.columns]
        if not text_cols:
            return df

        # Об'єднуємо текстові колонки
        df['_combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1).str.lower()

        # Фільтруємо за ключовими словами або тікерами
        mask = df['_combined_text'].str.contains(
            '|'.join(keywords + tickers), case=False, na=False, regex=True
        )
        result = df[mask].drop(columns=['_combined_text'])
        return result

    def _filter_market_data_by_tickers(
        self, df: pd.DataFrame, tickers: list[str]
    ) -> pd.DataFrame:
        """Фільтрує ринкові дані за тікерами."""
        if df.empty or 'ticker' not in df.columns:
            return df

        mask = df['ticker'].str.lower().isin(tickers)
        return df[mask]

    def _filter_trends_by_keywords_and_tickers(
        self, df: pd.DataFrame, keywords: list[str], tickers: list[str]
    ) -> pd.DataFrame:
        """Фільтрує тренди за ключовими словами та тікерами."""
        if df.empty:
            return df

        text_cols = [c for c in ['keyword', 'term', 'query', 'search_term'] if c in df.columns]
        if not text_cols:
            return df

        df['_combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1).str.lower()
        mask = df['_combined_text'].str.contains(
            '|'.join(keywords + tickers), case=False, na=False, regex=True
        )
        result = df[mask].drop(columns=['_combined_text'])
        return result

