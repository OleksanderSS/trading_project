from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher
from src.features.nlp.extractors.entity_extractor import EntityExtractor
from src.features.nlp.extractors.keyword_extractor import KeywordExtractor
from src.features.utils.datetime_utils import parse_mixed_datetimes

logger = ProjectLogger.get_logger('KeywordEntityEnricher')
DATETIME64_NS = 'datetime64[ns]'
TEXT_COLUMNS = ['title', 'text', 'description', 'content']
TIME_COLUMNS = ['published_at', 'publishedAt', 'published_date', 'date',
    'timestamp', 'datetime']


class KeywordEntityEnricher(BaseEnricher):
    """
    Enriches DataFrame with keyword and entity features from news.
    Extracts keywords and named entities, then aggregates them per timestamp.
    """

    def __init__(self, config: dict[str, Any] | None=None):
        """Initialize with optional config from FeatureOrchestrator."""
        super().__init__()
        self.config = config or {}
        # Union, not "config wins". The enricher's own config carries three
        # groups -- 14 terms after the extractor drops noise and tickers --
        # and because that is non-empty the fallback added on 2026-08-13
        # never fired: `config or knowledge_base` took the smaller list every
        # time, and keyword_count stayed 0 on all 42,541 rows of the next
        # rebuild.
        #
        # knowledge_base.keywords holds 167 terms in 14 categories and is
        # what the collection stage already filters news with. Anything the
        # enricher config adds on top is kept; the shared list is not
        # replaced by it.
        keyword_config = self._merge_keyword_sources(
            self.config.get('keywords'), self._knowledge_base_keywords()
        )
        self.keyword_extractor = KeywordExtractor(keyword_config)
        entity_config = self.config.get('entities', {'spacy_model':
            'en_core_web_sm', 'disable_components': ['parser', 'lemmatizer',
            'attribute_ruler']})
        self.entity_extractor: EntityExtractor | None = None
        try:
            self.entity_extractor = EntityExtractor(entity_config)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            logger.warning(
                f'Failed to initialize EntityExtractor: {e}. Entity features will be skipped.'
                )
            self.entity_extractor = None
        logger.info('KeywordEntityEnricher initialized')

    @property
    def name(self) ->str:
        return 'keyword_entity'

    @property
    def priority(self) ->int:
        """Run after NLP (30), before sentiment (40)"""
        return 35

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) ->pd.DataFrame:
        """
        Adds keyword and entity features to the DataFrame.

        Args:
            df: Input DataFrame with DatetimeIndex
            **kwargs: Should contain 'news' DataFrame

        Returns:
            DataFrame with added keyword_count, entity_count, ticker_mentions features
        """
        if not self._validate_input(df):
            return df
        news_df = kwargs.get('news')
        if not self._validate_news_data(news_df):
            return df
        text_col = self._find_text_column(news_df)
        if text_col is None:
            return df
        time_col = self._find_time_column(news_df)
        if time_col is None:
            return df
        try:
            return self._process_enrichment(df, news_df, text_col, time_col)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f'Error during keyword/entity enrichment: {e}')
            return df

    def _validate_input(self, df: pd.DataFrame) ->bool:
        """Validate input DataFrame."""
        if df.empty:
            logger.warning(
                'Input DataFrame is empty. Skipping keyword/entity enrichment.'
                )
            return False
        return True

    def _validate_news_data(self, news_df: pd.DataFrame) ->bool:
        """Validate news data."""
        if news_df is None or not isinstance(news_df, pd.DataFrame
            ) or news_df.empty:
            logger.warning(
                'No news data available in kwargs. Skipping keyword/entity enrichment.'
                )
            return False
        return True

    def _find_text_column(self, news_df: pd.DataFrame) ->str | None:
        """Pick the column that carries text, not the first one that exists.

        This returned `title` because `title` is in TEXT_COLUMNS first and
        the column is present. Presence is not content: this database stores
        blanks as '' rather than NaN, and on the 2026-08-13 batch the
        enricher reported

            Extracting keywords and entities from 15274 news items...
            ✅ Added keyword/entity features. Avg keywords: 0.0, Avg entities: 0.0

        after 32 seconds of work, on every timeframe. Reproduced directly:
        with `title` populated the same twenty articles yield 144 keywords
        and 72 entities across forty bars; with `title` empty and the same
        body text in `text`, both are zero.

        The identical mistake sat in the sentiment path (fixed 2026-08-13) and
        in news_impact, which reported "Successfully calculated" over a score
        range of [0.000, 0.000]. Wherever this news frame is read, "the column
        exists" has to mean "the column has something in it" — so the check
        now lives on BaseEnricher and all three callers share it.
        """
        return self.choose_text_column(news_df, list(TEXT_COLUMNS))

    def _find_time_column(self, news_df: pd.DataFrame) ->str | None:
        """Find time column in news DataFrame."""
        for col in TIME_COLUMNS:
            if col in news_df.columns:
                return col
        logger.error(
            f'No time column found in news data. Available columns: {news_df.columns.tolist()[:10]}. Skipping keyword/entity enrichment.'
            )
        return None

    def _process_enrichment(self, df: pd.DataFrame, news_df: pd.DataFrame,
        text_col: str, time_col: str) ->pd.DataFrame:
        """Process the enrichment workflow."""
        news_copy = self._prepare_news_data(news_df, time_col)
        logger.info(
            f"✅ Found time column '{time_col}' with {len(news_copy)} valid timestamps"
            )
        logger.info(
            f'Extracting keywords and entities from {len(news_copy)} news items...'
            )
        news_copy = self._extract_features(news_copy, text_col)
        window = self._bar_interval(df)
        aggregated = self._aggregate_by_time(news_copy, time_col, window)
        return self._merge_with_main_df(df, aggregated, time_col)

    @staticmethod
    def _bar_interval(df: pd.DataFrame) -> pd.Timedelta:
        """How far apart these bars are, which is how wide a news window is.

        The window was hardcoded to one hour, and `merge_asof` takes the most
        recent CLOSED bucket. An hourly bucket is right for a 15-minute bar and
        useless for a daily one: a bar stamped midnight reads the 23:00-00:00
        bucket, so a story published at 09:00 that same session is twenty-three
        buckets behind it and is never counted. Measured on the 2026-08-14
        batch, bars receiving a non-zero keyword count:

            15m   60.6%
            60m   42.5%
            1d     8.4%

        Counting a day's news over a day makes the daily figure mean what its
        name says: 19.0% of daily bars and 49 times the total count.

        Never NARROWER than an hour, though the bars may be. Buckets narrower
        than the bar spacing still tile the timeline correctly -- 15-minute
        buckets read by 15-minute bars lose nothing -- but they answer a
        different question, "news in the last quarter hour", which is empty far
        more often: measured, 33.0% of 15m bars against 60.6% with the hourly
        window each hour being read by four consecutive bars. Neither is wrong,
        so the intraday behaviour is left exactly as it was and only the case
        that was broken changes.

        Inferred from the bars rather than plumbed through from the caller,
        because every caller already has the answer in its index and one more
        parameter is one more thing to pass wrongly.

        Falls back to an hour when the spacing cannot be read — an unchanged
        result rather than a guessed one.
        """
        hour = pd.Timedelta(hours=1)
        try:
            stamps = (df['datetime'] if 'datetime' in df.columns
                      else pd.Series(df.index))
            stamps = pd.to_datetime(stamps, errors='coerce').dropna()
            if 'ticker' in df.columns and len(df) == len(stamps):
                per = pd.DataFrame({'t': stamps.to_numpy(),
                                    'k': df['ticker'].to_numpy()})
                gaps = per.sort_values('t').groupby('k')['t'].diff().dropna()
            else:
                gaps = stamps.sort_values().diff().dropna()
            positive = gaps[gaps > pd.Timedelta(0)]
            if positive.empty:
                return hour
            return max(positive.median(), hour)
        except (ValueError, TypeError, AttributeError, KeyError):
            return hour

    def _prepare_news_data(self, news_df: pd.DataFrame, time_col: str
        ) ->pd.DataFrame:
        """Prepare news data with normalized datetime."""
        news_copy = news_df.copy()
        # Four news tables, four date conventions. A single inferred format
        # kept 12,252 of 35,673 rows on the live database and turned all
        # 23,421 SEC filings into NaT -- and those filings are the only news
        # rows carrying a ticker, so losing them also decided what the hourly
        # aggregation below had left to group by.
        news_copy[time_col] = parse_mixed_datetimes(news_copy[time_col], utc=True)
        if news_copy[time_col].dt.tz is not None:
            news_copy[time_col] = news_copy[time_col].dt.tz_localize(None)
        news_copy[time_col] = news_copy[time_col].astype(DATETIME64_NS)
        return news_copy.dropna(subset=[time_col])

    @staticmethod
    def _merge_keyword_sources(
        configured: dict | None, shared: dict | None
    ) -> dict[str, list[str]]:
        """Both lists, by category, with duplicates removed."""
        merged: dict[str, list[str]] = {}
        for source in (shared or {}, configured or {}):
            if not isinstance(source, dict):
                continue
            for category, terms in source.items():
                if isinstance(terms, str):
                    terms = [terms]
                if not isinstance(terms, list):
                    continue
                existing = merged.setdefault(str(category), [])
                for term in terms:
                    if isinstance(term, str) and term not in existing:
                        existing.append(term)
        total = sum(len(v) for v in merged.values())
        logger.info(
            "Keyword extraction over %d terms in %d categories "
            "(knowledge base + enricher config).", total, len(merged),
        )
        return merged

    @staticmethod
    def _knowledge_base_keywords() -> dict[str, list[str]]:
        """The list this project already keeps, rather than a second one.

        `KeywordExtractor` matches pre-defined terms; it does not discover
        them. Handed an empty config it matches nothing, and that is what it
        was handed: `keyword_count` is 0 on all 55,565 rows of the
        2026-08-13 batch, on every timeframe, with `keyword_entity_available`
        0 beside it. The enricher still ran for 34-90 seconds per timeframe,
        because entity extraction does work — so the cost was paid and half
        the output was structurally zero.

        Meanwhile `knowledge_base.keywords` holds 167 terms in 14 categories
        (market_terms, finance_economy, sectors_tech, healthcare_biotech and
        so on), and the collection stage already uses exactly this list to
        decide which news to keep. Its shape -- category -> list of terms --
        is the shape KeywordExtractor wants, so the two now read the same
        source instead of one of them reading nothing.
        """
        try:
            from src.config.unified_config_manager import get_current_config
            knowledge_base = get_current_config().get_config('knowledge_base') or {}
        except (ImportError, AttributeError, KeyError, TypeError) as exc:
            logger.warning(
                "Could not read knowledge_base keywords (%s: %s); keyword "
                "counts will be zero for this run.", type(exc).__name__, exc,
            )
            return {}

        keywords = knowledge_base.get('keywords') or {}
        if isinstance(keywords, list):
            keywords = {'knowledge_base': keywords}
        if not isinstance(keywords, dict) or not keywords:
            logger.warning(
                "knowledge_base declares no keywords; keyword_count will be "
                "zero for this run."
            )
            return {}

        total = sum(len(v) for v in keywords.values() if isinstance(v, list))
        logger.info(
            "Keyword extraction using %d terms from %d knowledge_base "
            "categories.", total, len(keywords),
        )
        return keywords

    def _extract_features(self, news_copy: pd.DataFrame, text_col: str
        ) ->pd.DataFrame:
        """Extract keywords and entities from news."""
        # Пре-фільтрація: обробляємо лише заповнені тексти
        mask = news_copy[text_col].notna() & (news_copy[text_col] != '')

        # Keywords: використовують lru_cache, apply все ще прийнятний, але зробимо чистішим
        news_copy['keywords'] = ''
        news_copy.loc[mask, 'keywords'] = news_copy.loc[mask, text_col].apply(
            lambda x: self.keyword_extractor.extract(x))
        news_copy['keyword_count'] = news_copy['keywords'].apply(len)

        # Entities: використовуємо batch-обробку
        news_copy['entities'] = ''
        news_copy['entity_count'] = 0

        if self.entity_extractor and mask.any():
            texts = news_copy.loc[mask, text_col].tolist()
            entities_batch = self.entity_extractor.extract_batch(texts, entity_types=['ORG', 'GPE', 'PERSON'])

            # Виправлення: Присвоюємо список об'єктів як Series, щоб Pandas не сприймав це як 2D-масив
            news_copy.loc[mask, 'entities'] = pd.Series(entities_batch, index=news_copy.index[mask])
            news_copy.loc[mask, 'entity_count'] = [len(ent) for ent in entities_batch]

        return news_copy

    def _aggregate_by_time(self, news_copy: pd.DataFrame, time_col: str,
        window: pd.Timedelta | None = None) ->pd.DataFrame:
        """Aggregate news data by time (hourly), split by ticker.

        news_df carries per-company vs general news via a 'ticker' or
        'type' column (see sentiment_features_enricher.py for the same
        convention) - aggregating without it collapses every article's
        keyword/entity counts into one global series, which then leaks
        one ticker's news into every other ticker's feature rows on merge.
        """
        news_copy = news_copy.copy()
        # `fillna('general')` was not enough: this database writes blanks as ''
        # rather than NaN, so on the 2026-08-14 batch all 15,661 cleaned news
        # rows had ticker == '' and none were NaN. The fill did nothing, every
        # row was grouped under '', and the merge below then looked for
        # 'general' and for 'AAPL' and found neither -- so nothing was
        # attached and the counts were absent from the feature set again.
        if 'ticker' in news_copy.columns:
            news_copy['_agg_ticker'] = self.normalise_news_ticker(news_copy['ticker'])
        elif 'type' in news_copy.columns:
            news_copy['_agg_ticker'] = self.normalise_news_ticker(news_copy['type'])
        else:
            news_copy['_agg_ticker'] = 'general'
        # The window is the bar's own spacing, not a fixed hour. `merge_asof`
        # takes the most recent CLOSED bucket, so hourly buckets left a daily
        # bar reading only 23:00-00:00: on the 2026-08-14 batch just 8.4% of
        # daily bars received a non-zero count, against 60.6% of 15m bars.
        window = window or pd.Timedelta(hours=1)
        news_copy = news_copy.set_index(time_col)
        aggregated = news_copy.groupby('_agg_ticker').resample(window).agg(
            {'keyword_count': 'sum', 'entity_count': 'sum'})
        aggregated = aggregated.rename_axis(index={'_agg_ticker': 'ticker'})

        # `resample` labels a bucket with its START, so an article published
        # at 14:50 is filed under 14:00. `_merge_with_main_df` then runs
        # merge_asof backward, which hands the bar at 14:00 -- and 14:15,
        # 14:30, 14:45 -- counts drawn from articles up to 14:59. Up to a full
        # window of look-ahead on every bar.
        #
        # A bucket covering [T, T+window) is knowable at T+window. Moving the
        # label there makes it mean "available from", which is what the
        # backward merge is entitled to assume.
        levels = list(aggregated.index.names)
        time_level = levels[-1]
        aggregated.index = aggregated.index.set_levels(
            aggregated.index.levels[levels.index(time_level)] + window,
            level=time_level,
        )
        return aggregated

    def _merge_with_main_df(self, df: pd.DataFrame, aggregated: pd.
        DataFrame, time_col: str) ->pd.DataFrame:
        """Merge aggregated features with main DataFrame, per ticker."""
        df_enriched = df.copy()
        if not self._ensure_datetime_index(df_enriched):
            return df
        self._normalize_timezones(df_enriched, aggregated)
        df_reset = self._prepare_df_for_merge(df_enriched)
        aggregated_reset = self._prepare_aggregated_for_merge(aggregated,
            time_col)
        if 'ticker' in df_reset.columns:
            df_merged = self._merge_per_ticker(df_reset, aggregated_reset)
        else:
            general = aggregated_reset[aggregated_reset['ticker'] ==
                'general'].drop(columns=['ticker'])
            df_merged = pd.merge_asof(df_reset.sort_values('datetime'),
                general.sort_values('datetime'), on='datetime',
                direction='backward',
                tolerance=self.bar_window(df_reset['datetime']))
        return self._finalize_merge_result(df_merged)

    def _merge_per_ticker(self, df_reset: pd.DataFrame, aggregated_reset:
        pd.DataFrame) ->pd.DataFrame:
        """Merge aggregated news features per ticker group (no cross-ticker leakage)."""
        window = self.bar_window(df_reset['datetime'])
        # General news is ADDED to a ticker's own, not used only when the
        # ticker has none.
        #
        # Of the four news tables, only sec_filings carries a ticker column,
        # and it names six of them. So in the concatenated frame every RSS,
        # Google News and NewsAPI headline has ticker NaN -> 'general', while
        # those six tickers have their filings. The old rule -- take the
        # ticker's rows, fall back to general only if there are none -- then
        # handed those six ONLY their 8-K and 10-Q titles, which contain no
        # market vocabulary, and dropped the 15,000 headlines where every
        # keyword hit actually lives. Hence "Avg keywords: 0.0" on rebuild
        # after rebuild, while the same extractor scored 139 hits across 300
        # of those headlines when run directly.
        #
        # These are counts of market attention in an hour. A filing about
        # AAPL and a market-wide headline in the same hour are both attention
        # in that hour, so they sum.
        # Name the columns rather than assume them. The aggregated frame's
        # time column is whatever `_prepare_aggregated_for_merge` produced,
        # and taking "everything that is not ticker or datetime" as the
        # counts swept the time column in when it was called something else:
        #
        #   Error during keyword/entity enrichment: "None of
        #   [Index(['keyword_count', 'entity_count'])] are in the [columns]"
        #
        # — 77 seconds of extraction discarded on every timeframe, three
        # rebuilds running.
        counted = [c for c in ('keyword_count', 'entity_count')
                   if c in aggregated_reset.columns]
        if not counted or 'datetime' not in aggregated_reset.columns:
            logger.error(
                "Aggregated news lacks the columns needed to merge (has %s); "
                "keyword/entity features not attached.",
                list(aggregated_reset.columns)[:8],
            )
            return df_reset
        # Both comparisons case-folded. They used to disagree: 'general' was
        # matched case-insensitively and the ticker exactly, so a news frame
        # whose tickers differed in case from the bars' matched NEITHER
        # branch, every group was appended without counts, and the finalizer
        # died on columns that were never merged in.
        keyed = aggregated_reset.assign(
            _key=aggregated_reset['ticker'].astype(str).str.strip().str.lower()
        )
        general = keyed[keyed['_key'] == 'general'].drop(columns=['ticker', '_key'])

        parts = []
        for ticker, group in df_reset.groupby('ticker'):
            own = keyed[keyed['_key'] == str(ticker).strip().lower()]
            own = own.drop(columns=['ticker', '_key']) if not own.empty else own
            pieces = [frame for frame in (general, own) if not frame.empty]
            if not pieces:
                parts.append(group)
                continue
            combined = pd.concat(pieces, ignore_index=True)
            if counted:
                combined = (
                    combined.groupby('datetime', as_index=False)[counted].sum()
                )
            else:
                combined = combined.drop_duplicates(subset=['datetime'], keep='last')
            # Bounded by the bar's own spacing. Unbounded, the resampler's
            # zero-filled buckets mean every bar inside the collected era
            # matches something, so `keyword_entity_available` became the
            # constant 1 wherever price history sits inside news history --
            # 1.0000 on 15m against 0.2153 on 60m, which is the era and not
            # the bar.
            merged_group = pd.merge_asof(group.sort_values('datetime'),
                combined.sort_values('datetime'), on='datetime',
                direction='backward', tolerance=window)
            parts.append(merged_group)
        if not parts:
            return df_reset
        return pd.concat(parts).sort_index()

    def _ensure_datetime_index(self, df_enriched: pd.DataFrame) ->bool:
        """Ensure DataFrame has DatetimeIndex."""
        if isinstance(df_enriched.index, pd.DatetimeIndex):
            return True
        if 'datetime' in df_enriched.columns:
            df_enriched.set_index('datetime', inplace=True)
            return True
        logger.error(
            "Cannot merge: df has no DatetimeIndex or 'datetime' column")
        return False

    def _normalize_timezones(self, df_enriched: pd.DataFrame, aggregated:
        pd.DataFrame):
        """Normalize timezones in both DataFrames.

        aggregated now has a (ticker, time) MultiIndex (see
        _aggregate_by_time) rather than a plain DatetimeIndex, so its
        tz-awareness can't be read via a plain .index.tz - it doesn't
        need to be here anyway, since _prepare_aggregated_for_merge's
        reset_index() + _normalize_datetime_column() already normalizes
        the resulting 'datetime' column right after this is called.
        """
        if df_enriched.index.tz is not None:
            df_enriched.index = df_enriched.index.tz_localize(None)

    def _prepare_df_for_merge(self, df_enriched: pd.DataFrame) ->pd.DataFrame:
        """Prepare main DataFrame for merge."""
        df_reset = df_enriched.reset_index()
        df_reset = df_reset.rename(columns={'index': 'datetime'} if 'index' in
            df_reset.columns else {})
        self._normalize_datetime_column(df_reset, 'datetime')
        return df_reset

    def _prepare_aggregated_for_merge(self, aggregated: pd.DataFrame,
        time_col: str) ->pd.DataFrame:
        """Prepare aggregated DataFrame for merge."""
        aggregated_reset = aggregated.reset_index()
        aggregated_reset = aggregated_reset.rename(columns={time_col:
            'datetime'})
        self._normalize_datetime_column(aggregated_reset, 'datetime')
        return aggregated_reset

    def _normalize_datetime_column(self, df: pd.DataFrame, col: str):
        """Normalize datetime column timezone and precision."""
        if col not in df.columns:
            return
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if hasattr(df[col].dtype, 'tz') and df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
            if df[col].dtype != DATETIME64_NS:
                df[col] = df[col].astype(DATETIME64_NS)

    def _finalize_merge_result(self, df_merged: pd.DataFrame) ->pd.DataFrame:
        """Finalize merge result.

        The merge has three paths that legitimately return the bars untouched
        — no countable columns, no aggregated rows for any ticker, nothing to
        concatenate — and this method used to index straight into
        ['keyword_count', 'entity_count'] regardless. The KeyError propagated
        to `_enrich_impl`'s handler, which returns the ORIGINAL frame, so a
        missed merge cost not just the counts but the 45-to-137 seconds of
        extraction that produced them, on every timeframe of three rebuilds.

        A merge that attached nothing is a fact to report, not an exception.
        """
        missing = [c for c in ('keyword_count', 'entity_count')
                   if c not in df_merged.columns]
        if missing:
            logger.error(
                "Keyword/entity counts were extracted but never attached to "
                "the bars: %s absent after the merge. Frame has %s. The bars "
                "are returned unchanged.",
                missing, list(df_merged.columns)[:10],
            )
            return df_merged.set_index('datetime') if 'datetime' in df_merged.columns else df_merged
        df_merged = df_merged.set_index('datetime')
        df_merged['keyword_entity_available'] = (
            df_merged[['keyword_count', 'entity_count']].notna().any(axis=1).astype(int)
        )
        df_merged['keyword_count'] = df_merged['keyword_count'].where(
            df_merged['keyword_count'].notna(), 0).astype(int)
        df_merged['entity_count'] = df_merged['entity_count'].where(
            df_merged['entity_count'].notna(), 0).astype(int)
        avg_keywords = df_merged['keyword_count'].values.mean()
        avg_entities = df_merged['entity_count'].values.mean()
        logger.info(
            f'✅ Added keyword/entity features. Avg keywords: {avg_keywords:.1f}, Avg entities: {avg_entities:.1f}'
            )
        return df_merged
