from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher
from src.features.nlp.extractors.entity_extractor import EntityExtractor
from src.features.nlp.extractors.keyword_extractor import KeywordExtractor

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
        keyword_config = self.config.get('keywords') or self._knowledge_base_keywords()
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

        The identical mistake sat in the sentiment path (fixed 2026-08-13) --
        `notna().any()` was true for 15,274 empty strings there too. Wherever
        this news frame is read, "the column exists" has to mean "the column
        has something in it".
        """
        filled = {}
        for col in TEXT_COLUMNS:
            if col not in news_df.columns:
                continue
            values = news_df[col].fillna('').astype(str).str.strip()
            filled[col] = int((values != '').sum())

        best = max(filled, key=filled.get) if filled else None
        if best is None or filled[best] == 0:
            logger.error(
                'No usable text in news data (non-empty counts: %s). Skipping '
                'keyword/entity enrichment.', filled or news_df.columns.tolist()[:10],
            )
            return None

        if filled[best] < len(news_df):
            logger.info(
                "Keyword/entity extraction reading '%s': %d of %d items carry "
                "text.", best, filled[best], len(news_df),
            )
        return best

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
        aggregated = self._aggregate_by_time(news_copy, time_col)
        return self._merge_with_main_df(df, aggregated, time_col)

    def _prepare_news_data(self, news_df: pd.DataFrame, time_col: str
        ) ->pd.DataFrame:
        """Prepare news data with normalized datetime."""
        news_copy = news_df.copy()
        news_copy[time_col] = pd.to_datetime(news_copy[time_col], errors=
            'coerce', utc=True)
        if news_copy[time_col].dt.tz is not None:
            news_copy[time_col] = news_copy[time_col].dt.tz_localize(None)
        news_copy[time_col] = news_copy[time_col].astype(DATETIME64_NS)
        return news_copy.dropna(subset=[time_col])

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

    def _aggregate_by_time(self, news_copy: pd.DataFrame, time_col: str
        ) ->pd.DataFrame:
        """Aggregate news data by time (hourly), split by ticker.

        news_df carries per-company vs general news via a 'ticker' or
        'type' column (see sentiment_features_enricher.py for the same
        convention) - aggregating without it collapses every article's
        keyword/entity counts into one global series, which then leaks
        one ticker's news into every other ticker's feature rows on merge.
        """
        news_copy = news_copy.copy()
        if 'ticker' in news_copy.columns:
            news_copy['_agg_ticker'] = news_copy['ticker'].fillna('general')
        elif 'type' in news_copy.columns:
            news_copy['_agg_ticker'] = news_copy['type'].fillna('general')
        else:
            news_copy['_agg_ticker'] = 'general'
        news_copy = news_copy.set_index(time_col)
        aggregated = news_copy.groupby('_agg_ticker').resample('1h').agg(
            {'keyword_count': 'sum', 'entity_count': 'sum'})
        aggregated = aggregated.rename_axis(index={'_agg_ticker': 'ticker'})

        # `resample('1h')` labels a bucket with its START, so an article
        # published at 14:50 is filed under 14:00. `_merge_with_main_df` then
        # runs merge_asof backward, which hands the bar at 14:00 -- and
        # 14:15, 14:30, 14:45 -- counts drawn from articles up to 14:59.
        # Up to 59 minutes of look-ahead on every intraday bar.
        #
        # A window covering [H, H+1) is knowable at H+1. Moving the label
        # there makes it mean "available from", which is what the backward
        # merge is entitled to assume.
        levels = list(aggregated.index.names)
        time_level = levels[-1]
        aggregated.index = aggregated.index.set_levels(
            aggregated.index.levels[levels.index(time_level)]
            + pd.Timedelta(hours=1),
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
                direction='backward')
        return self._finalize_merge_result(df_merged)

    def _merge_per_ticker(self, df_reset: pd.DataFrame, aggregated_reset:
        pd.DataFrame) ->pd.DataFrame:
        """Merge aggregated news features per ticker group (no cross-ticker leakage)."""
        parts = []
        for ticker, group in df_reset.groupby('ticker'):
            ticker_features = aggregated_reset[aggregated_reset['ticker']
                == ticker]
            if ticker_features.empty:
                ticker_features = aggregated_reset[aggregated_reset[
                    'ticker'] == 'general']
            ticker_features = ticker_features.drop(columns=['ticker']
                ).drop_duplicates(subset=['datetime'], keep='last')
            merged_group = pd.merge_asof(group.sort_values('datetime'),
                ticker_features.sort_values('datetime'), on='datetime',
                direction='backward')
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
        """Finalize merge result."""
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
