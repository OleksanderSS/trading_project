from datetime import datetime
from typing import Any

import contextlib
import time

import numpy as np
import pandas as pd

from src.analytics.calculators.advanced_econometrics_calculator import AdvancedEconometricsCalculator
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.features.selection.enhanced_smart_selector import get_enhanced_smart_selector
from src.pipeline.stages.base_stage import BaseStage
from src.pipeline.target_column_utils import is_target_like_column
from src.pipeline.timeframe_lineage import (
    normalize_timeframe,
    partition_market_frame_by_timeframe,
    timeframe_lineage_report,
)

from .enricher import FeatureEnricher
from .guards import FeatureGuards
from .targets import TargetGenerator
from .timeframe_context import BackwardTimeframeContextAssembler


class FeatureEngineeringStage(BaseStage):
    """
    Modular Stage 3: Advanced Feature Engineering Hub.
    Delegates to specialized components for enrichment, target generation, and safety.
    """

    #: Wall time per phase of the last run(), filled by _phase().
    _phase_seconds: dict[str, float] = {}

    def __init__(self, config_manager: UnifiedConfigManager, error_handler: ErrorHandler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger('FeatureEngineeringStage')

        # Initialize Core Components
        self.selector = get_enhanced_smart_selector(config_manager)

        # Initialize Specialized Modular Components
        self.guards = FeatureGuards(mode=kwargs.get('mode', 'full'))
        self.enricher = FeatureEnricher(config_manager)
        self.target_gen = TargetGenerator(config_manager)
        self.timeframe_context_assembler = BackwardTimeframeContextAssembler()
        self._last_timeframe_context_report: dict[str, Any] = {}
        self._last_causal_evidence: dict[str, Any] = {}

        self.logger.info("✅ FeatureEngineeringStage (Modular) initialized")

    _SCRATCH_MARKERS = ('_backup', 'backup_', 'test_', '_prepurge',
                        '_orphan', 'snapshot', '__run_')

    @classmethod
    def _is_scratch_table(cls, name: str) -> bool:
        """Backups and scratch tables are not sources.

        Forwarding every frame to the enrichers fixed eight collectors that
        had never reached a feature, but it also handed them
        market_data_raw_backup_15m_20260805, market_data_raw_prepurge_20260805,
        experience_diary_backup_20260730, test_table and test_fts_tiny — six
        frames of hundreds of thousands of rows that no enricher asks for.

        Harmless in the sense that nothing reads them, expensive in the sense
        that they are held in memory for every timeframe, and dangerous in the
        one case that matters: a backup of market_data_raw carries the same
        column names as the live table, so a future enricher matching on
        columns rather than on key could silently pick the wrong one.
        """
        lowered = str(name).lower()
        return any(marker in lowered for marker in cls._SCRATCH_MARKERS)

    @contextlib.contextmanager
    def _phase(self, name: str):
        """Accumulate wall time per phase of stage 3.

        Added 2026-08-21 after a five-hour run was killed on a wrong guess.
        Asked whether it was behaving abnormally, I had no measurement -- only
        per-enricher lines and nothing for the phases around them -- so I
        reasoned from a plausible theory, blamed a change of my own, and had to
        reconstruct the truth afterwards by diffing timestamps out of a 31 MB
        log. The run had been normal: the previous SUCCESSFUL rebuild spent
        3.8 hours in the same pauses against this one's 4.5.

        What the forensics found is worth having printed every run instead:

            TechnicalAnalysisEnricher    99 min
            RedundancyDetector           88 min
            smart_selector               51 min

        None of it new, and the largest consumer computes variance-inflation
        factors across 2,192 columns -- including the 1,494 ctx_/state_ columns
        measured to contribute nothing to any decision.
        """
        start = time.perf_counter()
        try:
            yield
        except BaseException:
            self._phase_seconds[name] = (
                self._phase_seconds.get(name, 0.0) + time.perf_counter() - start
            )
            # A crash five hours in is exactly when the breakdown is worth
            # having, so print it on the way out rather than losing it.
            self._log_phase_breakdown()
            raise
        else:
            self._phase_seconds[name] = (
                self._phase_seconds.get(name, 0.0) + time.perf_counter() - start
            )

    def _log_phase_breakdown(self) -> None:
        """Print where the stage's time went, longest first."""
        if not self._phase_seconds:
            return
        total = sum(self._phase_seconds.values())
        self.logger.info('⏱️ Stage 3 phase breakdown (total %.1f min):', total / 60)
        for name, seconds in sorted(self._phase_seconds.items(),
                                    key=lambda kv: -kv[1]):
            share = seconds / total * 100 if total else 0.0
            self.logger.info('    %7.1f min  %5.1f%%  %s', seconds / 60, share, name)

    async def run(self, **kwargs) -> dict[str, Any]:
        """Runs the feature engineering cycle."""
        self._phase_seconds: dict[str, float] = {}
        self.logger.info('Starting modular feature engineering stage...')

        with self._phase('validate + prepare market data'):
            cleaned_data, market_data_dict = self._validate_and_prepare_market_data(**kwargs)
        if not market_data_dict:
            raise ValueError(
                "FeatureEngineeringStage: No market data available. "
                "Ensure Stage 2 (ProcessingStage) produced 'prices' in cleaned_data."
            )

        enriched_data: dict[str, pd.DataFrame] = {}
        all_targets: dict[str, pd.DataFrame] = {}

        if 'news' in cleaned_data:
            with self._phase('score news sentiment'):
                cleaned_data['news'] = self._score_news_sentiment(cleaned_data['news'])

        # 1. Enrichment for each timeframe
        for tf, df in market_data_dict.items():
            # Every collected table reaches the enrichers, not a list of two.
            #
            # This forwarded `macro_data` and `news` and nothing else, which is
            # the single reason eight enabled collectors produced no feature
            # column at all. Measured on the 2026-08-15 batch: cftc (2,610 rows,
            # weekly since 2016), fear_greed (267 daily rows since 2025),
            # wikimedia_attention (11,417), insider_trades (1,395),
            # sociological_sentiment_data (737), economic_calendar, sdmx_macro —
            # all collected, all cleaned, none ever handed to this step. They
            # were not missing enrichers; they were missing a hand-off.
            #
            # A whitelist here fails silently and invisibly: adding a collector
            # and an enricher is not enough, and nothing says so. Passing the
            # frames through instead means the enricher's own kwargs decide what
            # it reads, and an enricher that wants a source it was never given
            # can say so in its own log.
            #
            # Keyed under BOTH the bare stem and the table name, because the
            # two halves of the codebase disagree: stage inputs arrive as
            # `cftc_data` / `fear_greed_data`, while enrichers and their tests
            # ask for `cftc` / `fear_greed`.
            enrich_kwargs: dict[str, Any] = {}
            for source in (kwargs, cleaned_data):
                for key, value in (source or {}).items():
                    if key == 'prices' or not isinstance(value, pd.DataFrame):
                        continue
                    if value.empty or self._is_scratch_table(key):
                        continue
                    enrich_kwargs.setdefault(key, value)
                    if key.endswith('_data'):
                        enrich_kwargs.setdefault(key[:-5], value)
            # cleaned_data wins where both carry the same name: stage 2 filtered
            # and normalised those frames, and `news` in particular has just
            # been scored with FinBERT above.
            for key in ('macro_data', 'news'):
                if key in cleaned_data:
                    enrich_kwargs[key] = cleaned_data[key]
            if kwargs.get('offline_only'):
                enrich_kwargs['offline_only'] = True
            self.logger.info(
                "Enrichment inputs for %s: %s", tf,
                sorted(k for k, v in enrich_kwargs.items()
                       if isinstance(v, pd.DataFrame)),
            )

            with self._phase(f'enrich {tf}'):
                enriched_df = self.enricher.enrich_features(df, timeframe=tf, **enrich_kwargs)
            enriched_df = self._restore_service_columns(enriched_df, df)

            # 2. Target Generation (for all timeframes, not just 1d)
            targets_df = self.target_gen.generate_targets(
                enriched_df,
                timeframe=tf,
            )
            all_targets[tf] = targets_df
            target_cols = [col for col in targets_df.columns if col.startswith('target_')]
            for col in target_cols:
                enriched_df[col] = targets_df[col].reindex(enriched_df.index)

            # 3. Apply Safety Guards
            with self._phase(f'guards {tf}'):
                enriched_df = self.guards.apply_guards(enriched_df)
            enriched_df = self._restore_service_columns(enriched_df, df)

            enriched_data[tf] = enriched_df

        # 4. Feature Selection (on the primary timeframe)
        with self._phase('combine timeframes'):
            final_features = self._combine_timeframes(enriched_data)
        with self._phase('list numeric columns'):
            selected_features = self._initial_feature_columns(final_features)
        feature_importance: dict[str, float] = {}
        if not final_features.empty:
            target_col = kwargs.get('target_column', 'target_up_1d')
            if target_col in final_features.columns:
                with self._phase('feature selection (VIF + selector)'):
                    selected_features, feature_importance = await self._select_features(
                        final_features,
                        target_col,
                        kwargs,
                    )

        self._log_phase_breakdown()

        return {
            'status': 'success',
            'features': final_features,
            'enriched_data': final_features,
            'all_timeframes': enriched_data,
            'enriched_prices': enriched_data,
            'all_targets': all_targets,
            'combined_features': final_features,
            'selected_features': selected_features,
            'feature_importance': feature_importance,
            'timeframe_context_report': self._last_timeframe_context_report,
            'causal_evidence': self._last_causal_evidence,
            'timestamp': datetime.now().isoformat()
        }

    def _validate_and_prepare_market_data(self, **kwargs):
        cleaned_data = kwargs.get('cleaned_data', {})
        stage_logger = getattr(self, "logger", None)
        if stage_logger is not None:
            stage_logger.info(
                f"Stage3 input keys: {list(kwargs.keys())}; "
                f"cleaned_data keys: {list(cleaned_data.keys()) if isinstance(cleaned_data, dict) else type(cleaned_data).__name__}"
            )
        market_data_raw = cleaned_data.get('prices')
        if market_data_raw is None:
            market_data_raw = cleaned_data.get('market_data')
        if market_data_raw is None:
            market_data_raw = kwargs.get('market_data')

        if market_data_raw is None:
            if stage_logger is not None:
                stage_logger.error("No market data found in cleaned_data['prices'], cleaned_data['market_data'], or kwargs['market_data']")
        elif isinstance(market_data_raw, dict):
            inner_types = {k: type(v).__name__ for k, v in market_data_raw.items()}
            if stage_logger is not None:
                stage_logger.info(f"Market data dict contents: {inner_types}")
        else:
            if stage_logger is not None:
                stage_logger.info(f"Market data type: {type(market_data_raw).__name__}")

        if isinstance(market_data_raw, pd.DataFrame):
            market_data_raw = partition_market_frame_by_timeframe(
                market_data_raw
            )
        elif isinstance(market_data_raw, dict):
            validated = {}
            for raw_timeframe, frame in market_data_raw.items():
                if not isinstance(frame, pd.DataFrame) or frame.empty:
                    # An entire requested timeframe leaving here without a
                    # word is how 15m disappeared from the 2026-08-04 batch:
                    # batch_metadata recorded timeframes ['15m','1d','1h'],
                    # features.parquet held only 1d and 60m, targets.parquet
                    # carried no 15m column, and 0 of 506 champions were 15m.
                    # Every later stage then reported success on two thirds
                    # of the requested scope.
                    if stage_logger is not None:
                        stage_logger.error(
                            "Timeframe '%s' reached feature engineering with "
                            "%s and is being dropped. No features, no targets "
                            "and no models will exist for it, and nothing "
                            "downstream will mention it again.",
                            raw_timeframe,
                            "no DataFrame"
                            if not isinstance(frame, pd.DataFrame)
                            else "0 rows",
                        )
                    continue
                declared = normalize_timeframe(raw_timeframe)
                report = timeframe_lineage_report(
                    frame,
                    declared_timeframe=declared,
                )
                if report.get("status") in {
                    "timeframe_cadence_mismatch",
                    "timeframe_cadence_ambiguous",
                }:
                    raise ValueError(
                        f"Market frame {raw_timeframe} conflicts with "
                        f"observed {report.get('observed_timeframe')} "
                        "cadence"
                    )
                resolved = report.get("resolved_timeframe")
                if not resolved:
                    raise ValueError(
                        f"Market frame {raw_timeframe} has no "
                        "cadence-validated timeframe"
                    )
                candidate = frame.copy()
                candidate["interval"] = resolved
                candidate.attrs["timeframe_lineage"] = report
                candidate.attrs["timeframe_source"] = (
                    "stage3_input_mapping_key"
                )
                validated[resolved] = candidate
            market_data_raw = validated

        return cleaned_data, market_data_raw

    def _combine_timeframes(self, enriched_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build point-in-time higher-timeframe context for every base frame."""
        # Filter each dataframe to only contain its own interval
        filtered_data = {}
        for tf, df in enriched_data.items():
            if 'interval' in df.columns:
                # Compared on NORMALISED names. A raw `df['interval'] == tf`
                # is one spelling away from selecting nothing: this project
                # writes '1h' in config and '60m' in data, and the same pair
                # has already cost it a rolling-window budget lookup and a
                # macro availability column. partition_market_frame_by_timeframe
                # normalises both sides today, so this is defence rather than
                # a live bug -- but it costs nothing and the failure mode is
                # an entire timeframe silently emptying.
                normalized_tf = normalize_timeframe(tf)
                matches = df['interval'].map(normalize_timeframe) == normalized_tf
                filtered_df = df[matches].copy()
                self.logger.info(f"Filtered {tf} timeframe: {len(df)} -> {len(filtered_df)} rows")
                if filtered_df.empty and not df.empty:
                    self.logger.error(
                        "Timeframe '%s' had %d row(s) but none whose interval "
                        "matches it (present: %s); it will contribute nothing "
                        "to the combined features.",
                        tf, len(df), sorted(set(df['interval'].dropna().astype(str)))[:5],
                    )
                filtered_data[tf] = filtered_df
            else:
                filtered_data[tf] = df

        assembler = getattr(
            self,
            "timeframe_context_assembler",
            BackwardTimeframeContextAssembler(),
        )
        combined, report = assembler.assemble(filtered_data)
        self._last_timeframe_context_report = report
        self.logger.info(
            "Assembled %s causal timeframe contexts: %s total rows",
            report["summary"]["base_context_count"],
            report["summary"]["output_rows"],
        )
        return combined

    def _initial_feature_columns(self, frame: pd.DataFrame) -> list[str]:
        """Return numeric model candidates without labels or service metadata."""
        if frame.empty:
            return []
        metadata_columns = {
            "datetime",
            "date",
            "timestamp",
            "ticker",
            "interval",
            "timeframe",
        }
        # `frame.dtypes`, not `frame.select_dtypes(...)`.
        #
        # select_dtypes CONSOLIDATES and copies the matching columns into one
        # block just to hand back their names. On the 2026-08-21 batch that is
        # 2,192 columns x 258,397 rows of float64 -- **4.22 GiB allocated to
        # answer a question about column NAMES**, and it is what failed the
        # rebuild with MemoryError twice: once at 256,208 rows and once here.
        #
        # Deepening the daily history from 2 years to 30 is what pushed it over,
        # but the allocation was always pointless: dtypes are metadata and cost
        # nothing to read. Casting to float32 would have halved a number that
        # should be zero.
        return [
            column
            for column, dtype in frame.dtypes.items()
            if dtype.kind in "iuf"
            and column not in metadata_columns
            and not is_target_like_column(column)
        ]

    #: Columns restored onto an enriched frame when an enricher drops them.
    _SERVICE_COLUMNS = (
        "datetime",
        "timestamp",
        "date",
        "ticker",
        "interval",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "hash",
    )

    #: Columns that identify a row well enough to prove two frames are still in
    #: the same order. `hash` is the collector's per-row SHA-256 and is exact;
    #: the price columns are a strong practical check when it was dropped.
    _ALIGNMENT_ANCHORS = ("hash", "close", "volume", "open")

    @staticmethod
    def _score_news_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
        """Give the news a sentiment before any enricher asks it for one.

        The collectors store a `sentiment` column and leave it empty. On the
        2026-08-13 batch it held 15,165 empty strings, which the enricher
        reported plainly --

            News sentiment column 'sentiment': 15165/15165 non-null, values: ['']

        -- and `pd.to_numeric` turned every one into NaN. Every sentiment
        feature was therefore built from nothing, and `sentiment_available`
        read 0 on all 55,565 rows.

        FinBERT was running the whole time. It runs inside `news_impact`,
        which the priority order places AFTER `sentiment_features`, and its
        scores go into that analyzer rather than back onto the news. The model
        was loaded, the texts were scored, and the result never reached the
        frame the other enrichers read.

        Scoring once here, before the per-timeframe loop, means every enricher
        and all three timeframes see the same values, and the model is loaded
        once per run instead of per enricher.

        The sign convention is this project's existing one, from
        `SentimentIntegrator._extract_sentiment_scores`: +score for positive,
        -score for negative, 0 for neutral. A second convention would be a
        second answer to a question already answered.
        """
        logger = ProjectLogger.get_logger(__name__)
        if news_df is None or getattr(news_df, 'empty', True):
            return news_df

        if 'sentiment' in news_df.columns:
            existing = pd.to_numeric(news_df['sentiment'], errors='coerce')
            if existing.notna().any():
                logger.info(
                    "News already carries %d numeric sentiment values; "
                    "leaving them.", int(existing.notna().sum()),
                )
                return news_df

        # Pick by CONTENT, not by presence. `notna().any()` was true for a
        # `text` column that is 15,274 empty strings -- this database stores
        # blanks as '' rather than NaN, which is the same trap the empty
        # `sentiment` column set. FinBERT then scored the word "neutral"
        # (what _prepare_batch_texts substitutes for an empty string), hit its
        # cache 15,273 times, and returned all-neutral in 2.8 seconds on CPU:
        #
        #   Scored 15274 news items with FinBERT from 'text': 0 non-neutral
        #
        # A real forward pass over that many texts takes minutes. The timing
        # was the tell.
        filled = {}
        for candidate in ('content', 'text', 'description', 'title'):
            if candidate not in news_df.columns:
                continue
            values = news_df[candidate].fillna('').astype(str).str.strip()
            filled[candidate] = int((values != '').sum())

        text_col = max(filled, key=filled.get) if filled else None
        if text_col is None or filled.get(text_col, 0) == 0:
            logger.error(
                "News carries no usable text to score (non-empty counts: %s); "
                "sentiment features will be empty for this run.",
                filled or list(news_df.columns)[:12],
            )
            return news_df

        usable = filled[text_col]
        if usable < len(news_df):
            logger.warning(
                "Scoring sentiment from '%s': %d of %d items have text, the "
                "rest are blank and will score neutral.",
                text_col, usable, len(news_df),
            )

        texts = news_df[text_col].fillna('').astype(str).tolist()
        try:
            from src.sentiment.sentiment_models import analyze_sentiment
            scored = analyze_sentiment(texts)
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            logger.error(
                "Could not score news sentiment (%s: %s); sentiment features "
                "will be empty for this run.", type(exc).__name__, exc,
            )
            return news_df

        if scored is None or len(scored) != len(news_df):
            logger.error(
                "Sentiment scoring returned %s rows for %d news items; "
                "refusing to attach them by position.",
                'no' if scored is None else len(scored), len(news_df),
            )
            return news_df

        labels = scored['label'].to_numpy()
        scores = scored['score'].to_numpy(dtype=float)
        signed = np.where(
            labels == 'positive', scores,
            np.where(labels == 'negative', -scores, 0.0),
        )

        news_df = news_df.copy()
        news_df['sentiment'] = signed
        logger.info(
            "Scored %d news items with FinBERT from '%s': %d non-neutral, "
            "range %.3f..%.3f", len(signed), text_col,
            int((signed != 0).sum()), float(signed.min()), float(signed.max()),
        )
        return news_df

    def _restore_service_columns(
        self,
        enriched_df: pd.DataFrame,
        source_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Restore service columns an enricher dropped, WITHOUT trusting order.

        This method wrote `source_df[column].to_numpy()` straight onto the
        enriched frame, guarded only by equal row counts. Row count is not row
        identity: an enricher that returns the same rows in a different order
        (a groupby-apply, a sort, a merge) makes that assignment paste every
        value onto the wrong row.

        That is not hypothetical -- it is what produced the 2026-08-06 batch.
        Measured on it: for AAPL 1d, all 327 exported bars are genuine bars
        from the database (matched by `hash`), every OHLCV field and every
        calendar feature belongs to that real bar, and NOT ONE row carries the
        date the bar actually has. Only `datetime` was wrong, because
        `datetime` was the only column the enricher dropped and this method
        restored. `apply_guards` then sorted by that wrong date (see run()),
        which is why the file looks tidy: dates ascending, bars shuffled
        against them, offsets up to 686 days. Every indicator, every
        shift(-n) target and every model trained on that batch was built on
        bars in a random order.

        The repair: prove the rows correspond before copying anything.
        `hash` (the collector's per-row SHA-256) gives an exact join. Failing
        that, a surviving price column proves positional alignment. If neither
        is available the frames cannot be matched, and this raises rather than
        guessing -- a silent wrong answer here is invisible for weeks.
        """
        if len(enriched_df) != len(source_df):
            return enriched_df

        result = enriched_df.copy()
        missing = [
            column
            for column in self._SERVICE_COLUMNS
            if column not in result.columns and column in source_df.columns
        ]
        if not missing:
            return result

        # Exact route: join on the row hash, which survives reordering.
        if "hash" in result.columns and "hash" in source_df.columns:
            source_by_hash = source_df.drop_duplicates("hash").set_index("hash")
            if result["hash"].is_unique and result["hash"].isin(source_by_hash.index).all():
                for column in missing:
                    result[column] = result["hash"].map(source_by_hash[column]).to_numpy()
                return result

        # Fallback: a column present in BOTH frames proves the row order is
        # unchanged, so a positional copy is safe.
        anchor = next(
            (
                column
                for column in self._ALIGNMENT_ANCHORS
                if column in result.columns and column in source_df.columns
            ),
            None,
        )
        if anchor is not None:
            left = result[anchor].to_numpy()
            right = source_df[anchor].to_numpy()
            aligned = (
                np.array_equal(left, right)
                if left.dtype == object or not np.issubdtype(left.dtype, np.number)
                else np.allclose(
                    left.astype(float), right.astype(float), rtol=1e-9, equal_nan=True
                )
            )
            if aligned:
                for column in missing:
                    result[column] = source_df[column].to_numpy(copy=True)
                return result
            raise ValueError(
                f"Cannot restore {missing} onto the enriched frame: the "
                f"enricher reordered its rows (anchor column '{anchor}' no "
                f"longer matches the source row for row order). Restoring by "
                f"position here would attach these values to the wrong bars."
            )

        raise ValueError(
            f"Cannot restore {missing} onto the enriched frame: no 'hash' and "
            f"no shared anchor column to prove the rows still correspond. "
            f"Enrichers must preserve either the row hash or a price column."
        )

    # Held-out fraction (chronological tail, per ticker) that feature selection
    # must never see. Matches the test_size=0.2 convention already used by
    # src/models/adapters/data_preparation.py's train/val/test split, so the
    # rows Stage 4's walk-forward validator eventually treats as unseen were
    # never used to decide which columns exist.
    _SELECTION_HOLDOUT_FRACTION = 0.2

    def _train_only_index(self, frame: pd.DataFrame) -> pd.Index:
        """Chronological prefix of `frame`, excluding the final holdout tail.

        Grouped per ticker (when a 'ticker' column is present) so a
        multi-symbol frame doesn't leak one ticker's future into another's
        selection window. Falls back to positional order (frame is expected
        to already be time-sorted per timeframe) when no explicit time
        column is available.
        """
        time_col = next((c for c in ('datetime', 'timestamp', 'date') if c in frame.columns), None)

        def _prefix(group: pd.DataFrame) -> pd.Index:
            ordered = group.sort_values(time_col) if time_col else group
            cutoff = int(len(ordered) * (1.0 - self._SELECTION_HOLDOUT_FRACTION))
            return ordered.index[:cutoff]

        if 'ticker' in frame.columns:
            parts = [_prefix(g) for _, g in frame.groupby('ticker', sort=False)]
            return parts[0].append(parts[1:]) if parts else frame.index[:0]
        return _prefix(frame)

    # Column-name prefixes/substrings that identify an "external" predictor
    # (macro, sentiment, news, economic-calendar) as opposed to a technical
    # indicator derived from the ticker's own price — Granger-testing a
    # technical indicator against the price it was computed from is close
    # to circular, so those are deliberately excluded from this diagnostic.
    _EXTERNAL_PREDICTOR_MARKERS: tuple[str, ...] = (
        'FRED_', 'sentiment', 'nlp_sentiment', 'finbert_', 'news_', 'economic_', 'macro_', 'surprise_index',
    )
    _MAX_CAUSAL_DIAGNOSTIC_PREDICTORS = 15

    def _diagnose_external_predictor_causality(
        self,
        candidate_features: pd.DataFrame,
        target_series: pd.Series,
        target_col: str,
    ) -> dict[str, Any]:
        """Diagnostic-only Granger/stationarity/cointegration check for
        external (macro/sentiment/news) predictors against the target.

        Does NOT filter or reweight feature selection — this is attached to
        the Stage 3 output as `causal_evidence` purely so a human (or a
        later, explicitly-approved change) can see which external
        predictors show real lead-lag structure versus which are along for
        the ride on correlation alone. Best-effort: any failure returns an
        empty/partial result rather than blocking feature selection, and a
        predictor count cap keeps this from dominating Stage 3 runtime,
        since each predictor fits a VAR model plus stationarity/
        cointegration/impulse-response/variance-decomposition tests.
        """
        external_cols = [
            col
            for col in candidate_features.columns
            if any(marker in col for marker in self._EXTERNAL_PREDICTOR_MARKERS)
        ][: self._MAX_CAUSAL_DIAGNOSTIC_PREDICTORS]

        if not external_cols:
            return {}

        frame = candidate_features[external_cols].copy()
        frame[target_col] = target_series

        try:
            causality_results = AdvancedEconometricsCalculator.run_comprehensive_causal_analysis(
                frame, target_col, external_cols, maxlag=10, lag_selection='aic',
            )
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.warning(f"Causal diagnostic failed, skipping: {e}")
            return {}

        evidence = {
            col: {
                'is_significant': result.get('is_significant'),
                'causality_strength': result.get('causality_strength'),
                'p_value': (result.get('granger_test') or {}).get('p_value'),
                'is_cointegrated': (result.get('cointegration') or {}).get('is_cointegrated'),
            }
            for col, result in causality_results.items()
            if col != '_summary' and isinstance(result, dict) and 'error' not in result
        }

        if evidence:
            significant = sum(1 for v in evidence.values() if v.get('is_significant'))
            self.logger.info(
                f"Causal diagnostic: {significant}/{len(evidence)} external predictors show "
                f"significant Granger causality vs '{target_col}' (diagnostic only, selection unaffected)."
            )
        return evidence

    async def _select_features(
        self,
        final_features: pd.DataFrame,
        target_col: str,
        kwargs: dict[str, Any],
    ) -> tuple[list[str], dict[str, float]]:
        target_cols = [col for col in final_features.columns if is_target_like_column(col)]
        metadata_cols = {'datetime', 'date', 'timestamp', 'ticker', 'interval'}
        candidate_features = final_features.drop(columns=target_cols, errors='ignore')
        candidate_features = candidate_features.drop(
            columns=[col for col in metadata_cols if col in candidate_features.columns],
            errors='ignore',
        )
        candidate_features = candidate_features.select_dtypes(include='number')
        target_series = final_features[target_col]
        # Ensure we do not leak the target into features
        candidate_features = candidate_features.drop(columns=[target_col], errors='ignore')
        valid_index = candidate_features.index.intersection(target_series.dropna().index)

        # LEAKAGE FIX: restrict the index used for selection to a chronological
        # train-only prefix. Selecting against the full dataset (including the
        # rows Stage 4 later holds out) lets feature choice "see" future
        # target values, inflating apparent walk-forward skill.
        train_only_index = self._train_only_index(final_features).intersection(valid_index)
        if len(train_only_index) < 5:
            # Not enough train-only rows (e.g. tiny fixture data) — fall back
            # to the full valid_index rather than failing selection outright.
            train_only_index = valid_index

        if candidate_features.empty or len(valid_index) < 5 or target_series.loc[valid_index].nunique() < 2:
            fallback = list(candidate_features.columns)
            return fallback, dict.fromkeys(fallback, 1.0)

        self._last_causal_evidence = self._diagnose_external_predictor_causality(
            candidate_features.loc[train_only_index],
            target_series.loc[train_only_index],
            target_col,
        )

        try:
            selection_result = await self.selector.select_with_full_analysis(
                candidate_features.loc[train_only_index],
                target_series.loc[train_only_index],
                context_id=kwargs.get('context_id', f'stage3_{target_col}'),
                market_data=final_features.loc[train_only_index],
                max_features=kwargs.get('max_features'),
            )
            selected = selection_result.get('selected_features') or []
            selected = [feature for feature in selected if feature in candidate_features.columns]
            if not selected:
                selected = list(candidate_features.columns)
            importance = {
                feature: 1.0 / (rank + 1)
                for rank, feature in enumerate(selected)
            }
            return selected, importance
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Feature selection failed critically: {e}', exc_info=True)
            fallback = list(candidate_features.columns)
            return fallback, dict.fromkeys(fallback, 1.0)
