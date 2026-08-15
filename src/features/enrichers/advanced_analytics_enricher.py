from typing import Any

import pandas as pd

from src.analytics.calculators.macro_score_calculator import MacroScoreCalculator
from src.analytics.calculators.sentiment_stats_calculator import SentimentStatsCalculator
from src.analytics.context.market_phase_analyzer import MarketPhaseAnalyzer
from src.core.logging.logger import ProjectLogger
from src.features.enrichers.base import BaseEnricher

logger = ProjectLogger.get_logger('AdvancedAnalyticsEnricher')


class AdvancedAnalyticsEnricher(BaseEnricher):
    """
    Enriches DataFrame with advanced analytics features:
    - Sentiment statistics (mean, std, thresholds)
    - Macro composite score
    - Market phase detection
    """

    def __init__(self, config: (dict[str, Any] | None)=None):
        """Initialize with optional config from FeatureOrchestrator."""
        super().__init__()
        self.config = config or {}
        self.macro_calculator: MacroScoreCalculator | None = None
        self.phase_analyzer: MarketPhaseAnalyzer | None = None
        self.sentiment_calculator = SentimentStatsCalculator()
        macro_indicators = self.config.get('macro_indicators', {'FRED_GDP':
            {'weight': 0.3, 'direction': 'positive'}, 'FRED_UNRATE': {
            'weight': 0.2, 'direction': 'negative'}, 'FRED_VIXCLS': {
            'weight': 0.2, 'direction': 'negative'}, 'FRED_DGS10': {
            'weight': 0.15, 'direction': 'positive'}, 'FRED_CPIAUCSL': {
            'weight': 0.15, 'direction': 'negative'}})
        try:
            self.macro_calculator = MacroScoreCalculator(macro_indicators)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            logger.warning(f'Failed to initialize MacroScoreCalculator: {e}')
            self.macro_calculator = None
        # The regime is a WORD, and these rules compared it to 0 and 1.
        # MARKET_REGIME holds TRENDING_UP, TRENDING_DOWN, RANGING,
        # MEAN_REVERSION, NORMAL — so all four rules were false on every row,
        # each fell through to the catch-all, and market_phase was the
        # constant 'neutral' (code 4) on all three timeframes in every export
        # this project has produced.
        #
        # RANGING and MEAN_REVERSION stay neutral deliberately: they really
        # are neither bull nor bear, so the catch-all is the right answer for
        # them rather than a failure to classify.
        phase_config = self.config.get('market_phase', {'indicators': {
            'volatility': 'VOLATILITY_20', 'trend': 'SMA_50', 'regime':
            'MARKET_REGIME'}, 'rules': [{'condition':
            "volatility < 0.02 and regime == 'TRENDING_UP'", 'phase':
            'calm_bull'}, {'condition':
            "volatility < 0.02 and regime == 'TRENDING_DOWN'", 'phase':
            'calm_bear'}, {'condition':
            "volatility >= 0.02 and regime == 'TRENDING_UP'", 'phase':
            'volatile_bull'}, {'condition':
            "volatility >= 0.02 and regime == 'TRENDING_DOWN'", 'phase':
            'volatile_bear'}, {'condition': 'True', 'phase': 'neutral'}]})
        try:
            self.phase_analyzer = MarketPhaseAnalyzer(phase_config)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            logger.warning(f'Failed to initialize MarketPhaseAnalyzer: {e}')
            self.phase_analyzer = None
        logger.info('AdvancedAnalyticsEnricher initialized')

    @property
    def name(self) ->str:
        return 'advanced_analytics'

    @property
    def priority(self) ->int:
        """Run after all basic enrichers, before context_map (80)"""
        return 78

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) ->pd.DataFrame:
        """
        Adds advanced analytics features to the DataFrame.

        Args:
            df: Input DataFrame
            **kwargs: May contain 'news' DataFrame

        Returns:
            DataFrame with added analytics features
        """
        if df.empty:
            logger.warning(
                'Input DataFrame is empty. Skipping advanced analytics enrichment.'
                )
            return df
        df_enriched = df.copy()
        self._add_sentiment_statistics(df_enriched, kwargs.get('news'))
        self._add_macro_composite_score(df_enriched)
        self._add_market_phase_detection(df_enriched)
        self._log_market_regime_info(df_enriched)
        logger.info('Advanced analytics enrichment completed')
        return df_enriched

    def _add_sentiment_statistics(self, df_enriched: pd.DataFrame, news_df:
        (pd.DataFrame | None)) ->None:
        """Add sentiment statistics to DataFrame."""
        if news_df is None or not isinstance(news_df, pd.DataFrame
            ) or news_df.empty:
            return
        if ('sentiment' not in news_df.columns and 'nlp_sentiment_score' not in
            df_enriched.columns):
            return
        try:
            # These were one mean, one standard deviation and two thresholds
            # computed over the WHOLE series and written onto every row. Four
            # columns with a single value each: nothing for a model to learn
            # from, and computed from news published after most of the bars
            # they were attached to.
            #
            # Expanding statistics say the same thing causally -- "how does
            # sentiment now compare with its own history up to this bar" --
            # and they vary, which is the point of a feature. min_periods=2
            # because a standard deviation of one observation is not a
            # number.
            if 'nlp_sentiment_score' not in df_enriched.columns:
                # Only the news frame carries sentiment, and it has a
                # different length and ordering than the bars. Per-bar
                # statistics need a per-bar series; broadcasting a corpus
                # scalar is what produced the constants.
                logger.info(
                    "Sentiment statistics skipped: no per-bar sentiment column "
                    "on this frame (news-only sentiment cannot be expanded "
                    "per bar without inventing an alignment)."
                )
                return

            sentiment_col = 'nlp_sentiment_score'
            series = pd.to_numeric(df_enriched[sentiment_col], errors='coerce')
            if 'ticker' in df_enriched.columns:
                grouped = series.groupby(df_enriched['ticker'])
                mean = grouped.transform(
                    lambda s: s.expanding(min_periods=2).mean())
                std = grouped.transform(
                    lambda s: s.expanding(min_periods=2).std())
            else:
                mean = series.expanding(min_periods=2).mean()
                std = series.expanding(min_periods=2).std()

            df_enriched['sentiment_mean'] = mean
            df_enriched['sentiment_std_stat'] = std
            df_enriched['sentiment_pos_threshold'] = mean + std
            df_enriched['sentiment_neg_threshold'] = mean - std
            logger.info(
                "Added expanding sentiment statistics over %d bars "
                "(last mean=%.3f, std=%.3f)",
                len(series),
                float(mean.iloc[-1]) if len(mean) and pd.notna(mean.iloc[-1]) else float('nan'),
                float(std.iloc[-1]) if len(std) and pd.notna(std.iloc[-1]) else float('nan'),
            )
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f'Error calculating sentiment stats: {e}')

    def _add_macro_composite_score(self, df_enriched: pd.DataFrame) ->None:
        """Add macro composite score to DataFrame."""
        if not self.macro_calculator:
            return
        fred_cols = [col for col in df_enriched.columns if col.startswith(
            'FRED_')]
        if len(fred_cols) < 3:
            return
        try:
            available_indicators = [ind for ind in self.macro_calculator.
                indicators_config.keys() if ind in df_enriched.columns]
            if not available_indicators:
                return
            macro_subset = df_enriched[available_indicators].copy()
            scores_df = self.macro_calculator.calculate_composite_score(
                macro_subset, rolling_window=min(252, len(df_enriched) // 2))
            if (not scores_df.empty and 'composite_macro_score' in
                scores_df.columns):
                df_enriched['macro_composite_score'] = scores_df[
                    'composite_macro_score'].values
                logger.info(
                    f"Added macro composite score (range: [{df_enriched['macro_composite_score'].min():.1f}, {df_enriched['macro_composite_score'].max():.1f}])"
                    )
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f'Error calculating macro composite score: {e}')

    def _add_market_phase_detection(self, df_enriched: pd.DataFrame) ->None:
        """Add market phase detection to DataFrame.

        Computed per-row (point-in-time), not once from the physically
        last row of the whole batch: the old approach used
        market_data.iloc[-1] and broadcast that single scalar to every
        row, which both mixed tickers together in a multi-ticker batch
        and leaked future information into every historical row's
        feature. MarketPhaseAnalyzer._determine_market_phase() only ever
        needs a single row's own indicator values (no trailing window),
        so evaluating it per-row is correct and needs no groupby('ticker')
        - each row's own already-per-ticker indicator columns are enough.
        """
        if not self.phase_analyzer:
            return
        try:
            required_cols = list(self.phase_analyzer.indicators.values())
            if not all(col in df_enriched.columns for col in required_cols):
                missing = [col for col in required_cols if col not in
                    df_enriched.columns]
                logger.warning(
                    f'Cannot detect market phase: missing columns {missing}')
                return
            phase_map = self._get_phase_mapping()
            phases = [
                phase_map.get(
                    self.phase_analyzer.analyze(
                        {'market_data': df_enriched.iloc[[i]]}
                    ).get('market_phase', 'unknown'),
                    5,
                )
                for i in range(len(df_enriched))
            ]
            df_enriched['market_phase'] = phases
            logger.info(
                f'Added per-row market phase detection ({len(set(phases))} distinct phases across {len(phases)} rows)'
                )
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f'Error detecting market phase: {e}')

    def _get_phase_mapping(self) ->dict[str, int]:
        """Get market phase mapping to numeric values."""
        return {'calm_bull': 0, 'calm_bear': 1, 'volatile_bull': 2,
            'volatile_bear': 3, 'neutral': 4, 'unknown': 5, 'error': 6}

    def _log_market_regime_info(self, df_enriched: pd.DataFrame) ->None:
        """Log information about MARKET_REGIME handling."""
        if 'MARKET_REGIME' not in df_enriched.columns:
            logger.info(
                'MARKET_REGIME will be added by TechnicalAnalysisEnricher')
