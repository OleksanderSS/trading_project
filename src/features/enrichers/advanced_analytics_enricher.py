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
        phase_config = self.config.get('market_phase', {'indicators': {
            'volatility': 'VOLATILITY_20', 'trend': 'SMA_50', 'regime':
            'MARKET_REGIME'}, 'rules': [{'condition':
            'volatility < 0.02 and regime == 0', 'phase': 'calm_bull'}, {
            'condition': 'volatility < 0.02 and regime == 1', 'phase':
            'calm_bear'}, {'condition':
            'volatility >= 0.02 and regime == 0', 'phase': 'volatile_bull'},
            {'condition': 'volatility >= 0.02 and regime == 1', 'phase':
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
            if 'nlp_sentiment_score' in df_enriched.columns:
                sentiment_col = 'nlp_sentiment_score'
                temp_df = df_enriched[[sentiment_col]].copy()
            else:
                sentiment_col = 'sentiment'
                temp_df = news_df[[sentiment_col]].copy()
            stats = self.sentiment_calculator.calculate_sentiment_stats(temp_df
                , sentiment_col)
            df_enriched['sentiment_mean'] = stats['mean']
            df_enriched['sentiment_std_stat'] = stats['std']
            df_enriched['sentiment_pos_threshold'] = stats['positive_threshold'
                ]
            df_enriched['sentiment_neg_threshold'] = stats['negative_threshold'
                ]
            logger.info(
                f"Added sentiment statistics: mean={stats['mean']:.3f}, std={stats['std']:.3f}"
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
