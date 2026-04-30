import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

from src.features.enrichers.base import BaseEnricher
from src.analytics.calculators.sentiment_stats_calculator import SentimentStatsCalculator
from src.analytics.calculators.macro_score_calculator import MacroScoreCalculator
from src.analytics.context.market_phase_analyzer import MarketPhaseAnalyzer
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("AdvancedAnalyticsEnricher")

class AdvancedAnalyticsEnricher(BaseEnricher):
    """
    Enriches DataFrame with advanced analytics features:
    - Sentiment statistics (mean, std, thresholds)
    - Macro composite score
    - Market phase detection
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional config from FeatureOrchestrator."""
        self.config = config or {}
        
        # Initialize calculators
        self.sentiment_calculator = SentimentStatsCalculator()
        
        # Macro score calculator config
        macro_indicators = self.config.get('macro_indicators', {
            'FRED_GDP': {'weight': 0.3, 'direction': 'positive'},
            'FRED_UNRATE': {'weight': 0.2, 'direction': 'negative'},
            'FRED_VIXCLS': {'weight': 0.2, 'direction': 'negative'},
            'FRED_DGS10': {'weight': 0.15, 'direction': 'positive'},
            'FRED_CPIAUCSL': {'weight': 0.15, 'direction': 'negative'}
        })
        
        try:
            self.macro_calculator = MacroScoreCalculator(macro_indicators)
        except Exception as e:
            logger.warning(f"Failed to initialize MacroScoreCalculator: {e}")
            self.macro_calculator = None
        
        # Market phase analyzer config
        phase_config = self.config.get('market_phase', {
            'indicators': {
                'volatility': 'VOLATILITY_20',
                'trend': 'SMA_50',
                'regime': 'MARKET_REGIME'
            },
            'rules': [
                {'condition': 'volatility < 0.02 and regime == 0', 'phase': 'calm_bull'},
                {'condition': 'volatility < 0.02 and regime == 1', 'phase': 'calm_bear'},
                {'condition': 'volatility >= 0.02 and regime == 0', 'phase': 'volatile_bull'},
                {'condition': 'volatility >= 0.02 and regime == 1', 'phase': 'volatile_bear'},
                {'condition': 'True', 'phase': 'neutral'}
            ]
        })
        
        try:
            self.phase_analyzer = MarketPhaseAnalyzer(phase_config)
        except Exception as e:
            logger.warning(f"Failed to initialize MarketPhaseAnalyzer: {e}")
            self.phase_analyzer = None
        
        logger.info("AdvancedAnalyticsEnricher initialized")

    @property
    def name(self) -> str:
        return "advanced_analytics"

    @property
    def priority(self) -> int:
        """Run after all basic enrichers, before context_map (80)"""
        return 78

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Adds advanced analytics features to the DataFrame.

        Args:
            df: Input DataFrame
            **kwargs: May contain 'news' DataFrame

        Returns:
            DataFrame with added analytics features
        """
        if df.empty:
            logger.warning("Input DataFrame is empty. Skipping advanced analytics enrichment.")
            return df

        df_enriched = df.copy()

        # 1. Sentiment Statistics
        self._add_sentiment_statistics(df_enriched, kwargs.get('news'))

        # 2. Macro Composite Score
        self._add_macro_composite_score(df_enriched)

        # 3. Market Phase Detection
        self._add_market_phase_detection(df_enriched)

        # 4. MARKET_REGIME is now handled by TechnicalAnalysisEnricher (dual encoding)
        self._log_market_regime_info(df_enriched)

        logger.info("Advanced analytics enrichment completed")
        return df_enriched

    def _add_sentiment_statistics(self, df_enriched: pd.DataFrame, news_df: Optional[pd.DataFrame]) -> None:
        """Add sentiment statistics to DataFrame."""
        if news_df is None or not isinstance(news_df, pd.DataFrame) or news_df.empty:
            return
            
        if 'sentiment' not in news_df.columns and 'nlp_sentiment_score' not in df_enriched.columns:
            return

        try:
            # Use sentiment from df if available
            if 'nlp_sentiment_score' in df_enriched.columns:
                sentiment_col = 'nlp_sentiment_score'
                temp_df = df_enriched[[sentiment_col]].copy()
            else:
                sentiment_col = 'sentiment'
                temp_df = news_df[[sentiment_col]].copy()
            
            stats = self.sentiment_calculator.calculate_sentiment_stats(temp_df, sentiment_col)
            
            # Add as constant features (same value for all rows)
            df_enriched['sentiment_mean'] = stats['mean']
            df_enriched['sentiment_std_stat'] = stats['std']
            df_enriched['sentiment_pos_threshold'] = stats['positive_threshold']
            df_enriched['sentiment_neg_threshold'] = stats['negative_threshold']
            
            logger.info(f"Added sentiment statistics: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        except Exception as e:
            logger.error(f"Error calculating sentiment stats: {e}", exc_info=True)

    def _add_macro_composite_score(self, df_enriched: pd.DataFrame) -> None:
        """Add macro composite score to DataFrame."""
        if not self.macro_calculator:
            return

        # Check if we have FRED columns
        fred_cols = [col for col in df_enriched.columns if col.startswith('FRED_')]
        if len(fred_cols) < 3:  # Need at least 3 indicators
            return

        try:
            # Select only the indicators we configured
            available_indicators = [ind for ind in self.macro_calculator.indicators_config.keys() 
                                  if ind in df_enriched.columns]
            
            if not available_indicators:
                return

            macro_subset = df_enriched[available_indicators].copy()
            
            # Calculate composite score
            scores_df = self.macro_calculator.calculate_composite_score(
                macro_subset, 
                rolling_window=min(252, len(df_enriched) // 2)
            )
            
            if not scores_df.empty and 'composite_macro_score' in scores_df.columns:
                df_enriched['macro_composite_score'] = scores_df['composite_macro_score'].values
                logger.info(f"Added macro composite score (range: [{df_enriched['macro_composite_score'].min():.1f}, {df_enriched['macro_composite_score'].max():.1f}])")
        except Exception as e:
            logger.error(f"Error calculating macro composite score: {e}", exc_info=True)

    def _add_market_phase_detection(self, df_enriched: pd.DataFrame) -> None:
        """Add market phase detection to DataFrame."""
        if not self.phase_analyzer:
            return

        try:
            # Check if required indicators exist
            required_cols = list(self.phase_analyzer.indicators.values())
            if not all(col in df_enriched.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df_enriched.columns]
                logger.warning(f"Cannot detect market phase: missing columns {missing}")
                return

            # Analyze phase for each row (or just use last row for all)
            # For efficiency, we'll detect phase once and apply to all rows
            phase_result = self.phase_analyzer.analyze({'market_data': df_enriched})
            market_phase = phase_result.get('market_phase', 'unknown')
            
            # Map phase to numeric for ML models
            phase_map = self._get_phase_mapping()
            df_enriched['market_phase'] = phase_map.get(market_phase, 5)
            logger.info(f"Detected market phase: {market_phase} (encoded as {df_enriched['market_phase'].iloc[0]})")
            
        except Exception as e:
            logger.error(f"Error detecting market phase: {e}", exc_info=True)

    def _get_phase_mapping(self) -> Dict[str, int]:
        """Get market phase mapping to numeric values."""
        return {
            'calm_bull': 0,
            'calm_bear': 1,
            'volatile_bull': 2,
            'volatile_bear': 3,
            'neutral': 4,
            'unknown': 5,
            'error': 6
        }

    def _log_market_regime_info(self, df_enriched: pd.DataFrame) -> None:
        """Log information about MARKET_REGIME handling."""
        # MARKET_REGIME is now handled by TechnicalAnalysisEnricher (dual encoding)
        # We skip it here to avoid conflicts
        if 'MARKET_REGIME' not in df_enriched.columns:
            logger.info("MARKET_REGIME will be added by TechnicalAnalysisEnricher")
