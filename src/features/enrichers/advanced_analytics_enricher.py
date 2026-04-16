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
        news_df = kwargs.get('news')
        if news_df is not None and isinstance(news_df, pd.DataFrame) and not news_df.empty:
            if 'sentiment' in news_df.columns or 'nlp_sentiment_score' in df_enriched.columns:
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
                    
                    logger.info(f"✅ Added sentiment statistics: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
                except Exception as e:
                    logger.error(f"Error calculating sentiment stats: {e}", exc_info=True)

        # 2. Macro Composite Score
        if self.macro_calculator:
            # Check if we have FRED columns
            fred_cols = [col for col in df_enriched.columns if col.startswith('FRED_')]
            if len(fred_cols) >= 3:  # Need at least 3 indicators
                try:
                    # Select only the indicators we configured
                    available_indicators = [ind for ind in self.macro_calculator.indicators_config.keys() 
                                          if ind in df_enriched.columns]
                    
                    if available_indicators:
                        macro_subset = df_enriched[available_indicators].copy()
                        
                        # Calculate composite score
                        scores_df = self.macro_calculator.calculate_composite_score(
                            macro_subset, 
                            rolling_window=min(252, len(df_enriched) // 2)
                        )
                        
                        if not scores_df.empty and 'composite_macro_score' in scores_df.columns:
                            df_enriched['macro_composite_score'] = scores_df['composite_macro_score'].values
                            logger.info(f"✅ Added macro composite score (range: [{df_enriched['macro_composite_score'].min():.1f}, {df_enriched['macro_composite_score'].max():.1f}])")
                except Exception as e:
                    logger.error(f"Error calculating macro composite score: {e}", exc_info=True)

        # 3. Market Phase Detection
        if self.phase_analyzer:
            try:
                # Check if required indicators exist
                required_cols = list(self.phase_analyzer.indicators.values())
                if all(col in df_enriched.columns for col in required_cols):
                    # Analyze phase for each row (or just use last row for all)
                    # For efficiency, we'll detect phase once and apply to all rows
                    phase_result = self.phase_analyzer.analyze({'market_data': df_enriched})
                    market_phase = phase_result.get('market_phase', 'unknown')
                    
                    # Map phase to numeric for ML models
                    phase_map = {
                        'calm_bull': 0,
                        'calm_bear': 1,
                        'volatile_bull': 2,
                        'volatile_bear': 3,
                        'neutral': 4,
                        'unknown': 5,
                        'error': 6
                    }
                    
                    df_enriched['market_phase'] = phase_map.get(market_phase, 5)
                    logger.info(f"✅ Detected market phase: {market_phase} (encoded as {df_enriched['market_phase'].iloc[0]})")
                else:
                    missing = [col for col in required_cols if col not in df_enriched.columns]
                    logger.warning(f"Cannot detect market phase: missing columns {missing}")
            except Exception as e:
                logger.error(f"Error detecting market phase: {e}", exc_info=True)
        
        # 4. MARKET_REGIME is now handled by TechnicalAnalysisEnricher (dual encoding)
        # We skip it here to avoid conflicts
        if 'MARKET_REGIME' not in df_enriched.columns:
            logger.info("MARKET_REGIME will be added by TechnicalAnalysisEnricher")


        logger.info("Advanced analytics enrichment completed")
        return df_enriched
