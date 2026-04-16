"""
Market Context Enricher - додає макроекономічні та ринкові індикатори до датасету.

Інтегрує MarketContextAnalyzer як enricher для додавання context features
безпосередньо в features DataFrame.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base import BaseEnricher
from src.core.logging.logger import ProjectLogger
from src.analytics.context.market_context_analyzer import MarketContextAnalyzer

logger = ProjectLogger.get_logger("MarketContextEnricher")

class MarketContextEnricher(BaseEnricher):
    """
    Додає макроекономічні та ринкові індикатори до датасету.
    
    Використовує MarketContextAnalyzer для розрахунку 18 контекстних показників:
    - Volatility metrics (5d, 20d, ratio)
    - Trend metrics (5d, 20d, alignment)
    - Technical indicators (RSI, volume ratio, price to MA20)
    - Temporal features (hour, day of week)
    - Macro indicators (yield curve, Fed Funds, market breadth, dollar strength, put/call ratio)
    """
    
    @property
    def name(self) -> str:
        return "market_context"
    
    @property
    def priority(self) -> int:
        return 85  # Після context_map (80), перед фінальними enrichers
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Список context features для розрахунку
        self.context_features = self.config.get('context_features', [
            # Volatility
            "volatility_5d", "volatility_20d", "volatility_ratio",
            # Trend
            "trend_5d", "trend_20d", "trend_alignment",
            # Technical
            "rsi_current", "volume_ratio", "price_to_ma20",
            # Temporal
            "hour_of_day", "day_of_week",
            # Macro (нові)
            "yield_curve_slope", "yield_curve_inverted",
            "fed_funds_trend", "fed_funds_velocity",
            "market_breadth", "dollar_strength", "put_call_ratio"
        ])
        
        # Ініціалізуємо analyzer
        self.analyzer = MarketContextAnalyzer(context_features=self.context_features)
        
        logger.info(f"MarketContextEnricher initialized with {len(self.context_features)} features")
    
    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Додає market context features до DataFrame.
        
        Args:
            df: DataFrame з OHLCV та іншими features
            **kwargs: Додаткові параметри (VIX, DGS10, DXY, etc.)
        
        Returns:
            DataFrame з доданими market_context_* колонками
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, skipping enrichment")
            return df
        
        result_df = df.copy()
        
        # Розраховуємо context vector для кожного рядка
        # (в реальності можна оптимізувати - розраховувати тільки для останнього рядка
        # і forward-fill для історичних даних)
        
        try:
            # Викликаємо analyzer для всього DataFrame
            analysis_result = self.analyzer.analyze(df, **kwargs)
            context_vector = analysis_result.get('market_context_vector')
            
            if context_vector is not None:
                # Додаємо кожен показник як окрему колонку з префіксом
                for feature_name, feature_value in context_vector.items():
                    col_name = f"market_context_{feature_name}"
                    
                    # Для часових показників (hour, day_of_week) - беремо з datetime
                    if feature_name in ['hour_of_day', 'day_of_week']:
                        if isinstance(df.index, pd.DatetimeIndex):
                            if feature_name == 'hour_of_day':
                                result_df[col_name] = df.index.hour
                            elif feature_name == 'day_of_week':
                                result_df[col_name] = df.index.weekday
                        else:
                            result_df[col_name] = feature_value
                    else:
                        # Для інших показників - forward-fill (вони змінюються рідко)
                        result_df[col_name] = feature_value
                
                logger.info(f"✅ Added {len(context_vector)} market context features")
                
                # Логуємо останні значення для перевірки
                if len(result_df) > 0:
                    last_values = {k: v for k, v in context_vector.items()}
                    logger.debug(f"Latest market context: {last_values}")
            else:
                logger.warning("⚠️ MarketContextAnalyzer returned None")
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate market context: {e}", exc_info=True)
        
        return result_df
