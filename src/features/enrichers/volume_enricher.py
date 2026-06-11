from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

from .base import BaseEnricher

logger = ProjectLogger.get_logger("VolumeEnricher")

class VolumeEnricher(BaseEnricher):
    """
    Volume analysis enricher.
    Adds volume-based indicators.
    """

    @property
    def name(self) -> str:
        return "volume"

    @property
    def priority(self) -> int:
        return 25

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Enrich DataFrame with volume indicators."""
        try:
            df_enriched = df.copy()

            if 'volume' in df_enriched.columns and 'close' in df_enriched.columns:
                # Volume Moving Averages
                df_enriched['volume_sma_5'] = df_enriched['volume'].rolling(window=5).mean()
                df_enriched['volume_sma_10'] = df_enriched['volume'].rolling(window=10).mean()

                # Volume Rate of Change
                df_enriched['volume_roc'] = df_enriched['volume'].pct_change(periods=5)

                # Price-Volume indicators
                df_enriched['price_volume_trend'] = df_enriched['volume'] * df_enriched['close'].pct_change()

                # On-Balance Volume
                price_change = df_enriched['close'].diff()
                obv = np.where(price_change > 0, df_enriched['volume'],
                             np.where(price_change < 0, -df_enriched['volume'], 0))
                df_enriched['obv'] = pd.Series(obv).cumsum()

                # Volume Relative Strength
                df_enriched['volume_rs'] = df_enriched['volume'] / df_enriched['volume_sma_10']

                logger.info(f"✅ Added {6} volume indicators")

            return df_enriched

        except Exception as e:
            logger.error(f"❌ Error in volume enrichment: {e}", exc_info=True)
            return df
