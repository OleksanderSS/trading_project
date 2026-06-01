from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

from .base import BaseEnricher

logger = ProjectLogger.get_logger("VolatilityEnricher")


class VolatilityEnricher(BaseEnricher):
    """
    Volatility analysis enricher.
    Adds volatility-based indicators.
    """

    @property
    def name(self) -> str:
        return "volatility"

    @property
    def priority(self) -> int:
        return 30

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Enrich DataFrame with volatility indicators."""
        try:
            df_enriched = df.copy()

            if "close" in df_enriched.columns:
                # Returns
                df_enriched["returns"] = df_enriched["close"].pct_change(fill_method=None).fillna(0)

                # Historical Volatility
                df_enriched["volatility_5"] = df_enriched["returns"].rolling(window=5, min_periods=1).std().shift(1) * np.sqrt(252)
                df_enriched["volatility_10"] = df_enriched["returns"].rolling(window=10, min_periods=1).std().shift(1) * np.sqrt(252)
                df_enriched["volatility_20"] = df_enriched["returns"].rolling(window=20, min_periods=1).std().shift(1) * np.sqrt(252)

                # Average True Range (ATR)
                if all(col in df_enriched.columns for col in ["high", "low", "close"]):
                    tr1 = df_enriched["high"] - df_enriched["low"]
                    tr2 = abs(df_enriched["high"] - df_enriched["close"].shift())
                    tr3 = abs(df_enriched["low"] - df_enriched["close"].shift())
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    df_enriched["atr_14"] = true_range.rolling(window=14, min_periods=1).mean().shift(1)

                # Garman-Klass Volatility
                if "high" in df_enriched.columns and "low" in df_enriched.columns:
                    gk = (
                        0.5 * (np.log(df_enriched["high"]) - np.log(df_enriched["low"])) ** 2
                        - (2 * np.log(2) - 1)
                        * (np.log(df_enriched["close"]) - np.log(df_enriched["close"].shift())) ** 2
                    )
                    df_enriched["gk_volatility"] = gk.rolling(window=20, min_periods=1).sum().shift(1)

                # Volatility Regime
                df_enriched["volatility_regime"] = pd.cut(
                    df_enriched["volatility_10"],
                    bins=[0, 0.15, 0.25, 0.35, float("inf")],
                    labels=["low", "normal", "high", "extreme"],
                ).fillna("normal")

                logger.info(f"✅ Added {8} volatility indicators")

            return df_enriched

        except Exception as e:
            logger.error(f"❌ Error in volatility enrichment: {e}", exc_info=True)
            return df
