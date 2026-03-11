import pandas as pd
from typing import Dict, Any

from .base import BaseEnricher
from src.features.utils.technical_indicators_lib import TechnicalIndicators
from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import get_current_config

logger = ProjectLogger.get_logger("TechnicalAnalysisEnricher")

class TechnicalAnalysisEnricher(BaseEnricher):
    """
    Enriches a DataFrame with technical indicators specified in the configuration.
    This enricher dynamically calls calculation methods from the TechnicalIndicators library
    based on the settings in `src/config/features.yaml`.
    """
    
    def __init__(self):
        self.config = get_current_config().get_config('technical_analysis') or {}
        logger.info("TechnicalAnalysisEnricher initialized with dynamic configuration.")

    @property
    def name(self) -> str:
        return "technical_analysis"

    @property
    def priority(self) -> int:
        return 20

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Dynamically adds configured technical indicators to the DataFrame.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty. Skipping enrichment.")
            return df

        df_enriched = df.copy()
        required_cols = ['close', 'high', 'low', 'volume']
        if not all(col in df_enriched.columns for col in required_cols):
            logger.error(f"Missing one or more required columns {required_cols}. Aborting.")
            return df

        logger.info(f"Applying technical analysis to {len(df_enriched)} rows.")

        # Mapping from config keys to TechnicalIndicators methods and parameters
        indicator_map = {
            'sma': (TechnicalIndicators.calculate_sma, ['close'], ['window'], ['SMA']),
            'ema': (TechnicalIndicators.calculate_ema, ['close'], ['window'], ['EMA']),
            'rsi': (TechnicalIndicators.calculate_rsi, ['close'], ['period'], ['RSI']),
            'macd': (TechnicalIndicators.calculate_macd, ['close'], ['fast', 'slow', 'signal'], ['MACD', 'MACD_SIGNAL', 'MACD_HIST']),
            'bollinger_bands': (TechnicalIndicators.calculate_bollinger_bands, ['close'], ['period', 'std'], ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']),
            'atr': (TechnicalIndicators.calculate_atr, ['high', 'low', 'close'], ['period'], ['ATR']),
            'stochastic': (TechnicalIndicators.calculate_stochastic, ['high', 'low', 'close'], ['k_period', 'd_period'], ['STOCH_K', 'STOCH_D']),
            'williams_r': (TechnicalIndicators.calculate_williams_r, ['high', 'low', 'close'], ['period'], ['WILLIAMS_R']),
            'cci': (TechnicalIndicators.calculate_cci, ['high', 'low', 'close'], ['period'], ['CCI']),
        }

        for indicator, settings in self.config.items():
            if not settings.get('enabled', False):
                logger.debug(f"Skipping disabled indicator: {indicator}")
                continue

            if indicator not in indicator_map:
                logger.warning(f"Unknown indicator '{indicator}' in config. Skipping.")
                continue

            # Unpack the mapping
            method, input_cols, param_keys, output_cols = indicator_map[indicator]
            params = {key: settings.get(key) for key in param_keys}

            # Check if all parameters are provided
            if any(p is None for p in params.values()):
                logger.error(f"Missing parameters for {indicator}: required {param_keys}. Skipping.")
                continue

            try:
                logger.debug(f"Calculating {indicator} with params: {params}")
                # Prepare input data for the method
                input_data = [df_enriched[col] for col in input_cols]
                
                # Call the calculation method
                results = method(*input_data, **params)

                # Assign results to output columns
                if isinstance(results, tuple):
                    for i, col_name in enumerate(output_cols):
                        df_enriched[col_name] = results[i]
                else:
                    df_enriched[output_cols[0]] = results
                    
                logger.info(f"Successfully calculated {indicator}.")

            except Exception as e:
                logger.error(f"Error calculating {indicator}: {e}", exc_info=True)

        logger.info("Technical analysis enrichment complete.")
        return df_enriched