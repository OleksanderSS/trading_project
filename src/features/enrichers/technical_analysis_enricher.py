import pandas as pd
from typing import Dict, Any

from .base import BaseEnricher
from src.features.utils.technical_indicators_lib import TechnicalIndicators
from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import get_current_config

# Advanced calculators for enhanced features
from src.analytics.calculators.volatility_calculator import VolatilityCalculator
from src.analytics.calculators.market_regime_calculator import MarketRegimeCalculator
from src.analytics.calculators.fama_french_factors import FamaFrenchFactors
from src.analytics.calculators.drawdown_calculator import DrawdownCalculator
from src.analytics.calculators.econometrics_calculator import EconometricsCalculator
from src.analytics.calculators.risk_reward_calculator import RiskRewardCalculator
from src.analytics.calculators.macro_score_calculator import MacroScoreCalculator
from src.analytics.calculators.sentiment_stats_calculator import SentimentStatsCalculator
from src.analytics.calculators.explainability_calculator import ExplainabilityCalculator

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
            
            # ✅ ПІДТРИМКА МНОЖИННИХ ВІКОН для SMA/EMA
            if indicator in ['sma', 'ema'] and 'windows' in settings:
                windows = settings['windows']
                if not isinstance(windows, list):
                    windows = [windows]
                
                for window in windows:
                    try:
                        logger.debug(f"Calculating {indicator.upper()}_{window}")
                        input_data = [df_enriched[col] for col in input_cols]
                        result = method(*input_data, window=window)
                        df_enriched[f'{indicator.upper()}_{window}'] = result
                        logger.info(f"Successfully calculated {indicator.upper()}_{window}.")
                    except Exception as e:
                        logger.error(f"Error calculating {indicator.upper()}_{window}: {e}", exc_info=True)
                continue
            
            # Стандартна обробка для інших індикаторів
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

        # Add advanced calculator features (ALWAYS included regardless of config)
        logger.info("Adding advanced calculator features...")
        try:
            # Volatility features
            if 'close' in df_enriched.columns:
                returns = df_enriched['close'].pct_change()
                df_enriched['VOLATILITY_20'] = VolatilityCalculator.calculate_rolling_volatility(returns, 20)
                df_enriched['VOLATILITY_50'] = VolatilityCalculator.calculate_rolling_volatility(returns, 50)
                logger.info("Added volatility features")
            
            # Market regime features (dual encoding: text + numeric)
            if 'close' in df_enriched.columns:
                # Text labels for humans
                df_enriched['MARKET_REGIME'] = MarketRegimeCalculator.calculate_regime(df_enriched['close'], 20, return_encoded=False)
                # Numeric encoding for models
                df_enriched['MARKET_REGIME_ENCODED'] = MarketRegimeCalculator.calculate_regime(df_enriched['close'], 20, return_encoded=True)
                logger.info("Added market regime features (text + numeric encoding)")
            
            # Drawdown features
            if 'close' in df_enriched.columns and 'high' in df_enriched.columns:
                try:
                    returns = df_enriched['close'].pct_change()
                    df_enriched['MAX_DRAWDOWN'] = DrawdownCalculator.calculate_max_drawdown_from_returns(returns)
                    df_enriched['CURRENT_DRAWDOWN'] = DrawdownCalculator.calculate_max_drawdown_from_prices(df_enriched)
                    logger.info("Added drawdown features")
                except Exception as e:
                    logger.warning(f"Could not add drawdown features: {e}")
            
            # Risk-Reward features
            if 'close' in df_enriched.columns:
                try:
                    rr_calc = RiskRewardCalculator()
                    returns = df_enriched['close'].pct_change()
                    df_enriched['SHARPE_RATIO'] = rr_calc.calculate_sharpe_ratio(returns)
                    df_enriched['SORTINO_RATIO'] = rr_calc.calculate_sortino_ratio(returns)
                    logger.info("Added risk-reward features")
                except Exception as e:
                    logger.warning(f"Could not add risk-reward features: {e}")
            
            # Econometrics features
            if 'close' in df_enriched.columns:
                try:
                    returns = df_enriched['close'].pct_change()
                    # Add autocorrelation
                    df_enriched['AUTOCORR'] = returns.autocorr(lag=1)
                    # Add Hurst exponent approximation
                    df_enriched['HURST_EXPONENT'] = self._calculate_hurst_exponent(returns)
                    # Add additional econometric features
                    df_enriched['SKEWNESS'] = returns.skew()
                    df_enriched['KURTOSIS'] = returns.kurtosis()
                    logger.info("Added econometrics features")
                except Exception as e:
                    logger.warning(f"Could not add econometrics features: {e}")
            
            # Fama-French factors (if available)
            try:
                ff_factors = FamaFrenchFactors()
                if 'close' in df_enriched.columns:
                    # Simplified Fama-French calculation for single ticker
                    market_return = df_enriched['close'].pct_change()
                    df_enriched['MARKET_PREMIUM'] = market_return - market_return.rolling(252).mean()
                    logger.info("Added Fama-French factors")
            except Exception as e:
                logger.warning(f"Could not add Fama-French factors: {e}")
                
        except Exception as e:
            logger.error(f"Error adding advanced calculator features: {e}", exc_info=True)

        logger.info("Technical analysis enrichment complete.")
        return df_enriched
    
    def _calculate_hurst_exponent(self, ts):
        """Calculate the Hurst exponent of a time series."""
        try:
            import numpy as np
            
            # Create the range of lag values
            lags = range(2, min(100, len(ts)//2))
            
            # Calculate the array of the variances of the lagged differences
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            
            # Use a linear fit to estimate the Hurst Exponent
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            
            # Return the Hurst exponent from the polyfit output
            hurst = poly[0] * 2.0
            
            return hurst
        except Exception as e:
            logger.warning(f"Could not calculate Hurst exponent: {e}")
            return 0.5  # Return neutral value