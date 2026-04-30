import pandas as pd
from typing import Dict, Any, List

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
        super().__init__()  # Initialize logger from BaseEnricher
        self.config = get_current_config().get_config('technical_analysis') or {}
        logger.info("TechnicalAnalysisEnricher initialized with dynamic configuration.")
        # Lazy import calculators
        self._calculators_loaded = False

    def _load_calculators(self):
        """Lazy load calculators only when needed."""
        if not self._calculators_loaded:
            from src.analytics.calculators.volatility_calculator import VolatilityCalculator
            from src.analytics.calculators.market_regime_calculator import MarketRegimeCalculator
            from src.analytics.calculators.fama_french_factors import FamaFrenchFactors
            from src.analytics.calculators.drawdown_calculator import DrawdownCalculator
            from src.analytics.calculators.econometrics_calculator import EconometricsCalculator
            from src.analytics.calculators.risk_reward_calculator import RiskRewardCalculator
            from src.analytics.calculators.macro_score_calculator import MacroScoreCalculator
            from src.analytics.calculators.sentiment_stats_calculator import SentimentStatsCalculator
            from src.analytics.calculators.explainability_calculator import ExplainabilityCalculator
            
            self.VolatilityCalculator = VolatilityCalculator
            self.MarketRegimeCalculator = MarketRegimeCalculator
            self.FamaFrenchFactors = FamaFrenchFactors
            self.DrawdownCalculator = DrawdownCalculator
            self.EconometricsCalculator = EconometricsCalculator
            self.RiskRewardCalculator = RiskRewardCalculator
            self.MacroScoreCalculator = MacroScoreCalculator
            self.SentimentStatsCalculator = SentimentStatsCalculator
            self.ExplainabilityCalculator = ExplainabilityCalculator
            self._calculators_loaded = True

    @property
    def name(self) -> str:
        return "technical_analysis"

    @property
    def priority(self) -> int:
        return 20

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Dynamically adds configured technical indicators to the DataFrame.
        """
        if not self._validate_input(df):
            return df

        df_enriched = df.copy()
        logger.info(f"Applying technical analysis to {len(df_enriched)} rows.")

        indicator_map = self._get_indicator_mapping()
        
        for indicator, settings in self.config.items():
            if not self._is_indicator_enabled(indicator, settings):
                continue
                
            if indicator not in indicator_map:
                logger.warning(f"Unknown indicator '{indicator}' in config. Skipping.")
                continue

            self._process_indicator(df_enriched, indicator, settings, indicator_map)

        self._add_advanced_features(df_enriched)
        logger.info("Technical analysis enrichment complete.")
        return df_enriched
    
    def _validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input DataFrame."""
        if df.empty:
            logger.warning("Input DataFrame is empty. Skipping enrichment.")
            return False

        required_cols = ['close', 'high', 'low', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing one or more required columns {required_cols}. Aborting.")
            return False
        
        return True
    
    def _get_indicator_mapping(self) -> Dict[str, tuple]:
        """Get mapping from config keys to TechnicalIndicators methods and parameters."""
        return {
            'sma': (TechnicalIndicators.calculate_sma, ['close'], ['window'], ['SMA']),
            'ema': (TechnicalIndicators.calculate_ema, ['close'], ['window'], ['EMA']),
            'rsi': (TechnicalIndicators.calculate_rsi, ['close'], ['period'], ['RSI_14']),  # ✅ Fixed: RSI → RSI_14
            'macd': (TechnicalIndicators.calculate_macd, ['close'], ['fast', 'slow', 'signal'], ['MACD', 'MACD_Signal', 'MACD_Histogram']),  # ✅ Fixed: MACD_HIST → MACD_Histogram, MACD_SIGNAL → MACD_Signal
            'bollinger_bands': (TechnicalIndicators.calculate_bollinger_bands, ['close'], ['period', 'std'], ['BB_Upper', 'BB_Middle', 'BB_Lower']),  # ✅ Fixed: BB_UPPER → BB_Upper, BB_MIDDLE → BB_Middle, BB_LOWER → BB_Lower
            'atr': (TechnicalIndicators.calculate_atr, ['high', 'low', 'close'], ['period'], ['ATR_14']),  # ✅ Fixed: ATR → ATR_14
            'stochastic': (TechnicalIndicators.calculate_stochastic, ['high', 'low', 'close'], ['k_period', 'd_period'], ['Stoch_K', 'Stoch_D']),  # ✅ Fixed: STOCH_K → Stoch_K, STOCH_D → Stoch_D
            'williams_r': (TechnicalIndicators.calculate_williams_r, ['high', 'low', 'close'], ['period'], ['Williams_R']),  # ✅ Fixed: WILLIAMS_R → Williams_R
            'cci': (TechnicalIndicators.calculate_cci, ['high', 'low', 'close'], ['period'], ['CCI']),
        }
    
    def _is_indicator_enabled(self, indicator: str, settings: Dict) -> bool:
        """Check if indicator is enabled in config."""
        if not settings.get('enabled', False):
            logger.debug(f"Skipping disabled indicator: {indicator}")
            return False
        return True
    
    def _process_indicator(self, df_enriched: pd.DataFrame, indicator: str, settings: Dict, indicator_map: Dict[str, tuple]):
        """Process a single indicator."""
        method, input_cols, param_keys, output_cols = indicator_map[indicator]
        
        # Handle multiple windows for SMA/EMA
        if indicator in ['sma', 'ema'] and 'windows' in settings:
            self._process_multiple_windows(df_enriched, indicator, settings, method, input_cols)
            return
        
        # Standard processing for other indicators
        self._process_standard_indicator(df_enriched, indicator, settings, method, input_cols, param_keys, output_cols)
    
    def _process_multiple_windows(self, df_enriched: pd.DataFrame, indicator: str, settings: Dict, method, input_cols: List[str]):
        """Process indicators with multiple windows (SMA/EMA)."""
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
    
    def _process_standard_indicator(self, df_enriched: pd.DataFrame, indicator: str, settings: Dict, method, input_cols: List[str], param_keys: List[str], output_cols: List[str]):
        """Process standard indicators with single parameter set."""
        params = {key: settings.get(key) for key in param_keys}

        # Check if all parameters are provided
        if any(p is None for p in params.values()):
            logger.error(f"Missing parameters for {indicator}: required {param_keys}. Skipping.")
            return

        try:
            logger.debug(f"Calculating {indicator} with params: {params}")
            input_data = [df_enriched[col] for col in input_cols]
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
    
    def _add_advanced_features(self, df_enriched: pd.DataFrame):
        """Add advanced calculator features to the DataFrame."""
        logger.info("Adding advanced calculator features...")
        try:
            # ✅ FIX: Load calculators before using them
            self._load_calculators()
            
            self._add_volatility_features(df_enriched)
            self._add_market_regime_features(df_enriched)
            self._add_drawdown_features(df_enriched)
            self._add_risk_reward_features(df_enriched)
            self._add_econometrics_features(df_enriched)
            self._add_fama_french_features(df_enriched)
        except Exception as e:
            logger.error(f"Error adding advanced calculator features: {e}", exc_info=True)
    
    def _add_volatility_features(self, df_enriched: pd.DataFrame):
        """Add volatility features."""
        if 'close' in df_enriched.columns:
            returns = df_enriched['close'].pct_change()
            df_enriched['VOLATILITY_20'] = self.VolatilityCalculator.calculate_rolling_volatility(returns, 20)
            df_enriched['VOLATILITY_50'] = self.VolatilityCalculator.calculate_rolling_volatility(returns, 50)
            logger.info("Added volatility features")
    
    def _add_market_regime_features(self, df_enriched: pd.DataFrame):
        """Add market regime features (dual encoding: text + numeric)."""
        if 'close' in df_enriched.columns:
            # Text labels for humans
            df_enriched['MARKET_REGIME'] = self.MarketRegimeCalculator.calculate_regime(df_enriched['close'], 20, return_encoded=False)
            # Numeric encoding for models
            df_enriched['MARKET_REGIME_ENCODED'] = self.MarketRegimeCalculator.calculate_regime(df_enriched['close'], 20, return_encoded=True)
            logger.info("Added market regime features (text + numeric encoding)")
    
    def _add_drawdown_features(self, df_enriched: pd.DataFrame):
        """Add drawdown features."""
        if 'close' in df_enriched.columns and 'high' in df_enriched.columns:
            try:
                returns = df_enriched['close'].pct_change()
                df_enriched['MAX_DRAWDOWN'] = self.DrawdownCalculator.calculate_max_drawdown_from_returns(returns)
                df_enriched['CURRENT_DRAWDOWN'] = self.DrawdownCalculator.calculate_max_drawdown_from_prices(df_enriched)
                logger.info("Added drawdown features")
            except Exception as e:
                logger.warning(f"Could not add drawdown features: {e}")
    
    def _add_risk_reward_features(self, df_enriched: pd.DataFrame):
        """Add risk-reward features."""
        if 'close' in df_enriched.columns:
            try:
                rr_calc = self.RiskRewardCalculator()
                returns = df_enriched['close'].pct_change()
                df_enriched['SHARPE_RATIO'] = rr_calc.calculate_sharpe_ratio(returns)
                df_enriched['SORTINO_RATIO'] = rr_calc.calculate_sortino_ratio(returns)
                logger.info("Added risk-reward features")
            except Exception as e:
                logger.warning(f"Could not add risk-reward features: {e}")
    
    def _add_econometrics_features(self, df_enriched: pd.DataFrame):
        """Add econometrics features."""
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
    
    def _add_fama_french_features(self, df_enriched: pd.DataFrame):
        """Add Fama-French factors."""
        try:
            if 'close' in df_enriched.columns:
                # Simplified Fama-French calculation for single ticker
                market_return = df_enriched['close'].pct_change()
                df_enriched['MARKET_PREMIUM'] = market_return - market_return.rolling(252).mean()
                logger.info("Added Fama-French factors")
        except Exception as e:
            logger.warning(f"Could not add Fama-French factors: {e}")
    
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