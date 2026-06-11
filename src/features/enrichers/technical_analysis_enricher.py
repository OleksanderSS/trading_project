import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.features.utils.modular_adaptive_technical_indicators import ModularAdaptiveTechnicalIndicators
from src.features.utils.technical_indicators_lib import TechnicalIndicators

from .base import BaseEnricher

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
            from src.algorithms.regime_detector import MarketRegimeDetector  # Unified regime detection
            from src.analytics.calculators.drawdown_calculator import DrawdownCalculator
            from src.analytics.calculators.econometrics_calculator import EconometricsCalculator
            from src.analytics.calculators.explainability_calculator import ExplainabilityCalculator
            from src.analytics.calculators.fama_french_factors import FamaFrenchFactors
            from src.analytics.calculators.macro_score_calculator import MacroScoreCalculator
            from src.analytics.calculators.risk_reward_calculator import RiskRewardCalculator
            from src.analytics.calculators.sentiment_stats_calculator import SentimentStatsCalculator
            from src.analytics.calculators.volatility_calculator import VolatilityCalculator

            self.VolatilityCalculator = VolatilityCalculator
            self.market_regime_detector = MarketRegimeDetector()  # Instantiate for instance methods
            self.FamaFrenchFactors = FamaFrenchFactors
            self.DrawdownCalculator = DrawdownCalculator
            self.EconometricsCalculator = EconometricsCalculator
            self.RiskRewardCalculator = RiskRewardCalculator
            self.MacroScoreCalculator = MacroScoreCalculator
            self.SentimentStatsCalculator = SentimentStatsCalculator
            self.ExplainabilityCalculator = ExplainabilityCalculator

            # Initialize adaptive indicators
            self.adaptive_indicators = ModularAdaptiveTechnicalIndicators()

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
                if indicator != 'market_regime': # Handled in _add_advanced_features
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

    def _is_indicator_enabled(self, indicator: str, settings: dict) -> bool:
        """Check if an indicator is enabled in the configuration."""
        if not isinstance(settings, dict):
            return False
        return settings.get('enabled', False)

    def _process_indicator(self, df: pd.DataFrame, indicator: str, settings: dict, indicator_map: dict):
        """Processes and adds a single technical indicator to the DataFrame."""
        try:
            method, price_cols, param_keys, suffix_keys = indicator_map[indicator]

            # Prepare parameters
            params = {k: settings[k] for k in param_keys if k in settings}

            # Special handling for indicators with multiple output columns (windows)
            if 'windows' in settings and 'window' in param_keys:
                for window in settings['windows']:
                    self._apply_indicator_method(df, method, price_cols, {'window': window}, suffix_keys, f"_{window}")
            else:
                self._apply_indicator_method(df, method, price_cols, params, suffix_keys)

        except Exception as e:
            logger.error(f"Error processing indicator '{indicator}': {e}")

    def _apply_indicator_method(self, df: pd.DataFrame, method, price_cols: list, params: dict, suffix_keys: list, custom_suffix: str = ""):
        """Helper to apply a specific indicator method and handle its output."""
        # Get required price series
        args = [df[col] for col in price_cols]

        # Call method
        result = method(*args, **params)

        # Handle single vs multiple outputs and clean NaNs
        if isinstance(result, tuple):
            for i, res in enumerate(result):
                col_name = f"{suffix_keys[i].lower()}{custom_suffix}"
                # ✅ ENHANCED: Fill NaNs immediately after calculation
                df[col_name] = res.fillna(0) if isinstance(res, (pd.Series, pd.DataFrame)) else res
        else:
            col_name = f"{suffix_keys[0].lower()}{custom_suffix}"
            # ✅ ENHANCED: Fill NaNs immediately after calculation
            df[col_name] = result.fillna(0) if isinstance(result, (pd.Series, pd.DataFrame)) else result

    def _get_indicator_mapping(self) -> dict[str, tuple]:
        """Get mapping from config keys to TechnicalIndicators methods and parameters."""
        return {
            'sma': (TechnicalIndicators.calculate_sma, ['close'], ['window'], ['SMA']),
            'ema': (TechnicalIndicators.calculate_ema, ['close'], ['window'], ['EMA']),
            'rsi': (TechnicalIndicators.calculate_rsi, ['close'], ['period'], ['RSI']),
            'macd': (TechnicalIndicators.calculate_macd, ['close'], ['fast', 'slow', 'signal'], ['MACD', 'MACD_Signal', 'MACD_Hist']),
            'bollinger_bands': (TechnicalIndicators.calculate_bollinger_bands, ['close'], ['period', 'std'], ['BB_Upper', 'BB_Middle', 'BB_Lower']),
            'atr': (TechnicalIndicators.calculate_atr, ['high', 'low', 'close'], ['period'], ['ATR']),
            'stochastic': (TechnicalIndicators.calculate_stochastic, ['high', 'low', 'close'], ['k_period', 'd_period'], ['Stoch_K', 'Stoch_D']),
            'williams_r': (TechnicalIndicators.calculate_williams_r, ['high', 'low', 'close'], ['period'], ['Williams_R']),
            'cci': (TechnicalIndicators.calculate_cci, ['high', 'low', 'close'], ['period'], ['CCI'])
        }

    def _add_advanced_features(self, df: pd.DataFrame):
        """Adds advanced features using lazy-loaded calculators."""
        if not self._is_indicator_enabled('market_regime', self.config.get('market_regime', {})):
            return

        self._load_calculators()

        try:
            # Add adaptive indicators from ModularAdaptiveTechnicalIndicators
            adaptive_results = self.adaptive_indicators.calculate_all_adaptive_indicators(df)
            for name, result in adaptive_results.items():
                if isinstance(result, tuple):
                    for i, res in enumerate(result):
                        df[f"{name}_{i}"] = res
                else:
                    df[name] = result

            # Add market regime if enabled
            self.config.get('market_regime', {})
            # window = regime_settings.get('window', 20) # Not used in detector.detect_regime

            # Calculate returns for regime detection
            returns = df['close'].pct_change().fillna(0).values
            data_bundle = {
                'prices': df['close'].values,
                'volume': df['volume'].values if 'volume' in df.columns else None
            }

            regime_result = self.market_regime_detector.detect_regime(returns, data_bundle=data_bundle)
            df['market_regime'] = regime_result.get('regime', 'NORMAL')

        except Exception as e:
            logger.error(f"Error adding advanced technical features: {e}")
