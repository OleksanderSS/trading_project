import logging

import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.features.utils.technical_indicators_lib import TechnicalIndicators

from .base import BaseEnricher

logger = ProjectLogger.get_logger('TechnicalAnalysisEnricher')


class TechnicalAnalysisEnricher(BaseEnricher):
    """
    Enriches a DataFrame with technical indicators specified in the configuration.
    This enricher dynamically calls calculation methods from the TechnicalIndicators library
    based on the settings in `src/config/features.yaml`.
    """

    def __init__(self):
        super().__init__()
        self.config = get_current_config().get_config('technical_analysis'
            ) or {}
        logger.info(
            'TechnicalAnalysisEnricher initialized with dynamic configuration.'
            )
        self._calculators_loaded = False

    def _load_calculators(self):
        """Lazy load calculators only when needed."""
        if not self._calculators_loaded:
            from src.algorithms.regime_detector import MarketRegimeDetector
            from src.analytics.calculators.drawdown_calculator import DrawdownCalculator
            from src.analytics.calculators.econometrics_calculator import EconometricsCalculator
            from src.analytics.calculators.explainability_calculator import ExplainabilityCalculator
            from src.analytics.calculators.fama_french_factors import FamaFrenchFactors
            from src.analytics.calculators.macro_score_calculator import MacroScoreCalculator
            from src.analytics.calculators.risk_reward_calculator import RiskRewardCalculator
            from src.analytics.calculators.sentiment_stats_calculator import SentimentStatsCalculator
            from src.analytics.calculators.volatility_calculator import VolatilityCalculator
            self.VolatilityCalculator = VolatilityCalculator()
            self.MarketRegimeCalculator = MarketRegimeDetector()
            self.FamaFrenchFactors = FamaFrenchFactors()
            self.DrawdownCalculator = DrawdownCalculator()
            self.EconometricsCalculator = EconometricsCalculator()
            self.RiskRewardCalculator = RiskRewardCalculator()
            self.MacroScoreCalculator = MacroScoreCalculator(indicators_config={
                'FRED_FEDFUNDS': {'weight': 0.3, 'direction': 'negative'},
                'FRED_UNRATE':   {'weight': 0.2, 'direction': 'negative'},
                'FRED_VIXCLS':   {'weight': 0.2, 'direction': 'negative'},
                'FRED_DGS10':    {'weight': 0.15, 'direction': 'positive'},
                'FRED_CPIAUCSL': {'weight': 0.15, 'direction': 'negative'},
            })
            self.SentimentStatsCalculator = SentimentStatsCalculator()
            self.ExplainabilityCalculator = ExplainabilityCalculator()
            self._calculators_loaded = True

    @property
    def name(self) ->str:
        return 'technical_analysis'

    @property
    def priority(self) ->int:
        return 20

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) ->pd.DataFrame:
        """
        Dynamically adds configured technical indicators to the DataFrame.
        """
        if not self._validate_input(df):
            return df
        if 'ticker' in df.columns and df['ticker'].nunique() > 1:
            enriched_groups = [self._enrich_single_group(group) for _,
                group in df.groupby('ticker', sort=False)]
            return pd.concat(enriched_groups).sort_index()
        return self._enrich_single_group(df)

    def _enrich_single_group(self, df: pd.DataFrame) ->pd.DataFrame:
        """Apply technical indicators within one ticker/time-series group."""
        df_enriched = df.copy()
        logger.info(f'Applying technical analysis to {len(df_enriched)} rows.')
        indicator_map = self._get_indicator_mapping()
        for indicator, settings in self.config.items():
            if not self._is_indicator_enabled(indicator, settings):
                continue
            if indicator not in indicator_map:
                if indicator != 'market_regime':
                    logger.warning(
                        f"Unknown indicator '{indicator}' in config. Skipping."
                        )
                continue
            self._process_indicator(df_enriched, indicator, settings,
                indicator_map)
        # Market regime is now handled as an advanced feature in _add_advanced_features
        # which is already called below.

        # Calculate returns once - canonical source
        returns = (
            df_enriched['close']
            .pct_change(fill_method=None)
            .replace([float('inf'), float('inf')], float('nan'))
        )
        self._add_advanced_features(df_enriched, returns)
        logger.info('Technical analysis enrichment complete.')
        return df_enriched

    def _validate_input(self, df: pd.DataFrame) ->bool:
        """Validate input DataFrame."""
        if df.empty:
            logger.warning('Input DataFrame is empty. Skipping enrichment.')
            return False
        required_cols = ['close', 'high', 'low', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(
                f'Missing one or more required columns {required_cols}. Aborting.'
                )
            return False
        return True

    def _get_indicator_mapping(self) ->dict[str, tuple]:
        """Get mapping from config keys to TechnicalIndicators methods and parameters."""
        return {'sma': (TechnicalIndicators.calculate_sma, ['close'], [
            'window'], ['SMA']), 'ema': (TechnicalIndicators.calculate_ema,
            ['close'], ['window'], ['EMA']), 'rsi': (TechnicalIndicators.
            calculate_rsi, ['close'], ['period'], ['RSI_14']), 'macd': (
            TechnicalIndicators.calculate_macd, ['close'], ['fast', 'slow',
            'signal'], ['MACD', 'MACD_Signal', 'MACD_Histogram']),
            'bollinger_bands': (TechnicalIndicators.
            calculate_bollinger_bands, ['close'], ['period', 'std'], [
            'BB_Upper', 'BB_Middle', 'BB_Lower']), 'atr': (
            TechnicalIndicators.calculate_atr, ['high', 'low', 'close'], [
            'period'], ['ATR_14']), 'stochastic': (TechnicalIndicators.
            calculate_stochastic, ['high', 'low', 'close'], ['k_period',
            'd_period'], ['Stoch_K', 'Stoch_D']), 'williams_r': (
            TechnicalIndicators.calculate_williams_r, ['high', 'low',
            'close'], ['period'], ['Williams_R']), 'cci': (
            TechnicalIndicators.calculate_cci, ['high', 'low', 'close'], [
            'period'], ['CCI'])}

    def _is_indicator_enabled(self, indicator: str, settings: dict) ->bool:
        """Check if indicator is enabled in config."""
        if not settings.get('enabled', False):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Skipping disabled indicator: {indicator}')
            return False
        return True

    def _process_indicator(self, df_enriched: pd.DataFrame, indicator: str,
        settings: dict, indicator_map: dict[str, tuple]):
        """Process a single indicator."""
        method, input_cols, param_keys, output_cols = indicator_map[indicator]
        if indicator in ['sma', 'ema'] and 'windows' in settings:
            self._process_multiple_windows(df_enriched, indicator, settings,
                method, input_cols)
            return
        self._process_standard_indicator(df_enriched, indicator, settings,
            method, input_cols, param_keys, output_cols)

    def _process_multiple_windows(self, df_enriched: pd.DataFrame,
        indicator: str, settings: dict, method, input_cols: list[str]):
        """Process indicators with multiple windows (SMA/EMA)."""
        windows = settings['windows']
        if not isinstance(windows, list):
            windows = [windows]
        for window in windows:
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'Calculating {indicator.upper()}_{window}')
                input_data = [df_enriched[col] for col in input_cols]
                result = method(*input_data, window=window)
                df_enriched[f'{indicator.upper()}_{window}'] = result
                logger.info(
                    f'Successfully calculated {indicator.upper()}_{window}.')
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.error(
                    f'Error calculating {indicator.upper()}_{window}: {e}',
                    exc_info=True)

    def _process_standard_indicator(self, df_enriched: pd.DataFrame,
        indicator: str, settings: dict, method, input_cols: list[str],
        param_keys: list[str], output_cols: list[str]):
        """Process standard indicators with single parameter set."""
        params = {key: settings.get(key) for key in param_keys}
        if any(p is None for p in params.values()):
            logger.error(
                f'Missing parameters for {indicator}: required {param_keys}. Skipping.'
                )
            return
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Calculating {indicator} with params: {params}')
            input_data = [df_enriched[col] for col in input_cols]
            results = method(*input_data, **params)
            if isinstance(results, tuple):
                for i, col_name in enumerate(output_cols):
                    df_enriched[col_name] = results[i]
            else:
                df_enriched[output_cols[0]] = results
            logger.info(f'Successfully calculated {indicator}.')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Error calculating {indicator}: {e}', exc_info=True)

    def _add_advanced_features(self, df_enriched: pd.DataFrame, returns: pd.Series):
        """Add advanced calculator features to the DataFrame."""
        logger.info('Adding advanced calculator features...')
        try:
            self._load_calculators()
            # No need to recalculate returns here - it's passed in

            # --- NEW: Short-term Volatility (5d) ---
            df_enriched['VOLATILITY_5'] = returns.rolling(5, min_periods=2).std()

            # --- NEW: Momentum Z-Score (20d) ---
            # Normalizes returns to see how "extreme" the current move is
            mean_ret = returns.rolling(20, min_periods=1).mean()
            std_ret = returns.rolling(20, min_periods=1).std()
            df_enriched['MOMENTUM_ZSCORE'] = (returns - mean_ret) / (std_ret + 1e-9)

            # --- NEW: RSI Velocity ---
            if 'RSI_14' in df_enriched.columns:
                df_enriched['RSI_VELOCITY'] = df_enriched['RSI_14'].diff(3)

            try:
                self._add_volatility_features(df_enriched, returns)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.exception(f'Error adding volatility features: {e}')
            try:
                self._add_market_regime_features(df_enriched, returns)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.exception(f'Error adding market regime features: {e}')
            try:
                self._add_drawdown_features(df_enriched, returns)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.exception(f'Error adding drawdown features: {e}')
            try:
                self._add_risk_reward_features(df_enriched, returns)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.exception(f'Error adding risk-reward features: {e}')
            try:
                self._add_econometrics_features(df_enriched, returns)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.exception(f'Error adding econometrics features: {e}')
            try:
                self._add_fama_french_features(df_enriched, returns)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.exception(f'Error adding Fama-French features: {e}')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f'Error adding advanced calculator features: {e}')

    def _add_volatility_features(self, df_enriched: pd.DataFrame, returns: pd.Series = None):
        """Add volatility features."""
        if 'close' in df_enriched.columns:
            if returns is None:
                returns = (
                    df_enriched['close']
                    .pct_change(fill_method=None)
                    .replace([float('inf'), float('-inf')], float('nan'))
                )
            df_enriched['VOLATILITY_20'
                ] = self.VolatilityCalculator.calculate_rolling_volatility(
                returns, 20)
            df_enriched['VOLATILITY_50'
                ] = self.VolatilityCalculator.calculate_rolling_volatility(
                returns, 50)
            logger.info('Added volatility features')

    def _add_market_regime_features(self, df_enriched: pd.DataFrame, returns: pd.Series = None):
        """Add market regime features (dual encoding: text + numeric)."""
        if 'close' in df_enriched.columns:
            if returns is None:
                returns = df_enriched['close'].pct_change(fill_method=None)
            valid_returns = returns.replace([float('inf'), float('-inf')], float('nan')).dropna()
            if valid_returns.empty:
                df_enriched['MARKET_REGIME'] = 'UNKNOWN'
                df_enriched['MARKET_REGIME_ENCODED'] = float('nan')
                return
            regime_result = self.MarketRegimeCalculator.detect_regime(
                valid_returns.values if hasattr(valid_returns, 'values') else valid_returns
            )
            df_enriched['MARKET_REGIME'] = regime_result.get('regime',
                'UNKNOWN')
            df_enriched['MARKET_REGIME_ENCODED'] = regime_result.get(
                'confidence', 0.0)
            logger.info(
                'Added market regime features (text + numeric encoding)')

    def _add_drawdown_features(self, df_enriched: pd.DataFrame, returns: pd.Series = None):
        """Add drawdown features."""
        if 'close' in df_enriched.columns and 'high' in df_enriched.columns:
            try:
                if returns is None:
                    returns = (
                        df_enriched['close']
                        .pct_change(fill_method=None)
                        .replace([float('inf'), float('-inf')], float('nan'))
                    )
                df_enriched['MAX_DRAWDOWN'] = (self.DrawdownCalculator.
                    calculate_max_drawdown_from_returns(returns))
                df_enriched['CURRENT_DRAWDOWN'
                    ] = self.DrawdownCalculator.calculate_max_drawdown_from_prices(
                    df_enriched)
                logger.info('Added drawdown features')
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                logger.warning(f'Could not add drawdown features: {e}')
                raise

    def _add_risk_reward_features(self, df_enriched: pd.DataFrame, returns: pd.Series = None):
        """Add risk-reward features on a rolling basis to avoid look-ahead bias."""
        if 'close' in df_enriched.columns:
            try:
                import numpy as np
                if returns is None:
                    returns = df_enriched['close'].pct_change(fill_method=None)

                window = 252
                min_periods = 30

                rolling_mean = returns.rolling(window=window, min_periods=min_periods).mean()
                rolling_std = returns.rolling(window=window, min_periods=min_periods).std()

                # Sharpe Ratio
                # Guard against zero/near-zero std to prevent inf/nan
                sharpe_denominator = rolling_std.copy()
                sharpe_denominator[sharpe_denominator < 1e-10] = np.nan
                sharpe = (rolling_mean / sharpe_denominator).replace([float('inf'), float('-inf')], float('nan')) * np.sqrt(252)
                df_enriched['SHARPE_RATIO'] = sharpe.fillna(np.nan)

                # Sortino Ratio
                downside_returns = returns.copy()
                downside_returns[downside_returns > 0] = 0.0
                rolling_downside_var = downside_returns.pow(2).rolling(window=window, min_periods=min_periods).mean()
                rolling_downside_std = np.sqrt(rolling_downside_var)
                
                # Guard against zero/near-zero std to prevent inf/nan
                sortino_denominator = rolling_downside_std.copy()
                sortino_denominator[sortino_denominator < 1e-10] = np.nan
                sortino = (rolling_mean / sortino_denominator).replace([float('inf'), float('-inf')], float('nan')) * np.sqrt(252)
                df_enriched['SORTINO_RATIO'] = sortino.fillna(np.nan)

                logger.info('Added rolling risk-reward features')
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.exception(f'Виникла помилка: {e}')
                logger.warning(f'Could not add risk-reward features: {e}')
                raise

    def _add_econometrics_features(self, df_enriched: pd.DataFrame, returns: pd.Series = None):
        """Add econometrics features on a rolling basis to avoid look-ahead bias."""
        if 'close' in df_enriched.columns:
            try:
                if returns is None:
                    returns = df_enriched['close'].pct_change(fill_method=None)

                window = 252
                min_periods = 30

                # Rolling autocorrelation (correlation of returns with returns.shift(1))
                df_enriched['AUTOCORR'] = returns.rolling(window=window, min_periods=min_periods).corr(returns.shift(1)).fillna(np.nan)

                # Rolling Hurst Exponent
                df_enriched['HURST_EXPONENT'] = returns.rolling(window=window, min_periods=100).apply(
                    self._calculate_hurst_exponent, raw=True
                ).fillna(0.5)

                # Rolling Skewness
                df_enriched['SKEWNESS'] = returns.rolling(window=window, min_periods=min_periods).skew().fillna(np.nan)

                # Rolling Kurtosis
                df_enriched['KURTOSIS'] = returns.rolling(window=window, min_periods=min_periods).kurt().fillna(np.nan)

                logger.info('Added rolling econometrics features')
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                logger.warning(f'Could not add econometrics features: {e}')
                raise

    def _add_fama_french_features(self, df_enriched: pd.DataFrame, returns: pd.Series = None):
        """Add Fama-French factors."""
        try:
            if 'close' in df_enriched.columns:
                if returns is None:
                    market_return = df_enriched['close'].pct_change(fill_method=None)
                else:
                    market_return = returns
                df_enriched['MARKET_PREMIUM'
                    ] = market_return - market_return.rolling(252, min_periods=1).mean()
                logger.info('Added Fama-French factors')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f'Виникла помилка: {e}')
            logger.warning(f'Could not add Fama-French factors: {e}')
            raise

    def _calculate_hurst_exponent(self, ts):
        """Calculate the Hurst exponent of a time series safely and efficiently."""
        try:
            import numpy as np
            ts_clean = ts[~np.isnan(ts)]
            if len(ts_clean) < 30:
                return 0.5
            lags = range(2, min(20, len(ts_clean) // 2))
            tau = []
            for lag in lags:
                diffs = ts_clean[lag:] - ts_clean[:-lag]
                std = np.std(diffs)
                tau.append(std if std > 0 else 1e-9)
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return float(poly[0] * 2.0)
        except Exception:
            return 0.5
        return 0.5
