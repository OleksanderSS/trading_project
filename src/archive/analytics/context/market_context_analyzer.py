import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..interfaces import IAnalyzer

logger = logging.getLogger(__name__)


class MarketContextAnalyzer(IAnalyzer):
    """
    Analyzes raw market data to generate a standardized context vector.
    This vector captures the 'DNA' of the market at a specific moment, including
    volatility, trend, momentum, and other user-defined features.
    """

    def __init__(self, context_features: list[str]):
        """
        Initializes the MarketContextAnalyzer.

        Args:
            context_features (List[str]): A list of feature names that define the market context.
        """
        if not context_features:
            raise ValueError('The context_features list cannot be empty.')
        self.context_features = context_features
        self.logger = logger
        logger.info(
            f'MarketContextAnalyzer initialized with {len(context_features)} features.'
            )

    def analyze(self, data: pd.DataFrame, **kwargs) ->dict[str, Any]:
        """
        Analyzes the provided market data to compute a context vector.

        Args:
            data (pd.DataFrame): A DataFrame containing market data (OHLCV, indicators, etc.).
                                 Must have a datetime index.
            **kwargs: Can include external data like 'vix_level' or 'macro_bias'.

        Returns:
            Dict[str, Any]: A dictionary containing the computed context vector as a pd.Series.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            logger.error(
                'Invalid input: Market data must be a non-empty pd.DataFrame.')
            return {'error': 'Invalid input data'}
        context_vector = pd.Series(index=self.context_features, dtype=float)
        self._find_price_columns(data)
        for feature in self.context_features:
            calc_method_name = f'_calculate_{feature}'
            if hasattr(self, calc_method_name):
                try:
                    value = getattr(self, calc_method_name)(data, **kwargs)
                    context_vector[feature] = value
                except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                    self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                    logger.warning(
                        f"Could not calculate feature '{feature}': {e}")
                    context_vector[feature] = np.nan
                    raise
            elif feature in kwargs:
                context_vector[feature] = kwargs[feature]
        final_vector, missing_features = self._apply_context_defaults(context_vector)
        return {
            'market_context_vector': final_vector,
            'missing_context_features': missing_features,
        }

    def _apply_context_defaults(self, context_vector: pd.Series) ->tuple[pd.Series, list[str]]:
        final_vector = context_vector.copy()
        missing_features = [str(feature) for feature in final_vector.index[final_vector.isna()]]
        defaults = {
            'rsi_current': 50.0,
            'volume_ratio': 1.0,
            'volatility_ratio': 1.0,
            'trend_alignment': 0.0,
            'price_to_ma20': 0.0,
            'yield_curve_slope': 0.0,
            'credit_spread': 0.0,
            'macro_bias': 0.0,
        }
        for feature in missing_features:
            final_vector.loc[feature] = defaults.get(feature, 0.0)
        return final_vector, missing_features

    def _find_price_columns(self, df: pd.DataFrame):
        """
        Знаходить колонки з OHLCV даними в DataFrame.
        Підтримує як стандартний формат (close, open, high, low, volume),
        так і event-centric формат (AMD_1d_close, AMD_15m_open, тощо).
        """
        self.close_col = self._find_column(df, 'close')
        self.open_col = self._find_column(df, 'open')
        self.high_col = self._find_column(df, 'high')
        self.low_col = self._find_column(df, 'low')
        self.volume_col = self._find_column(df, 'volume')
        self.rsi_col = self._find_rsi_column(df)

    def _find_column(self, df: pd.DataFrame, column_type: str) ->str | None:
        """Знайти колонку певного типу"""
        if column_type in df.columns:
            return column_type
        suffix = f'_{column_type}'
        cols = [col for col in df.columns if col.endswith(suffix) and not
            col.endswith('_+1') and not col.endswith('_+2')]
        if cols:
            return next((col for col in cols if '_1d_' in col), cols[0])
        return None

    def _find_rsi_column(self, df: pd.DataFrame) ->str | None:
        """Знайти RSI колонку"""
        rsi_cols = [col for col in df.columns if 'rsi' in col.lower() and
            not col.endswith('_+1') and not col.endswith('_+2')]
        if rsi_cols:
            return next((col for col in rsi_cols if '_1d_' in col), rsi_cols[0]
                )
        return None

    def _calculate_volatility_5d(self, df: pd.DataFrame, **kwargs) ->float:
        if self.close_col and self.close_col in df.columns:
            returns = df[self.close_col].pct_change(fill_method=None).replace(
                [np.inf, -np.inf], np.nan).dropna()
            return returns.tail(5).std() if len(returns) >= 2 else np.nan
        return np.nan

    def _calculate_volatility_20d(self, df: pd.DataFrame, **kwargs) ->float:
        if self.close_col and self.close_col in df.columns:
            returns = df[self.close_col].pct_change(fill_method=None).replace(
                [np.inf, -np.inf], np.nan).dropna()
            return returns.tail(20).std() if len(returns) >= 2 else np.nan
        return np.nan

    def _calculate_volatility_ratio(self, df: pd.DataFrame, **kwargs) ->float:
        vol_5d = self._calculate_volatility_5d(df)
        vol_20d = self._calculate_volatility_20d(df)
        if not np.isfinite(vol_5d) or not np.isfinite(vol_20d) or vol_20d <= 1e-12:
            return np.nan
        return vol_5d / (vol_20d + 1e-09)

    def _calculate_trend_5d(self, df: pd.DataFrame, **kwargs) ->float:
        if self.close_col and self.close_col in df.columns:
            prices = df[self.close_col].tail(5).values
            if len(prices) >= 2:
                return np.polyfit(np.arange(len(prices)), prices, 1)[0]
        return 0.0

    def _calculate_trend_20d(self, df: pd.DataFrame, **kwargs) ->float:
        if self.close_col and self.close_col in df.columns:
            prices = df[self.close_col].tail(20).values
            if len(prices) >= 2:
                return np.polyfit(np.arange(len(prices)), prices, 1)[0]
        return 0.0

    def _calculate_trend_alignment(self, df: pd.DataFrame, **kwargs) ->float:
        trend_5d = self._calculate_trend_5d(df)
        trend_20d = self._calculate_trend_20d(df)
        return np.sign(trend_5d * trend_20d)

    def _calculate_rsi_current(self, df: pd.DataFrame, **kwargs) ->float:
        if self.rsi_col and self.rsi_col in df.columns:
            return df[self.rsi_col].iloc[-1]
        return 50.0

    def _calculate_volume_ratio(self, df: pd.DataFrame, **kwargs) ->float:
        if not self._can_calculate_volume_ratio(df):
            return 1.0
        avg_vol_5 = df[self.volume_col].tail(5).mean()
        avg_vol_20 = df[self.volume_col].tail(20).mean()
        return avg_vol_5 / (avg_vol_20 + 1e-09)

    def _can_calculate_volume_ratio(self, df: pd.DataFrame) ->bool:
        """Check if volume ratio can be calculated."""
        return self.volume_col and self.volume_col in df.columns and len(df
            ) >= 20

    def _calculate_price_to_ma20(self, df: pd.DataFrame, **kwargs) ->float:
        if not self._can_calculate_price_to_ma20(df):
            return 0.0
        ma20 = df[self.close_col].tail(20).mean()
        return df[self.close_col].iloc[-1] / ma20 - 1

    def _can_calculate_price_to_ma20(self, df: pd.DataFrame) ->bool:
        """Check if price to MA20 can be calculated."""
        return self.close_col and self.close_col in df.columns and len(df
            ) >= 20

    def _calculate_hour_of_day(self, df: pd.DataFrame, **kwargs) ->int:
        return df.index[-1].hour if isinstance(df.index, pd.DatetimeIndex
            ) else datetime.now().hour

    def _calculate_day_of_week(self, df: pd.DataFrame, **kwargs) ->int:
        return df.index[-1].weekday() if isinstance(df.index, pd.DatetimeIndex
            ) else datetime.now().weekday()

    def _calculate_yield_curve_slope(self, df: pd.DataFrame, **kwargs) ->float:
        """
        Розраховує нахил кривої дохідності (10Y - 2Y).
        Негативне значення = інверсія кривої = можлива рецесія.
        """
        dgs10 = self._get_yield_rate('DGS10', df, kwargs)
        dgs2 = self._get_yield_rate('DGS2', df, kwargs)
        if self._is_yield_data_invalid(dgs10, dgs2):
            return 0.0
        slope = dgs10 - dgs2
        self._log_yield_curve_slope(slope, dgs10, dgs2)
        return slope

    def _get_yield_rate(self, rate_name: str, df: pd.DataFrame, kwargs:
        dict[str, Any]) ->float:
        """Get yield rate from kwargs or DataFrame."""
        if rate_name in df.columns:
            return df[rate_name].iloc[-1]
        return kwargs.get(rate_name, np.nan)

    def _is_yield_data_invalid(self, dgs10: float, dgs2: float) ->bool:
        """Check if yield data is invalid."""
        return pd.isna(dgs10) or pd.isna(dgs2)

    def _log_yield_curve_slope(self, slope: float, dgs10: float, dgs2: float):
        """Log yield curve slope information."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f'Yield curve slope: {slope:.4f} (10Y={dgs10:.2f}%, 2Y={dgs2:.2f}%)'
                )

    def _calculate_yield_curve_inverted(self, df: pd.DataFrame, **kwargs
        ) ->int:
        """
        Прапорець інверсії кривої дохідності (0 або 1).
        1 = інверсія (10Y < 2Y) = сигнал рецесії.
        """
        slope = self._calculate_yield_curve_slope(df, **kwargs)
        return 1 if slope < 0 else 0

    def _calculate_fed_funds_trend(self, df: pd.DataFrame, **kwargs) ->float:
        """
        Тренд ставки Fed Funds (зміна за останні 3 місяці).
        Позитивне = підвищення ставок = жорсткіша монетарна політика.
        """
        if not self._can_calculate_fed_funds_trend(df):
            return 0.0
        current = df['FEDFUNDS'].iloc[-1]
        three_months_ago = self._get_fed_funds_three_months_ago(df)
        trend = current - three_months_ago
        self._log_fed_funds_trend(trend, current, three_months_ago)
        return trend

    def _can_calculate_fed_funds_trend(self, df: pd.DataFrame) ->bool:
        """Check if Fed Funds trend can be calculated."""
        return 'FEDFUNDS' in df.columns and len(df) >= 60

    def _get_fed_funds_three_months_ago(self, df: pd.DataFrame) ->float:
        """Get Fed Funds rate from three months ago."""
        return df['FEDFUNDS'].iloc[-60] if len(df) >= 60 else df['FEDFUNDS'
            ].iloc[0]

    def _log_fed_funds_trend(self, trend: float, current: float,
        three_months_ago: float):
        """Log Fed Funds trend information."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f'Fed Funds trend: {trend:.4f}% (current={current:.2f}%, 3m ago={three_months_ago:.2f}%)'
                )

    def _calculate_fed_funds_velocity(self, df: pd.DataFrame, **kwargs
        ) ->float:
        """
        Швидкість зміни ставки Fed Funds (% за місяць).
        Висока швидкість = агресивна зміна політики.
        """
        if not self._can_calculate_fed_funds_velocity(df):
            return 0.0
        current = df['FEDFUNDS'].iloc[-1]
        one_month_ago = self._get_fed_funds_one_month_ago(df)
        velocity = current - one_month_ago
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Fed Funds velocity: {velocity:.4f}%/month')
        return velocity

    def _can_calculate_fed_funds_velocity(self, df: pd.DataFrame) ->bool:
        """Check if Fed Funds velocity can be calculated."""
        return 'FEDFUNDS' in df.columns and len(df) >= 20

    def _get_fed_funds_one_month_ago(self, df: pd.DataFrame) ->float:
        """Get Fed Funds rate from one month ago."""
        return df['FEDFUNDS'].iloc[-20] if len(df) >= 20 else df['FEDFUNDS'
            ].iloc[0]

    def _calculate_market_breadth(self, df: pd.DataFrame, **kwargs) ->float:
        """
        Ширина ринку (advance/decline ratio).
        > 1 = більше акцій зростає = здоровий тренд.
        < 1 = більше акцій падає = слабкий тренд.

        Якщо немає даних про advance/decline, використовуємо proxy:
        % акцій вище SMA(50) / % акцій нижче SMA(50).
        """
        if self._has_advance_decline_data(df):
            return self._calculate_advance_decline_breadth(df)
        if self._can_use_price_proxy(df):
            return self._calculate_price_proxy_breadth(df)
        return 1.0

    def _has_advance_decline_data(self, df: pd.DataFrame) ->bool:
        """Check if DataFrame has advance/decline data."""
        return 'advances' in df.columns and 'declines' in df.columns

    def _calculate_advance_decline_breadth(self, df: pd.DataFrame) ->float:
        """Calculate breadth using advance/decline data."""
        advances = df['advances'].iloc[-1]
        declines = df['declines'].iloc[-1]
        breadth = advances / (declines + 1e-09)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f'Market breadth: {breadth:.2f} (advances={advances}, declines={declines})'
                )
        return breadth

    def _can_use_price_proxy(self, df: pd.DataFrame) ->bool:
        """Check if price proxy can be used for breadth."""
        return 'close' in df.columns and len(df) >= 50

    def _calculate_price_proxy_breadth(self, df: pd.DataFrame) ->float:
        """Calculate breadth using price proxy."""
        sma50 = df['close'].tail(50).mean()
        current_price = df['close'].iloc[-1]
        breadth_proxy = 1.0 if current_price > sma50 else 0.5
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f'Market breadth (proxy): {breadth_proxy:.2f} (price vs SMA50)')
        return breadth_proxy

    def _calculate_dollar_strength(self, df: pd.DataFrame, **kwargs) -> float:
        """
        Сила долара (DXY index).
        Висока сила долара = тиск на commodities та emerging markets.
        """
        if 'DXY' in df.columns:
            dxy = df['DXY'].iloc[-1]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Dollar strength (DXY): {dxy:.2f}')
            return dxy
        dxy = kwargs.get('DXY', 100.0)
        return dxy

    def _calculate_put_call_ratio(self, df: pd.DataFrame, **kwargs) -> float:
        """
        Put/Call ratio (страх на ринку).
        > 1 = більше puts = песимізм.
        < 1 = більше calls = оптимізм.
        """
        if 'put_call_ratio' in df.columns:
            ratio = df['put_call_ratio'].iloc[-1]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Put/Call ratio: {ratio:.2f}')
            return ratio
        ratio = kwargs.get('put_call_ratio', 1.0)
        return ratio

    # --- COGNITIVE FEATURES (NEWS & SENTIMENT) ---

    def _calculate_sentiment_score(self, df: pd.DataFrame, **kwargs) -> float:
        """Поточний агрегований сентимент ринку."""
        sentiment_cols = ['nlp_sentiment_score', 'sentiment_score', 'sentiment']
        for col in sentiment_cols:
            if col in df.columns:
                return df[col].tail(5).mean()
        return kwargs.get('sentiment_score', 0.0)

    def _calculate_sentiment_momentum(self, df: pd.DataFrame, **kwargs) -> float:
        """Швидкість зміни сентименту (короткострокова vs довгострокова)."""
        sentiment_col = None
        for col in ['nlp_sentiment_score', 'sentiment_score', 'sentiment']:
            if col in df.columns:
                sentiment_col = col
                break

        if sentiment_col:
            short_ema = df[sentiment_col].ewm(span=5).mean().iloc[-1]
            long_ema = df[sentiment_col].ewm(span=20).mean().iloc[-1]
            return short_ema - long_ema
        return 0.0

    def _calculate_news_intensity(self, df: pd.DataFrame, **kwargs) -> float:
        """Інтенсивність новинного потоку (Hype factor)."""
        if 'news_intensity' in df.columns:
            return df['news_intensity'].iloc[-1]

        sentiment_col = None
        for col in ['nlp_sentiment_score', 'sentiment_score', 'sentiment']:
            if col in df.columns:
                sentiment_col = col
                break

        if sentiment_col:
            # Скільки ненульових значень сентименту за останні 20 періодів
            return (df[sentiment_col] != 0).tail(20).mean()
        return 0.0

