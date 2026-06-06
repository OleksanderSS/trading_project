"""
Pre-News Target Calculator
Розраховує таргети на основі свічок ДО публікації новини
"""


import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("PreNewsTargetCalculator")


class PreNewsTargetCalculator:
    """
    Розраховує таргети для свічок до новини.

    Логіка:
    - Знаходить N-у свічку ДО публікації новини
    - Розраховує return від тієї свічки до моменту публікації новини
    """

    def _filter_by_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Filter data by timeframe."""
        if 'interval' in df.columns:
            return df[df['interval'] == timeframe].copy()
        else:
            return df.copy()

    def _ensure_datetime_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure datetime column exists."""
        if 'datetime' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df = df.rename(columns={'index': 'datetime'})
        return df

    def _get_upcoming_news(self, ticker: str, current_time: pd.Timestamp, news_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Find upcoming news for a ticker."""
        ticker_news = news_df[
            (news_df['ticker'] == ticker) | (news_df.get('news_type', 'general') == 'general')
        ]

        time_window = pd.Timedelta(hours=24) if timeframe == '1d' else pd.Timedelta(hours=1)
        return ticker_news[
            (ticker_news['published_date'] >= current_time) &
            (ticker_news['published_date'] <= current_time + time_window)
        ]

    def _calculate_target_return(self, df_tf: pd.DataFrame, ticker: str, current_time: pd.Timestamp,
                                 current_price: float, news_df: pd.DataFrame, candle_num: int, timeframe: str) -> float:
        """Calculate target return from pre-news candle to news time."""
        upcoming_news = self._get_upcoming_news(ticker, current_time, news_df, timeframe)

        if upcoming_news.empty:
            return np.nan

        upcoming_news = upcoming_news.sort_values('published_date')
        news_time = upcoming_news.iloc[0]['published_date']

        past_candles = df_tf[
            (df_tf['ticker'] == ticker) &
            (df_tf['datetime'] < news_time)
        ].tail(candle_num)

        if len(past_candles) < candle_num:
            return np.nan

        target_candle = past_candles.iloc[0]
        past_price = target_candle['close']

        news_candle = df_tf[
            (df_tf['ticker'] == ticker) &
            (df_tf['datetime'] >= news_time)
        ].head(1)

        if news_candle.empty:
            return np.nan

        news_price = news_candle.iloc[0]['close']
        return (news_price - past_price) / past_price

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Розрахувати pre-news таргет

        Args:
            df: DataFrame з ціновими даними
            **kwargs: Параметри
                - timeframe: '15m', '60m', '1d'
                - candle_num: номер свічки до новини (1, 2, ...)
                - news_df: DataFrame з новинами

        Returns:
            Series з таргетами
        """
        timeframe = kwargs.get('timeframe', '1d')
        candle_num = kwargs.get('candle_num', 1)
        news_df = kwargs.get('news_df')

        if news_df is None or news_df.empty:
            logger.warning("No news data provided for pre_news target")
            return pd.Series(np.nan, index=df.index)

        df_tf = self._filter_by_timeframe(df, timeframe)

        if df_tf.empty:
            logger.warning(f"No data for timeframe {timeframe}")
            return pd.Series(np.nan, index=df.index)

        df_tf = self._ensure_datetime_column(df_tf)

        targets = []

        for _idx, row in df_tf.iterrows():
            ticker = row.get('ticker')
            current_time = row.get('datetime')
            current_price = row.get('close')

            if pd.isna(ticker) or pd.isna(current_time) or pd.isna(current_price):
                targets.append(np.nan)
                continue

            target_return = self._calculate_target_return(df_tf, ticker, current_time, current_price, news_df, candle_num, timeframe)
            targets.append(target_return)

        return pd.Series(targets, index=df_tf.index)
