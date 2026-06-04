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
        # Отримати параметри
        timeframe = kwargs.get('timeframe', '1d')
        candle_num = kwargs.get('candle_num', 1)
        news_df = kwargs.get('news_df')

        if news_df is None or news_df.empty:
            logger.warning("No news data provided for pre_news target")
            return pd.Series(np.nan, index=df.index)

        # Фільтруємо дані по таймфрейму
        if 'interval' in df.columns:
            df_tf = df[df['interval'] == timeframe].copy()
        else:
            df_tf = df.copy()

        if df_tf.empty:
            logger.warning(f"No data for timeframe {timeframe}")
            return pd.Series(np.nan, index=df.index)

        # Переконуємося що datetime є колонкою
        if 'datetime' not in df_tf.columns:
            if isinstance(df_tf.index, pd.DatetimeIndex):
                df_tf = df_tf.reset_index()
                df_tf = df_tf.rename(columns={'index': 'datetime'})

        # Розрахувати таргети для кожного рядка
        targets = []

        for _idx, row in df_tf.iterrows():
            ticker = row.get('ticker')
            current_time = row.get('datetime')
            current_price = row.get('close')

            if pd.isna(ticker) or pd.isna(current_time) or pd.isna(current_price):
                targets.append(np.nan)
                continue

            # Знайти новини для цього тікера близько до поточного часу
            # (в межах 1 свічки вперед)
            ticker_news = news_df[
                (news_df['ticker'] == ticker) | (news_df.get('news_type', 'general') == 'general')
            ]

            # Знайти новини що будуть опубліковані незабаром після поточної свічки
            time_window = pd.Timedelta(hours=24) if timeframe == '1d' else pd.Timedelta(hours=1)
            upcoming_news = ticker_news[
                (ticker_news['published_date'] >= current_time) &
                (ticker_news['published_date'] <= current_time + time_window)
            ]

            if upcoming_news.empty:
                targets.append(np.nan)
                continue

            # Взяти найближчу новину
            upcoming_news = upcoming_news.sort_values('published_date')
            news_time = upcoming_news.iloc[0]['published_date']

            # Знайти N-у свічку ДО новини
            past_candles = df_tf[
                (df_tf['ticker'] == ticker) &
                (df_tf['datetime'] < news_time)
            ].tail(candle_num)

            if len(past_candles) < candle_num:
                targets.append(np.nan)
                continue

            # Взяти ціну N-ої свічки до новини
            target_candle = past_candles.iloc[0]  # Найдавніша з останніх N
            past_price = target_candle['close']

            # Знайти ціну на момент новини (або найближчу)
            news_candle = df_tf[
                (df_tf['ticker'] == ticker) &
                (df_tf['datetime'] >= news_time)
            ].head(1)

            if news_candle.empty:
                targets.append(np.nan)
                continue

            news_price = news_candle.iloc[0]['close']

            # Розрахувати return від минулої свічки до новини
            target_return = (news_price - past_price) / past_price
            targets.append(target_return)

        return pd.Series(targets, index=df_tf.index)
