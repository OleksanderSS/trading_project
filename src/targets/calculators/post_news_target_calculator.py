"""
Post-News Target Calculator
Розраховує таргети на основі свічок ПІСЛЯ публікації новини
"""


import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("PostNewsTargetCalculator")


class PostNewsTargetCalculator:
    """
    Розраховує таргети для свічок після новини.

    Логіка:
    - Знаходить найближчу свічку після публікації новини
    - Розраховує return від ціни на момент новини до N-ої свічки після
    """

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Розрахувати post-news таргет

        Args:
            df: DataFrame з ціновими даними
            **kwargs: Параметри
                - timeframe: '15m', '60m', '1d'
                - candle_num: номер свічки після новини (1, 2, ...)
                - news_df: DataFrame з новинами

        Returns:
            Series з таргетами
        """
        # Отримати параметри
        timeframe = kwargs.get('timeframe', '1d')
        candle_num = kwargs.get('candle_num', 1)
        news_df = kwargs.get('news_df')

        if news_df is None or news_df.empty:
            logger.warning("No news data provided for post_news target")
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
            # (в межах 1 свічки назад)
            ticker_news = news_df[
                (news_df['ticker'] == ticker) | (news_df.get('news_type', 'general') == 'general')
            ]

            # Знайти новини що були опубліковані незадовго до поточної свічки
            time_window = pd.Timedelta(hours=24) if timeframe == '1d' else pd.Timedelta(hours=1)
            recent_news = ticker_news[
                (ticker_news['published_date'] >= current_time - time_window) &
                (ticker_news['published_date'] <= current_time)
            ]

            if recent_news.empty:
                targets.append(np.nan)
                continue

            # Взяти найближчу новину
            recent_news = recent_news.sort_values('published_date')
            news_time = recent_news.iloc[-1]['published_date']

            # Знайти N-у свічку після новини
            future_candles = df_tf[
                (df_tf['ticker'] == ticker) &
                (df_tf['datetime'] > news_time)
            ].head(candle_num)

            if len(future_candles) < candle_num:
                targets.append(np.nan)
                continue

            # Взяти ціну N-ої свічки
            target_candle = future_candles.iloc[candle_num - 1]
            target_price = target_candle['close']

            # Розрахувати return
            target_return = (target_price - current_price) / current_price
            targets.append(target_return)

        return pd.Series(targets, index=df_tf.index)
