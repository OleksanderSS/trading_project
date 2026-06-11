"""
Pre-News Target Calculator
Ð Ð¾Ð·Ñ€Ð°Ñ…Ð¾Ð²ÑƒÑ” Ñ‚Ð°Ñ€Ð³ÐµÑ‚Ð¸ Ð½Ð° Ð¾ÑÐ½Ð¾Ð²Ñ– ÑÐ²Ñ–Ñ‡Ð¾Ðº Ð”Ðž Ð¿ÑƒÐ±Ð»Ñ–ÐºÐ°Ñ†Ñ–Ñ— Ð½Ð¾Ð²Ð¸Ð½Ð¸
"""


import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("PreNewsTargetCalculator")


class PreNewsTargetCalculator:
    """
    Ð Ð¾Ð·Ñ€Ð°Ñ…Ð¾Ð²ÑƒÑ” Ñ‚Ð°Ñ€Ð³ÐµÑ‚Ð¸ Ð´Ð»Ñ ÑÐ²Ñ–Ñ‡Ð¾Ðº Ð´Ð¾ Ð½Ð¾Ð²Ð¸Ð½Ð¸.

    Ð›Ð¾Ð³Ñ–ÐºÐ°:
    - Ð—Ð½Ð°Ñ…Ð¾Ð´Ð¸Ñ‚ÑŒ N-Ñƒ ÑÐ²Ñ–Ñ‡ÐºÑƒ Ð”Ðž Ð¿ÑƒÐ±Ð»Ñ–ÐºÐ°Ñ†Ñ–Ñ— Ð½Ð¾Ð²Ð¸Ð½Ð¸
    - Ð Ð¾Ð·Ñ€Ð°Ñ…Ð¾Ð²ÑƒÑ” return Ð²Ñ–Ð´ Ñ‚Ñ–Ñ”Ñ— ÑÐ²Ñ–Ñ‡ÐºÐ¸ Ð´Ð¾ Ð¼Ð¾Ð¼ÐµÐ½Ñ‚Ñƒ Ð¿ÑƒÐ±Ð»Ñ–ÐºÐ°Ñ†Ñ–Ñ— Ð½Ð¾Ð²Ð¸Ð½Ð¸
    """

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Ð Ð¾Ð·Ñ€Ð°Ñ…ÑƒÐ²Ð°Ñ‚Ð¸ pre-news Ñ‚Ð°Ñ€Ð³ÐµÑ‚

        Args:
            df: DataFrame Ð· Ñ†Ñ–Ð½Ð¾Ð²Ð¸Ð¼Ð¸ Ð´Ð°Ð½Ð¸Ð¼Ð¸
            **kwargs: ÐŸÐ°Ñ€Ð°Ð¼ÐµÑ‚Ñ€Ð¸
                - timeframe: '15m', '60m', '1d'
                - candle_num: Ð½Ð¾Ð¼ÐµÑ€ ÑÐ²Ñ–Ñ‡ÐºÐ¸ Ð´Ð¾ Ð½Ð¾Ð²Ð¸Ð½Ð¸ (1, 2, ...)
                - news_df: DataFrame Ð· Ð½Ð¾Ð²Ð¸Ð½Ð°Ð¼Ð¸

        Returns:
            Series Ð· Ñ‚Ð°Ñ€Ð³ÐµÑ‚Ð°Ð¼Ð¸
        """
        # ÐžÑ‚Ñ€Ð¸Ð¼Ð°Ñ‚Ð¸ Ð¿Ð°Ñ€Ð°Ð¼ÐµÑ‚Ñ€Ð¸
        timeframe = kwargs.get('timeframe', '1d')
        candle_num = kwargs.get('candle_num', 1)
        news_df = kwargs.get('news_df')

        if news_df is None or news_df.empty:
            logger.warning("No news data provided for pre_news target")
            return pd.Series(np.nan, index=df.index)

        # Ð¤Ñ–Ð»ÑŒÑ‚Ñ€ÑƒÑ”Ð¼Ð¾ Ð´Ð°Ð½Ñ– Ð¿Ð¾ Ñ‚Ð°Ð¹Ð¼Ñ„Ñ€ÐµÐ¹Ð¼Ñƒ
        if 'interval' in df.columns:
            df_tf = df[df['interval'] == timeframe].copy()
        else:
            df_tf = df.copy()

        if df_tf.empty:
            logger.warning(f"No data for timeframe {timeframe}")
            return pd.Series(np.nan, index=df.index)

        # ÐŸÐµÑ€ÐµÐºÐ¾Ð½ÑƒÑ”Ð¼Ð¾ÑÑ Ñ‰Ð¾ datetime Ñ” ÐºÐ¾Ð»Ð¾Ð½ÐºÐ¾ÑŽ
        if 'datetime' not in df_tf.columns:
            if isinstance(df_tf.index, pd.DatetimeIndex):
                df_tf = df_tf.copy()
                df_tf['datetime'] = df_tf.index

        # Ð Ð¾Ð·Ñ€Ð°Ñ…ÑƒÐ²Ð°Ñ‚Ð¸ Ñ‚Ð°Ñ€Ð³ÐµÑ‚Ð¸ Ð´Ð»Ñ ÐºÐ¾Ð¶Ð½Ð¾Ð³Ð¾ Ñ€ÑÐ´ÐºÐ°
        targets = []

        for _idx, row in df_tf.iterrows():
            ticker = row.get('ticker')
            current_time = row.get('datetime')
            current_price = row.get('close')

            if pd.isna(ticker) or pd.isna(current_time) or pd.isna(current_price):
                targets.append(np.nan)
                continue

            # Ð—Ð½Ð°Ð¹Ñ‚Ð¸ Ð½Ð¾Ð²Ð¸Ð½Ð¸ Ð´Ð»Ñ Ñ†ÑŒÐ¾Ð³Ð¾ Ñ‚Ñ–ÐºÐµÑ€Ð° Ð±Ð»Ð¸Ð·ÑŒÐºÐ¾ Ð´Ð¾ Ð¿Ð¾Ñ‚Ð¾Ñ‡Ð½Ð¾Ð³Ð¾ Ñ‡Ð°ÑÑƒ
            # (Ð² Ð¼ÐµÐ¶Ð°Ñ… 1 ÑÐ²Ñ–Ñ‡ÐºÐ¸ Ð²Ð¿ÐµÑ€ÐµÐ´)
            ticker_news = news_df[
                (news_df['ticker'] == ticker) | (news_df.get('news_type', 'general') == 'general')
            ]

            # Ð—Ð½Ð°Ð¹Ñ‚Ð¸ Ð½Ð¾Ð²Ð¸Ð½Ð¸ Ñ‰Ð¾ Ð±ÑƒÐ´ÑƒÑ‚ÑŒ Ð¾Ð¿ÑƒÐ±Ð»Ñ–ÐºÐ¾Ð²Ð°Ð½Ñ– Ð½ÐµÐ·Ð°Ð±Ð°Ñ€Ð¾Ð¼ Ð¿Ñ–ÑÐ»Ñ Ð¿Ð¾Ñ‚Ð¾Ñ‡Ð½Ð¾Ñ— ÑÐ²Ñ–Ñ‡ÐºÐ¸
            time_window = pd.Timedelta(hours=24) if timeframe == '1d' else pd.Timedelta(hours=1)
            upcoming_news = ticker_news[
                (ticker_news['published_date'] >= current_time) &
                (ticker_news['published_date'] <= current_time + time_window)
            ]

            if upcoming_news.empty:
                targets.append(np.nan)
                continue

            # Ð’Ð·ÑÑ‚Ð¸ Ð½Ð°Ð¹Ð±Ð»Ð¸Ð¶Ñ‡Ñƒ Ð½Ð¾Ð²Ð¸Ð½Ñƒ
            upcoming_news = upcoming_news.sort_values('published_date')
            news_time = upcoming_news.iloc[0]['published_date']

            # Ð—Ð½Ð°Ð¹Ñ‚Ð¸ N-Ñƒ ÑÐ²Ñ–Ñ‡ÐºÑƒ Ð”Ðž Ð½Ð¾Ð²Ð¸Ð½Ð¸
            past_candles = df_tf[
                (df_tf['ticker'] == ticker) &
                (df_tf['datetime'] < news_time)
            ].tail(candle_num)
