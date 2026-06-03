from abc import ABC, abstractmethod

import pandas as pd

from src.core.logging.logger import ProjectLogger


class BaseNewsTargetCalculator(ABC):
    def __init__(self, name: str):
        self.logger = ProjectLogger.get_logger(name)

    def _prepare_data(
        self,
        df_tf: pd.DataFrame,
        ticker: str,
        news_df: pd.DataFrame,
        current_time: pd.Timestamp,
        time_window: pd.Timedelta,
    ):
        if news_df is None or news_df.empty:
            self.logger.warning("No news data provided")
            return None, None

        ticker_news = news_df[news_df["ticker"] == ticker]
        return ticker_news, ticker_news[
            (ticker_news["published_date"] >= current_time - time_window)
            if self.is_post
            else (ticker_news["published_date"] >= current_time) & (ticker_news["published_date"] <= current_time)
            if self.is_post
            else (ticker_news["published_date"] <= current_time + time_window)
        ]

    @abstractmethod
    def calculate(
        self, df_tf: pd.DataFrame, ticker: str, news_df: pd.DataFrame, candle_num: int, timeframe: str
    ) -> pd.Series:
        pass
