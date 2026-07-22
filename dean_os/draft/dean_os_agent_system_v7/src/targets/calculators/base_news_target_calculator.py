from abc import ABC, abstractmethod

import pandas as pd

from src.core.logging.logger import ProjectLogger


class BaseNewsTargetCalculator(ABC):
    def __init__(self, name: str, is_post: bool = False):
        self.logger = ProjectLogger.get_logger(name)
        self.is_post = is_post

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
        if self.is_post:
            filtered_news = ticker_news[ticker_news["published_date"] >= current_time - time_window]
        else:
            filtered_news = ticker_news[
                (ticker_news["published_date"] >= current_time) &
                (ticker_news["published_date"] <= current_time + time_window)
            ]
        return ticker_news, filtered_news

    @abstractmethod
    def calculate(
        self, df_tf: pd.DataFrame, ticker: str, news_df: pd.DataFrame, candle_num: int, timeframe: str
    ) -> pd.Series:
        pass
