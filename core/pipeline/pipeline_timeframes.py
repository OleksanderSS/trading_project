# core/pipeline_timeframes.py

import pandas as pd
from typing import Dict
from collectors.news_collector import NewsCollector
from collectors.price_collector import PriceCollector
from utils.features import FeatureEngineer
from utils.features_utils import FeatureUtils
from utils.logger import ProjectLogger
import logging


# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger = ProjectLogger.get_logger("PipelineTimeframes")


class PipelineTimeframes:
    def __init__(self,
        news_collector: NewsCollector,
        price_collector: PriceCollector,
        feature_engineer: FeatureEngineer):
        self.news_collector = news_collector
        self.price_collector = price_collector
        self.feature_engineer = feature_engineer
        self.feature_utils = FeatureUtils(short_window=14, long_window=200)

    def build_contexts(self, forecast_date: pd.Timestamp) -> Dict[str, Dict[str, pd.DataFrame]]:
        start_date = forecast_date - pd.DateOffset(years=1)
        trading_days = pd.date_range(start=start_date, end=forecast_date, freq="B")  # торговand днand

        df_news = self.news_collector.fetch(start_date=start_date, end=forecast_date)
        df_prices = self.price_collector.fetch(start_date=start_date, end=forecast_date)

        if df_news.empty or df_prices.empty:
            logger.warning("[PipelineTimeframes] Данand вandдсутнand")
            return {}

        # --- Перевandрка ключових колонок ---
        for col in ["published_at", "sentiment_score", "news_count"]:
            if col not in df_news.columns:
                df_news[col] = 0.0
        for col in ["datetime", "close", "volume"]:
            if col not in df_prices.columns:
                df_prices[col] = 0.0

        df_news = df_news.set_index('published_at').sort_index()
        df_prices = df_prices.set_index('datetime').sort_index()

        # --- Master timeline ---
        df_news = df_news.resample("B").agg({
            'sentiment_score': 'mean',
            'news_count': 'count'
        }).reindex(trading_days).fillna(0)

        df_prices = df_prices.reindex(trading_days).ffill()

        contexts = {}
        contexts['year'] = self._build_context_slice(df_prices, df_news, lookback_days=252, key='year')
        contexts['month'] = self._build_context_slice(df_prices, df_news, lookback_days=21, key='month')
        contexts['week'] = self._build_context_slice(df_prices, df_news, lookback_days=5, key='week')
        contexts['day'] = self._build_context_slice(df_prices,
            df_news,
            lookback_days=1,
            key='day',
            intraday_freqs=['H',
            '15T'])

        # --- Логування реwithульandтandв ---
        for k, v in contexts.items():
            logger.info(f"[PipelineTimeframes] [OK] {k}  price={v['price'].shape}, news={v['news'].shape}")
            if v['price'].empty or v['news'].empty:
                logger.warning(f"[PipelineTimeframes] [WARN] {k} контекст порожнandй")

        return contexts

    def _build_context_slice(self,
        df_prices: pd.DataFrame,
        df_news: pd.DataFrame,
        lookback_days: int,
        key: str,
        intraday_freqs=None):
        price_slice = df_prices.tail(lookback_days).copy()
        news_slice = df_news.tail(lookback_days).copy()

        if price_slice.empty or news_slice.empty:
            logger.warning(f"[PipelineTimeframes] Порожнandй slice for {key}")
            return {'price': pd.DataFrame(), 'news': pd.DataFrame()}

        try:
            # Баwithовand контекстнand фandчand (беwith derived)
            price_feat = self.feature_utils.transform(price_slice, key=key)
            news_feat = self.feature_utils.transform(news_slice, key=key)
            news_feat = self._sync_news_with_prices(news_feat, price_feat)

            # Intraday ресемплandнг
            if intraday_freqs:
                for freq in intraday_freqs:
                    try:
                        price_intraday = df_prices.tail(lookback_days * 24).resample(freq).ffill()
                        price_intraday_feat = self.feature_utils.transform(price_intraday, key=f"{key}_{freq}")
                        price_feat = pd.concat([price_feat, price_intraday_feat]).sort_index()

                        news_intraday = df_news.tail(lookback_days * 24).resample(freq).ffill()
                        news_intraday_feat = self.feature_utils.transform(news_intraday, key=f"{key}_{freq}")
                        news_intraday_feat = self._sync_news_with_prices(news_intraday_feat, price_intraday_feat)
                        news_feat = pd.concat([news_feat, news_intraday_feat]).sort_index()
                    except Exception as e:
                        logger.error(f"[PipelineTimeframes] [ERROR] Intraday ресемплandнг {freq} for {key} not вдався: {e}")

            return {'price': price_feat, 'news': news_feat}

        except Exception as e:
            logger.error(f"[PipelineTimeframes] Error контексту {key}: {e}")
            return {'price': pd.DataFrame(), 'news': pd.DataFrame()}

    @staticmethod
    def _sync_news_with_prices(df_news: pd.DataFrame, df_price: pd.DataFrame) -> pd.DataFrame:
        if df_news.empty or df_price.empty:
            return pd.DataFrame()
        
        # Use merge_asof instead of reindex for better timestamp matching
        # This handles news that don't exactly match price timestamps
        try:
            merged = pd.merge_asof(
                df_news.reset_index().sort_values('published_at'),
                df_price.reset_index().sort_values('datetime'),
                left_on='published_at',
                right_on='datetime',
                direction='nearest',
                allow_exact_matches=True
            )
            # Set index back to datetime for consistency
            merged = merged.set_index('datetime').drop(columns=['published_at'])
            return merged
        except Exception:
            # Fallback to original method if merge_asof fails
            return df_news.reindex(df_price.index).ffill()