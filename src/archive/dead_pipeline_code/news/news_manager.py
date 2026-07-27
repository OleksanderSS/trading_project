from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.news_clusterer import cluster_news_simple
from src.features.news_dataset_builder import NewsContextDatasetBuilder

logger = ProjectLogger.get_logger("FeatureEngineeringNewsManager")

class FeatureEngineeringNewsManager:
    def __init__(self, config_manager: Any):
        self.news_builder = NewsContextDatasetBuilder(config_manager)
        logger.info("✅ News manager initialized")

    def process_news(self, news_df: pd.DataFrame, prices_dict: dict[str, pd.DataFrame],
                     macro_df: pd.DataFrame | None, market_sentiment_df: pd.DataFrame | None,
                     output_dir: Any) -> pd.DataFrame | None:
        if news_df is None or news_df.empty:
            logger.info("ℹ️ No news data available, skipping news-based dataset generation")
            return None

        logger.info(f"📰 Generating news-based dataset from {len(news_df)} news articles...")
        try:
            news_clustered = cluster_news_simple(news_df, similarity_threshold=0.85, text_column="title")
            logger.info(f"✅ Clustered {len(news_df)} → {len(news_clustered)} news")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"News clustering failed: {e}. Falling back to unclustered news.")
            news_clustered = news_df

        news_features_df = self.news_builder.build_dataset(
            news_df=news_clustered,
            prices_dict=prices_dict,
            macro_df=macro_df,
            market_sentiment_df=market_sentiment_df,
        )

        if news_features_df is not None and not news_features_df.empty:
            news_output_path = output_dir / "news_features.parquet"
            self.news_builder.save_dataset(news_features_df, news_output_path)
            logger.info(f"✅ News dataset built: {len(news_features_df)} rows")

        return news_features_df
