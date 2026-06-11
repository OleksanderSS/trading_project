"""
Intelligent Data Filter - Orchestrator for data quality and filtering.
Delegates specialized filtering to sub-modules in src/processing/filters/.
"""
from typing import Any

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.processing.filters.price_filter import PriceFilter
from src.processing.filters.social_filter import SocialFilter

logger = ProjectLogger.get_logger("IntelligentDataFilter")

class IntelligentDataFilter:
    """
    Main orchestrator for data filtering.
    Classifies data imperfections and ensures high-quality datasets for models.
    """

    def __init__(self, config_manager: Any = None):
        self.config = config_manager or get_current_config()
        self.filter_config = self.config.get('data.filtering', {})

        # Initialize specialized filters
        self.price_filter = PriceFilter(self.filter_config)
        self.social_filter = SocialFilter(self.filter_config)

        logger.info("✅ IntelligentDataFilter initialized as orchestrator.")

    def filter_quality_data(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """
        Main filtering function. Delegates to specialized filters.
        """
        filtered_data = {}
        quality_report = {}

        # 1. Price Data
        if 'prices' in raw_data:
            filtered_prices, price_quality = self.price_filter.filter(raw_data['prices'])
            filtered_data['prices'] = filtered_prices
            quality_report['prices'] = price_quality

        # 2. News Data
        if 'news' in raw_data:
            filtered_news, news_quality = self.social_filter.filter_news(raw_data['news'])
            filtered_data['news'] = filtered_news
            quality_report['news'] = news_quality

        # 3. Reddit Data
        if 'reddit_sentiment' in raw_data:
            filtered_reddit, reddit_quality = self.social_filter.filter_reddit(raw_data['reddit_sentiment'])
            filtered_data['reddit_sentiment'] = filtered_reddit
            quality_report['reddit_sentiment'] = reddit_quality

        # 4. Macro Data (Direct pass-through as it's pre-cleaned)
        if 'macro_data' in raw_data:
            filtered_data['macro_data'] = raw_data['macro_data']
            quality_report['macro_data'] = {'status': 'accepted', 'rows': len(raw_data['macro_data'])}

        return {
            'filtered_data': filtered_data,
            'quality_report': quality_report,
            'filtering_summary': self._create_summary(quality_report)
        }

    def _create_summary(self, quality_report: dict[str, Any]) -> dict[str, Any]:
        """Generates a high-level summary of the filtering results."""
        summary = {
            'accepted_sources': [],
            'rejected_sources': [],
            'overall_status': 'good'
        }
        for source, report in quality_report.items():
            if isinstance(report, dict):
                status = report.get('status', 'unknown')
                if status == 'accepted':
                    summary['accepted_sources'].append(source)
                else:
                    summary['rejected_sources'].append(source)

        if len(summary['rejected_sources']) > len(summary['accepted_sources']):
            summary['overall_status'] = 'critical'

        return summary

def filter_data_for_model_training(raw_data: dict[str, Any], config: Any = None) -> dict[str, Any]:
    """Helper function for backward compatibility."""
    return IntelligentDataFilter(config).filter_quality_data(raw_data)
