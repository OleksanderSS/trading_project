from src.core.logging.logger import ProjectLogger

from .news_filter import NewsFilter
from .pattern_extractor import PatternExtractor
from .price_filter import PriceFilter
from .social_filter import SocialFilter

logger = ProjectLogger.get_logger("IntelligentDataFilter")

class IntelligentDataFilter:
    """
    Modular Orchestrator for data filtering.
    Delegates to specialized filters for prices, news, and social data.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.logger = logger

        # Initialize modular components
        self.price_filter = PriceFilter(self.config)
        self.news_filter = NewsFilter(self.config)
        self.social_filter = SocialFilter(self.config)
        self.pattern_extractor = PatternExtractor()

    def filter_quality_data(self, raw_data: dict) -> dict:
        """Main entry point for filtering quality data."""
        self.logger.info("Starting intelligent data filtering cycle...")

        filtered_data = {}
        quality_report = {}

        # 1. Price Data
        if 'prices' in raw_data:
            f_prices, p_quality = self.price_filter.filter_price_data(raw_data['prices'])
            filtered_data['prices'] = f_prices
            quality_report['prices'] = p_quality

        # 2. News Data
        if 'news' in raw_data:
            f_news, n_quality = self.news_filter.filter_news_data(raw_data['news'])
            filtered_data['news'] = f_news
            quality_report['news'] = n_quality

        # 3. Macro Data (pass-through as per original)
        if 'macro_data' in raw_data:
            filtered_data['macro_data'] = raw_data['macro_data']
            quality_report['macro_data'] = {'status': 'accepted', 'rows': len(raw_data['macro_data'])}

        # 4. Reddit Sentiment
        if 'reddit_sentiment' in raw_data:
            f_reddit, r_quality = self.social_filter.filter_reddit_data(raw_data['reddit_sentiment'])
            filtered_data['reddit_sentiment'] = f_reddit
            quality_report['reddit_sentiment'] = r_quality

        # 5. Extract Patterns
        patterns = self.pattern_extractor.extract_patterns(filtered_data)

        return {
            'filtered_data': filtered_data,
            'quality_report': quality_report,
            'patterns': patterns,
            'filtering_summary': self._create_filtering_summary(quality_report)
        }

    def _create_filtering_summary(self, quality_report: dict) -> dict:
        """Creates a high-level summary of the filtering results."""
        summary = {
            'status': 'success',
            'accepted_sources': [],
            'rejected_sources': [],
            'total_anomalies': 0
        }

        for source, report in quality_report.items():
            if isinstance(report, dict) and report.get('status') == 'accepted':
                summary['accepted_sources'].append(source)
            else:
                summary['rejected_sources'].append(source)

        return summary
