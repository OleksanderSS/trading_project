
from typing import Any

import pandas as pd

from src.analytics.interfaces import IAnalyzer
from src.core.logging.logger import ProjectLogger
from src.sentiment.sentiment_models import analyze_sentiment

logger = ProjectLogger.get_logger("NewsImpactAnalyzer")

class NewsImpactAnalyzer(IAnalyzer):
    """
    Analyzes raw news text with sentiment analysis and persistent caching.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.sentiment_weights = self.config.get('sentiment_weights', {
            'positive': 1.0, 'negative': -1.0, 'neutral': 0.0
        })
        self.half_life_hours = self.config.get('half_life_hours', 48)

    def analyze(self, data: Any, **kwargs) -> dict[str, Any] | Any:
        """Main entry point for news analysis."""
        news_data = data
        if isinstance(news_data, dict):
            news_data = news_data.get('news_data')

        if news_data is None or not isinstance(news_data, pd.DataFrame) or news_data.empty:
            return {}

        sentiment_results = self._perform_sentiment_analysis(news_data)
        if sentiment_results.empty:
            return {}

        # Calculation pipeline
        weighted = self._calculate_weighted_scores(sentiment_results)
        # Use existing timestamp index
        aggregated = weighted['weighted_score'].groupby(weighted.index).sum()

        # Apply time decay
        impact_scores = self._apply_time_decay(aggregated)

        return {
            'news_impact_scores': impact_scores,
            'news_significance_levels': self._determine_significance(impact_scores)
        }

    def _perform_sentiment_analysis(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Performs sentiment analysis with robust batch-based DuckDB caching."""
        from src.config.unified_config_manager import get_current_config
        from src.data.management.data_manager import DataManager

        try:
            db = DataManager(config_manager=get_current_config())
            db.execute_query("CREATE TABLE IF NOT EXISTS news_sentiment_cache (news_hash VARCHAR PRIMARY KEY, sentiment DOUBLE, confidence DOUBLE)")

            # Identify news by hash
            news_data = news_data.copy()
            # Try multiple text sources
            text_col = 'text' if 'text' in news_data.columns else ('content' if 'content' in news_data.columns else 'title')
            if text_col not in news_data.columns:
                logger.warning("No text column found for sentiment analysis. Using empty strings.")
                news_data['text_to_analyze'] = ""
            else:
                news_data['text_to_analyze'] = news_data[text_col].fillna("")

            if 'hash' not in news_data.columns:
                news_data['hash'] = [str(hash(t)) for t in news_data['text_to_analyze']]

            # Check cache
            cached_data = db.fetch_all("SELECT news_hash, sentiment, confidence FROM news_sentiment_cache")
            cached_df = pd.DataFrame(cached_data).set_index('news_hash') if cached_data else pd.DataFrame()

            results = []
            to_analyze = []

            for _, row in news_data.iterrows():
                h = row['hash']
                if not cached_df.empty and h in cached_df.index:
                    # Safe retrieval
                    cached_row = cached_df.loc[h]
                    # Handle both Series and DataFrame (if duplicate hashes exist)
                    s_val = cached_row['sentiment'] if isinstance(cached_row, pd.Series) else cached_row.iloc[0]['sentiment']
                    c_val = cached_row['confidence'] if isinstance(cached_row, pd.Series) else cached_row.iloc[0]['confidence']
                    results.append({'sentiment': s_val, 'confidence': c_val, 'hash': h})
                else:
                    to_analyze.append(row)

            if to_analyze:
                logger.info(f"Analyzing {len(to_analyze)} new news items...")
                analyze_df = pd.DataFrame(to_analyze)

                # Process in batches of 50
                batch_size = 50
                for i in range(0, len(analyze_df), batch_size):
                    batch = analyze_df.iloc[i:i+batch_size]
                    logger.info(f"   Batch {i//batch_size + 1}/{(len(analyze_df)-1)//batch_size + 1}...")

                    batch_results = analyze_sentiment(batch['text_to_analyze'].tolist())

                    for j, res in enumerate(batch_results.to_dict('records')):
                        h = batch.iloc[j]['hash']
                        # Map label/score to numeric sentiment (-1 to 1)
                        label = res.get('label', 'neutral')
                        score = res.get('score', 0.0)

                        sentiment_val = score if label == 'positive' else (-score if label == 'negative' else 0.0)
                        confidence_val = score # For FinBERT, score IS confidence

                        results.append({'sentiment': sentiment_val, 'confidence': confidence_val, 'hash': h})
                        db.execute_query("INSERT OR IGNORE INTO news_sentiment_cache VALUES (?, ?, ?)",
                                         [h, sentiment_val, confidence_val])

            final_df = pd.DataFrame(results)
            final_df.index = news_data.index
            return final_df

        except Exception as e:
            logger.warning(f"⚠️ Sentiment Cache failed: {e}. Falling back to full analysis.")
            text_col = 'text' if 'text' in news_data.columns else ('content' if 'content' in news_data.columns else 'title')
            texts = news_data[text_col].fillna("").tolist() if text_col in news_data.columns else [""] * len(news_data)

            raw_results = analyze_sentiment(texts)
            # Map labels to sentiment column
            raw_results['sentiment'] = raw_results.apply(
                lambda x: x['score'] if x['label'] == 'positive' else (-x['score'] if x['label'] == 'negative' else 0.0),
                axis=1
            )
            return raw_results

    def _calculate_weighted_scores(self, sentiment_results: pd.DataFrame) -> pd.DataFrame:
        sentiment_results['weighted_score'] = sentiment_results['sentiment']
        return sentiment_results

    def _apply_time_decay(self, aggregated_scores: pd.Series) -> pd.Series:
        if len(aggregated_scores) < 2: return aggregated_scores
        # Safe alpha calculation
        return aggregated_scores.ewm(alpha=0.1, adjust=False).mean()

    def _determine_significance(self, scores: pd.Series) -> pd.Series:
        return scores.apply(lambda x: 'high' if abs(x) > 0.5 else 'low').astype('category')
