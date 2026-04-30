# src/features/nlp/processors/article_processor.py

import logging
from typing import Dict, Any, List, Optional

# --- NLP Components ---
from src.features.nlp.models.roberta_sentiment import RobertaSentimentAnalyzer
from src.features.nlp.scoring.summarizer import Summarizer
from src.features.nlp.extractors.keyword_extractor import KeywordExtractor

# --- Scoring Components ---
from src.features.nlp.scoring.simple_sentiment import SimpleSentimentAnalyzer
from src.features.nlp.scoring.news_scorer import NewsScorer

logger = logging.getLogger(__name__)

class ArticleProcessor:
    """
    A comprehensive processor for news articles that orchestrates enrichment and scoring.
    It extracts sentiment, keywords, entities, and a summary, then calculates a final score.
    """

    def __init__(self, processor_config: Dict[str, Any], primary_tickers: Optional[List[str]] = None):
        """
        Initializes the ArticleProcessor and its sub-components based on configuration.

        Args:
            processor_config (Dict[str, Any]): A dictionary containing configurations for all components,
                                                e.g., {'sentiment': {...}, 'summarizer': {...}}.
            primary_tickers (Optional[List[str]]): A list of primary tickers for scoring significance.
        """
        if not processor_config:
            raise ValueError("ArticleProcessor requires a valid configuration.")

        # --- 1. Initialize Sentiment Analyzer ---
        sentiment_config = processor_config.get('sentiment', {})
        analyzer_type = sentiment_config.get('analyzer', 'simple').lower()
        
        if analyzer_type == 'roberta' and 'roberta_model' in sentiment_config:
            self.sentiment_analyzer = RobertaSentimentAnalyzer(sentiment_config['roberta_model'])
            logger.info("Using RobertaSentimentAnalyzer for sentiment analysis.")
        else:
            self.sentiment_analyzer = SimpleSentimentAnalyzer(sentiment_config.get('simple_model'))
            logger.info("Using SimpleSentimentAnalyzer for sentiment analysis.")

        # --- 2. Initialize NLP Feature Extractors ---
        self.summarizer = Summarizer(processor_config.get('summarizer', {}))
        self.keyword_extractor = KeywordExtractor(processor_config.get('knowledge_base', {}))
        # In a real scenario, an entity extractor would be initialized here as well.
        # self.entity_extractor = EntityExtractor(processor_config.get('entity_model', {}))
        logger.info("NLP feature extractors (Summarizer, KeywordExtractor) initialized.")

        # --- 3. Initialize Scorer ---
        self.scorer = NewsScorer(processor_config.get('scoring'), primary_tickers)
        logger.info("NewsScorer initialized.")

    def process(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single article to enrich it with NLP features and a final score.

        Args:
            article (Dict[str, Any]): The article to process, containing 'content' or 'description'.

        Returns:
            Dict[str, Any]: The enriched article with new keys: 'sentiment', 'summary', 
                            'keywords', 'entities', and 'score'.
        """
        text = article.get('content') or article.get('description') or article.get('title')
        if not text:
            logger.warning(f"Article with title '{article.get('title', 'N/A')}' has no content to process.")
            article['score'] = 0.0
            return article

        # --- Step 1: Feature Extraction ---
        sentiment_result = self.sentiment_analyzer.analyze(text)
        summary = self.summarizer.summarize(text)
        
        # In a real implementation, you'd extract entities here.
        # For this example, we'll simulate it with keywords or predefined tickers.
        # entities = self.entity_extractor.extract(text)
        entities = self.keyword_extractor.extract_keywords(text) # Using keywords as a proxy for entities

        # Combine extracted keywords with any existing ones
        all_keywords = sorted(set(article.get('keywords', [])) | set(entities))

        # --- Step 2: Scoring ---
        final_score = self.scorer.score(
            sentiment_details=sentiment_result.get('details', {}),
            keywords=all_keywords,
            entities=entities # Passing keywords as a stand-in for entities
        )

        # --- Step 3: Assemble Final Result ---
        enriched_data = {
            'sentiment': sentiment_result,
            'summary': summary,
            'keywords': all_keywords,
            'entities': entities,
            'score': final_score
        }

        # Update the original article dictionary with the new data
        article.update(enriched_data)

        logger.info(
            f"Article processed. Score: {final_score:.3f}, Sentiment: {sentiment_result['label']}. "
            f"Title: '{article.get('title', 'N/A')[:50]}...'"
        )

        return article