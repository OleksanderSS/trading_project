
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.enrichers.nlp_features_enricher import NLPFeaturesEnricher

logger = ProjectLogger.get_logger("ValidationScript")

def test_nlp_enricher():
    logger.info("Starting NLP Enricher validation...")
    
    # 1. Initialize Enricher
    try:
        enricher = NLPFeaturesEnricher()
        logger.info("✅ NLPFeaturesEnricher initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize NLPFeaturesEnricher: {e}")
        return

    # 2. Create mock data
    df = pd.DataFrame({
        'datetime': pd.to_datetime(['2026-05-23 10:00:00', '2026-05-23 11:00:00']),
        'ticker': ['AAPL', 'AAPL'],
        'close': [150.0, 151.0]
    }).set_index('datetime')
    
    news_df = pd.DataFrame({
        'published_at': ['2026-05-23 09:30:00'],
        'title': ['Apple announces new product'],
        'ticker': ['AAPL'],
        'sentiment': [0.8]
    })
    
    # 3. Test Enrichment
    try:
        enriched_df = enricher.enrich(df, news=news_df)
        logger.info("✅ NLP enrichment executed successfully.")
        logger.info(f"Columns after enrichment: {enriched_df.columns.tolist()}")
    except Exception as e:
        logger.error(f"❌ NLP enrichment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nlp_enricher()
