import os
import sys

import pandas as pd

# Додаємо кореневу директорію проекту до sys.path
sys.path.append(os.getcwd())

from src.core.logging.logger import ProjectLogger
from src.sentiment.sentiment_models import aggregate_sentiment, analyze_sentiment


def test_sentiment_analysis():
    logger = ProjectLogger.get_logger("TestSentiment")
    logger.info("--- Testing Sentiment Analysis ---")
    
    texts = [
        "Apple reports record-breaking earnings this quarter, exceeding analyst expectations.",
        "The market is crashing and everyone is panicking about the recession.",
        "The weather is fine today, nothing much happening in the markets.",
        "" # Empty string test
    ]
    
    # Test analyze_sentiment
    # Note: This might take a while if it downloads the model
    logger.info("Starting analysis (might download model)...")
    results = analyze_sentiment(texts)
    
    logger.info("Analysis Results:")
    print(results)
    
    assert len(results) == len(texts)
    assert "label" in results.columns
    assert "score" in results.columns
    
    # Test aggregate_sentiment
    agg = aggregate_sentiment(results)
    logger.info(f"Aggregated Sentiment: {agg}")
    
    assert "positive" in agg
    assert "negative" in agg
    assert "neutral" in agg
    
    # Test caching
    logger.info("Testing cache (should be instant)...")
    results_cached = analyze_sentiment(texts)
    pd.testing.assert_frame_equal(results, results_cached)
    logger.info("Cache test passed!")

def test_aggregation_methods():
    logger = ProjectLogger.get_logger("TestSentiment")
    logger.info("--- Testing Aggregation Methods ---")
    
    data = [
        {"text": "t1", "label": "positive", "score": 0.9},
        {"text": "t2", "label": "positive", "score": 0.8},
        {"text": "t3", "label": "negative", "score": 0.7},
        {"text": "t4", "label": "neutral", "score": 0.5},
    ]
    df = pd.DataFrame(data)
    
    # Mean
    agg_mean = aggregate_sentiment(df, method="mean", normalize=False)
    logger.info(f"Mean (non-normalized): {agg_mean}")
    # Total count = 4. Positive sum = 1.7. 1.7/4 = 0.425
    assert abs(agg_mean["positive"] - 0.425) < 1e-6
    
    # Sum
    agg_sum = aggregate_sentiment(df, method="sum", normalize=False)
    logger.info(f"Sum: {agg_sum}")
    assert agg_sum["positive"] == 1.7
    
    # Count
    agg_count = aggregate_sentiment(df, method="count", normalize=False)
    logger.info(f"Count: {agg_count}")
    assert agg_count["positive"] == 2
    
    logger.info("Aggregation methods tests passed!")

if __name__ == "__main__":
    try:
        # We might want to mock the pipeline if we don't want to download 400MB
        # but for a real audit, it's better to see if it works.
        # However, in a restricted environment, I'll check if torch/transformers are available first.
        
        try:
            import torch
            import transformers
            HAS_LIBS = True
        except ImportError:
            HAS_LIBS = False
            print("Torch or Transformers not found. Test will run in fallback mode.")
            
        test_sentiment_analysis()
        test_aggregation_methods()
        print("\nALL SENTIMENT TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nTESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
