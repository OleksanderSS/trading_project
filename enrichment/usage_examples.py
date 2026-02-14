# enrichment/usage_examples.py

"""
Usage examples for the improved enrichment modules
"""

from enrichment.sentiment_factory import SentimentFactory
from enrichment.enrichment_config import EnrichmentConfig
import pandas as pd

def example_basic_usage():
    """Basic usage with automatic fallback"""
    print("=== Basic Usage Example ===")
    
    # Create analyzer with automatic fallback
    analyzer = SentimentFactory.create_analyzer()
    
    # Analyze text
    text = "Tesla stock is doing great today with strong growth!"
    sentiment, confidence = analyzer.predict(text)
    
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}")

def example_advanced_usage():
    """Advanced usage with specific model"""
    print("\n=== Advanced Usage Example ===")
    
    # Try to use RoBERTa analyzer
    analyzer = SentimentFactory.create_analyzer("roberta")
    
    # Batch processing
    texts = [
        "Tesla stock is rising!",
        "Market is crashing today",
        "Neutral market conditions"
    ]
    
    results = analyzer.predict_batch(texts)
    for i, (sentiment, confidence) in enumerate(results):
        print(f"Text {i+1}: {sentiment} ({confidence:.2f})")

def example_configuration():
    """Configuration example"""
    print("\n=== Configuration Example ===")
    
    # Get available models
    available = SentimentFactory.get_available_analyzers()
    print(f"Available analyzers: {available}")
    
    # Get model info
    info = SentimentFactory.get_analyzer_info()
    for name, model_info in info.items():
        print(f"{name}: {model_info}")

def example_entity_extraction():
    """Entity extraction example"""
    print("\n=== Entity Extraction Example ===")
    
    from enrichment.entity_extractor import EntityExtractor
    
    extractor = EntityExtractor()
    
    text = "Tesla (TSLA) and NVIDIA (NVDA) are showing strong performance in the market"
    entities = extractor.extract_entities(text)
    
    print(f"Text: {text}")
    print(f"Entities: {entities}")

def example_impact_analysis():
    """Impact analysis example"""
    print("\n=== Impact Analysis Example ===")
    
    from enrichment.reverse_impact_analyzer import ReverseImpactAnalyzer
    
    analyzer = ReverseImpactAnalyzer()
    
    # Sample data
    news_df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'sentiment': [0.05, -0.15],  # neutral, negative
        'text': ['Market update', 'Bad news']
    })
    
    price_df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'price_change': [0.01, -0.03]  # 1%, -3%
    })
    
    print(f"News data:\n{news_df}")
    print(f"Price data:\n{price_df}")
    print("Use ReverseImpactAnalyzer.analyze_daily_impact() for full analysis")

def example_factory_pattern():
    """Factory pattern example"""
    print("\n=== Factory Pattern Example ===")
    
    # Different analyzer preferences
    preferences = ["simple", "roberta", "nonexistent"]
    
    for pref in preferences:
        try:
            analyzer = SentimentFactory.create_analyzer(pref)
            print(f"Created {pref} analyzer: {analyzer.__class__.__name__}")
        except Exception as e:
            print(f"Failed to create {pref} analyzer: {e}")

if __name__ == "__main__":
    example_basic_usage()
    example_advanced_usage()
    example_configuration()
    example_entity_extraction()
    example_impact_analysis()
    example_factory_pattern()
