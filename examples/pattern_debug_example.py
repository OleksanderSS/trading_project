"""
PATTERN ANALYZER DEBUG EXAMPLE
Demonstrates how to use the PatternAnalyzer with comprehensive debugging.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.patterns.pattern_analyzer import PatternAnalyzer

def create_sample_data():
    """Create sample data for pattern analysis demonstration."""
    
    # Create sample price data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    rng = np.random.default_rng(42)
    
    # Generate realistic price data with some patterns
    base_price = 100
    returns = rng.normal(0.001, 0.02, 100)  # Daily returns
    prices = base_price * (1 + np.cumsum(returns))
    
    # Add some candlestick patterns
    # Create a bullish pinbar around day 50
    pinbar_idx = 50
    if pinbar_idx < len(prices):
        # Make lower wick much longer than body
        prices[pinbar_idx] = prices[pinbar_idx] * 0.98  # Lower close
        high_prices = prices.copy()
        low_prices = prices.copy()
        high_prices[pinbar_idx] = prices[pinbar_idx] * 1.05  # High wick
        low_prices[pinbar_idx] = prices[pinbar_idx] * 0.92  # Long lower wick
    else:
        high_prices = prices
        low_prices = prices
    
    price_df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + rng.normal(0, 0.005, 100)),
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': rng.integers(1000, 10000, 100)
    })
    price_df.set_index('date', inplace=True)
    
    # Create sample news data
    news_data = [
        {
            'title': 'AI Technology Breakthrough Announced by NVIDIA',
            'text': 'NVIDIA announced groundbreaking AI technology that could revolutionize the industry. Artificial intelligence stocks surge on positive sentiment.',
            'published_at': datetime.now() - timedelta(days=1)
        },
        {
            'title': 'Federal Reserve Considers Rate Hike',
            'text': 'The Federal Reserve is considering another rate hike to combat inflation. Fed officials indicate monetary tightening may continue.',
            'published_at': datetime.now() - timedelta(hours=12)
        },
        {
            'title': 'Geopolitical Tensions Rise in Global Markets',
            'text': 'Geopolitical conflict escalates as war tensions affect global trade. Market volatility expected due to geopolitical risks.',
            'published_at': datetime.now() - timedelta(hours=6)
        },
        {
            'title': 'Market Volatility Concerns Grow',
            'text': 'Market correction fears grow as volatility spikes. Investors worry about potential market crash and economic uncertainty.',
            'published_at': datetime.now() - timedelta(hours=2)
        }
    ]
    
    # Create market metrics
    market_metrics = {
        'vix': 32.5,  # High volatility
        'tech_concentration': 0.75,  # High tech concentration
        'market_cap': 2500000000000,  # 2.5T market cap
        'volume': 5000000000  # 5B daily volume
    }
    
    return price_df, news_data, market_metrics

def demonstrate_pattern_analysis():
    """Demonstrate PatternAnalyzer with comprehensive debugging."""
    
    print("=" * 80)
    print("PATTERN ANALYZER DEBUG DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    print("\n1. Creating sample data...")
    price_df, news_data, market_metrics = create_sample_data()
    
    print(f"Price data shape: {price_df.shape}")
    print(f"News articles: {len(news_data)}")
    print(f"Latest price: ${price_df['close'].iloc[-1]:.2f}")
    
    # Initialize PatternAnalyzer with debugging enabled
    print("\n2. Initializing PatternAnalyzer with debugging...")
    analyzer = PatternAnalyzer(enable_debug=True)
    
    # Prepare input data
    input_data = {
        'price_data': price_df,
        'news_data': news_data,
        'market_metrics': market_metrics
    }
    
    # Run pattern analysis
    print("\n3. Running pattern analysis with comprehensive debugging...")
    print("   (Check logs/pattern_debug/ directory for detailed debug logs)")
    
    results = analyzer.analyze(input_data)
    
    # Display results
    print("\n4. Pattern Analysis Results:")
    print("-" * 40)
    
    if 'price_patterns' in results:
        print(f"Price Patterns: {results['price_patterns']}")
    
    if 'news_patterns' in results:
        print(f"News Patterns: {results['news_patterns']}")
    
    if 'regime_warnings' in results:
        print(f"Regime Warnings: {results['regime_warnings']}")
        print(f"Market State: {results['market_state']}")
    
    if 'signal_bias' in results:
        print(f"Signal Bias: {results['signal_bias']:.3f}")
    
    if 'fractal_match' in results:
        fractal = results['fractal_match']
        if fractal:
            print(f"Fractal Similarity: {fractal['similarity_score']:.3f}")
            print(f"Historical Outcome: {fractal['historical_outcome']:.4f}")
    
    print(f"Analysis Timestamp: {results['analysis_timestamp']}")
    
    return results

def demonstrate_debug_session_analysis():
    """Analyze the debug session logs."""
    
    print("\n5. Debug Session Analysis:")
    print("-" * 40)
    
    # The debug logs are saved to logs/pattern_debug/ directory
    debug_dir = "logs/pattern_debug"
    
    try:
        import os
        if os.path.exists(debug_dir):
            debug_files = [f for f in os.listdir(debug_dir) if f.endswith('.json')]
            if debug_files:
                latest_file = max(debug_files)
                print(f"Latest debug session: {latest_file}")
                print(f"Debug directory: {debug_dir}")
                print("\nTo view detailed debug logs:")
                print(f"1. Open: {debug_dir}/{latest_file}")
                print("2. Look for sections:")
                print("   - session_info: Timing and execution summary")
                print("   - debug_log: Step-by-step execution details")
                print("   - final_results: Complete analysis results")
            else:
                print("No debug files found yet.")
        else:
            print(f"Debug directory {debug_dir} not found.")
    except Exception as e:
        print(f"Error analyzing debug session: {e}")

def demonstrate_pattern_insights():
    """Provide insights from the pattern analysis."""
    
    print("\n6. Pattern Analysis Insights:")
    print("-" * 40)
    
    print("Key Features of the Debug System:")
    print("1. Step-by-step execution tracking")
    print("2. Data shape and type validation")
    print("3. Pattern detection logging")
    print("4. Bias calculation transparency")
    print("5. Performance timing for each step")
    print("6. Complete session persistence")
    
    print("\nDebug Information Available:")
    print("- Input data validation")
    print("- Price pattern detection details")
    print("- News theme analysis results")
    print("- Regime detection reasoning")
    print("- Signal bias calculation steps")
    print("- Fractal similarity analysis")
    
    print("\nHow to Use Debug Information:")
    print("1. Check debug logs for execution flow")
    print("2. Verify data input quality")
    print("3. Understand pattern detection results")
    print("4. Debug bias calculation issues")
    print("5. Optimize performance bottlenecks")
    print("6. Validate analysis completeness")

if __name__ == "__main__":
    try:
        # Run the demonstration
        results = demonstrate_pattern_analysis()
        demonstrate_debug_session_analysis()
        demonstrate_pattern_insights()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext Steps:")
        print("1. Review debug logs in logs/pattern_debug/")
        print("2. Integrate PatternAnalyzer in your pipeline")
        print("3. Customize debug settings as needed")
        print("4. Monitor performance with debug timing")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
