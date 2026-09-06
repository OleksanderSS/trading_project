# PatternAnalyzer Debug System Guide

## Overview

The PatternAnalyzer now includes a comprehensive debugging system that provides detailed insights into pattern analysis execution, data validation, and performance tracking.

## Features

### 1. Execution Tracking
- **Session Management**: Each analysis session is tracked with unique IDs
- **Step Timing**: Detailed timing for each analysis step
- **Performance Metrics**: Execution time for optimization
- **Error Logging**: Comprehensive error tracking and reporting

### 2. Data Validation
- **Input Data Logging**: Shape, type, and sample data verification
- **Data Quality Checks**: Null values, missing columns detection
- **Type Validation**: Ensure correct data types for analysis

### 3. Pattern Detection Debugging
- **Pattern Logging**: Detailed logging of detected patterns
- **Pattern Confidence**: Pattern strength and reliability metrics
- **Pattern Context**: Market conditions affecting pattern detection

### 4. Bias Calculation Transparency
- **Step-by-Step Calculation**: Detailed bias calculation process
- **Contribution Analysis**: How each pattern affects final bias
- **Adjustment Tracking**: News pattern adjustments and multipliers

### 5. Session Persistence
- **JSON Export**: Complete session data saved to files
- **Historical Analysis**: Compare sessions over time
- **Debug Replay**: Analyze past sessions for debugging

## Usage

### Basic Usage

```python
from src.patterns.pattern_analyzer import PatternAnalyzer

# Initialize with debugging enabled
analyzer = PatternAnalyzer(enable_debug=True)

# Run analysis with comprehensive debugging
data = {
    'price_data': price_dataframe,
    'news_data': news_list,
    'market_metrics': market_dict
}
results = analyzer.analyze(data)
```

### Debug Session Management

```python
# Access debug information
debugger = analyzer.debugger

# Check recent logs
recent_logs = debugger.get_debug_logs(lines=10)

# Load previous session
session_data = debugger.load_session("pattern_143022")
```

## Debug Output Structure

### Session Files
Debug sessions are saved to `logs/pattern_debug/` with JSON structure:

```json
{
  "session_info": {
    "start_time": "2024-01-20T14:30:22.123456",
    "total_time": 0.234,
    "step_times": {
      "price_pattern_detection": 0.045,
      "news_pattern_detection": 0.012,
      "regime_detection": 0.008,
      "signal_bias_calculation": 0.003,
      "fractal_similarity_analysis": 0.166
    }
  },
  "debug_log": [
    {
      "timestamp": "2024-01-20T14:30:22.123456",
      "level": "INFO",
      "message": "Pattern analysis started: pattern_143022",
      "data": null
    }
  ],
  "final_results": {
    "price_patterns": {...},
    "news_patterns": {...},
    "signal_bias": 0.234,
    "regime_warnings": ["HIGH_VOLATILITY_REGIME"]
  }
}
```

### Log Levels

- **DEBUG**: Detailed execution information
- **INFO**: Major milestones and results
- **WARNING**: Data quality issues and missing data
- **ERROR**: Exceptions and critical issues

## Analysis Steps

### 1. Price Pattern Detection
```python
# Debug output example:
[PATTERN_DEBUG] Starting price pattern detection
[PATTERN_DEBUG] Data info for 'price_df': {
  "type": "DataFrame",
  "shape": (100, 5),
  "columns": ["open", "high", "low", "close", "volume"],
  "null_count": 0
}
[PATTERN_DEBUG] Detected price patterns: ["bullish_pinbar"]
[PATTERN_DEBUG] price pattern 'bullish_pinbar': 100
[PATTERN_DEBUG] Step 'price_pattern_detection' completed in 0.045s
```

### 2. News Pattern Detection
```python
# Debug output example:
[PATTERN_DEBUG] Starting news pattern detection
[PATTERN_DEBUG] Data info for 'news_list': {
  "type": "List",
  "length": 4,
  "sample": {...}
}
[PATTERN_DEBUG] Detected news patterns: ["ai_euphoria", "geopolitical_risk"]
[PATTERN_DEBUG] news pattern 'ai_euphoria': 2
[PATTERN_DEBUG] Step 'news_pattern_detection' completed in 0.012s
```

### 3. Bias Calculation
```python
# Debug output example:
[PATTERN_DEBUG] Starting signal bias calculation
[PATTERN_DEBUG] Bias calculation completed: 0.234
[PATTERN_DEBUG] Calculation steps:
  - Price pattern 'bullish_pinbar': value=100, contribution=0.2, running_total=0.2
  - AI euphoria synergy: multiplier=1.2, before=0.2, after=0.24
  - Geopolitical risk adjustment: adjustment=-0.3, before=0.24, after=-0.06
  - Final bias: 0.234 (clamped)
```

## Configuration

### Enable/Disable Debugging
```python
# Enable debugging (default)
analyzer = PatternAnalyzer(enable_debug=True)

# Disable debugging for production
analyzer = PatternAnalyzer(enable_debug=False)
```

### Custom Debug Directory
```python
# Modify PatternDebugger to use custom directory
class CustomPatternDebugger(PatternDebugger):
    def __init__(self, enable_debug=True):
        super().__init__(enable_debug)
        self.debug_dir = Path("custom_debug_logs")
```

## Performance Analysis

### Step Timing
The debug system tracks execution time for each step:

1. **price_pattern_detection**: Candlestick and chart pattern analysis
2. **news_pattern_detection**: News theme analysis
3. **regime_detection**: Market regime identification
4. **signal_bias_calculation**: Bias computation
5. **fractal_similarity_analysis**: Historical pattern matching

### Optimization Opportunities
- Identify slow steps for optimization
- Compare performance across different data sizes
- Monitor memory usage during analysis
- Track pattern detection accuracy

## Troubleshooting

### Common Issues

1. **Missing Data**: Check data validation logs
2. **Pattern Detection Failures**: Review pattern calculation steps
3. **Performance Issues**: Analyze step timing
4. **Bias Calculation Errors**: Check bias calculation details

### Debug Commands

```python
# Check if debugging is enabled
if analyzer.debugger.enable_debug:
    print("Debugging is enabled")

# Get recent debug logs
logs = analyzer.debugger.debug_log[-10:]

# Check step times
times = analyzer.debugger.step_times
```

## Integration Examples

### With Pipeline Integration
```python
class TradingPipeline:
    def __init__(self):
        self.pattern_analyzer = PatternAnalyzer(enable_debug=True)
    
    def analyze_market(self, data):
        # Pattern analysis with debugging
        patterns = self.pattern_analyzer.analyze(data)
        
        # Access debug information
        if self.pattern_analyzer.debugger.enable_debug:
            debug_info = {
                'execution_time': self.pattern_analyzer.debugger.step_times,
                'patterns_detected': patterns
            }
        
        return patterns
```

### With Model Training
```python
def train_with_pattern_analysis(training_data):
    analyzer = PatternAnalyzer(enable_debug=True)
    
    # Analyze patterns in training data
    patterns = analyzer.analyze(training_data)
    
    # Use pattern insights for model training
    if patterns['signal_bias'] > 0.5:
        # Adjust training strategy for bullish bias
        pass
    
    return patterns
```

## Best Practices

1. **Enable Debugging in Development**: Use debugging during development and testing
2. **Disable in Production**: Turn off debugging for production performance
3. **Regular Session Review**: Periodically review debug sessions for insights
4. **Performance Monitoring**: Track step times for optimization opportunities
5. **Data Validation**: Always check data validation logs for quality issues

## Advanced Features

### Custom Debug Handlers
```python
class CustomPatternDebugger(PatternDebugger):
    def log_pattern_insights(self, patterns, context):
        """Custom pattern analysis logging"""
        insights = self.analyze_pattern_context(patterns, context)
        self.log("INFO", "Pattern insights", insights)
    
    def analyze_pattern_context(self, patterns, context):
        """Analyze patterns in market context"""
        return {
            'market_regime': context.get('regime'),
            'pattern_strength': self.calculate_pattern_strength(patterns),
            'reliability_score': self.assess_reliability(patterns)
        }
```

### Real-time Monitoring
```python
class RealTimePatternMonitor:
    def __init__(self):
        self.analyzer = PatternAnalyzer(enable_debug=True)
        self.alert_thresholds = {
            'bias_extreme': 0.8,
            'regime_change_count': 3
        }
    
    def monitor_patterns(self, data):
        results = self.analyzer.analyze(data)
        
        # Check for alerts
        if abs(results['signal_bias']) > self.alert_thresholds['bias_extreme']:
            self.trigger_bias_alert(results)
        
        if len(results['regime_warnings']) > self.alert_thresholds['regime_change_count']:
            self.trigger_regime_alert(results)
        
        return results
```

## Conclusion

The PatternAnalyzer debug system provides comprehensive insights into pattern analysis execution, making it easier to:

- **Debug Issues**: Identify and resolve analysis problems
- **Optimize Performance**: Track and improve execution speed
- **Validate Results**: Ensure analysis accuracy and reliability
- **Monitor Quality**: Track data quality and pattern detection effectiveness

Use this system to develop, test, and optimize your trading pattern analysis pipeline.
