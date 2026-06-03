#!/usr/bin/env python3
"""
Diagnose zero features in continue mode training.

This script analyzes why models are getting zero features during continue training.
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager

logger = ProjectLogger.get_logger(__name__)


class ZeroFeatureDiagnoser:
    """Diagnoses zero features in training data."""
    
    def __init__(self):
        self.config_manager = UnifiedConfigManager()
        self.data_manager = DataManager(self.config_manager)
        
    def analyze_zero_features(self) -> Dict[str, Any]:
        """Analyze zero features in the dataset."""
        results = {
            'total_records': 0,
            'zero_features': {},
            'feature_stats': {},
            'recommendations': []
        }
        
        try:
            # Load enriched features
            df = self.data_manager.fetch_data_from_table('enriched_features')
            
            if df is None or df.empty:
                logger.error("No enriched_features table found")
                return results
            
            results['total_records'] = len(df)
            
            # Analyze each feature
            for col in df.columns:
                if col in ['datetime', 'ticker', 'hash']:
                    continue
                    
                col_stats = self._analyze_column(df[col])
                results['feature_stats'][col] = col_stats
                
                # Check for excessive zeros
                if col_stats['zero_ratio'] > 0.8:  # More than 80% zeros
                    results['zero_features'][col] = col_stats
                    logger.warning(f"Feature {col} has {col_stats['zero_ratio']:.1%} zeros")
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing zero features: {e}")
            return results
    
    def _analyze_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a single column for zero patterns."""
        stats = {
            'total_count': len(series),
            'zero_count': 0,
            'nan_count': 0,
            'zero_ratio': 0.0,
            'nan_ratio': 0.0,
            'unique_values': 0,
            'data_type': str(series.dtype),
            'mean_value': 0.0,
            'std_value': 0.0,
            'min_value': 0.0,
            'max_value': 0.0
        }
        
        try:
            # Count zeros and NaNs
            stats['zero_count'] = (series == 0).sum()
            stats['nan_count'] = series.isna().sum()
            
            stats['zero_ratio'] = stats['zero_count'] / stats['total_count']
            stats['nan_ratio'] = stats['nan_count'] / stats['total_count']
            
            # Basic statistics for non-NaN values
            non_nan_series = series.dropna()
            if not non_nan_series.empty:
                stats['unique_values'] = non_nan_series.nunique()
                stats['mean_value'] = float(non_nan_series.mean())
                stats['std_value'] = float(non_nan_series.std())
                stats['min_value'] = float(non_nan_series.min())
                stats['max_value'] = float(non_nan_series.max())
            
        except Exception as e:
            logger.error(f"Error analyzing column: {e}")
        
        return stats
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> list:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        zero_features = results['zero_features']
        
        if not zero_features:
            recommendations.append("✅ No excessive zero features detected")
            return recommendations
        
        # Analyze patterns in zero features
        news_features = [f for f in zero_features if 'news' in f.lower()]
        sentiment_features = [f for f in zero_features if 'sentiment' in f.lower()]
        macro_features = [f for f in zero_features if 'fred_' in f.lower()]
        technical_features = [f for f in zero_features if any(x in f.lower() for x in ['rsi', 'macd', 'bollinger'])]
        
        if news_features:
            recommendations.append(f"📰 News features have excessive zeros: {news_features}")
            recommendations.append("   → Check NewsImpactEnricher._add_zero_scores() method")
            recommendations.append("   → Consider using last known scores instead of zeros")
        
        if sentiment_features:
            recommendations.append(f"😊 Sentiment features have excessive zeros: {sentiment_features}")
            recommendations.append("   → Check SentimentFeaturesEnricher fillna(0.0) calls")
            recommendations.append("   → Consider using neutral sentiment (0.5) instead of zeros")
        
        if macro_features:
            recommendations.append(f"📈 Macro features have excessive zeros: {macro_features}")
            recommendations.append("   → Check MacroFeaturesEnricher fillna() calls")
            recommendations.append("   → Consider using interpolation or last known values")
        
        if technical_features:
            recommendations.append(f"📊 Technical features have excessive zeros: {technical_features}")
            recommendations.append("   → Check technical indicator calculations")
            recommendations.append("   → Verify data quality and time alignment")
        
        # General recommendations
        recommendations.append("\n🔧 General fixes:")
        recommendations.append("1. Replace fillna(0) with smart interpolation")
        recommendations.append("2. Use last known values for time series")
        recommendations.append("3. Add data quality checks before enrichment")
        recommendations.append("4. Implement caching for missing data periods")
        
        return recommendations


def main():
    """Main diagnostic function."""
    print("🔍 Diagnosing zero features in continue mode...")
    
    diagnoser = ZeroFeatureDiagnoser()
    results = diagnoser.analyze_zero_features()
    
    print(f"\n📊 Analysis Results:")
    print(f"Total records: {results['total_records']:,}")
    print(f"Features with excessive zeros: {len(results['zero_features'])}")
    
    if results['zero_features']:
        print(f"\n🚨 Problem Features:")
        for feature, stats in results['zero_features'].items():
            print(f"  {feature}: {stats['zero_ratio']:.1%} zeros ({stats['zero_count']:,}/{stats['total_count']:,})")
    
    print(f"\n💡 Recommendations:")
    for rec in results['recommendations']:
        print(rec)
    
    # Save detailed report
    report_path = project_root / "zero_features_report.txt"
    with open(report_path, 'w') as f:
        f.write("Zero Features Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total records: {results['total_records']:,}\n")
        f.write(f"Features with excessive zeros: {len(results['zero_features'])}\n\n")
        
        for feature, stats in results['zero_features'].items():
            f.write(f"{feature}: {stats['zero_ratio']:.1%} zeros\n")
        
        f.write("\nRecommendations:\n")
        for rec in results['recommendations']:
            f.write(f"{rec}\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
