#!/usr/bin/env python3
"""
Diagnose feature selection issues in continue mode.

This script analyzes why models are getting limited features during continue training.
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager

logger = ProjectLogger.get_logger(__name__)


class FeatureSelectionDiagnoser:
    """Diagnose feature selection issues in continue mode."""
    
    def __init__(self):
        self.config_manager = UnifiedConfigManager()
        self.data_manager = DataManager(self.config_manager)
        
    def analyze_feature_selection(self) -> Dict[str, Any]:
        """Analyze feature selection issues."""
        results = {
            'total_tickers': 0,
            'ticker_data_stats': {},
            'feature_counts': {},
            'model_feature_limits': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Load enriched features
            df = self.data_manager.fetch_data_from_table('enriched_features')
            
            if df is None or df.empty:
                logger.error("No enriched_features table found")
                return results
            
            # Analyze ticker data distribution
            if 'ticker' in df.columns:
                ticker_stats = df['ticker'].value_counts()
                results['total_tickers'] = len(ticker_stats)
                results['ticker_data_stats'] = ticker_stats.to_dict()
                
                # Check for data imbalance
                min_data = ticker_stats.min()
                max_data = ticker_stats.max()
                data_ratio = max_data / min_data if min_data > 0 else float('inf')
                
                if data_ratio > 10:
                    results['issues'].append(f"Data imbalance: {data_ratio:.1f}x between most/least data")
                    results['recommendations'].append("Consider balancing data across tickers")
                
                # Check for insufficient data
                insufficient_tickers = ticker_stats[ticker_stats < 100].index.tolist()
                if insufficient_tickers:
                    results['issues'].append(f"Insufficient data for {len(insufficient_tickers)} tickers (<100 rows)")
                    results['recommendations'].append(f"Collect more data for: {', '.join(insufficient_tickers[:5])}")
            
            # Analyze feature counts per model type
            results['model_feature_limits'] = self._get_model_feature_limits()
            
            # Check actual feature count
            numeric_features = df.select_dtypes(include=[np.number]).columns
            results['feature_counts'] = {
                'total_features': len(df.columns),
                'numeric_features': len(numeric_features),
                'text_features': len(df.columns) - len(numeric_features)
            }
            
            # Check for feature selection bottlenecks
            for model_type, max_features in results['model_feature_limits'].items():
                if len(numeric_features) < max_features:
                    results['issues'].append(f"Insufficient features for {model_type}: {len(numeric_features)} < {max_features}")
                    results['recommendations'].append(f"Generate more features or reduce max_features for {model_type}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing feature selection: {e}")
            return results
    
    def _get_model_feature_limits(self) -> Dict[str, int]:
        """Get feature limits from config."""
        return {
            'mlp': 256,
            'lstm': 128,
            'gru': 128,
            'cnn': 64,
            'transformer': 128,
            'tabnet': 256,
            'autoencoder': 128,
            'catboost': 45,
            'lightgbm': 50,
            'xgboost': 48,
            'random_forest': 40,
            'linear': 35,
            'svm': 42,
            'knn': 38
        }
    
    def analyze_cache_issues(self) -> Dict[str, Any]:
        """Analyze potential caching issues."""
        results = {
            'cache_status': 'unknown',
            'cache_size': 0,
            'cache_tickers': [],
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check cache directory
            cache_dir = project_root / "data" / "cache"
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.json"))
                results['cache_size'] = len(cache_files)
                results['cache_status'] = 'exists'
                
                # Analyze cache content
                for cache_file in cache_files:
                    try:
                        import json
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                            if 'ticker' in cache_data:
                                results['cache_tickers'].append(cache_data['ticker'])
                    except (json.JSONDecodeError, KeyError):
                        continue
            else:
                results['cache_status'] = 'missing'
                results['issues'].append("Cache directory not found")
                results['recommendations'].append("Create cache directory for feature selection")
            
            # Check if cache covers all tickers
            if 'total_tickers' in self.analyze_feature_selection():
                total_tickers = self.analyze_feature_selection()['total_tickers']
                cached_tickers = len(set(results['cache_tickers']))
                
                if cached_tickers < total_tickers:
                    results['issues'].append(f"Cache covers only {cached_tickers}/{total_tickers} tickers")
                    results['recommendations'].append("Generate cache for all tickers")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing cache: {e}")
            return results


def main():
    """Main diagnostic function."""
    print("🔍 Diagnosing feature selection issues in continue mode...")
    
    diagnoser = FeatureSelectionDiagnoser()
    
    # Analyze feature selection
    selection_results = diagnoser.analyze_feature_selection()
    
    print(f"\n📊 Feature Selection Analysis:")
    print(f"Total tickers: {selection_results['total_tickers']}")
    print(f"Total features: {selection_results['feature_counts']['total_features']}")
    print(f"Numeric features: {selection_results['feature_counts']['numeric_features']}")
    
    if selection_results['ticker_data_stats']:
        print(f"\n📈 Ticker Data Distribution:")
        for ticker, count in list(selection_results['ticker_data_stats'].items())[:5]:
            print(f"  {ticker}: {count:,} rows")
    
    print(f"\n🎯 Model Feature Limits:")
    for model, limit in list(selection_results['model_feature_limits'].items())[:5]:
        available = selection_results['feature_counts']['numeric_features']
        status = "✅" if available >= limit else "❌"
        print(f"  {model}: {limit} features (available: {available}) {status}")
    
    if selection_results['issues']:
        print(f"\n🚨 Issues Found:")
        for issue in selection_results['issues']:
            print(f"  ❌ {issue}")
    
    if selection_results['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in selection_results['recommendations']:
            print(f"  💡 {rec}")
    
    # Analyze cache issues
    cache_results = diagnoser.analyze_cache_issues()
    
    print(f"\n🗂️ Cache Analysis:")
    print(f"Cache status: {cache_results['cache_status']}")
    print(f"Cache size: {cache_results['cache_size']} files")
    print(f"Cached tickers: {len(cache_results['cache_tickers'])}")
    
    if cache_results['issues']:
        print(f"\n🚨 Cache Issues:")
        for issue in cache_results['issues']:
            print(f"  ❌ {issue}")
    
    if cache_results['recommendations']:
        print(f"\n💡 Cache Recommendations:")
        for rec in cache_results['recommendations']:
            print(f"  💡 {rec}")
    
    # Save detailed report
    report_path = project_root / "feature_selection_diagnosis.txt"
    with open(report_path, 'w') as f:
        f.write("Feature Selection Diagnosis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total tickers: {selection_results['total_tickers']}\n")
        f.write(f"Total features: {selection_results['feature_counts']['total_features']}\n")
        f.write(f"Numeric features: {selection_results['feature_counts']['numeric_features']}\n\n")
        
        f.write("Model Feature Limits:\n")
        for model, limit in selection_results['model_feature_limits'].items():
            available = selection_results['feature_counts']['numeric_features']
            f.write(f"{model}: {limit} (available: {available})\n")
        
        if selection_results['issues']:
            f.write("\nIssues:\n")
            for issue in selection_results['issues']:
                f.write(f"- {issue}\n")
        
        if selection_results['recommendations']:
            f.write("\nRecommendations:\n")
            for rec in selection_results['recommendations']:
                f.write(f"- {rec}\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
