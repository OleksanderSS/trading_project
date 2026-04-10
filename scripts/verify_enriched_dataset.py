#!/usr/bin/env python3
"""
Enriched Dataset Verification Script
Перевірка структури збагаченого датасету та event-series формату

Перевіряє:
1. Структуру даних (columns, types)
2. Event-series формат
3. Цілісність даних
4. Наявність усіх збагачувачів
"""

import sys
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager

logger = ProjectLogger.get_logger("DatasetVerification")

class EnrichedDatasetVerifier:
    """Верифікатор структури збагаченого датасету"""
    
    def __init__(self, config_manager: UnifiedConfigManager):
        self.config_manager = config_manager
        self.data_manager = DataManager(config_manager)
        self.verification_report = {}
    
    def run(self) -> dict:
        """Запустити повну перевірку"""
        logger.info("=" * 80)
        logger.info("🔍 ENRICHED DATASET VERIFICATION - Starting")
        logger.info("=" * 80)
        
        try:
            # 1. Check DuckDB tables
            logger.info("\n[Step 1] Checking DuckDB tables...")
            self._verify_tables()
            
            # 2. Check raw_data structure
            logger.info("\n[Step 2] Checking raw_data structure...")
            self._verify_raw_data()
            
            # 3. Check enriched_features structure
            logger.info("\n[Step 3] Checking enriched_features structure...")
            self._verify_enriched_features()
            
            # 4. Check event-series format
            logger.info("\n[Step 4] Checking event-series format...")
            self._verify_event_series_format()
            
            # 5. Check data integrity
            logger.info("\n[Step 5] Checking data integrity...")
            self._verify_data_integrity()
            
            # 6. Check enricher coverage
            logger.info("\n[Step 6] Checking enricher coverage...")
            self._verify_enricher_coverage()
            
            # 7. Generate summary
            logger.info("\n[Summary] Generating verification report...")
            summary = self._generate_summary()
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ VERIFICATION COMPLETED")
            logger.info("=" * 80)
            
            return summary
            
        except Exception as e:
            logger.exception(f"❌ Verification failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _verify_tables(self):
        """Перевірити наявність таблиць"""
        tables = self.data_manager.get_all_table_names()
        logger.info(f"Found {len(tables)} tables: {tables}")
        
        self.verification_report['tables'] = {
            'count': len(tables),
            'names': tables,
            'has_raw_data': 'raw_data' in tables,
            'has_enriched_features': 'enriched_features' in tables,
            'has_targets': 'targets' in tables
        }
        
        if not tables:
            logger.warning("⚠️  No tables found in DuckDB!")
        else:
            logger.info(f"✓ Found {len(tables)} tables")
    
    def _verify_raw_data(self):
        """Перевірити структуру raw_data"""
        raw_df = self.data_manager.fetch_data_from_table('raw_data')
        
        if raw_df is None or raw_df.empty:
            logger.warning("⚠️  raw_data table is empty or not found")
            self.verification_report['raw_data'] = {'status': 'empty'}
            return
        
        # Expected columns for raw data
        expected_cols = ['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        report = {
            'rows': len(raw_df),
            'columns': len(raw_df.columns),
            'column_names': list(raw_df.columns),
            'dtypes': {col: str(raw_df[col].dtype) for col in raw_df.columns},
            'has_expected_columns': all(col in raw_df.columns for col in expected_cols),
            'missing_columns': [col for col in expected_cols if col not in raw_df.columns],
            'null_counts': raw_df.isnull().sum().to_dict(),
            'date_range': {
                'min': str(raw_df['timestamp'].min()) if 'timestamp' in raw_df.columns else None,
                'max': str(raw_df['timestamp'].max()) if 'timestamp' in raw_df.columns else None
            }
        }
        
        self.verification_report['raw_data'] = report
        
        logger.info(f"✓ raw_data: {report['rows']} rows, {report['columns']} columns")
        logger.info(f"  Expected columns: {expected_cols}")
        logger.info(f"  Has all expected: {report['has_expected_columns']}")
        
        if report['missing_columns']:
            logger.warning(f"  ⚠️  Missing columns: {report['missing_columns']}")
        
        # Check for nulls
        null_cols = [col for col, count in report['null_counts'].items() if count > 0]
        if null_cols:
            logger.warning(f"  ⚠️  Columns with nulls: {null_cols}")
    
    def _verify_enriched_features(self):
        """Перевірити структуру enriched_features"""
        enriched_df = self.data_manager.fetch_data_from_table('enriched_features')
        
        if enriched_df is None or enriched_df.empty:
            logger.warning("⚠️  enriched_features table is empty or not found")
            self.verification_report['enriched_features'] = {'status': 'empty'}
            return
        
        # Base columns (from raw data)
        base_cols = ['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Enriched columns (added by enrichers)
        enriched_cols = [col for col in enriched_df.columns if col not in base_cols]
        
        report = {
            'rows': len(enriched_df),
            'columns': len(enriched_df.columns),
            'base_columns': len(base_cols),
            'enriched_columns': len(enriched_cols),
            'total_column_names': list(enriched_df.columns),
            'enriched_column_names': enriched_cols[:20],  # First 20
            'dtypes_sample': {col: str(enriched_df[col].dtype) for col in enriched_df.columns[:10]},
            'null_counts': enriched_df.isnull().sum().to_dict(),
            'memory_usage_mb': enriched_df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        self.verification_report['enriched_features'] = report
        
        logger.info(f"✓ enriched_features: {report['rows']} rows, {report['columns']} columns")
        logger.info(f"  Base columns: {report['base_columns']}")
        logger.info(f"  Enriched columns: {report['enriched_columns']}")
        logger.info(f"  Memory usage: {report['memory_usage_mb']:.2f} MB")
        
        # Show sample enriched columns
        if enriched_cols:
            logger.info(f"  Sample enriched columns: {enriched_cols[:10]}")
    
    def _verify_event_series_format(self):
        """Перевірити event-series формат"""
        enriched_df = self.data_manager.fetch_data_from_table('enriched_features')
        
        if enriched_df is None or enriched_df.empty:
            logger.warning("⚠️  Cannot verify event-series format: no data")
            self.verification_report['event_series_format'] = {'status': 'no_data'}
            return
        
        # Event-series format requirements:
        # 1. Has timestamp column
        # 2. Has ticker column
        # 3. Rows are sorted by timestamp
        # 4. Each row represents one event (one time point for one ticker)
        
        has_timestamp = 'timestamp' in enriched_df.columns
        has_ticker = 'ticker' in enriched_df.columns
        
        # Check if sorted by timestamp
        if has_timestamp:
            is_sorted = enriched_df['timestamp'].is_monotonic_increasing
        else:
            is_sorted = False
        
        # Check for duplicate events (same ticker + timestamp)
        if has_ticker and has_timestamp:
            duplicates = enriched_df.groupby(['ticker', 'timestamp']).size()
            has_duplicates = (duplicates > 1).any()
            duplicate_count = (duplicates > 1).sum()
        else:
            has_duplicates = False
            duplicate_count = 0
        
        report = {
            'has_timestamp': has_timestamp,
            'has_ticker': has_ticker,
            'is_sorted_by_timestamp': is_sorted,
            'has_duplicate_events': has_duplicates,
            'duplicate_event_count': duplicate_count,
            'is_valid_event_series': has_timestamp and has_ticker and is_sorted and not has_duplicates
        }
        
        self.verification_report['event_series_format'] = report
        
        logger.info(f"✓ Event-series format check:")
        logger.info(f"  Has timestamp: {report['has_timestamp']}")
        logger.info(f"  Has ticker: {report['has_ticker']}")
        logger.info(f"  Sorted by timestamp: {report['is_sorted_by_timestamp']}")
        logger.info(f"  Has duplicates: {report['has_duplicate_events']}")
        logger.info(f"  Valid event-series: {report['is_valid_event_series']}")
        
        if not report['is_valid_event_series']:
            logger.warning("⚠️  Dataset is NOT in valid event-series format!")
    
    def _verify_data_integrity(self):
        """Перевірити цілісність даних"""
        enriched_df = self.data_manager.fetch_data_from_table('enriched_features')
        
        if enriched_df is None or enriched_df.empty:
            logger.warning("⚠️  Cannot verify data integrity: no data")
            self.verification_report['data_integrity'] = {'status': 'no_data'}
            return
        
        report = {
            'total_rows': len(enriched_df),
            'total_columns': len(enriched_df.columns),
            'null_percentage': (enriched_df.isnull().sum().sum() / (len(enriched_df) * len(enriched_df.columns))) * 100,
            'duplicate_rows': enriched_df.duplicated().sum(),
            'numeric_columns': len(enriched_df.select_dtypes(include=[np.number]).columns),
            'object_columns': len(enriched_df.select_dtypes(include=['object']).columns),
            'datetime_columns': len(enriched_df.select_dtypes(include=['datetime64']).columns)
        }
        
        self.verification_report['data_integrity'] = report
        
        logger.info(f"✓ Data integrity check:")
        logger.info(f"  Total rows: {report['total_rows']}")
        logger.info(f"  Total columns: {report['total_columns']}")
        logger.info(f"  Null percentage: {report['null_percentage']:.2f}%")
        logger.info(f"  Duplicate rows: {report['duplicate_rows']}")
        logger.info(f"  Numeric columns: {report['numeric_columns']}")
        logger.info(f"  Object columns: {report['object_columns']}")
        logger.info(f"  Datetime columns: {report['datetime_columns']}")
        
        if report['null_percentage'] > 10:
            logger.warning(f"⚠️  High null percentage: {report['null_percentage']:.2f}%")
        
        if report['duplicate_rows'] > 0:
            logger.warning(f"⚠️  Found {report['duplicate_rows']} duplicate rows")
    
    def _verify_enricher_coverage(self):
        """Перевірити покриття збагачувачів"""
        enriched_df = self.data_manager.fetch_data_from_table('enriched_features')
        
        if enriched_df is None or enriched_df.empty:
            logger.warning("⚠️  Cannot verify enricher coverage: no data")
            self.verification_report['enricher_coverage'] = {'status': 'no_data'}
            return
        
        # Expected enricher patterns
        enricher_patterns = {
            'technical_indicators': ['rsi', 'macd', 'bb_', 'atr', 'ema', 'sma'],
            'volatility': ['volatility', 'std_dev', 'range'],
            'momentum': ['momentum', 'roc', 'stoch'],
            'volume': ['volume_', 'obv', 'ad_'],
            'context_map': ['context_', 'regime', 'market_phase'],
            'sentiment': ['sentiment', 'news_', 'social_'],
            'macro': ['macro_', 'fred_', 'economic_']
        }
        
        report = {}
        for enricher_name, patterns in enricher_patterns.items():
            matching_cols = [col for col in enriched_df.columns 
                           if any(pattern.lower() in col.lower() for pattern in patterns)]
            report[enricher_name] = {
                'found': len(matching_cols) > 0,
                'column_count': len(matching_cols),
                'sample_columns': matching_cols[:5]
            }
        
        self.verification_report['enricher_coverage'] = report
        
        logger.info(f"✓ Enricher coverage check:")
        for enricher_name, info in report.items():
            status = "✓" if info['found'] else "✗"
            logger.info(f"  {status} {enricher_name}: {info['column_count']} columns")
            if info['sample_columns']:
                logger.info(f"     Sample: {info['sample_columns']}")
    
    def _generate_summary(self) -> dict:
        """Генерувати підсумковий звіт"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'status': 'verified',
            'verification_report': self.verification_report,
            'recommendations': []
        }
        
        # Add recommendations based on verification
        if not self.verification_report.get('event_series_format', {}).get('is_valid_event_series'):
            summary['recommendations'].append("Fix event-series format: ensure timestamp sorting and no duplicates")
        
        if self.verification_report.get('data_integrity', {}).get('null_percentage', 0) > 10:
            summary['recommendations'].append("Handle missing values: null percentage is high")
        
        enricher_coverage = self.verification_report.get('enricher_coverage', {})
        missing_enrichers = [name for name, info in enricher_coverage.items() if not info.get('found')]
        if missing_enrichers:
            summary['recommendations'].append(f"Missing enrichers: {missing_enrichers}")
        
        return summary


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify enriched dataset structure')
    parser.add_argument('--config-path', default='src/config', help='Path to config directory')
    parser.add_argument('--output', default='results/verification_report.json', help='Output file for report')
    
    args = parser.parse_args()
    
    # Initialize config
    config_manager = UnifiedConfigManager(config_path=args.config_path)
    
    # Run verification
    verifier = EnrichedDatasetVerifier(config_manager)
    result = verifier.run()
    
    # Save report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"\n✓ Verification report saved to {args.output}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY:")
    logger.info(json.dumps(result, indent=2, default=str))
    logger.info("=" * 80)
    
    return 0 if result.get('status') == 'verified' else 1


if __name__ == '__main__':
    sys.exit(main())
