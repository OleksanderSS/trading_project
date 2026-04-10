#!/usr/bin/env python3
"""
Real Data Accumulation Script
Накопичення реальних даних БЕЗ тренування моделей

Stages: 0 (Setup) → 1 (Collection) → 2 (Processing) → 3 (Feature Engineering) → DuckDB
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager
from src.features.feature_orchestrator import FeatureOrchestrator
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator

logger = ProjectLogger.get_logger("DataAccumulation")

class RealDataAccumulator:
    """Накопичувач реальних даних без тренування"""
    
    def __init__(self, config_manager: UnifiedConfigManager):
        self.config_manager = config_manager
        self.data_manager = DataManager(config_manager)
        self.orchestrator = PipelineOrchestrator(config_manager)
        self.feature_orchestrator = FeatureOrchestrator.create_from_config(config_manager)
        
    def run(self, tickers: list = None, days_back: int = 30):
        """
        Запустити накопичення реальних даних
        
        Args:
            tickers: Список тікерів для накопичення (за замовчуванням з конфігу)
            days_back: Кількість днів історії для завантаження
        """
        logger.info("=" * 80)
        logger.info("🚀 REAL DATA ACCUMULATION - Starting")
        logger.info("=" * 80)
        
        try:
            # 1. STAGE 0: Setup
            logger.info("\n[Stage 0] Setup & Validation")
            self._stage_0_setup()
            
            # 2. STAGE 1: Collection
            logger.info("\n[Stage 1] Data Collection")
            tickers = tickers or self.config_manager.get_config('assets', {}).get('tickers', ['AMD', 'NVDA'])
            collected_data = self._stage_1_collection(tickers, days_back)
            
            if collected_data is None or collected_data.empty:
                logger.error("❌ No data collected. Aborting.")
                return {'status': 'failed', 'reason': 'No data collected'}
            
            # 3. STAGE 2: Processing & Context
            logger.info("\n[Stage 2] Data Processing & Cleaning")
            processed_data = self._stage_2_processing(collected_data)
            
            if processed_data is None or processed_data.empty:
                logger.error("❌ No data after processing. Aborting.")
                return {'status': 'failed', 'reason': 'No data after processing'}
            
            # 4. STAGE 3: Feature Engineering
            logger.info("\n[Stage 3] Feature Engineering & Enrichment")
            enriched_data = self._stage_3_enrichment(processed_data)
            
            if enriched_data is None or enriched_data.empty:
                logger.error("❌ No data after enrichment. Aborting.")
                return {'status': 'failed', 'reason': 'No data after enrichment'}
            
            # 5. Store in DuckDB
            logger.info("\n[Storage] Saving to DuckDB")
            self._store_in_duckdb(enriched_data, collected_data)
            
            # 6. Verification
            logger.info("\n[Verification] Checking stored data")
            verification_result = self._verify_storage()
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ REAL DATA ACCUMULATION - Completed Successfully")
            logger.info("=" * 80)
            
            return {
                'status': 'success',
                'tickers': tickers,
                'days_back': days_back,
                'collected_rows': len(collected_data),
                'enriched_rows': len(enriched_data),
                'enriched_features': len(enriched_data.columns),
                'verification': verification_result
            }
            
        except Exception as e:
            logger.exception(f"❌ Error during data accumulation: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _stage_0_setup(self):
        """Stage 0: Setup & Validation"""
        logger.info("✓ Configuration loaded")
        logger.info(f"✓ DuckDB path: {self.data_manager.db_path}")
        logger.info(f"✓ Feature orchestrator initialized with {len(self.feature_orchestrator.enrichers)} enrichers")
        
        # Verify enrichers
        enricher_names = [e.name for e in self.feature_orchestrator.enrichers]
        logger.info(f"  Enrichers: {', '.join(enricher_names)}")
    
    def _stage_1_collection(self, tickers: list, days_back: int) -> pd.DataFrame:
        """Stage 1: Collect real data"""
        logger.info(f"Collecting data for {len(tickers)} tickers (last {days_back} days)...")
        
        try:
            # Use PipelineOrchestrator to run Stage 1
            # This will collect market, macro, and news data
            collection_result = self.orchestrator.run_stage(
                stage_num=1,
                tickers=tickers,
                days_back=days_back
            )
            
            if collection_result is None or collection_result.empty:
                logger.warning("⚠️  Stage 1 returned empty data")
                return None
            
            logger.info(f"✓ Collected {len(collection_result)} rows")
            logger.info(f"✓ Columns: {list(collection_result.columns)[:10]}... ({len(collection_result.columns)} total)")
            
            return collection_result
            
        except Exception as e:
            logger.error(f"❌ Stage 1 collection failed: {e}")
            return None
    
    def _stage_2_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 2: Process & clean data"""
        logger.info(f"Processing {len(df)} rows...")
        
        try:
            # Use PipelineOrchestrator to run Stage 2
            # This will clean, normalize, and identify market regime
            processing_result = self.orchestrator.run_stage(
                stage_num=2,
                data=df
            )
            
            if processing_result is None or processing_result.empty:
                logger.warning("⚠️  Stage 2 returned empty data")
                return None
            
            logger.info(f"✓ Processed {len(processing_result)} rows")
            logger.info(f"✓ Removed {len(df) - len(processing_result)} rows (duplicates/anomalies)")
            
            return processing_result
            
        except Exception as e:
            logger.error(f"❌ Stage 2 processing failed: {e}")
            return None
    
    def _stage_3_enrichment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 3: Feature engineering & enrichment"""
        logger.info(f"Enriching {len(df)} rows with {len(self.feature_orchestrator.enrichers)} enrichers...")
        
        try:
            # Run FeatureOrchestrator
            enriched_df = self.feature_orchestrator.run(df)
            
            if enriched_df is None or enriched_df.empty:
                logger.warning("⚠️  Feature orchestrator returned empty data")
                return None
            
            logger.info(f"✓ Enriched {len(enriched_df)} rows")
            logger.info(f"✓ Features: {len(df.columns)} → {len(enriched_df.columns)}")
            
            # Show sample of enriched data
            logger.info(f"✓ Sample enriched row:")
            if len(enriched_df) > 0:
                sample_row = enriched_df.iloc[0]
                logger.info(f"  Columns: {list(enriched_df.columns)[:15]}...")
                logger.info(f"  Shape: {enriched_df.shape}")
            
            return enriched_df
            
        except Exception as e:
            logger.error(f"❌ Stage 3 enrichment failed: {e}")
            return None
    
    def _store_in_duckdb(self, enriched_data: pd.DataFrame, raw_data: pd.DataFrame):
        """Store data in DuckDB"""
        logger.info("Storing data in DuckDB...")
        
        try:
            # Store raw data
            self.data_manager.upsert(
                table_name='raw_data',
                df=raw_data,
                unique_on=['ticker', 'timestamp'] if 'ticker' in raw_data.columns else []
            )
            logger.info(f"✓ Stored {len(raw_data)} raw data rows")
            
            # Store enriched features
            self.data_manager.upsert(
                table_name='enriched_features',
                df=enriched_data,
                unique_on=['ticker', 'timestamp'] if 'ticker' in enriched_data.columns else []
            )
            logger.info(f"✓ Stored {len(enriched_data)} enriched feature rows")
            
        except Exception as e:
            logger.error(f"❌ DuckDB storage failed: {e}")
            raise
    
    def _verify_storage(self) -> dict:
        """Verify stored data"""
        logger.info("Verifying stored data...")
        
        verification = {}
        
        try:
            # Check tables
            tables = self.data_manager.get_all_table_names()
            logger.info(f"✓ Tables in DuckDB: {tables}")
            verification['tables'] = tables
            
            # Check raw_data
            if 'raw_data' in tables:
                raw_df = self.data_manager.fetch_data_from_table('raw_data')
                if raw_df is not None:
                    logger.info(f"✓ raw_data: {len(raw_df)} rows, {len(raw_df.columns)} columns")
                    verification['raw_data_rows'] = len(raw_df)
                    verification['raw_data_columns'] = len(raw_df.columns)
            
            # Check enriched_features
            if 'enriched_features' in tables:
                enriched_df = self.data_manager.fetch_data_from_table('enriched_features')
                if enriched_df is not None:
                    logger.info(f"✓ enriched_features: {len(enriched_df)} rows, {len(enriched_df.columns)} columns")
                    verification['enriched_features_rows'] = len(enriched_df)
                    verification['enriched_features_columns'] = len(enriched_df.columns)
                    
                    # Check for event-series format
                    expected_cols = ['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
                    has_event_series = all(col in enriched_df.columns for col in expected_cols)
                    verification['has_event_series_format'] = has_event_series
                    logger.info(f"✓ Event-series format: {has_event_series}")
            
            verification['status'] = 'verified'
            return verification
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            verification['status'] = 'failed'
            verification['error'] = str(e)
            return verification


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Accumulate real data without training')
    parser.add_argument('--tickers', nargs='+', default=['AMD', 'NVDA'], help='Tickers to accumulate')
    parser.add_argument('--days', type=int, default=30, help='Days of history to collect')
    parser.add_argument('--config-path', default='src/config', help='Path to config directory')
    
    args = parser.parse_args()
    
    # Initialize config
    config_manager = UnifiedConfigManager(config_path=args.config_path)
    
    # Run accumulation
    accumulator = RealDataAccumulator(config_manager)
    result = accumulator.run(tickers=args.tickers, days_back=args.days)
    
    # Print result
    logger.info("\n" + "=" * 80)
    logger.info("RESULT:")
    logger.info(json.dumps(result, indent=2, default=str))
    logger.info("=" * 80)
    
    return 0 if result['status'] == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
