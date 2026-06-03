#!/usr/bin/env python3
"""
Simplified Real Data Accumulation Script
Збирає дані БЕЗ використання PipelineOrchestrator
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager
from src.features.feature_orchestrator import FeatureOrchestrator

logger = ProjectLogger.get_logger("SimpleAccumulation")


def main():
    """Main entry point"""
    logger.info("=" * 80)
    logger.info("🚀 SIMPLE DATA ACCUMULATION - Starting")
    logger.info("=" * 80)
    
    try:
        # Initialize config
        config_manager = UnifiedConfigManager()
        logger.info("✓ Configuration loaded")
        
        # Initialize data manager
        data_manager = DataManager(config_manager)
        logger.info(f"✓ DataManager initialized: {data_manager.db_path}")
        
        # Check if we have market data
        logger.info("\n[Step 1] Checking existing data...")
        market_df = data_manager.fetch_data_from_table('market_data_raw')
        
        if market_df is None or market_df.empty:
            logger.error("❌ No market data found in database!")
            logger.info("Please run Stage 1 collection first to gather data.")
            return 1
        
        logger.info(f"✓ Found {len(market_df)} market data rows")
        logger.info(f"✓ Columns: {list(market_df.columns)}")
        
        # Rename datetime to timestamp if needed
        if 'datetime' in market_df.columns and 'timestamp' not in market_df.columns:
            market_df = market_df.rename(columns={'datetime': 'timestamp'})
            logger.info("✓ Renamed 'datetime' to 'timestamp'")
        
        # Filter duplicates
        if 'ticker' in market_df.columns and 'timestamp' in market_df.columns:
            before = len(market_df)
            market_df = market_df.drop_duplicates(subset=['ticker', 'timestamp'], keep='last')
            after = len(market_df)
            if before != after:
                logger.info(f"✓ Removed {before - after} duplicates")
        
        # Sort by timestamp
        if 'timestamp' in market_df.columns:
            market_df = market_df.sort_values('timestamp')
            # Set timestamp as index for enrichers that need datetime index
            market_df = market_df.set_index('timestamp')
            logger.info("✓ Sorted by timestamp and set as index")
        
        logger.info(f"\n[Step 2] Processed data: {len(market_df)} rows")
        
        # Initialize feature orchestrator
        logger.info("\n[Step 3] Initializing Feature Orchestrator...")
        feature_orchestrator = FeatureOrchestrator.create_from_config(config_manager)
        
        logger.info(f"✓ Loaded {len(feature_orchestrator.enrichers)} enrichers")
        logger.info(f"✓ Enabled: {[e.name for e in feature_orchestrator.enrichers]}")
        
        # Run enrichment
        logger.info(f"\n[Step 4] Enriching {len(market_df)} rows...")
        enriched_df = feature_orchestrator.run(market_df)
        
        if enriched_df is None or enriched_df.empty:
            logger.error("❌ Enrichment failed!")
            return 1
        
        logger.info(f"✓ Enriched: {len(enriched_df)} rows, {len(enriched_df.columns)} columns")
        logger.info(f"✓ Added {len(enriched_df.columns) - len(market_df.columns)} new features")
        
        # Store in DuckDB
        logger.info("\n[Step 5] Storing in DuckDB...")
        
        # Store raw data
        data_manager.upsert(
            table_name='raw_data',
            df=market_df,
            unique_on=['ticker', 'timestamp'] if 'ticker' in market_df.columns else []
        )
        logger.info(f"✓ Stored {len(market_df)} raw data rows")
        
        # Store enriched features
        data_manager.upsert(
            table_name='enriched_features',
            df=enriched_df,
            unique_on=['ticker', 'timestamp'] if 'ticker' in enriched_df.columns else []
        )
        logger.info(f"✓ Stored {len(enriched_df)} enriched feature rows")
        
        # Verification
        logger.info("\n[Step 6] Verification...")
        tables = data_manager.get_all_table_names()
        logger.info(f"✓ Tables in DuckDB: {tables}")
        
        if 'enriched_features' in tables:
            verify_df = data_manager.fetch_data_from_table('enriched_features')
            if verify_df is not None:
                logger.info(f"✓ Verified: {len(verify_df)} rows, {len(verify_df.columns)} columns")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ DATA ACCUMULATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"📊 Summary:")
        logger.info(f"   - Raw data rows: {len(market_df)}")
        logger.info(f"   - Enriched rows: {len(enriched_df)}")
        logger.info(f"   - Total features: {len(enriched_df.columns)}")
        logger.info(f"   - Database: {data_manager.db_path}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"❌ Error during data accumulation: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
