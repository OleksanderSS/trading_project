#!/usr/bin/env python3
"""
Analyze stages 0-3 execution and data flow.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.data.management.data_manager import DataManager

logger = ProjectLogger.get_logger(__name__)


def _log_result_details(result, stage_name):
    """Log details of stage result."""
    if not isinstance(result, dict):
        logger.info(f"✅ {stage_name} Result: {result}")
        return
    
    logger.info(f"✅ {stage_name} Result Keys: {result.keys()}")
    for key, value in result.items():
        _log_value_info(key, value)


def _log_value_info(key, value):
    """Log information about a value."""
    if hasattr(value, 'shape'):
        if hasattr(value, 'dtypes'):
            logger.info(f"   - {key}: shape={value.shape}, dtypes={value.dtypes.to_dict()}")
        else:
            logger.info(f"   - {key}: shape={value.shape}")
    elif isinstance(value, dict):
        logger.info(f"   - {key}: dict with keys={list(value.keys())}")
    else:
        logger.info(f"   - {key}: {type(value)}")


def _log_database_state(db_manager):
    """Log database tables and row counts."""
    try:
        tables = db_manager.get_all_tables()
        logger.info(f"✅ Tables in database: {tables}")
        
        for table in tables:
            try:
                count = db_manager.get_row_count(table)
                logger.info(f"   - {table}: {count} rows")
            except (ValueError, TypeError, Exception) as e:
                logger.error(f"   - {table}: Error getting count - {e}", exc_info=True)
                raise RuntimeError(f"Failed to get row count for table {table}: {e}") from e
    except (ValueError, TypeError, Exception) as e:
        logger.error(f"❌ Database error: {e}", exc_info=True)
        raise RuntimeError(f"Database analysis failed: {e}") from e


async def analyze_stages():
    """Analyze each stage execution."""
    
    logger.info("=" * 80)
    logger.info("STAGE ANALYSIS: 0-3 Pipeline Execution")
    logger.info("=" * 80)
    
    # Initialize
    config_manager = UnifiedConfigManager()
    db_manager = DataManager()
    orchestrator = PipelineOrchestrator(config_manager, db_manager)
    
    # Run stages 0-3
    logger.info("\n📋 STAGE 0: Setup")
    logger.info("-" * 80)
    stage_0_result = await orchestrator.run_stage(0)
    _log_result_details(stage_0_result, "Stage 0")
    
    logger.info("\n📋 STAGE 1: Collection")
    logger.info("-" * 80)
    stage_1_result = await orchestrator.run_stage(1, tickers=['AMD'], keywords=['tech'])
    _log_result_details(stage_1_result, "Stage 1")
    
    logger.info("\n📋 STAGE 2: Processing")
    logger.info("-" * 80)
    stage_2_result = await orchestrator.run_stage(2, **stage_1_result)
    _log_result_details(stage_2_result, "Stage 2")
    
    logger.info("\n📋 STAGE 3: Feature Engineering")
    logger.info("-" * 80)
    stage_3_result = await orchestrator.run_stage(3, **stage_2_result)
    _log_result_details(stage_3_result, "Stage 3")
    
    # Database check
    logger.info("\n📊 DATABASE STATE")
    logger.info("-" * 80)
    _log_database_state(db_manager)
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(analyze_stages())
