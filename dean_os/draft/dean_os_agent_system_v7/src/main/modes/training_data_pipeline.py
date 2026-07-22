
"""
End-to-End Pipeline for Creating a Training Dataset.

This script orchestrates the process of data collection, feature engineering,
and target calculation using the new, modular components.
"""

import asyncio
from pathlib import Path

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager
from src.features.feature_orchestrator import FeatureOrchestrator
from src.pipeline.stages.stage_1_collection import CollectionStage

# audit-ignore: ARCHITECTURAL_USAGE
from src.targets.target_orchestrator import TargetOrchestrator

logger = ProjectLogger.get_logger("TrainingDataPipeline")

async def run_pipeline(config_manager: UnifiedConfigManager, db_manager: DataManager):
    """Executes the full data generation pipeline."""
    logger.info("--- Starting Training Data Pipeline (Modular) ---")

    error_handler = ErrorHandler(config_manager)

    # 1. Collection Stage
    collection_stage = CollectionStage(config_manager=config_manager, db_manager=db_manager, error_handler=error_handler)
    collected_data = await collection_stage.run()
    raw_data = collected_data.get('raw_data', {})

    if not raw_data:
        logger.critical("No data collected. Aborting pipeline.")
        return

    # Combine all collected data into a single DataFrame if needed, or process them based on type
    # For now, let's assume we primarily work with market_data
    market_data = raw_data.get('yahoo_finance')
    if market_data is None or market_data.empty:
        logger.critical("No market data from yahoo_finance. Aborting.")
        return

    logger.info(f"Successfully collected data. Types: {list(raw_data.keys())}")

    # 2. Feature Engineering Stage
    feature_orchestrator = FeatureOrchestrator.create_from_config(config_manager)
    features_df = feature_orchestrator.run(market_data, **raw_data)
    logger.info(f"Feature engineering complete. DataFrame shape: {features_df.shape}")

    # 3. Target Generation Stage
    targets_list = config_manager.get_config('targets', [])
    # audit-ignore: ARCHITECTURAL_USAGE
    target_orchestrator = TargetOrchestrator(targets_list=targets_list)
    # audit-ignore: ARCHITECTURAL_USAGE
    targets_df = target_orchestrator.generate_targets(features_df)
    # audit-ignore: ARCHITECTURAL_USAGE
    target_cols = [col for col in targets_df.columns if col.startswith('target_')]
    final_df = features_df.copy()
    # audit-ignore: ARCHITECTURAL_USAGE
    for col in target_cols:
        final_df[col] = targets_df[col].reindex(final_df.index)
    logger.info(f"Target generation complete. Final DataFrame shape: {final_df.shape}")

    # 4. Save the final dataset
    try:
        paths_config = config_manager.get_config('paths')
        output_dir_str = paths_config.get('processed_data', 'data/processed/')
        output_filename = "training_dataset.parquet"
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        final_df.to_parquet(output_path, index=False)
        logger.info("--- Pipeline Finished Successfully ---")
        logger.info(f"Final dataset saved to: {output_path}")

    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.critical(f"Failed to save final dataset: {e}", exc_info=True)

if __name__ == "__main__":
    from src.config.unified_config_manager import get_current_config

    config = get_current_config()
    db_manager = DataManager(config) # Create the DataManager

    asyncio.run(run_pipeline(config, db_manager))
