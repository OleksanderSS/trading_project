#!/usr/bin/env python3
"""
Main pipeline for data processing, feature engineering, and model training.
Orchestrates the execution of sequential stages, passing data from one to the next.
"""

import argparse
import logging
import pandas as pd
import os
import shutil
from datetime import datetime
from pathlib import Path
import sys

# --- Stage Imports ---
from core.stages.stage_1_collectors_layer import run_stage_1
from core.stages.stage_2_enrichment_fixed import run_stage_2
from core.stages.stage_3_features import run_stage_3
from core.stages.stage_4_modeling import run_stage_4_v2 as run_stage_4
from core.stages.stage_5_prediction import run_stage_5

# --- Utility Imports ---
from utils.automated_reporting import AutomatedReporting
from utils.config_manager import ConfigManager
from utils.results_manager import ResultsManager, ComprehensiveReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def _accumulate_and_save_data(new_data: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """
    Accumulates new data with existing data, saves it, and returns the combined DataFrame.
    """
    if new_data.empty:
        logger.info("No new data to accumulate.")
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        return pd.DataFrame()

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(file_path):
        try:
            existing_data = pd.read_parquet(file_path)
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        except Exception:
            combined_data = new_data
    else:
        combined_data = new_data

    key_columns = [col for col in ['published_at', 'ticker', 'title'] if col in combined_data.columns]
    if key_columns:
        combined_data.drop_duplicates(subset=key_columns, keep='last', inplace=True)

    combined_data.to_parquet(file_path, index=False)
    return combined_data

def run_pipeline(stages_to_run: list, pause_after_stage_3: bool):
    """
    Main pipeline orchestrator.
    """
    logger.info(f"🚀 Starting pipeline for stages: {', '.join(stages_to_run)}")
    results_manager = ResultsManager()
    reporter = ComprehensiveReporter(results_manager)
    reporter.generate_comprehensive_report()

    stage1_output_path = "data/stages/stage_1_collected.pkl"
    stage2_output_path = "data/stages/stage_2_enriched.parquet"
    stage3_output_path = "data/stages/stage_3_master_featureset.parquet"
    accumulated_data_path = "data/accumulated/full_data.parquet"

    stage1_output = {}
    stage2_output = pd.DataFrame()

    # --- STAGE 1: Data Collection ---
    if '1' in stages_to_run:
        logger.info("📡 Running Stage 1: Data Collection...")
        stage1_output = run_stage_1()
        if stage1_output:
            Path(stage1_output_path).parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(stage1_output, stage1_output_path)
            logger.info(f"Stage 1 finished. Saved data to {stage1_output_path}")

    # --- STAGE 2: Data Enrichment ---
    if '2' in stages_to_run:
        logger.info("🔧 Running Stage 2: Data Enrichment...")
        if not stage1_output and os.path.exists(stage1_output_path):
            stage1_output = pd.read_pickle(stage1_output_path)
        
        stage2_output = run_stage_2(stage1_output)

        if not stage2_output.empty:
            stage2_output.to_parquet(stage2_output_path, index=False)
            logger.info(f"Stage 2 finished. Saved data to {stage2_output_path}")
            _accumulate_and_save_data(stage2_output, accumulated_data_path)

    # --- STAGE 3: Feature Engineering ---
    if '3' in stages_to_run:
        logger.info("⚙️ Running Stage 3: Feature Engineering...")
        if not stage1_output:
            stage1_output = pd.read_pickle(stage1_output_path)
        if stage2_output.empty and os.path.exists(stage2_output_path):
            stage2_output = pd.read_parquet(stage2_output_path)

        run_stage_3(stage1_data=stage1_output, stage2_data={'events': stage2_output})

    if pause_after_stage_3:
        logger.warning(f"⏸️ PAUSED after Stage 3. The final dataset is ready for heavy training at: {stage3_output_path}")
        return

    # --- STAGE 4 & 5 (Local/Light Training) ---
    if '4' in stages_to_run:
        logger.info("🤖 Running Stage 4: Model Training...")
        if not os.path.exists(stage3_output_path):
            logger.error(f"Stage 4 cannot run. Missing master feature set: {stage3_output_path}")
            return
        master_df = pd.read_parquet(stage3_output_path)
        run_stage_4(master_df)

    if '5' in stages_to_run:
        logger.info("🧪 Running Stage 5: Prediction...")
        if not os.path.exists("data/stages/stage_4_models.pkl"):
             logger.error("Stage 5 cannot run. Missing models file.")
             return
        features_df = pd.read_parquet(stage3_output_path)
        run_stage_5(features_df)

    logger.info("✅ Pipeline execution completed.")
    reporter.generate_comprehensive_report()


def main():
    parser = argparse.ArgumentParser(description="Progressive Pipeline Orchestrator")
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['1', '2', '3', '4', '5'],
        default=['1', '2', '3', '4', '5'],
        help="A list of stages to run."
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help="If set, resets all cached data and results."
    )
    parser.add_argument(
        '--pause-after-stage-3',
        action='store_true',
        help="If set, the pipeline will stop after Stage 3 for manual intervention."
    )
    args = parser.parse_args()

    if args.reset:
        logger.warning("🔄 Resetting all pipeline data...")
        dirs_to_reset = ["data/stages", "data/accumulated", "output"]
        for d in dirs_to_reset:
            if os.path.exists(d):
                shutil.rmtree(d)
                logger.info(f"Removed directory: {d}")
        logger.info("✅ Data reset complete.")

    run_pipeline(args.stages, args.pause_after_stage_3)


if __name__ == "__main__":
    main()
