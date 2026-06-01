#!/usr/bin/env python3
"""
Full Pipeline Runner
Runs all pipeline stages sequentially with analysis after each stage group.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging.logger import ProjectLogger


async def run_full_pipeline():
    """Run the complete pipeline with stage-by-stage analysis."""
    logger = ProjectLogger.get_logger(__name__)
    
    print("=" * 80)
    print("FULL PIPELINE RUNNER")
    print("Running all stages with analysis after each stage group")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Import stage runners
    from scripts.run_stage_0_1 import run_stage_0_1
    from scripts.run_stage_2_3 import run_stage_2_3
    from scripts.run_stage_4_5 import run_stage_4_5
    from scripts.run_stage_6_7 import run_stage_6_7
    
    # Stage 0-1
    print("\n" + "=" * 80)
    print("PHASE 1: DATA GENERATION & COLLECTION")
    print("=" * 80)
    results_0_1 = await run_stage_0_1()
    
    if results_0_1 is None or not results_0_1.get('stage_1_analysis', {}).get('overall_passed', False):
        print("\n❌ Pipeline failed at Stage 0-1. Stopping.")
        return None
    
    # Stage 2-3
    print("\n" + "=" * 80)
    print("PHASE 2: PROCESSING & FEATURE ENGINEERING")
    print("=" * 80)
    results_2_3 = await run_stage_2_3()
    
    if results_2_3 is None or not results_2_3.get('stage_3_analysis', {}).get('overall_passed', False):
        print("\n❌ Pipeline failed at Stage 2-3. Stopping.")
        return None
    
    # Stage 4-5
    print("\n" + "=" * 80)
    print("PHASE 3: MODELING & PREDICTION")
    print("=" * 80)
    results_4_5 = await run_stage_4_5()
    
    if results_4_5 is None or not results_4_5.get('stage_5_analysis', {}).get('overall_passed', False):
        print("\n❌ Pipeline failed at Stage 4-5. Stopping.")
        return None
    
    # Stage 6-7
    print("\n" + "=" * 80)
    print("PHASE 4: TRADING EXECUTION & EVALUATION")
    print("=" * 80)
    results_6_7 = await run_stage_6_7()
    
    if results_6_7 is None or not results_6_7.get('stage_7_analysis', {}).get('overall_passed', False):
        print("\n❌ Pipeline failed at Stage 6-7. Stopping.")
        return None
    
    # Generate final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETION SUMMARY")
    print("=" * 80)
    print(f"Start Time: {start_time.isoformat()}")
    print(f"End Time: {end_time.isoformat()}")
    print(f"Duration: {duration}")
    print()
    
    # Stage summaries
    all_results = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration': str(duration),
        'stage_0_1': results_0_1,
        'stage_2_3': results_2_3,
        'stage_4_5': results_4_5,
        'stage_6_7': results_6_7
    }
    
    # Save final summary
    results_dir = Path('data/stage_results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'full_pipeline_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✅ Full pipeline completed successfully!")
    print(f"📁 Final summary saved to: {results_dir / 'full_pipeline_summary.json'}")
    
    return all_results


if __name__ == '__main__':
    asyncio.run(run_full_pipeline())
