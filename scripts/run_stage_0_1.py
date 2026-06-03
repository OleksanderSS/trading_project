#!/usr/bin/env python3
"""
Stage 0-1: Data Generation & Collection
Runs Stage 0 (Data Generation) and Stage 1 (Collection) with result analysis.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.unified_config_manager import get_current_config
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.pipeline.stages.stage_0_data_generation import DataGenerator
from src.pipeline.stages.stage_1_collection import CollectionStage
from src.data.management.data_manager import DataManager


def analyze_stage_0_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Stage 0 results for correctness and completeness."""
    analysis = {
        'stage': 'Stage 0 - Data Generation',
        'timestamp': datetime.now().isoformat(),
        'status': 'unknown',
        'checks': []
    }
    
    # Check if stage succeeded
    if results.get('status') != 'success':
        analysis['status'] = 'FAILED'
        analysis['checks'].append({
            'check': 'Stage status',
            'passed': False,
            'message': f"Stage failed: {results.get('message', 'Unknown error')}"
        })
        return analysis
    
    analysis['status'] = 'SUCCESS'
    
    # Check data points
    data_points = results.get('data_points', 0)
    analysis['checks'].append({
        'check': 'Data points generated',
        'passed': data_points > 0,
        'message': f"Generated {data_points} data points"
    })
    
    # Check features
    features_df = results.get('features_df')
    if features_df is not None:
        analysis['checks'].append({
            'check': 'Features DataFrame exists',
            'passed': True,
            'message': f"Features shape: {features_df.shape}"
        })
        analysis['checks'].append({
            'check': 'Features not empty',
            'passed': len(features_df) > 0,
            'message': f"Features count: {len(features_df)}"
        })
    else:
        analysis['checks'].append({
            'check': 'Features DataFrame exists',
            'passed': False,
            'message': 'Features DataFrame is None'
        })
    
    # Check targets
    targets_df = results.get('targets_df')
    if targets_df is not None:
        analysis['checks'].append({
            'check': 'Targets DataFrame exists',
            'passed': True,
            'message': f"Targets shape: {targets_df.shape}"
        })
        analysis['checks'].append({
            'check': 'Targets not empty',
            'passed': len(targets_df) > 0,
            'message': f"Targets count: {len(targets_df)}"
        })
    else:
        analysis['checks'].append({
            'check': 'Targets DataFrame exists',
            'passed': False,
            'message': 'Targets DataFrame is None'
        })
    
    # Overall assessment
    all_passed = all(check['passed'] for check in analysis['checks'])
    analysis['overall_passed'] = all_passed
    
    return analysis


def analyze_stage_1_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Stage 1 results for correctness and completeness."""
    analysis = {
        'stage': 'Stage 1 - Collection',
        'timestamp': datetime.now().isoformat(),
        'status': 'unknown',
        'checks': []
    }
    
    # Check if raw_data exists
    raw_data = results.get('raw_data', {})
    if not raw_data:
        analysis['status'] = 'FAILED'
        analysis['checks'].append({
            'check': 'Raw data collected',
            'passed': False,
            'message': 'No raw data collected'
        })
        return analysis
    
    analysis['status'] = 'SUCCESS'
    
    # Check data sources
    data_sources = list(raw_data.keys())
    analysis['checks'].append({
        'check': 'Data sources collected',
        'passed': len(data_sources) > 0,
        'message': f"Collected data from {len(data_sources)} sources: {data_sources}"
    })
    
    # Check each data source
    for source, data in raw_data.items():
        if data is not None:
            if hasattr(data, 'shape'):
                analysis['checks'].append({
                    'check': f'{source} data exists',
                    'passed': True,
                    'message': f"{source} shape: {data.shape}"
                })
            elif isinstance(data, dict):
                analysis['checks'].append({
                    'check': f'{source} data exists',
                    'passed': True,
                    'message': f"{source} has {len(data)} keys"
                })
            else:
                analysis['checks'].append({
                    'check': f'{source} data exists',
                    'passed': True,
                    'message': f"{source} type: {type(data)}"
                })
        else:
            analysis['checks'].append({
                'check': f'{source} data exists',
                'passed': False,
                'message': f"{source} is None"
            })
    
    # Overall assessment
    all_passed = all(check['passed'] for check in analysis['checks'])
    analysis['overall_passed'] = all_passed
    
    return analysis


async def run_stage_0_1():
    """Run Stage 0 and Stage 1 with analysis."""
    logger = ProjectLogger.get_logger(__name__)
    
    print("=" * 80)
    print("STAGE 0-1: DATA GENERATION & COLLECTION")
    print("=" * 80)
    
    # Initialize components
    config_manager = get_current_config()
    error_handler = ErrorHandler(config_manager)
    db_manager = DataManager(config_manager)
    
    # Run Stage 0
    print("\n🔄 Running Stage 0: Data Generation...")
    data_generator = DataGenerator(config_manager)
    stage_0_results = data_generator.generate_synthetic_data()
    
    # Analyze Stage 0
    stage_0_analysis = analyze_stage_0_results(stage_0_results)
    print(f"\n📊 Stage 0 Analysis:")
    print(f"   Status: {stage_0_analysis['status']}")
    print(f"   Overall Passed: {stage_0_analysis['overall_passed']}")
    for check in stage_0_analysis['checks']:
        status = "✅" if check['passed'] else "❌"
        print(f"   {status} {check['check']}: {check['message']}")
    
    if not stage_0_analysis['overall_passed']:
        print("\n❌ Stage 0 failed. Stopping pipeline.")
        return {'stage_0': stage_0_results, 'stage_0_analysis': stage_0_analysis}
    
    # Run Stage 1
    print("\n🔄 Running Stage 1: Collection...")
    collection_stage = CollectionStage(config_manager, db_manager, error_handler)
    stage_1_results = await collection_stage.run()
    
    # Analyze Stage 1
    stage_1_analysis = analyze_stage_1_results(stage_1_results)
    print(f"\n📊 Stage 1 Analysis:")
    print(f"   Status: {stage_1_analysis['status']}")
    print(f"   Overall Passed: {stage_1_analysis['overall_passed']}")
    for check in stage_1_analysis['checks']:
        status = "✅" if check['passed'] else "❌"
        print(f"   {status} {check['check']}: {check['message']}")
    
    if not stage_1_analysis['overall_passed']:
        print("\n❌ Stage 1 failed. Stopping pipeline.")
        return {
            'stage_0': stage_0_results,
            'stage_0_analysis': stage_0_analysis,
            'stage_1': stage_1_results,
            'stage_1_analysis': stage_1_analysis
        }
    
    # Save results
    results_dir = Path('data/stage_results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'stage_0_1_results.json', 'w') as f:
        json.dump({
            'stage_0': stage_0_results,
            'stage_0_analysis': stage_0_analysis,
            'stage_1': stage_1_results,
            'stage_1_analysis': stage_1_analysis
        }, f, indent=2, default=str)
    
    print(f"\n✅ Stage 0-1 completed successfully!")
    print(f"📁 Results saved to: {results_dir / 'stage_0_1_results.json'}")
    
    return {
        'stage_0': stage_0_results,
        'stage_0_analysis': stage_0_analysis,
        'stage_1': stage_1_results,
        'stage_1_analysis': stage_1_analysis
    }


if __name__ == '__main__':
    asyncio.run(run_stage_0_1())
