#!/usr/bin/env python3
"""
Stage 2-3: Processing & Feature Engineering
Runs Stage 2 (Processing) and Stage 3 (Feature Engineering) with result analysis.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.unified_config_manager import get_current_config
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.pipeline.stages.stage_2_processing import ProcessingStage
from src.pipeline.stages.stage_3_feature_engineering import FeatureEngineeringStage


def analyze_stage_2_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Stage 2 results for correctness and completeness."""
    analysis = {
        'stage': 'Stage 2 - Processing',
        'timestamp': datetime.now().isoformat(),
        'status': 'unknown',
        'checks': []
    }
    
    # Check if stage succeeded
    if results.get('status') == 'error':
        analysis['status'] = 'FAILED'
        analysis['checks'].append({
            'check': 'Stage status',
            'passed': False,
            'message': f"Stage failed: {results.get('error', 'Unknown error')}"
        })
        return analysis
    
    analysis['status'] = 'SUCCESS'
    
    # Check processed data
    processed_data = results.get('processed_data', {})
    if processed_data:
        analysis['checks'].append({
            'check': 'Processed data exists',
            'passed': True,
            'message': f"Processed {len(processed_data)} data sources"
        })
    else:
        analysis['checks'].append({
            'check': 'Processed data exists',
            'passed': False,
            'message': 'No processed data'
        })
    
    # Check price data
    price_data = results.get('price_data', {})
    if price_data:
        analysis['checks'].append({
            'check': 'Price data processed',
            'passed': True,
            'message': f"Processed {len(price_data)} timeframes"
        })
    else:
        analysis['checks'].append({
            'check': 'Price data processed',
            'passed': False,
            'message': 'No price data'
        })
    
    # Check for NaN values
    for source, data in processed_data.items():
        if isinstance(data, pd.DataFrame):
            nan_count = data.isna().sum().sum()
            total_count = data.size
            nan_ratio = nan_count / total_count if total_count > 0 else 0
            analysis['checks'].append({
                'check': f'{source} NaN ratio',
                'passed': nan_ratio < 0.1,
                'message': f"{source}: {nan_count}/{total_count} NaN ({nan_ratio:.2%})"
            })
    
    # Overall assessment
    all_passed = all(check['passed'] for check in analysis['checks'])
    analysis['overall_passed'] = all_passed
    
    return analysis


def analyze_stage_3_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Stage 3 results for correctness and completeness."""
    analysis = {
        'stage': 'Stage 3 - Feature Engineering',
        'timestamp': datetime.now().isoformat(),
        'status': 'unknown',
        'checks': []
    }
    
    # Check if stage succeeded
    if results.get('status') == 'error':
        analysis['status'] = 'FAILED'
        analysis['checks'].append({
            'check': 'Stage status',
            'passed': False,
            'message': f"Stage failed: {results.get('error', 'Unknown error')}"
        })
        return analysis
    
    analysis['status'] = 'SUCCESS'
    
    # Check features
    features = results.get('features')
    if features is not None:
        if isinstance(features, pd.DataFrame):
            analysis['checks'].append({
                'check': 'Features DataFrame exists',
                'passed': True,
                'message': f"Features shape: {features.shape}"
            })
            analysis['checks'].append({
                'check': 'Features not empty',
                'passed': len(features) > 0,
                'message': f"Features count: {len(features)}, Columns: {len(features.columns)}"
            })
        else:
            analysis['checks'].append({
                'check': 'Features DataFrame exists',
                'passed': False,
                'message': f"Features is not DataFrame: {type(features)}"
            })
    else:
        analysis['checks'].append({
            'check': 'Features DataFrame exists',
            'passed': False,
            'message': 'Features is None'
        })
    
    # Check targets
    targets = results.get('targets')
    if targets is not None:
        if isinstance(targets, pd.DataFrame):
            analysis['checks'].append({
                'check': 'Targets DataFrame exists',
                'passed': True,
                'message': f"Targets shape: {targets.shape}"
            })
        else:
            analysis['checks'].append({
                'check': 'Targets DataFrame exists',
                'passed': False,
                'message': f"Targets is not DataFrame: {type(targets)}"
            })
    else:
        analysis['checks'].append({
            'check': 'Targets DataFrame exists',
            'passed': False,
            'message': 'Targets is None'
        })
    
    # Check feature-target alignment
    if features is not None and targets is not None:
        if isinstance(features, pd.DataFrame) and isinstance(targets, pd.DataFrame):
            aligned = len(features) == len(targets)
            analysis['checks'].append({
                'check': 'Feature-target alignment',
                'passed': aligned,
                'message': f"Features: {len(features)}, Targets: {len(targets)}"
            })
    
    # Overall assessment
    all_passed = all(check['passed'] for check in analysis['checks'])
    analysis['overall_passed'] = all_passed
    
    return analysis


async def run_stage_2_3():
    """Run Stage 2 and Stage 3 with analysis."""
    logger = ProjectLogger.get_logger(__name__)
    
    print("=" * 80)
    print("STAGE 2-3: PROCESSING & FEATURE ENGINEERING")
    print("=" * 80)
    
    # Load previous results
    results_dir = Path('data/stage_results')
    stage_0_1_file = results_dir / 'stage_0_1_results.json'
    
    if not stage_0_1_file.exists():
        print(f"\n❌ Previous results not found: {stage_0_1_file}")
        print("Please run stage_0_1.py first.")
        return None
    
    with open(stage_0_1_file, 'r') as f:
        previous_results = json.load(f)
    
    stage_1_results = previous_results.get('stage_1', {})
    
    # Initialize components
    config_manager = get_current_config()
    error_handler = ErrorHandler(config_manager)
    
    # Run Stage 2
    print("\n🔄 Running Stage 2: Processing...")
    processing_stage = ProcessingStage(config_manager, error_handler)
    stage_2_results = await processing_stage.run(raw_data=stage_1_results.get('raw_data', {}))
    
    # Analyze Stage 2
    stage_2_analysis = analyze_stage_2_results(stage_2_results)
    print(f"\n📊 Stage 2 Analysis:")
    print(f"   Status: {stage_2_analysis['status']}")
    print(f"   Overall Passed: {stage_2_analysis['overall_passed']}")
    for check in stage_2_analysis['checks']:
        status = "✅" if check['passed'] else "❌"
        print(f"   {status} {check['check']}: {check['message']}")
    
    if not stage_2_analysis['overall_passed']:
        print("\n❌ Stage 2 failed. Stopping pipeline.")
        return {
            'stage_2': stage_2_results,
            'stage_2_analysis': stage_2_analysis
        }
    
    # Run Stage 3
    print("\n🔄 Running Stage 3: Feature Engineering...")
    feature_stage = FeatureEngineeringStage(config_manager, error_handler)
    stage_3_results = await feature_stage.run(processed_data=stage_2_results.get('processed_data', {}))
    
    # Analyze Stage 3
    stage_3_analysis = analyze_stage_3_results(stage_3_results)
    print(f"\n📊 Stage 3 Analysis:")
    print(f"   Status: {stage_3_analysis['status']}")
    print(f"   Overall Passed: {stage_3_analysis['overall_passed']}")
    for check in stage_3_analysis['checks']:
        status = "✅" if check['passed'] else "❌"
        print(f"   {status} {check['check']}: {check['message']}")
    
    if not stage_3_analysis['overall_passed']:
        print("\n❌ Stage 3 failed. Stopping pipeline.")
        return {
            'stage_2': stage_2_results,
            'stage_2_analysis': stage_2_analysis,
            'stage_3': stage_3_results,
            'stage_3_analysis': stage_3_analysis
        }
    
    # Save results
    with open(results_dir / 'stage_2_3_results.json', 'w') as f:
        json.dump({
            'stage_2': stage_2_results,
            'stage_2_analysis': stage_2_analysis,
            'stage_3': stage_3_results,
            'stage_3_analysis': stage_3_analysis
        }, f, indent=2, default=str)
    
    print(f"\n✅ Stage 2-3 completed successfully!")
    print(f"📁 Results saved to: {results_dir / 'stage_2_3_results.json'}")
    
    return {
        'stage_2': stage_2_results,
        'stage_2_analysis': stage_2_analysis,
        'stage_3': stage_3_results,
        'stage_3_analysis': stage_3_analysis
    }


if __name__ == '__main__':
    asyncio.run(run_stage_2_3())
