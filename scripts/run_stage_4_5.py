#!/usr/bin/env python3
"""
Stage 4-5: Modeling & Prediction
Runs Stage 4 (Modeling) and Stage 5 (Prediction) with result analysis.
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
from src.pipeline.stages.stage_4_modeling import ModelingStage
from src.pipeline.stages.stage_5_prediction import PredictionStage


def analyze_stage_4_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Stage 4 results for correctness and completeness."""
    analysis = {
        'stage': 'Stage 4 - Modeling',
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
    
    # Check trained models
    trained_models = results.get('trained_models', {})
    if trained_models:
        analysis['checks'].append({
            'check': 'Models trained',
            'passed': True,
            'message': f"Trained {len(trained_models)} models"
        })
    else:
        analysis['checks'].append({
            'check': 'Models trained',
            'passed': False,
            'message': 'No models trained'
        })
    
    # Check model performance
    model_performance = results.get('model_performance', {})
    if model_performance:
        analysis['checks'].append({
            'check': 'Model performance metrics',
            'passed': True,
            'message': f"Performance metrics for {len(model_performance)} models"
        })
    else:
        analysis['checks'].append({
            'check': 'Model performance metrics',
            'passed': False,
            'message': 'No performance metrics'
        })
    
    # Check best model
    best_model = results.get('best_model')
    if best_model:
        analysis['checks'].append({
            'check': 'Best model selected',
            'passed': True,
            'message': f"Best model: {best_model}"
        })
    else:
        analysis['checks'].append({
            'check': 'Best model selected',
            'passed': False,
            'message': 'No best model selected'
        })
    
    # Overall assessment
    all_passed = all(check['passed'] for check in analysis['checks'])
    analysis['overall_passed'] = all_passed
    
    return analysis


def analyze_stage_5_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Stage 5 results for correctness and completeness."""
    analysis = {
        'stage': 'Stage 5 - Prediction',
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
    
    # Check predictions
    predictions = results.get('predictions')
    if predictions is not None:
        if isinstance(predictions, pd.DataFrame):
            analysis['checks'].append({
                'check': 'Predictions DataFrame exists',
                'passed': True,
                'message': f"Predictions shape: {predictions.shape}"
            })
            analysis['checks'].append({
                'check': 'Predictions not empty',
                'passed': len(predictions) > 0,
                'message': f"Predictions count: {len(predictions)}"
            })
        else:
            analysis['checks'].append({
                'check': 'Predictions DataFrame exists',
                'passed': False,
                'message': f"Predictions is not DataFrame: {type(predictions)}"
            })
    else:
        analysis['checks'].append({
            'check': 'Predictions DataFrame exists',
            'passed': False,
            'message': 'Predictions is None'
        })
    
    # Check prediction confidence
    confidence_scores = results.get('confidence_scores')
    if confidence_scores is not None:
        analysis['checks'].append({
            'check': 'Confidence scores calculated',
            'passed': True,
            'message': f"Confidence scores for {len(confidence_scores)} predictions"
        })
    else:
        analysis['checks'].append({
            'check': 'Confidence scores calculated',
            'passed': False,
            'message': 'No confidence scores'
        })
    
    # Check ensemble predictions
    ensemble_predictions = results.get('ensemble_predictions')
    if ensemble_predictions is not None:
        analysis['checks'].append({
            'check': 'Ensemble predictions generated',
            'passed': True,
            'message': 'Ensemble predictions available'
        })
    else:
        analysis['checks'].append({
            'check': 'Ensemble predictions generated',
            'passed': False,
            'message': 'No ensemble predictions'
        })
    
    # Overall assessment
    all_passed = all(check['passed'] for check in analysis['checks'])
    analysis['overall_passed'] = all_passed
    
    return analysis


async def run_stage_4_5():
    """Run Stage 4 and Stage 5 with analysis."""
    logger = ProjectLogger.get_logger(__name__)
    
    print("=" * 80)
    print("STAGE 4-5: MODELING & PREDICTION")
    print("=" * 80)
    
    # Load previous results
    results_dir = Path('data/stage_results')
    stage_2_3_file = results_dir / 'stage_2_3_results.json'
    
    if not stage_2_3_file.exists():
        print(f"\n❌ Previous results not found: {stage_2_3_file}")
        print("Please run stage_2_3.py first.")
        return None
    
    with open(stage_2_3_file, 'r') as f:
        previous_results = json.load(f)
    
    stage_3_results = previous_results.get('stage_3', {})
    
    # Initialize components
    config_manager = get_current_config()
    error_handler = ErrorHandler(config_manager)
    
    # Run Stage 4
    print("\n🔄 Running Stage 4: Modeling...")
    modeling_stage = ModelingStage(config_manager, error_handler)
    stage_4_results = await modeling_stage.run(
        features=stage_3_results.get('features'),
        targets=stage_3_results.get('targets')
    )
    
    # Analyze Stage 4
    stage_4_analysis = analyze_stage_4_results(stage_4_results)
    print(f"\n📊 Stage 4 Analysis:")
    print(f"   Status: {stage_4_analysis['status']}")
    print(f"   Overall Passed: {stage_4_analysis['overall_passed']}")
    for check in stage_4_analysis['checks']:
        status = "✅" if check['passed'] else "❌"
        print(f"   {status} {check['check']}: {check['message']}")
    
    if not stage_4_analysis['overall_passed']:
        print("\n❌ Stage 4 failed. Stopping pipeline.")
        return {
            'stage_4': stage_4_results,
            'stage_4_analysis': stage_4_analysis
        }
    
    # Run Stage 5
    print("\n🔄 Running Stage 5: Prediction...")
    prediction_stage = PredictionStage(config_manager, error_handler)
    stage_5_results = await prediction_stage.run(
        models=stage_4_results.get('trained_models'),
        features=stage_3_results.get('features')
    )
    
    # Analyze Stage 5
    stage_5_analysis = analyze_stage_5_results(stage_5_results)
    print(f"\n📊 Stage 5 Analysis:")
    print(f"   Status: {stage_5_analysis['status']}")
    print(f"   Overall Passed: {stage_5_analysis['overall_passed']}")
    for check in stage_5_analysis['checks']:
        status = "✅" if check['passed'] else "❌"
        print(f"   {status} {check['check']}: {check['message']}")
    
    if not stage_5_analysis['overall_passed']:
        print("\n❌ Stage 5 failed. Stopping pipeline.")
        return {
            'stage_4': stage_4_results,
            'stage_4_analysis': stage_4_analysis,
            'stage_5': stage_5_results,
            'stage_5_analysis': stage_5_analysis
        }
    
    # Save results
    with open(results_dir / 'stage_4_5_results.json', 'w') as f:
        json.dump({
            'stage_4': stage_4_results,
            'stage_4_analysis': stage_4_analysis,
            'stage_5': stage_5_results,
            'stage_5_analysis': stage_5_analysis
        }, f, indent=2, default=str)
    
    print(f"\n✅ Stage 4-5 completed successfully!")
    print(f"📁 Results saved to: {results_dir / 'stage_4_5_results.json'}")
    
    return {
        'stage_4': stage_4_results,
        'stage_4_analysis': stage_4_analysis,
        'stage_5': stage_5_results,
        'stage_5_analysis': stage_5_analysis
    }


if __name__ == '__main__':
    asyncio.run(run_stage_4_5())
