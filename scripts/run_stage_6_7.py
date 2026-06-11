#!/usr/bin/env python3
"""
Stage 6-7: Trading Execution & Evaluation
Runs Stage 6 (Trading Execution) and Stage 7 (Evaluation) with result analysis.
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
from src.pipeline.stages.stage_6_trading_execution import TradingExecutionStage
from src.pipeline.stages.stage_7_evaluation import EvaluationStage


def analyze_stage_6_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Stage 6 results for correctness and completeness."""
    analysis = {
        'stage': 'Stage 6 - Trading Execution',
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
    
    # Check trades executed
    trades = results.get('trades')
    if trades is not None:
        if isinstance(trades, pd.DataFrame):
            analysis['checks'].append({
                'check': 'Trades executed',
                'passed': True,
                'message': f"Executed {len(trades)} trades"
            })
        elif isinstance(trades, list):
            analysis['checks'].append({
                'check': 'Trades executed',
                'passed': True,
                'message': f"Executed {len(trades)} trades"
            })
        else:
            analysis['checks'].append({
                'check': 'Trades executed',
                'passed': False,
                'message': f"Trades is not DataFrame/list: {type(trades)}"
            })
    else:
        analysis['checks'].append({
            'check': 'Trades executed',
            'passed': False,
            'message': 'No trades'
        })
    
    # Check portfolio
    portfolio = results.get('portfolio')
    if portfolio is not None:
        analysis['checks'].append({
            'check': 'Portfolio updated',
            'passed': True,
            'message': 'Portfolio updated successfully'
        })
    else:
        analysis['checks'].append({
            'check': 'Portfolio updated',
            'passed': False,
            'message': 'Portfolio not updated'
        })
    
    # Check risk metrics
    risk_metrics = results.get('risk_metrics')
    if risk_metrics:
        analysis['checks'].append({
            'check': 'Risk metrics calculated',
            'passed': True,
            'message': f"Risk metrics: {list(risk_metrics.keys())}"
        })
    else:
        analysis['checks'].append({
            'check': 'Risk metrics calculated',
            'passed': False,
            'message': 'No risk metrics'
        })
    
    # Overall assessment
    all_passed = all(check['passed'] for check in analysis['checks'])
    analysis['overall_passed'] = all_passed
    
    return analysis


def analyze_stage_7_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Stage 7 results for correctness and completeness."""
    analysis = {
        'stage': 'Stage 7 - Evaluation',
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
    
    # Check backtest results
    backtest_results = results.get('backtest_results')
    if backtest_results:
        analysis['checks'].append({
            'check': 'Backtest completed',
            'passed': True,
            'message': 'Backtest results available'
        })
    else:
        analysis['checks'].append({
            'check': 'Backtest completed',
            'passed': False,
            'message': 'No backtest results'
        })
    
    # Check metrics
    metrics = results.get('metrics')
    if metrics:
        analysis['checks'].append({
            'check': 'Performance metrics calculated',
            'passed': True,
            'message': f"Metrics: {list(metrics.keys())}"
        })
    else:
        analysis['checks'].append({
            'check': 'Performance metrics calculated',
            'passed': False,
            'message': 'No performance metrics'
        })
    
    # Check reports generated
    reports = results.get('reports')
    if reports:
        analysis['checks'].append({
            'check': 'Reports generated',
            'passed': True,
            'message': f"Generated {len(reports)} reports"
        })
    else:
        analysis['checks'].append({
            'check': 'Reports generated',
            'passed': False,
            'message': 'No reports generated'
        })
    
    # Check visualizations
    visualizations = results.get('visualizations')
    if visualizations:
        analysis['checks'].append({
            'check': 'Visualizations created',
            'passed': True,
            'message': f"Created {len(visualizations)} visualizations"
        })
    else:
        analysis['checks'].append({
            'check': 'Visualizations created',
            'passed': False,
            'message': 'No visualizations'
        })
    
    # Overall assessment
    all_passed = all(check['passed'] for check in analysis['checks'])
    analysis['overall_passed'] = all_passed
    
    return analysis


async def run_stage_6_7():
    """Run Stage 6 and Stage 7 with analysis."""
    logger = ProjectLogger.get_logger(__name__)
    
    print("=" * 80)
    print("STAGE 6-7: TRADING EXECUTION & EVALUATION")
    print("=" * 80)
    
    # Load previous results
    results_dir = Path('data/stage_results')
    stage_4_5_file = results_dir / 'stage_4_5_results.json'
    
    if not stage_4_5_file.exists():
        print(f"\n❌ Previous results not found: {stage_4_5_file}")
        print("Please run stage_4_5.py first.")
        return None
    
    with open(stage_4_5_file, 'r') as f:
        previous_results = json.load(f)
    
    stage_5_results = previous_results.get('stage_5', {})
    
    # Initialize components
    config_manager = get_current_config()
    error_handler = ErrorHandler(config_manager)
    
    # Run Stage 6
    print("\n🔄 Running Stage 6: Trading Execution...")
    trading_stage = TradingExecutionStage(config_manager, error_handler)
    stage_6_results = await trading_stage.run(
        predictions=stage_5_results.get('predictions'),
        features=previous_results.get('stage_3', {}).get('features')
    )
    
    # Analyze Stage 6
    stage_6_analysis = analyze_stage_6_results(stage_6_results)
    print(f"\n📊 Stage 6 Analysis:")
    print(f"   Status: {stage_6_analysis['status']}")
    print(f"   Overall Passed: {stage_6_analysis['overall_passed']}")
    for check in stage_6_analysis['checks']:
        status = "✅" if check['passed'] else "❌"
        print(f"   {status} {check['check']}: {check['message']}")
    
    if not stage_6_analysis['overall_passed']:
        print("\n❌ Stage 6 failed. Stopping pipeline.")
        return {
            'stage_6': stage_6_results,
            'stage_6_analysis': stage_6_analysis
        }
    
    # Run Stage 7
    print("\n🔄 Running Stage 7: Evaluation...")
    evaluation_stage = EvaluationStage(config_manager, error_handler)
    stage_7_results = await evaluation_stage.run(
        trades=stage_6_results.get('trades'),
        portfolio=stage_6_results.get('portfolio')
    )
    
    # Analyze Stage 7
    stage_7_analysis = analyze_stage_7_results(stage_7_results)
    print(f"\n📊 Stage 7 Analysis:")
    print(f"   Status: {stage_7_analysis['status']}")
    print(f"   Overall Passed: {stage_7_analysis['overall_passed']}")
    for check in stage_7_analysis['checks']:
        status = "✅" if check['passed'] else "❌"
        print(f"   {status} {check['check']}: {check['message']}")
    
    if not stage_7_analysis['overall_passed']:
        print("\n❌ Stage 7 failed. Stopping pipeline.")
        return {
            'stage_6': stage_6_results,
            'stage_6_analysis': stage_6_analysis,
            'stage_7': stage_7_results,
            'stage_7_analysis': stage_7_analysis
        }
    
    # Save results
    with open(results_dir / 'stage_6_7_results.json', 'w') as f:
        json.dump({
            'stage_6': stage_6_results,
            'stage_6_analysis': stage_6_analysis,
            'stage_7': stage_7_results,
            'stage_7_analysis': stage_7_analysis
        }, f, indent=2, default=str)
    
    print(f"\n✅ Stage 6-7 completed successfully!")
    print(f"📁 Results saved to: {results_dir / 'stage_6_7_results.json'}")
    
    return {
        'stage_6': stage_6_results,
        'stage_6_analysis': stage_6_analysis,
        'stage_7': stage_7_results,
        'stage_7_analysis': stage_7_analysis
    }


if __name__ == '__main__':
    asyncio.run(run_stage_6_7())
