"""
Comprehensive analysis of signals and model performance.
"""
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def analyze_signals(signal_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs a detailed analysis of model performance based on signal data.

    Args:
        signal_data (Dict[str, Any]): The raw signal data loaded from a JSON file.

    Returns:
        Dict[str, Any]: A dictionary containing a comprehensive analysis.
    """
    analysis = {
        'summary': {},
        'models': {},
        'tickers': {},
        'timeframes': {},
        'warnings': [],
        'recommendations': []
    }

    model_performance = {}
    ticker_performance = {}
    timeframe_performance = {}

    # Analyze each model from the signal data
    for model_key, model_data in signal_data.items():
        model_name = model_key.split('_')[0]
        
        if model_name not in model_performance:
            model_performance[model_name] = {
                'mse_scores': [], 'mae_scores': [], 'accuracy_scores': [],
                'signals': [], 'combinations': []
            }
        
        for combination, results in model_data.items():
            if 'metrics' in results:
                metrics = results['metrics']
                model_performance[model_name]['mse_scores'].append(metrics.get('mse', 0))
                model_performance[model_name]['mae_scores'].append(metrics.get('mae', 0))
                model_performance[model_name]['accuracy_scores'].append(metrics.get('accuracy', 0))
                model_performance[model_name]['signals'].append(results.get('final_signal', 'HOLD'))
                model_performance[model_name]['combinations'].append(combination)
                
                # Ticker-level analysis
                ticker = combination.split('_')[0]
                if ticker not in ticker_performance:
                    ticker_performance[ticker] = {'models': {}, 'signals': []}
                if model_name not in ticker_performance[ticker]['models']:
                    ticker_performance[ticker]['models'][model_name] = []
                ticker_performance[ticker]['models'][model_name].append({'mse': metrics.get('mse', 0), 'mae': metrics.get('mae', 0), 'accuracy': metrics.get('accuracy', 0), 'signal': results.get('final_signal', 'HOLD')})
                ticker_performance[ticker]['signals'].append(results.get('final_signal', 'HOLD'))
                
                # Timeframe-level analysis
                timeframe = combination.split('_')[1]
                if timeframe not in timeframe_performance:
                    timeframe_performance[timeframe] = {'models': {}, 'signals': []}
                if model_name not in timeframe_performance[timeframe]['models']:
                    timeframe_performance[timeframe]['models'][model_name] = []
                timeframe_performance[timeframe]['models'][model_name].append({'mse': metrics.get('mse', 0), 'mae': metrics.get('mae', 0), 'accuracy': metrics.get('accuracy', 0), 'signal': results.get('final_signal', 'HOLD')})
                timeframe_performance[timeframe]['signals'].append(results.get('final_signal', 'HOLD'))

    # Calculate average metrics for each model
    for model_name, perf in model_performance.items():
        if perf['mse_scores']:
            avg_mse = np.mean(perf['mse_scores'])
            avg_mae = np.mean(perf['mae_scores'])
            avg_accuracy = np.mean(perf['accuracy_scores'])
            
            if avg_accuracy < 0:
                analysis['warnings'].append(f"CRITICAL: {model_name} has negative accuracy ({avg_accuracy:.2f})")
            elif avg_accuracy < 0.5:
                analysis['warnings'].append(f"WARNING: {model_name} has low accuracy ({avg_accuracy:.2f})")
            
            analysis['models'][model_name] = {
                'avg_mse': avg_mse, 'avg_mae': avg_mae, 'avg_accuracy': avg_accuracy,
                'signal_distribution': {'BUY': perf['signals'].count('BUY'), 'SELL': perf['signals'].count('SELL'), 'HOLD': perf['signals'].count('HOLD')},
                'total_combinations': len(perf['combinations']),
                'best_combination': _find_best_combination(perf),
                'worst_combination': _find_worst_combination(perf)
            }
    
    # Ticker analysis
    for ticker, perf in ticker_performance.items():
        total_signals = len(perf['signals'])
        buy_signals = perf['signals'].count('BUY')
        sell_signals = perf['signals'].count('SELL')
        hold_signals = perf['signals'].count('HOLD')
        analysis['tickers'][ticker] = {
            'signal_distribution': {'BUY': buy_signals, 'SELL': sell_signals, 'HOLD': hold_signals, 'total': total_signals},
            'buy_percentage': (buy_signals / total_signals * 100) if total_signals > 0 else 0,
            'sell_percentage': (sell_signals / total_signals * 100) if total_signals > 0 else 0,
            'hold_percentage': (hold_signals / total_signals * 100) if total_signals > 0 else 0,
            'model_performance': perf['models']
        }
    
    # Timeframe analysis
    for timeframe, perf in timeframe_performance.items():
        total_signals = len(perf['signals'])
        buy_signals = perf['signals'].count('BUY')
        sell_signals = perf['signals'].count('SELL')
        hold_signals = perf['signals'].count('HOLD')
        analysis['timeframes'][timeframe] = {
            'signal_distribution': {'BUY': buy_signals, 'SELL': sell_signals, 'HOLD': hold_signals, 'total': total_signals},
            'buy_percentage': (buy_signals / total_signals * 100) if total_signals > 0 else 0,
            'sell_percentage': (sell_signals / total_signals * 100) if total_signals > 0 else 0,
            'hold_percentage': (hold_signals / total_signals * 100) if total_signals > 0 else 0,
            'model_performance': perf['models']
        }
    
    # Final summary
    analysis['summary'] = {
        'total_models': len(model_performance),
        'total_combinations': sum(len(p['combinations']) for p in model_performance.values()),
        'analysis_timestamp': datetime.now().isoformat(),
        'warnings_count': len(analysis['warnings']),
        'best_model': _find_best_overall_model(analysis['models']),
        'worst_model': _find_worst_overall_model(analysis['models'])
    }
    
    analysis['recommendations'] = _generate_recommendations(analysis)
    
    return analysis

def _find_best_combination(performance: Dict) -> Dict:
    """Finds the best performing combination for a model based on accuracy."""
    if not performance['accuracy_scores']: return {}
    best_idx = np.argmax(performance['accuracy_scores'])
    return {
        'combination': performance['combinations'][best_idx],
        'accuracy': performance['accuracy_scores'][best_idx],
        'mse': performance['mse_scores'][best_idx],
        'mae': performance['mae_scores'][best_idx],
        'signal': performance['signals'][best_idx]
    }

def _find_worst_combination(performance: Dict) -> Dict:
    """Finds the worst performing combination for a model based on accuracy."""
    if not performance['accuracy_scores']: return {}
    worst_idx = np.argmin(performance['accuracy_scores'])
    return {
        'combination': performance['combinations'][worst_idx],
        'accuracy': performance['accuracy_scores'][worst_idx],
        'mse': performance['mse_scores'][worst_idx],
        'mae': performance['mae_scores'][worst_idx],
        'signal': performance['signals'][worst_idx]
    }

def _find_best_overall_model(models: Dict) -> str:
    """Finds the best overall model based on average accuracy."""
    if not models: return "N/A"
    return max(models.keys(), key=lambda x: models[x]['avg_accuracy'])

def _find_worst_overall_model(models: Dict) -> str:
    """Finds the worst overall model based on average accuracy."""
    if not models: return "N/A"
    return min(models.keys(), key=lambda x: models[x]['avg_accuracy'])

def _generate_recommendations(analysis: Dict) -> List[str]:
    """Generates actionable recommendations based on the analysis."""
    recommendations = []
    for name, model in analysis['models'].items():
        if model['avg_accuracy'] < 0.5:
            recommendations.append(f"RETRAIN {name}: Low accuracy ({model['avg_accuracy']:.2f})")
    for ticker, data in analysis['tickers'].items():
        if data.get('hold_percentage', 0) > 90:
            recommendations.append(f"ANALYZE {ticker}: High percentage of HOLD signals ({data['hold_percentage']:.1f}%)")
    if analysis['summary'].get('warnings_count', 0) > 5:
        recommendations.append("URGENT: Multiple critical issues detected - review all models.")
    return recommendations

def generate_report_string(analysis: Dict) -> str:
    """Generates a comprehensive, human-readable report string from the analysis."""
    report_lines = [
        "="*80, "COMPREHENSIVE SIGNAL ANALYSIS REPORT", "="*80,
        "\n[SUMMARY]:",
        f"   Total Models: {analysis['summary']['total_models']}",
        f"   Total Combinations: {analysis['summary']['total_combinations']}",
        f"   Best Model: {analysis['summary']['best_model']}",
        f"   Worst Model: {analysis['summary']['worst_model']}",
        f"   Warnings: {analysis['summary']['warnings_count']}",
    ]
    if analysis['warnings']:
        report_lines.append(f"\n[WARNINGS] ({len(analysis['warnings'])}):")
        report_lines.extend([f"    {w}" for w in analysis['warnings']])
    
    report_lines.append("\n[MODEL PERFORMANCE]:")
    for name, model in analysis['models'].items():
        status = "[OK] GOOD" if model['avg_accuracy'] > 0.7 else "[WARN] FAIR" if model['avg_accuracy'] > 0.5 else "[ERROR] POOR"
        report_lines.extend([
            f"\n   {name.upper()} {status}",
            f"   Accuracy: {model['avg_accuracy']:.4f}",
            f"   Signals: BUY:{model['signal_distribution']['BUY']} SELL:{model['signal_distribution']['SELL']} HOLD:{model['signal_distribution']['HOLD']}"
        ])

    if analysis['recommendations']:
        report_lines.append("\n[RECOMMENDATIONS]:")
        report_lines.extend([f"    {rec}" for rec in analysis['recommendations']])

    report_lines.append("\n" + "="*80)
    return "\n".join(report_lines)
