"""
Comprehensive analysis of signals and model performance.
"""
import logging
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

def analyze_signals(signal_data: dict[str, Any]) -> dict[str, Any]:
    """
    Performs a detailed analysis of model performance based on signal data.

    Args:
        signal_data (Dict[str, Any]): The raw signal data loaded from a JSON file.

    Returns:
        Dict[str, Any]: A dictionary containing a comprehensive analysis.
    """
    analysis: dict[str, Any] = {
        'summary': {},
        'models': {},
        'tickers': {},
        'timeframes': {},
        'warnings': [],
        'recommendations': []
    }

    # Initialize performance trackers
    model_performance, ticker_performance, timeframe_performance = _initialize_performance_trackers()

    # Process signal data
    _process_signal_data(signal_data, model_performance, ticker_performance, timeframe_performance)

    # Analyze model performance
    _analyze_model_performance(model_performance, analysis)

    # Analyze ticker performance
    _analyze_ticker_performance(ticker_performance, analysis)

    # Analyze timeframe performance
    _analyze_timeframe_performance(timeframe_performance, analysis)

    # Generate final summary
    _generate_final_summary(model_performance, analysis)

    # Generate recommendations
    analysis['recommendations'] = _generate_recommendations(analysis)

    return analysis

def _initialize_performance_trackers() -> tuple:
    """Initialize performance tracking dictionaries"""
    return {}, {}, {}

def _process_signal_data(signal_data: dict[str, Any], model_performance: dict,
                        ticker_performance: dict, timeframe_performance: dict) -> None:
    """Process raw signal data and populate performance trackers"""
    for model_key, model_data in signal_data.items():
        model_name = model_key.split('_')[0]

        # Initialize model performance if needed
        if model_name not in model_performance:
            model_performance[model_name] = {
                'mse_scores': [], 'mae_scores': [], 'accuracy_scores': [],
                'signals': [], 'combinations': []
            }

        for combination, results in model_data.items():
            if 'metrics' in results:
                _process_combination_metrics(combination, results, model_name,
                                           model_performance, ticker_performance, timeframe_performance)

def _process_combination_metrics(combination: str, results: dict, model_name: str,
                                model_performance: dict, ticker_performance: dict,
                                timeframe_performance: dict) -> None:
    """Process metrics for a single combination"""
    metrics = results['metrics']

    # Update model performance
    model_performance[model_name]['mse_scores'].append(metrics.get('mse', 0))
    model_performance[model_name]['mae_scores'].append(metrics.get('mae', 0))
    model_performance[model_name]['accuracy_scores'].append(metrics.get('accuracy', 0))
    model_performance[model_name]['signals'].append(results.get('final_signal', 'HOLD'))
    model_performance[model_name]['combinations'].append(combination)

    # Extract ticker and timeframe
    ticker = combination.split('_')[0]
    timeframe = combination.split('_')[1]

    # Update ticker performance
    _update_ticker_performance(ticker, model_name, metrics, results, ticker_performance)

    # Update timeframe performance
    _update_timeframe_performance(timeframe, model_name, metrics, results, timeframe_performance)

def _update_ticker_performance(ticker: str, model_name: str, metrics: dict,
                              results: dict, ticker_performance: dict) -> None:
    """Update performance data for a specific ticker"""
    if ticker not in ticker_performance:
        ticker_performance[ticker] = {'models': {}, 'signals': []}
    if model_name not in ticker_performance[ticker]['models']:
        ticker_performance[ticker]['models'][model_name] = []

    ticker_performance[ticker]['models'][model_name].append({
        'mse': metrics.get('mse', 0),
        'mae': metrics.get('mae', 0),
        'accuracy': metrics.get('accuracy', 0),
        'signal': results.get('final_signal', 'HOLD')
    })
    ticker_performance[ticker]['signals'].append(results.get('final_signal', 'HOLD'))

def _update_timeframe_performance(timeframe: str, model_name: str, metrics: dict,
                                results: dict, timeframe_performance: dict) -> None:
    """Update performance data for a specific timeframe"""
    if timeframe not in timeframe_performance:
        timeframe_performance[timeframe] = {'models': {}, 'signals': []}
    if model_name not in timeframe_performance[timeframe]['models']:
        timeframe_performance[timeframe]['models'][model_name] = []

    timeframe_performance[timeframe]['models'][model_name].append({
        'mse': metrics.get('mse', 0),
        'mae': metrics.get('mae', 0),
        'accuracy': metrics.get('accuracy', 0),
        'signal': results.get('final_signal', 'HOLD')
    })
    timeframe_performance[timeframe]['signals'].append(results.get('final_signal', 'HOLD'))

def _analyze_model_performance(model_performance: dict, analysis: dict) -> None:
    """Analyze and summarize model performance"""
    for model_name, perf in model_performance.items():
        if not perf['mse_scores']:
            continue

        avg_mse = np.mean(perf['mse_scores'])
        avg_mae = np.mean(perf['mae_scores'])
        avg_accuracy = np.mean(perf['accuracy_scores'])

        # Add warnings for poor performance
        if avg_accuracy < 0:
            analysis['warnings'].append(f"CRITICAL: {model_name} has negative accuracy ({avg_accuracy:.2f})")
        elif avg_accuracy < 0.5:
            analysis['warnings'].append(f"WARNING: {model_name} has low accuracy ({avg_accuracy:.2f})")

        analysis['models'][model_name] = {
            'avg_mse': avg_mse,
            'avg_mae': avg_mae,
            'avg_accuracy': avg_accuracy,
            'signal_distribution': _calculate_signal_distribution(perf['signals']),
            'total_combinations': len(perf['combinations']),
            'best_combination': _find_best_combination(perf),
            'worst_combination': _find_worst_combination(perf)
        }

def _calculate_signal_distribution(signals: list[str]) -> dict[str, int]:
    """Calculate distribution of signal types"""
    return {
        'BUY': signals.count('BUY'),
        'SELL': signals.count('SELL'),
        'HOLD': signals.count('HOLD')
    }

def _analyze_ticker_performance(ticker_performance: dict, analysis: dict) -> None:
    """Analyze and summarize ticker performance"""
    for ticker, perf in ticker_performance.items():
        signal_stats = _calculate_signal_percentages(perf['signals'])

        analysis['tickers'][ticker] = {
            'signal_distribution': signal_stats['distribution'],
            'buy_percentage': signal_stats['buy_percentage'],
            'sell_percentage': signal_stats['sell_percentage'],
            'hold_percentage': signal_stats['hold_percentage'],
            'model_performance': perf['models']
        }

def _analyze_timeframe_performance(timeframe_performance: dict, analysis: dict) -> None:
    """Analyze and summarize timeframe performance"""
    for timeframe, perf in timeframe_performance.items():
        signal_stats = _calculate_signal_percentages(perf['signals'])

        analysis['timeframes'][timeframe] = {
            'signal_distribution': signal_stats['distribution'],
            'buy_percentage': signal_stats['buy_percentage'],
            'sell_percentage': signal_stats['sell_percentage'],
            'hold_percentage': signal_stats['hold_percentage'],
            'model_performance': perf['models']
        }

def _calculate_signal_percentages(signals: list[str]) -> dict[str, Any]:
    """Calculate signal distribution and percentages"""
    total_signals = len(signals)
    buy_signals = signals.count('BUY')
    sell_signals = signals.count('SELL')
    hold_signals = signals.count('HOLD')

    # Calculate percentages separately to avoid nested conditionals
    buy_percentage = _calculate_percentage(buy_signals, total_signals)
    sell_percentage = _calculate_percentage(sell_signals, total_signals)
    hold_percentage = _calculate_percentage(hold_signals, total_signals)

    return {
        'distribution': {
            'BUY': buy_signals,
            'SELL': sell_signals,
            'HOLD': hold_signals,
            'total': total_signals
        },
        'buy_percentage': buy_percentage,
        'sell_percentage': sell_percentage,
        'hold_percentage': hold_percentage
    }

def _calculate_percentage(count: int, total: int) -> float:
    """Calculate percentage with zero division protection"""
    return (count / total * 100) if total > 0 else 0

def _generate_final_summary(model_performance: dict, analysis: dict) -> None:
    """Generate final summary statistics"""
    analysis['summary'] = {
        'total_models': len(model_performance),
        'total_combinations': sum(len(p['combinations']) for p in model_performance.values()),
        'analysis_timestamp': datetime.now().isoformat(),
        'warnings_count': len(analysis['warnings']),
        'best_model': _find_best_overall_model(analysis['models']),
        'worst_model': _find_worst_overall_model(analysis['models'])
    }

def _find_best_combination(performance: dict) -> dict:
    """Finds the best performing combination for a model based on accuracy."""
    if not performance['accuracy_scores']:
        return {}
    best_idx = np.argmax(performance['accuracy_scores'])
    return {
        'combination': performance['combinations'][best_idx],
        'accuracy': performance['accuracy_scores'][best_idx],
        'mse': performance['mse_scores'][best_idx],
        'mae': performance['mae_scores'][best_idx],
        'signal': performance['signals'][best_idx]
    }

def _find_worst_combination(performance: dict) -> dict:
    """Finds the worst performing combination for a model based on accuracy."""
    if not performance['accuracy_scores']:
        return {}
    worst_idx = np.argmin(performance['accuracy_scores'])
    return {
        'combination': performance['combinations'][worst_idx],
        'accuracy': performance['accuracy_scores'][worst_idx],
        'mse': performance['mse_scores'][worst_idx],
        'mae': performance['mae_scores'][worst_idx],
        'signal': performance['signals'][worst_idx]
    }

def _find_best_overall_model(models: dict) -> str:
    """Finds the best overall model based on average accuracy."""
    if not models:
        return "N/A"
    return str(max(models.keys(), key=lambda x: models[x]['avg_accuracy']))

def _find_worst_overall_model(models: dict) -> str:
    """Finds the worst overall model based on average accuracy."""
    if not models:
        return "N/A"
    return str(min(models.keys(), key=lambda x: models[x]['avg_accuracy']))

def _generate_recommendations(analysis: dict) -> list[str]:
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

def generate_report_string(analysis: dict) -> str:
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
        if model['avg_accuracy'] > 0.7:
            status = "[OK] GOOD"
        elif model['avg_accuracy'] > 0.5:
            status = "[WARN] FAIR"
        else:
            status = "[ERROR] POOR"
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
