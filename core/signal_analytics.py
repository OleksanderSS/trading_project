"""
Максимальний аналandwith сигналandв and продуктивностand моwhereлей
"""
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

class SignalAnalytics:
    def __init__(self):
        self.analysis_data = {}
        
    def analyze_model_performance(self, signals_file: str) -> Dict[str, Any]:
        """Деandльний аналandwith продуктивностand моwhereлей"""
        with open(signals_file, 'r') as f:
            data = json.load(f)
        
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
        
        # Аналandwithуємо кожну model
        for model_key, model_data in data.items():
            model_name = model_key.split('_')[0]
            
            if model_name not in model_performance:
                model_performance[model_name] = {
                    'mse_scores': [],
                    'mae_scores': [],
                    'accuracy_scores': [],
                    'signals': [],
                    'combinations': []
                }
            
            for combination, results in model_data.items():
                if 'metrics' in results:
                    metrics = results['metrics']
                    model_performance[model_name]['mse_scores'].append(metrics.get('mse', 0))
                    model_performance[model_name]['mae_scores'].append(metrics.get('mae', 0))
                    model_performance[model_name]['accuracy_scores'].append(metrics.get('accuracy', 0))
                    model_performance[model_name]['signals'].append(results.get('final_signal', 'HOLD'))
                    model_performance[model_name]['combinations'].append(combination)
                    
                    # Аналandwith for тandкерandв
                    ticker = combination.split('_')[0]
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
                    
                    # Аналandwith for andймфреймandв
                    timeframe = combination.split('_')[1]
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
        
        # Calculating середнand покаwithники for кожної моwhereлand
        for model_name, perf in model_performance.items():
            if perf['mse_scores']:
                avg_mse = np.mean(perf['mse_scores'])
                avg_mae = np.mean(perf['mae_scores'])
                avg_accuracy = np.mean(perf['accuracy_scores'])
                
                # Перевandрка на problemsнand моwhereлand
                if avg_accuracy < 0:
                    analysis['warnings'].append(f"CRITICAL: {model_name} has negative accuracy ({avg_accuracy:.2f})")
                elif avg_accuracy < 0.5:
                    analysis['warnings'].append(f"WARNING: {model_name} has low accuracy ({avg_accuracy:.2f})")
                elif avg_mse > 1:
                    analysis['warnings'].append(f"WARNING: {model_name} has high MSE ({avg_mse:.2f})")
                
                analysis['models'][model_name] = {
                    'avg_mse': avg_mse,
                    'avg_mae': avg_mae,
                    'avg_accuracy': avg_accuracy,
                    'signal_distribution': {
                        'BUY': perf['signals'].count('BUY'),
                        'SELL': perf['signals'].count('SELL'),
                        'HOLD': perf['signals'].count('HOLD')
                    },
                    'total_combinations': len(perf['combinations']),
                    'best_combination': self._find_best_combination(perf),
                    'worst_combination': self._find_worst_combination(perf)
                }
        
        # Аналandwith for тandкерandв
        for ticker, perf in ticker_performance.items():
            buy_signals = perf['signals'].count('BUY')
            sell_signals = perf['signals'].count('SELL')
            hold_signals = perf['signals'].count('HOLD')
            total_signals = len(perf['signals'])
            
            analysis['tickers'][ticker] = {
                'signal_distribution': {
                    'BUY': buy_signals,
                    'SELL': sell_signals,
                    'HOLD': hold_signals,
                    'total': total_signals
                },
                'buy_percentage': (buy_signals / total_signals * 100) if total_signals > 0 else 0,
                'sell_percentage': (sell_signals / total_signals * 100) if total_signals > 0 else 0,
                'hold_percentage': (hold_signals / total_signals * 100) if total_signals > 0 else 0,
                'model_performance': perf['models']
            }
        
        # Аналandwith for andймфреймandв
        for timeframe, perf in timeframe_performance.items():
            buy_signals = perf['signals'].count('BUY')
            sell_signals = perf['signals'].count('SELL')
            hold_signals = perf['signals'].count('HOLD')
            total_signals = len(perf['signals'])
            
            analysis['timeframes'][timeframe] = {
                'signal_distribution': {
                    'BUY': buy_signals,
                    'SELL': sell_signals,
                    'HOLD': hold_signals,
                    'total': total_signals
                },
                'buy_percentage': (buy_signals / total_signals * 100) if total_signals > 0 else 0,
                'sell_percentage': (sell_signals / total_signals * 100) if total_signals > 0 else 0,
                'hold_percentage': (hold_signals / total_signals * 100) if total_signals > 0 else 0,
                'model_performance': perf['models']
            }
        
        # Загальний withвandт
        total_models = len(model_performance)
        total_combinations = sum(len(perf['combinations']) for perf in model_performance.values())
        
        analysis['summary'] = {
            'total_models': total_models,
            'total_combinations': total_combinations,
            'analysis_timestamp': datetime.now().isoformat(),
            'warnings_count': len(analysis['warnings']),
            'best_model': self._find_best_overall_model(analysis['models']),
            'worst_model': self._find_worst_overall_model(analysis['models'])
        }
        
        # Рекомендацandї
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _find_best_combination(self, performance: Dict) -> Dict:
        """Знаходить найкращу комбandнацandю for моwhereлand"""
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
    
    def _find_worst_combination(self, performance: Dict) -> Dict:
        """Знаходить найгandршу комбandнацandю for моwhereлand"""
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
    
    def _find_best_overall_model(self, models: Dict) -> str:
        """Знаходить найкращу model forгалом"""
        if not models:
            return "N/A"
        
        best_model = max(models.keys(), key=lambda x: models[x]['avg_accuracy'])
        return best_model
    
    def _find_worst_overall_model(self, models: Dict) -> str:
        """Знаходить найгandршу model forгалом"""
        if not models:
            return "N/A"
        
        worst_model = min(models.keys(), key=lambda x: models[x]['avg_accuracy'])
        return worst_model
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Геnotрує рекомендацandї на основand аналandwithу"""
        recommendations = []
        
        # Рекомендацandї for problemsних моwhereлей
        for model_name, model_data in analysis['models'].items():
            if model_data['avg_accuracy'] < 0:
                recommendations.append(f"DISABLE {model_name}: Critical negative accuracy detected")
            elif model_data['avg_accuracy'] < 0.5:
                recommendations.append(f"RETRAIN {model_name}: Low accuracy ({model_data['avg_accuracy']:.2f})")
            elif model_data['avg_mse'] > 1:
                recommendations.append(f"CHECK {model_name}: High MSE ({model_data['avg_mse']:.2f})")
        
        # Рекомендацandї for тandкерandв
        for ticker, ticker_data in analysis['tickers'].items():
            if isinstance(ticker_data['hold_percentage'], (int, float)) and ticker_data['hold_percentage'] > 90:
                recommendations.append(f"ANALYZE {ticker}: Too many HOLD signals ({ticker_data['hold_percentage']:.1f}%)")
        
        # Загальнand рекомендацandї
        if isinstance(analysis['summary']['warnings_count'],
            (int,
            float)) and analysis['summary']['warnings_count'] > 5:
            recommendations.append("URGENT: Multiple critical issues detected - review all models")
        
        return recommendations
    
    def print_comprehensive_report(self, analysis: Dict):
        """
        Друкує комплексний withвandт
        """
        print("\n" + "="*80)
        print("[SEARCH] COMPREHENSIVE SIGNAL ANALYSIS REPORT")
        print("="*80)
        
        # Загальна andнформацandя
        summary = analysis['summary']
        print(f"\n[DATA] SUMMARY:")
        print(f"   Total Models: {summary['total_models']}")
        print(f"   Total Combinations: {summary['total_combinations']}")
        print(f"   Best Model: {summary['best_model']}")
        print(f"   Worst Model: {summary['worst_model']}")
        print(f"   Warnings: {summary['warnings_count']}")
        
        # Попередження
        if analysis['warnings']:
            print(f"\n[WARN]  WARNINGS ({len(analysis['warnings'])}):")
            for warning in analysis['warnings']:
                print(f"    {warning}")
        
        # Аналandwith моwhereлей
        print(f"\n MODEL PERFORMANCE:")
        for model_name, model_data in analysis['models'].items():
            avg_acc = model_data.get('avg_accuracy', 0)
            try:
                avg_acc = float(avg_acc)
            except (ValueError, TypeError):
                avg_acc = 0.0
            status = "[OK] GOOD" if avg_acc > 0.7 else "[WARN] FAIR" if avg_acc > 0.5 else "[ERROR] POOR"
            print(f"\n   {model_name.upper()} {status}")
            print(f"   Accuracy: {model_data['avg_accuracy']:.4f}")
            print(f"   MSE: {model_data['avg_mse']:.6f}")
            print(f"   MAE: {model_data['avg_mae']:.6f}")
            print(f"   Signals: BUY:{model_data['signal_distribution']['BUY']} SELL:{model_data['signal_distribution']['SELL']} HOLD:{model_data['signal_distribution']['HOLD']}")
            
            if model_data['best_combination']:
                best = model_data['best_combination']
                print(f"   Best: {best['combination']} (acc:{best['accuracy']:.4f})")
        
        # Аналandwith тandкерandв
        print(f"\n[UP] TICKER ANALYSIS:")
        for ticker, ticker_data in analysis['tickers'].items():
            print(f"\n   {ticker}:")
            print(f"   BUY: {ticker_data['buy_percentage']:.1f}% | SELL: {ticker_data['sell_percentage']:.1f}% | HOLD: {ticker_data['hold_percentage']:.1f}%")
        
        # Аналandwith andймфреймandв
        print(f"\n TIMEFRAME ANALYSIS:")
        for timeframe, timeframe_data in analysis['timeframes'].items():
            print(f"\n   {timeframe}:")
            print(f"   BUY: {timeframe_data['buy_percentage']:.1f}% | SELL: {timeframe_data['sell_percentage']:.1f}% | HOLD: {timeframe_data['hold_percentage']:.1f}%")
        
        # Рекомендацandї
        if analysis['recommendations']:
            print(f"\n[IDEA] RECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                print(f"    {rec}")
        
        print("\n" + "="*80)
    
    def save_analysis(self, analysis: Dict, filename: str = None):
        """Зберandгає аналandwith у file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/signals/signal_analysis_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f" Analysis saved: {filename}")
        return filename
