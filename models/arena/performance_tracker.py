# models/arena/performance_tracker.py - Performance Tracking System

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict

from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

@dataclass
class ModelPerformanceRecord:
    """Запис продуктивності моделі"""
    model_name: str
    model_type: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    execution_time: float
    confidence_score: float
    battle_id: Optional[int] = None
    opponent: Optional[str] = None
    result: Optional[str] = None

@dataclass
class LeaderboardEntry:
    """Запис в таблиці лідерів"""
    rank: int
    model_name: str
    model_type: str
    total_battles: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    points: int
    avg_accuracy: float
    avg_sharpe_ratio: float
    avg_win_rate: float
    last_updated: datetime

class ModelPerformanceTracker:
    """Трекер продуктивності моделей"""
    
    def __init__(self):
        self.performance_history: List[ModelPerformanceRecord] = []
        self.model_stats: Dict[str, Dict] = defaultdict(dict)
        self.leaderboard: List[LeaderboardEntry] = []
        self.battle_results: List[Dict] = []
        
        # Ініціалізуємо статистику для всіх моделей
        self._initialize_model_stats()
        
        logger.info("[TRACKER] Model Performance Tracker initialized")
    
    def _initialize_model_stats(self):
        """Ініціалізація статистики моделей"""
        # Список всіх можливих моделей
        all_models = [
            # Traditional Models
            "lgbm", "rf", "xgboost", "catboost", "linear", "mlp", "svm", "knn",
            # Heavy Models
            "lstm", "gru", "transformer", "cnn", "tabnet", "autoencoder",
            # Enhanced Models
            "dean_ensemble", "sentiment", "lgbm_bayesian"
        ]
        
        for model in all_models:
            self.model_stats[model] = {
                'total_battles': 0,
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'accuracy_scores': [],
                'sharpe_ratios': [],
                'win_rates': [],
                'confidence_scores': [],
                'execution_times': [],
                'last_battle': None,
                'current_streak': 0,
                'best_streak': 0,
                'model_type': self._get_model_type(model)
            }
    
    def _get_model_type(self, model_name: str) -> str:
        """Визначити тип моделі"""
        enhanced_models = ["dean_ensemble", "sentiment", "lgbm_bayesian"]
        heavy_models = ["lstm", "gru", "transformer", "cnn", "tabnet", "autoencoder"]
        
        if model_name in enhanced_models:
            return "enhanced"
        elif model_name in heavy_models:
            return "heavy"
        else:
            return "light"
    
    def record_battle_performance(self, battle_data: Dict[str, Any]) -> bool:
        """Запис продуктивності після бою"""
        try:
            battle_id = battle_data.get('battle_id')
            model1_name = battle_data.get('model1')
            model2_name = battle_data.get('model2')
            winner = battle_data.get('winner')
            
            if not all([battle_id, model1_name, model2_name]):
                logger.error("[TRACKER] Missing required battle data")
                return False
            
            # Записуємо продуктивність для обох моделей
            model1_metrics = battle_data.get('model1_metrics')
            model2_metrics = battle_data.get('model2_metrics')
            
            if model1_metrics:
                self._record_model_performance(model1_name, model1_metrics, battle_id, model2_name, 
                                              self._get_battle_result(model1_name, winner))
            
            if model2_metrics:
                self._record_model_performance(model2_name, model2_metrics, battle_id, model1_name,
                                              self._get_battle_result(model2_name, winner))
            
            # Оновлюємо статистику боїв
            self._update_battle_stats(model1_name, model2_name, winner)
            
            # Додаємо результат бою
            self.battle_results.append({
                'battle_id': battle_id,
                'model1': model1_name,
                'model2': model2_name,
                'winner': winner,
                'timestamp': datetime.now().isoformat()
            })
            
            # Оновлюємо таблицю лідерів
            self._update_leaderboard()
            
            logger.info(f"[TRACKER] Battle performance recorded: {model1_name} vs {model2_name}")
            return True
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to record battle performance: {e}")
            return False
    
    def _record_model_performance(self, model_name: str, metrics: Dict[str, Any], 
                                battle_id: int, opponent: str, result: str):
        """Запис продуктивності окремої моделі"""
        try:
            record = ModelPerformanceRecord(
                model_name=model_name,
                model_type=self.model_stats[model_name]['model_type'],
                timestamp=datetime.now(),
                accuracy=metrics.get('accuracy', 0.0),
                precision=metrics.get('precision', 0.0),
                recall=metrics.get('recall', 0.0),
                f1_score=metrics.get('f1_score', 0.0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                max_drawdown=metrics.get('max_drawdown', 0.0),
                win_rate=metrics.get('win_rate', 0.0),
                execution_time=metrics.get('execution_time', 0.0),
                confidence_score=metrics.get('confidence_score', 0.0),
                battle_id=battle_id,
                opponent=opponent,
                result=result
            )
            
            self.performance_history.append(record)
            
            # Оновлюємо агреговану статистику
            stats = self.model_stats[model_name]
            stats['accuracy_scores'].append(metrics.get('accuracy', 0.0))
            stats['sharpe_ratios'].append(metrics.get('sharpe_ratio', 0.0))
            stats['win_rates'].append(metrics.get('win_rate', 0.0))
            stats['confidence_scores'].append(metrics.get('confidence_score', 0.0))
            stats['execution_times'].append(metrics.get('execution_time', 0.0))
            stats['last_battle'] = datetime.now()
            
            # Обмежуємо історію до останніх 100 записів на модель
            if len(stats['accuracy_scores']) > 100:
                stats['accuracy_scores'] = stats['accuracy_scores'][-100:]
                stats['sharpe_ratios'] = stats['sharpe_ratios'][-100:]
                stats['win_rates'] = stats['win_rates'][-100:]
                stats['confidence_scores'] = stats['confidence_scores'][-100:]
                stats['execution_times'] = stats['execution_times'][-100:]
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to record model performance for {model_name}: {e}")
    
    def _get_battle_result(self, model_name: str, winner: str) -> str:
        """Отримати результат бою для моделі"""
        if winner == "draw":
            return "draw"
        elif winner == model_name:
            return "win"
        else:
            return "loss"
    
    def _update_battle_stats(self, model1: str, model2: str, winner: str):
        """Оновлення статистики боїв"""
        try:
            for model in [model1, model2]:
                stats = self.model_stats[model]
                stats['total_battles'] += 1
                
                result = self._get_battle_result(model, winner)
                
                if result == "win":
                    stats['wins'] += 1
                    stats['current_streak'] = max(stats['current_streak'], 0) + 1
                    stats['best_streak'] = max(stats['best_streak'], stats['current_streak'])
                elif result == "loss":
                    stats['losses'] += 1
                    stats['current_streak'] = min(stats['current_streak'], 0) - 1
                else:  # draw
                    stats['draws'] += 1
                    stats['current_streak'] = 0  # Скидаємо серію при нічиї
                    
        except Exception as e:
            logger.error(f"[TRACKER] Failed to update battle stats: {e}")
    
    def _update_leaderboard(self):
        """Оновлення таблиці лідерів"""
        try:
            leaderboard_entries = []
            
            for model_name, stats in self.model_stats.items():
                if stats['total_battles'] > 0:
                    # Розраховуємо середні показники
                    avg_accuracy = np.mean(stats['accuracy_scores']) if stats['accuracy_scores'] else 0.0
                    avg_sharpe = np.mean(stats['sharpe_ratios']) if stats['sharpe_ratios'] else 0.0
                    avg_win_rate = np.mean(stats['win_rates']) if stats['win_rates'] else 0.0
                    avg_confidence = np.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0.0
                    
                    # Розраховуємо win rate
                    win_rate = stats['wins'] / stats['total_battles']
                    
                    # Розраховуємо очки (3 за перемогу, 1 за нічию)
                    points = stats['wins'] * 3 + stats['draws']
                    
                    entry = LeaderboardEntry(
                        rank=0,  # Буде встановлено після сортування
                        model_name=model_name,
                        model_type=stats['model_type'],
                        total_battles=stats['total_battles'],
                        wins=stats['wins'],
                        losses=stats['losses'],
                        draws=stats['draws'],
                        win_rate=win_rate,
                        points=points,
                        avg_accuracy=avg_accuracy,
                        avg_sharpe_ratio=avg_sharpe,
                        avg_win_rate=avg_win_rate,
                        last_updated=datetime.now()
                    )
                    
                    leaderboard_entries.append(entry)
            
            # Сортуємо за очками (потім за win rate)
            leaderboard_entries.sort(key=lambda x: (x.points, x.win_rate), reverse=True)
            
            # Встановлюємо ранги
            for i, entry in enumerate(leaderboard_entries):
                entry.rank = i + 1
            
            self.leaderboard = leaderboard_entries
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to update leaderboard: {e}")
    
    def get_leaderboard(self, limit: int = 10) -> Dict[str, Any]:
        """Отримати таблицю лідерів"""
        try:
            top_entries = self.leaderboard[:limit]
            
            return {
                'leaderboard': [asdict(entry) for entry in top_entries],
                'total_models': len(self.leaderboard),
                'last_updated': datetime.now().isoformat(),
                'categories': self._get_leaderboard_categories()
            }
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to get leaderboard: {e}")
            return {'leaderboard': [], 'total_models': 0, 'last_updated': datetime.now().isoformat()}
    
    def _get_leaderboard_categories(self) -> Dict[str, List[Dict]]:
        """Отримати категорії таблиці лідерів"""
        try:
            categories = {
                'overall': [],
                'traditional': [],
                'enhanced': [],
                'heavy': [],
                'light': []
            }
            
            for entry in self.leaderboard:
                category_entry = asdict(entry)
                model_type = entry.model_type
                
                categories['overall'].append(category_entry)
                
                if model_type in ['traditional', 'light', 'heavy']:
                    categories[model_type].append(category_entry)
                elif model_type == 'enhanced':
                    categories['enhanced'].append(category_entry)
            
            # Сортуємо кожну категорію
            for category in categories:
                categories[category].sort(key=lambda x: (x['points'], x['win_rate']), reverse=True)
                # Встановлюємо ранги в категоріях
                for i, entry in enumerate(categories[category]):
                    entry['category_rank'] = i + 1
            
            return categories
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to get leaderboard categories: {e}")
            return {}
    
    def get_model_performance_history(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """Отримати історію продуктивності моделі"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            model_records = [
                record for record in self.performance_history
                if record.model_name == model_name and record.timestamp >= cutoff_date
            ]
            
            if not model_records:
                return {
                    'model_name': model_name,
                    'records': [],
                    'summary': {},
                    'period_days': days
                }
            
            # Розраховуємо статистику
            accuracies = [r.accuracy for r in model_records]
            sharpe_ratios = [r.sharpe_ratio for r in model_records]
            win_rates = [r.win_rate for r in model_records]
            
            summary = {
                'total_battles': len(model_records),
                'avg_accuracy': np.mean(accuracies),
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'avg_win_rate': np.mean(win_rates),
                'best_accuracy': max(accuracies),
                'best_sharpe_ratio': max(sharpe_ratios),
                'worst_accuracy': min(accuracies),
                'worst_sharpe_ratio': min(sharpe_ratios),
                'performance_trend': self._calculate_performance_trend(accuracies)
            }
            
            return {
                'model_name': model_name,
                'records': [asdict(r) for r in model_records],
                'summary': summary,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to get model performance history: {e}")
            return {'model_name': model_name, 'records': [], 'summary': {}, 'period_days': days}
    
    def _calculate_performance_trend(self, values: List[float]) -> str:
        """Розрахувати тренд продуктивності"""
        try:
            if len(values) < 2:
                return "insufficient_data"
            
            # Розраховуємо тренд за допомогою простої лінійної регресії
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            if slope > 0.01:
                return "improving"
            elif slope < -0.01:
                return "declining"
            else:
                return "stable"
                
        except:
            return "unknown"
    
    def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """Отримати детальну статистику моделі"""
        try:
            if model_name not in self.model_stats:
                return {'error': f'Model {model_name} not found'}
            
            stats = self.model_stats[model_name]
            
            # Розраховуємо середні показники
            avg_accuracy = np.mean(stats['accuracy_scores']) if stats['accuracy_scores'] else 0.0
            avg_sharpe = np.mean(stats['sharpe_ratios']) if stats['sharpe_ratios'] else 0.0
            avg_win_rate = np.mean(stats['win_rates']) if stats['win_rates'] else 0.0
            avg_confidence = np.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0.0
            
            # Знаходимо позицію в таблиці лідерів
            leaderboard_position = None
            for entry in self.leaderboard:
                if entry.model_name == model_name:
                    leaderboard_position = entry.rank
                    break
            
            return {
                'model_name': model_name,
                'model_type': stats['model_type'],
                'total_battles': stats['total_battles'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'draws': stats['draws'],
                'win_rate': stats['wins'] / stats['total_battles'] if stats['total_battles'] > 0 else 0.0,
                'current_streak': stats['current_streak'],
                'best_streak': stats['best_streak'],
                'leaderboard_position': leaderboard_position,
                'avg_accuracy': avg_accuracy,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_win_rate': avg_win_rate,
                'avg_confidence': avg_confidence,
                'last_battle': stats['last_battle'].isoformat() if stats['last_battle'] else None,
                'performance_trend': self._calculate_performance_trend(stats['accuracy_scores'])
            }
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to get model stats: {e}")
            return {'error': str(e)}
    
    def get_top_performers(self, metric: str = 'points', limit: int = 5) -> List[Dict]:
        """Отримати топ виконавців за метрикою"""
        try:
            if metric == 'points':
                sorted_models = sorted(self.leaderboard, key=lambda x: x.points, reverse=True)
            elif metric == 'win_rate':
                sorted_models = sorted(self.leaderboard, key=lambda x: x.win_rate, reverse=True)
            elif metric == 'accuracy':
                sorted_models = sorted(self.leaderboard, key=lambda x: x.avg_accuracy, reverse=True)
            elif metric == 'sharpe_ratio':
                sorted_models = sorted(self.leaderboard, key=lambda x: x.avg_sharpe_ratio, reverse=True)
            else:
                sorted_models = self.leaderboard
            
            top_models = sorted_models[:limit]
            return [asdict(model) for model in top_models]
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to get top performers: {e}")
            return []
    
    def save_performance_data(self, filepath: str) -> bool:
        """Зберегти дані продуктивності"""
        try:
            data = {
                'performance_history': [asdict(record) for record in self.performance_history],
                'model_stats': dict(self.model_stats),
                'leaderboard': [asdict(entry) for entry in self.leaderboard],
                'battle_results': self.battle_results,
                'last_updated': datetime.now().isoformat()
            }
            
            # Конвертуємо datetime об'єкти в строки для JSON
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj
            
            data = convert_datetime(data)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"[TRACKER] Performance data saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to save performance data: {e}")
            return False
    
    def load_performance_data(self, filepath: str) -> bool:
        """Завантажити дані продуктивності"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Відновлюємо історію продуктивності
            self.performance_history = []
            for record_data in data.get('performance_history', []):
                record = ModelPerformanceRecord(
                    model_name=record_data['model_name'],
                    model_type=record_data['model_type'],
                    timestamp=datetime.fromisoformat(record_data['timestamp']),
                    accuracy=record_data['accuracy'],
                    precision=record_data['precision'],
                    recall=record_data['recall'],
                    f1_score=record_data['f1_score'],
                    sharpe_ratio=record_data['sharpe_ratio'],
                    max_drawdown=record_data['max_drawdown'],
                    win_rate=record_data['win_rate'],
                    execution_time=record_data['execution_time'],
                    confidence_score=record_data['confidence_score'],
                    battle_id=record_data.get('battle_id'),
                    opponent=record_data.get('opponent'),
                    result=record_data.get('result')
                )
                self.performance_history.append(record)
            
            # Відновлюємо статистику моделей
            self.model_stats = defaultdict(dict)
            for model_name, stats in data.get('model_stats', {}).items():
                self.model_stats[model_name] = stats
            
            # Відновлюємо таблицю лідерів
            self.leaderboard = []
            for entry_data in data.get('leaderboard', []):
                entry = LeaderboardEntry(
                    rank=entry_data['rank'],
                    model_name=entry_data['model_name'],
                    model_type=entry_data['model_type'],
                    total_battles=entry_data['total_battles'],
                    wins=entry_data['wins'],
                    losses=entry_data['losses'],
                    draws=entry_data['draws'],
                    win_rate=entry_data['win_rate'],
                    points=entry_data['points'],
                    avg_accuracy=entry_data['avg_accuracy'],
                    avg_sharpe_ratio=entry_data['avg_sharpe_ratio'],
                    avg_win_rate=entry_data['avg_win_rate'],
                    last_updated=datetime.fromisoformat(entry_data['last_updated'])
                )
                self.leaderboard.append(entry)
            
            # Відновлюємо результати боїв
            self.battle_results = data.get('battle_results', [])
            
            logger.info(f"[TRACKER] Performance data loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"[TRACKER] Failed to load performance data: {e}")
            return False

# Глобальна функція
def get_performance_tracker() -> ModelPerformanceTracker:
    """Отримати глобальний трекер продуктивності"""
    global _performance_tracker
    if '_performance_tracker' not in globals():
        _performance_tracker = ModelPerformanceTracker()
    return _performance_tracker
