# models/arena/arena_battle.py - Arena Battle System for Trading Models

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import json
import logging
from dataclasses import dataclass
from enum import Enum

from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class BattleResult(Enum):
    """Результати бою між моделями"""
    MODEL1_WIN = "model1_win"
    MODEL2_WIN = "model2_win"
    DRAW = "draw"
    INCONCLUSIVE = "inconclusive"

@dataclass
class BattleMetrics:
    """Метрики для порівняння моделей"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    execution_time: float
    confidence_score: float

@dataclass
class Battle:
    """Інформація про бій між моделями"""
    model1_name: str
    model2_name: str
    battle_group: str
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[BattleResult] = None
    model1_metrics: Optional[BattleMetrics] = None
    model2_metrics: Optional[BattleMetrics] = None
    winner: Optional[str] = None
    vote_count: int = 0

class TradingModelArena:
    """
    Арена для порівняння трейдингових моделей side-by-side
    """
    
    def __init__(self):
        self.models = {}
        self.battle_history = []
        self.leaderboard = {}
        self.battle_groups = {}
        self.performance_tracker = None
        self.current_battles = []
        
        logger.info("[ARENA] Trading Model Arena initialized")
    
    def register_model(self, model_name: str, model_instance: Any, model_type: str = "traditional"):
        """Реєстрація моделі для арени"""
        try:
            self.models[model_name] = {
                'instance': model_instance,
                'type': model_type,
                'registered_at': datetime.now(),
                'battles_fought': 0,
                'wins': 0,
                'losses': 0,
                'draws': 0
            }
            
            logger.info(f"[ARENA] Model registered: {model_name} ({model_type})")
            return True
            
        except Exception as e:
            logger.error(f"[ARENA] Failed to register model {model_name}: {e}")
            return False
    
    def create_battle(self, model1_name: str, model2_name: str, battle_group: str = "custom") -> bool:
        """Створення бою між двома моделями"""
        try:
            if model1_name not in self.models or model2_name not in self.models:
                logger.error(f"[ARENA] Models not found: {model1_name}, {model2_name}")
                return False
            
            if model1_name == model2_name:
                logger.error(f"[ARENA] Cannot battle model against itself: {model1_name}")
                return False
            
            battle = Battle(
                model1_name=model1_name,
                model2_name=model2_name,
                battle_group=battle_group,
                start_time=datetime.now()
            )
            
            self.current_battles.append(battle)
            logger.info(f"[ARENA] Battle created: {model1_name} vs {model2_name} ({battle_group})")
            return True
            
        except Exception as e:
            logger.error(f"[ARENA] Failed to create battle: {e}")
            return False
    
    def run_battle(self, test_data: pd.DataFrame, actual_targets: pd.Series) -> Dict[str, Any]:
        """Виконання бою між моделями"""
        results = []
        
        for battle in self.current_battles:
            try:
                logger.info(f"[ARENA] Running battle: {battle.model1_name} vs {battle.model2_name}")
                
                # Отримуємо прогнози від обох моделей
                model1_predictions = self._get_model_predictions(battle.model1_name, test_data)
                model2_predictions = self._get_model_predictions(battle.model2_name, test_data)
                
                # Розраховуємо метрики
                model1_metrics = self._calculate_metrics(model1_predictions, actual_targets)
                model2_metrics = self._calculate_metrics(model2_predictions, actual_targets)
                
                # Визначаємо переможця
                winner = self._determine_battle_winner(model1_metrics, model2_metrics)
                
                # Оновлюємо інформацію про бій
                battle.end_time = datetime.now()
                battle.model1_metrics = model1_metrics
                battle.model2_metrics = model2_metrics
                battle.winner = winner
                battle.result = self._get_battle_result(winner)
                
                # Оновлюємо статистику моделей
                self._update_model_stats(battle)
                
                # Додаємо до історії
                self.battle_history.append(battle)
                
                battle_result = {
                    'battle_id': len(self.battle_history),
                    'model1': battle.model1_name,
                    'model2': battle.model2_name,
                    'winner': winner,
                    'model1_metrics': model1_metrics,
                    'model2_metrics': model2_metrics,
                    'battle_duration': (battle.end_time - battle.start_time).total_seconds(),
                    'timestamp': battle.end_time.isoformat()
                }
                
                results.append(battle_result)
                logger.info(f"[ARENA] Battle completed: {winner} wins")
                
            except Exception as e:
                logger.error(f"[ARENA] Battle failed: {e}")
                results.append({'error': str(e)})
        
        # Очищуємо поточні бої
        self.current_battles.clear()
        
        return {
            'battles_completed': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_model_predictions(self, model_name: str, test_data: pd.DataFrame) -> np.ndarray:
        """Отримання прогнозів від моделі"""
        try:
            model_info = self.models[model_name]
            model_instance = model_info['instance']
            
            # Для різних типів моделей використовуємо різні методи
            if hasattr(model_instance, 'predict'):
                predictions = model_instance.predict(test_data)
            elif hasattr(model_instance, 'forecast'):
                predictions = model_instance.forecast(test_data)
            elif hasattr(model_instance, 'predict_all_models_final'):
                # Для MoneyMaker моделей
                result = model_instance.predict_all_models_final(
                    models_dict={model_name: model_instance},
                    df_features=test_data
                )
                predictions = result.get('ensemble_enhanced', np.zeros(len(test_data)))
            else:
                # Імітація прогнозів для демонстрації
                predictions = np.random.normal(0, 0.01, len(test_data))
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"[ARENA] Failed to get predictions from {model_name}: {e}")
            return np.zeros(len(test_data))
    
    def _calculate_metrics(self, predictions: np.ndarray, actual_targets: pd.Series) -> BattleMetrics:
        """Розрахунок метрик для моделі"""
        try:
            # Конвертуємо actual_targets в numpy array
            actuals = actual_targets.values if hasattr(actual_targets, 'values') else actual_targets
            
            # Базові метрики
            mse = np.mean((predictions - actuals) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - actuals))
            
            # Accuracy (для бінарної класифікації напрямку)
            pred_direction = np.sign(predictions)
            actual_direction = np.sign(actuals)
            accuracy = np.mean(pred_direction == actual_direction)
            
            # Precision, Recall, F1 (для бінарної класифікації)
            precision = accuracy  # Спрощено для демонстрації
            recall = accuracy
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Фінансові метрики
            returns = predictions
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) if np.std(returns) > 0 else 0
            max_drawdown = self._calculate_max_drawdown(returns)
            win_rate = accuracy  # Спрощено
            
            # Confidence score
            confidence_score = min(accuracy + 0.1, 1.0)
            
            return BattleMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                execution_time=0.1,  # Спрощено
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"[ARENA] Failed to calculate metrics: {e}")
            return BattleMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Розрахунок максимального просідання"""
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        except:
            return 0.0
    
    def _determine_battle_winner(self, metrics1: BattleMetrics, metrics2: BattleMetrics) -> str:
        """Визначення переможця бою на основі метрик"""
        try:
            # Ваги для різних метрик
            weights = {
                'accuracy': 0.3,
                'sharpe_ratio': 0.25,
                'win_rate': 0.2,
                'max_drawdown': 0.15,  # менше краще
                'confidence_score': 0.1
            }
            
            # Розраховуємо загальний бал для кожної моделі
            score1 = (
                weights['accuracy'] * metrics1.accuracy +
                weights['sharpe_ratio'] * min(metrics1.sharpe_ratio, 2) +  # Обмежуємо Sharpe
                weights['win_rate'] * metrics1.win_rate +
                weights['max_drawdown'] * (1 - abs(metrics1.max_drawdown)) +  # Менше drawdown краще
                weights['confidence_score'] * metrics1.confidence_score
            )
            
            score2 = (
                weights['accuracy'] * metrics2.accuracy +
                weights['sharpe_ratio'] * min(metrics2.sharpe_ratio, 2) +
                weights['win_rate'] * metrics2.win_rate +
                weights['max_drawdown'] * (1 - abs(metrics2.max_drawdown)) +
                weights['confidence_score'] * metrics2.confidence_score
            )
            
            # Визначаємо переможця
            if abs(score1 - score2) < 0.05:  # Різниця менше 5% - нічиия
                return "draw"
            elif score1 > score2:
                return metrics1.model_name if hasattr(metrics1, 'model_name') else "model1"
            else:
                return metrics2.model_name if hasattr(metrics2, 'model_name') else "model2"
                
        except Exception as e:
            logger.error(f"[ARENA] Failed to determine winner: {e}")
            return "draw"
    
    def _get_battle_result(self, winner: str) -> BattleResult:
        """Отримання результату бою"""
        if winner == "draw":
            return BattleResult.DRAW
        elif winner == "model1":
            return BattleResult.MODEL1_WIN
        elif winner == "model2":
            return BattleResult.MODEL2_WIN
        else:
            return BattleResult.INCONCLUSIVE
    
    def _update_model_stats(self, battle: Battle):
        """Оновлення статистики моделей"""
        try:
            # Оновлюємо статистику для model1
            if battle.model1_name in self.models:
                self.models[battle.model1_name]['battles_fought'] += 1
                if battle.winner == battle.model1_name:
                    self.models[battle.model1_name]['wins'] += 1
                elif battle.winner == "draw":
                    self.models[battle.model1_name]['draws'] += 1
                else:
                    self.models[battle.model1_name]['losses'] += 1
            
            # Оновлюємо статистику для model2
            if battle.model2_name in self.models:
                self.models[battle.model2_name]['battles_fought'] += 1
                if battle.winner == battle.model2_name:
                    self.models[battle.model2_name]['wins'] += 1
                elif battle.winner == "draw":
                    self.models[battle.model2_name]['draws'] += 1
                else:
                    self.models[battle.model2_name]['losses'] += 1
                    
        except Exception as e:
            logger.error(f"[ARENA] Failed to update model stats: {e}")
    
    def get_leaderboard(self) -> Dict[str, Any]:
        """Отримання таблиці лідерів"""
        try:
            leaderboard = []
            
            for model_name, model_info in self.models.items():
                if model_info['battles_fought'] > 0:
                    win_rate = model_info['wins'] / model_info['battles_fought']
                    
                    leaderboard.append({
                        'model_name': model_name,
                        'model_type': model_info['type'],
                        'battles_fought': model_info['battles_fought'],
                        'wins': model_info['wins'],
                        'losses': model_info['losses'],
                        'draws': model_info['draws'],
                        'win_rate': win_rate,
                        'points': model_info['wins'] * 3 + model_info['draws']  # 3 points for win, 1 for draw
                    })
            
            # Сортуємо за очками
            leaderboard.sort(key=lambda x: x['points'], reverse=True)
            
            return {
                'leaderboard': leaderboard,
                'total_models': len(leaderboard),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"[ARENA] Failed to get leaderboard: {e}")
            return {'leaderboard': [], 'total_models': 0, 'last_updated': datetime.now().isoformat()}
    
    def run_tournament(self, battle_groups: List[str], test_data: pd.DataFrame, actual_targets: pd.Series) -> Dict[str, Any]:
        """Запуск турніру між моделями"""
        try:
            logger.info(f"[ARENA] Starting tournament with groups: {battle_groups}")
            
            tournament_results = {
                'tournament_id': len(self.battle_history) + 1,
                'battle_groups': battle_groups,
                'start_time': datetime.now().isoformat(),
                'battles': [],
                'leaderboard_before': self.get_leaderboard(),
                'leaderboard_after': None
            }
            
            # Створюємо бої для кожної групи
            for group in battle_groups:
                group_models = self._get_models_for_group(group)
                
                if len(group_models) < 2:
                    logger.warning(f"[ARENA] Not enough models for group {group}: {len(group_models)}")
                    continue
                
                # Створюємо попарні бої
                for i in range(len(group_models)):
                    for j in range(i + 1, len(group_models)):
                        model1 = group_models[i]
                        model2 = group_models[j]
                        
                        if self.create_battle(model1, model2, group):
                            logger.info(f"[ARENA] Tournament battle: {model1} vs {model2}")
            
            # Виконуємо бої
            battle_results = self.run_battle(test_data, actual_targets)
            tournament_results['battles'] = battle_results['results']
            tournament_results['battles_completed'] = battle_results['battles_completed']
            
            # Оновлюємо таблицю лідерів
            tournament_results['leaderboard_after'] = self.get_leaderboard()
            tournament_results['end_time'] = datetime.now().isoformat()
            
            logger.info(f"[ARENA] Tournament completed: {battle_results['battles_completed']} battles")
            
            return tournament_results
            
        except Exception as e:
            logger.error(f"[ARENA] Tournament failed: {e}")
            return {'error': str(e)}
    
    def _get_models_for_group(self, group: str) -> List[str]:
        """Отримання моделей для групи боїв"""
        try:
            if group == "traditional_vs_enhanced":
                traditional = [name for name, info in self.models.items() if info['type'] == 'traditional']
                enhanced = [name for name, info in self.models.items() if info['type'] == 'enhanced']
                return traditional[:3] + enhanced[:3]  # По 3 з кожної категорії
            
            elif group == "light_vs_heavy":
                light_models = ['lgbm', 'rf', 'xgboost', 'catboost', 'linear', 'mlp']
                heavy_models = ['lstm', 'gru', 'transformer', 'cnn', 'tabnet', 'autoencoder']
                
                available_light = [name for name in light_models if name in self.models]
                available_heavy = [name for name in heavy_models if name in self.models]
                
                return available_light[:3] + available_heavy[:3]
            
            elif group == "all_models":
                return list(self.models.keys())[:8]  # Перші 8 моделей
            
            else:
                return list(self.models.keys())
                
        except Exception as e:
            logger.error(f"[ARENA] Failed to get models for group {group}: {e}")
            return []
    
    def save_arena_state(self, filepath: str) -> bool:
        """Збереження стану арени"""
        try:
            arena_state = {
                'models': self.models,
                'battle_history': [
                    {
                        'model1_name': b.model1_name,
                        'model2_name': b.model2_name,
                        'battle_group': b.battle_group,
                        'start_time': b.start_time.isoformat(),
                        'end_time': b.end_time.isoformat() if b.end_time else None,
                        'result': b.result.value if b.result else None,
                        'winner': b.winner,
                        'vote_count': b.vote_count
                    } for b in self.battle_history
                ],
                'leaderboard': self.get_leaderboard(),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(arena_state, f, indent=2)
            
            logger.info(f"[ARENA] Arena state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"[ARENA] Failed to save arena state: {e}")
            return False
    
    def load_arena_state(self, filepath: str) -> bool:
        """Завантаження стану арени"""
        try:
            with open(filepath, 'r') as f:
                arena_state = json.load(f)
            
            # Відновлюємо моделі (without інстансів)
            self.models = arena_state.get('models', {})
            
            # Відновлюємо історію боїв
            self.battle_history = []
            for battle_data in arena_state.get('battle_history', []):
                battle = Battle(
                    model1_name=battle_data['model1_name'],
                    model2_name=battle_data['model2_name'],
                    battle_group=battle_data['battle_group'],
                    start_time=datetime.fromisoformat(battle_data['start_time']),
                    end_time=datetime.fromisoformat(battle_data['end_time']) if battle_data['end_time'] else None,
                    result=BattleResult(battle_data['result']) if battle_data['result'] else None,
                    winner=battle_data['winner'],
                    vote_count=battle_data['vote_count']
                )
                self.battle_history.append(battle)
            
            logger.info(f"[ARENA] Arena state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"[ARENA] Failed to load arena state: {e}")
            return False

# Глобальна функція для отримання арени
def get_trading_arena() -> TradingModelArena:
    """Отримання глобального екземпляру арени"""
    global _trading_arena
    if '_trading_arena' not in globals():
        _trading_arena = TradingModelArena()
    return _trading_arena
