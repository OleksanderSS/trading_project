import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.core.logging.logger import ProjectLogger
from src.utils.artifact_security import resolve_trusted_artifact_path

from .battle_groups import get_battle_group_manager
from src.analytics.context.dynamic_router import DynamicRouter

rng = np.random.default_rng(42)
logger = ProjectLogger.get_logger(__name__)


class BattleResult(Enum):
    """Результати бою між моделями"""
    MODEL1_WIN = 'model1_win'
    MODEL2_WIN = 'model2_win'
    DRAW = 'draw'
    INCONCLUSIVE = 'inconclusive'
    CHAMPION_RETAINED = 'champion_retained'
    CHAMPION_REPLACED = 'champion_replaced'


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
    log_loss: float = 0.0
    mse: float = 0.0
    financial_loss: float = 0.0
    structural_alignment: float = 0.0
    realization_gap: float = 0.0


@dataclass
class Battle:
    """Інформація про бій між моделями"""
    model1_name: str
    model2_name: str
    battle_group: str
    start_time: datetime
    end_time: datetime | None = None
    result: BattleResult | None = None
    model1_metrics: BattleMetrics | None = None
    model2_metrics: BattleMetrics | None = None
    winner: str | None = None
    vote_count: int = 0


class TradingModelArena:
    """
    Арена для порівняння трейдингових моделей side-by-side з правилами безпеки Champion.
    Включає етап 'The Reveal' (Blinded Simulation) для причинно-наслідкового аналізу.
    """

    def __init__(self, champion_dir: str='trained_models', safety_margin:
        float=0.05):
        self.models: dict[str, Any] = {}
        self.battle_history: list[Battle] = []
        self.leaderboard: dict[str, float] = {}
        self.battle_group_manager = get_battle_group_manager()
        self.performance_tracker = None
        self.current_battles: list[Battle] = []
        self.champion_dir = Path(champion_dir)
        self.safety_margin = safety_margin
        self.dynamic_router = DynamicRouter()
        logger.info(
            f'[ARENA] Trading Model Arena initialized. Safety margin: {safety_margin * 100}%'
            )
        logger.info(
            f'[ARENA] Available battle groups: {self.battle_group_manager.list_groups()}'
            )

    def register_model(self, model_name: str, model_instance: Any,
        model_type: str='traditional'):
        """Реєстрація моделі для арени"""
        try:
            activations = getattr(model_instance, 'activation_types', 'unknown'
                )
            params = getattr(model_instance, 'params', {})
            self.models[model_name] = {'instance': model_instance, 'type':
                model_type, 'registered_at': datetime.now(), 'activations':
                activations, 'params': params, 'battles_fought': 0, 'wins':
                0, 'losses': 0, 'draws': 0}
            logger.info(
                f'[ARENA] Model registered: {model_name} ({model_type}) | Activations: {activations}'
                )
            return True
        except (AttributeError, Exception) as e:
            raise RuntimeError(f"Failed to register model {model_name}: {e}") from e

    def calculate_loss_metrics(self, predictions: np.ndarray, actuals: np.
        ndarray, probs: np.ndarray | None=None) ->dict[str, float]:
        """Розрахунок розширених метрик втрат, включаючи фінансові втрати."""
        mse = np.mean((predictions - actuals) ** 2)
        l_loss = 0.0
        if probs is not None:
            try:
                binary_actuals = (actuals > 0).astype(int)
                l_loss = log_loss(binary_actuals, probs)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError):
                logger.exception('Помилка розрахунку метрик втрат (log_loss)')
                l_loss = 1.0
        financial_loss = np.mean(np.abs(predictions - actuals) * np.abs(
            actuals))
        return {'mse': float(mse), 'log_loss': float(l_loss),
            'financial_loss': float(financial_loss)}

    def compare_with_baseline(self, actual_targets: pd.Series) ->dict[str,
        float]:
        """Порівняння з базовим орієнтиром (Moving Average / VAR)."""
        baseline_series = actual_targets.shift(1).rolling(window=5).mean().dropna()
        aligned_actuals = actual_targets.loc[baseline_series.index]

        baseline_preds = baseline_series.values
        actuals = aligned_actuals.values

        baseline_loss = self.calculate_loss_metrics(baseline_preds, actuals)
        logger.info(
            f"[ARENA] Baseline established (MA-5). MSE: {baseline_loss['mse']:.6f} | FinLoss: {baseline_loss['financial_loss']:.6f}"
            )
        return baseline_loss

    def run_blind_challenge(self, model_name: str, context_data: pd.
        DataFrame, real_outcome: pd.Series) ->dict[str, Any]:
        """
        Фаза 'The Reveal' (Blinded Simulation): Оцінка причинно-наслідкового розуміння моделі.
        """
        logger.info(
            f'[ARENA] Phase 1 (Blinded): Model {model_name} conducting simulations...'
            )
        model_info = self.models.get(model_name)
        if not model_info:
            return {'error': 'Model not found'}
        model = model_info['instance']
        num_simulations = 50
        simulations = []
        for _ in range(num_simulations):
            # Only add noise to numeric columns
            noisy_context = context_data.copy()
            numeric_cols = noisy_context.select_dtypes(include=[np.number]).columns
            noise = rng.normal(0, 0.001, (len(noisy_context), len(numeric_cols)))
            noisy_context[numeric_cols] = noisy_context[numeric_cols] + noise
            sim_pred = self._get_instance_predictions(model, noisy_context)
            simulations.append(sim_pred)
        sim_matrix = np.array(simulations)
        sim_mean = np.mean(sim_matrix, axis=0)
        volatility = context_data.std(axis=1).values if len(context_data.
            columns) > 1 else np.ones(len(context_data))
        structural_alignment = np.corrcoef(np.abs(sim_mean), volatility)[0, 1
            ] if np.std(sim_mean) > 0 else 0.5
        logger.info(
            '[ARENA] Phase 3: The Reveal. Comparing simulations to real outcome.'
            )
        real_vals = real_outcome.values if hasattr(real_outcome, 'values'
            ) else real_outcome
        realization_gap = np.mean(np.abs(sim_mean - real_vals))
        logger.info(
            f'[ARENA] Insight: Model {model_name} | Alignment: {structural_alignment:.4f} | Gap: {realization_gap:.4f}'
            )
        return {'structural_alignment': float(structural_alignment),
            'realization_gap': float(realization_gap), 'sim_mean': sim_mean}

    def _load_current_champion(self, ticker: str, target: str) ->tuple[str, Any] | None:
        """Завантажує поточного чемпіона для конкретного тікера та таргету."""
        try:
            champ_files = list(self.champion_dir.glob(
                f'CHAMP_{ticker}_{target}_*.joblib'))
            if not champ_files:
                return None
            latest_champ = max(champ_files, key=os.path.getctime)
            trusted_champ = resolve_trusted_artifact_path(
                latest_champ,
                allowed_roots=(self.champion_dir,),
                allowed_suffixes={'.joblib'},
                must_exist=True,
            )
            model = joblib.load(trusted_champ)  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
            return latest_champ.name, model
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            raise RuntimeError(f'[ARENA] Could not load champion for {ticker}_{target}: {e}') from e

    def conduct_battle(self, ticker: str, target: str, candidate_name: str,
        test_data: pd.DataFrame, actual_targets: pd.Series) ->dict[str, Any]:
        """
        Головний метод проведення бою з фільтром строгості проти Baseline.
        """
        baseline_metrics = self.compare_with_baseline(actual_targets)
        self.run_blind_challenge(candidate_name, test_data, actual_targets)
        cand_preds = self._get_model_predictions(candidate_name, test_data)
        cand_probs = None
        cand_losses = self.calculate_loss_metrics(cand_preds,
            actual_targets.values, cand_probs)
        metric_to_check = 'log_loss' if cand_losses['log_loss'] > 0 else 'mse'
        cand_perf = cand_losses[metric_to_check]
        base_perf = baseline_metrics[metric_to_check]
        is_significant = cand_perf < base_perf * 0.9
        if not is_significant:
            logger.warning(
                f'[ARENA] Candidate {candidate_name} failed Strictness Filter. Improvement over baseline < 10%.'
                )
            return {'result': BattleResult.CHAMPION_RETAINED, 'reason':
                'Insufficient improvement over baseline'}
        champ_data = self._load_current_champion(ticker, target)
        if not champ_data:
            logger.info(f'[ARENA] Champion vacancy filled by {candidate_name}.'
                )
            return {'result': BattleResult.CHAMPION_REPLACED, 'reason':
                'No existing champion'}
        champ_name, champ_instance = champ_data
        self.register_model(champ_name, champ_instance, model_type='champion')
        result = self.run_champion_challenge(ticker, target, candidate_name,
            test_data, actual_targets)
        return result

    def run_champion_challenge(self, ticker: str, target: str,
        candidate_name: str, test_data: pd.DataFrame, actual_targets: pd.Series
        ) ->dict[str, Any]:
        """Бій претендента проти чинного чемпіона з використанням 'The Reveal'."""
        champ_name = f'CHAMP_{ticker}_{target}'
        cand_blind = self.run_blind_challenge(candidate_name, test_data,
            actual_targets)
        champ_blind = self.run_blind_challenge(champ_name, test_data,
            actual_targets)
        cand_preds = self._get_model_predictions(candidate_name, test_data)
        champ_preds = self._get_model_predictions(champ_name, test_data)
        cand_metrics = self._calculate_metrics(cand_preds, actual_targets)
        champ_metrics = self._calculate_metrics(champ_preds, actual_targets)
        cand_metrics.structural_alignment = cand_blind['structural_alignment']
        cand_metrics.realization_gap = cand_blind['realization_gap']
        champ_metrics.structural_alignment = champ_blind['structural_alignment'
            ]
        champ_metrics.realization_gap = champ_blind['realization_gap']
        cand_score = self._calculate_causal_weighted_score(cand_metrics)
        champ_score = self._calculate_causal_weighted_score(champ_metrics)

        # Apply Dynamic Routing Rules
        # To determine context, let's assume we use 'is_sharp_drop' if the market has fallen recently
        # For a full implementation, this should come from a ContextDetector. We will mock context here:
        mock_context = {'is_sharp_drop': True} # For testing the Shadow Battle Mode
        weight_multipliers = self.dynamic_router.adjust_weights({candidate_name: cand_preds, champ_name: champ_preds}, mock_context)
        
        cand_score *= weight_multipliers.get(candidate_name, 1.0)
        champ_score *= weight_multipliers.get(champ_name, 1.0)
        
        logger.info(f"[ARENA-ROUTER] Candidate {candidate_name} score adjusted to {cand_score:.4f}")
        logger.info(f"[ARENA-ROUTER] Champion {champ_name} score adjusted to {champ_score:.4f}")

        required_score = champ_score * (1 + self.safety_margin)
        model_info = self.models.get(candidate_name, {})
        logger.info(
            f"[COMBAT] {candidate_name} vs {champ_name} | Activations: {model_info.get('activations')} | Params: {model_info.get('params')}"
            )
        if cand_score > required_score:
            logger.info(
                f'[ARENA] 🏆 Champion Replaced! Score: {cand_score:.4f} vs {champ_score:.4f}'
                )
            return {'result': BattleResult.CHAMPION_REPLACED,
                'cand_metrics': cand_metrics}
        else:
            logger.info(
                f'[ARENA] 🛡️ Champion Retained. Diff: {(cand_score / champ_score - 1) * 100:.1f}%'
                )
            return {'result': BattleResult.CHAMPION_RETAINED,
                'champ_metrics': champ_metrics}

    def _calculate_causal_weighted_score(self, metrics: BattleMetrics) ->float:
        """Розраховує бал з фокусом на механіку та мінімізацію розриву реалізації."""
        weights = {'accuracy': 0.2, 'sharpe_ratio': 0.15,
            'structural_alignment': 0.35, 'realization_gap': 0.3}
        return weights['accuracy'] * metrics.accuracy + weights['sharpe_ratio'
            ] * min(metrics.sharpe_ratio, 2) + weights['structural_alignment'
            ] * metrics.structural_alignment + weights['realization_gap'] * (
            1 / (1 + metrics.realization_gap))

    def _calculate_weighted_score(self, metrics: BattleMetrics) ->float:
        """Legacy weighted score."""
        weights = {'accuracy': 0.3, 'sharpe_ratio': 0.25, 'win_rate': 0.2,
            'max_drawdown': 0.15, 'confidence_score': 0.1}
        return weights['accuracy'] * metrics.accuracy + weights['sharpe_ratio'
            ] * min(metrics.sharpe_ratio, 2) + weights['win_rate'
            ] * metrics.win_rate + weights['max_drawdown'] * (1 - abs(
            metrics.max_drawdown)) + weights['confidence_score'
            ] * metrics.confidence_score

    def create_battle(self, model1_name: str, model2_name: str,
        battle_group: str='custom') ->bool:
        """Створення бою між двома моделями"""
        try:
            if (model1_name not in self.models or model2_name not in self.
                models):
                raise ValueError(f"Models not found: {model1_name}, {model2_name}")
            battle = Battle(model1_name=model1_name, model2_name=
                model2_name, battle_group=battle_group, start_time=datetime
                .now())
            self.current_battles.append(battle)
            logger.info(
                f'[ARENA] Battle created: {model1_name} vs {model2_name} (group: {battle_group})'
                )
            return True
        except (ValueError, Exception) as e:
            raise RuntimeError(f"Failed to create battle: {e}") from e

    def create_battles_from_group(self, group_name: str) ->int:
        """
        Створює бої на основі попередньо визначеної групи боїв.
        """
        try:
            available_models = list(self.models.keys())
            battle_schedule = (self.battle_group_manager.
                generate_battle_schedule(group_name, available_models))
            battles_created = 0
            for model1, model2 in battle_schedule:
                if self.create_battle(model1, model2, battle_group=group_name):
                    battles_created += 1
            logger.info(
                f"[ARENA] Created {battles_created} battles from group '{group_name}'"
                )
            return battles_created
        except (Exception) as e:
            logger.error(
                f"[ARENA] Failed to create battles from group '{group_name}': {e}", exc_info=True
                )
            raise RuntimeError(f"Failed to create battles from group '{group_name}': {e}") from e

    def get_recommended_battle_groups(self) ->list[str]:
        """Отримує рекомендовані групи боїв на основі зареєстрованих моделей."""
        available_models = list(self.models.keys())
        return self.battle_group_manager.get_recommended_groups(
            available_models)

    def run_battle(self, test_data: pd.DataFrame, actual_targets: pd.Series
        ) ->dict[str, Any]:
        """Виконання бою між моделями"""
        results = []
        for battle in self.current_battles:
            try:
                model1_predictions = self._get_model_predictions(battle.
                    model1_name, test_data)
                model2_predictions = self._get_model_predictions(battle.
                    model2_name, test_data)
                model1_metrics = self._calculate_metrics(model1_predictions,
                    actual_targets)
                model2_metrics = self._calculate_metrics(model2_predictions,
                    actual_targets)
                winner = self._determine_battle_winner(model1_metrics,
                    model2_metrics)
                battle.end_time = datetime.now()
                battle.model1_metrics = model1_metrics
                battle.model2_metrics = model2_metrics
                battle.winner = winner
                battle.result = self._get_battle_result(winner)
                self._update_model_stats(battle)
                self.battle_history.append(battle)
                results.append({'battle_id': len(self.battle_history),
                    'winner': winner, 'timestamp': battle.end_time.isoformat()}
                    )
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError):
                logger.exception('[ARENA] Battle failed')
        self.current_battles.clear()
        return {'battles_completed': len(results), 'results': results,
            'timestamp': datetime.now().isoformat()}

    def _get_model_predictions(self, model_name: str, test_data: pd.DataFrame
        ) ->np.ndarray:
        """Отримання прогнозів від моделі"""
        try:
            model_info = self.models[model_name]
            return self._get_instance_predictions(model_info['instance'],
                test_data)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError):
            logger.exception(f'[ARENA] Failed to get predictions from {model_name}')
            return np.zeros(len(test_data))

    def _get_instance_predictions(self, model_instance: Any, data: pd.DataFrame
        ) ->np.ndarray:
        """Виконання прогнозу на конкретному інстансі моделі."""
        if hasattr(model_instance, 'predict'):
            predictions = model_instance.predict(data)
        elif hasattr(model_instance, 'forecast'):
            predictions = model_instance.forecast(data)
        else:
            predictions = np.zeros(len(data))
        return np.array(predictions)

    def _calculate_metrics(self, predictions: np.ndarray, actual_targets:
        pd.Series) ->BattleMetrics:
        """Розрахунок метрик для моделі"""
        try:
            actuals = actual_targets.values if hasattr(actual_targets, 'values'
                ) else actual_targets
            predictions_arr = np.asarray(predictions, dtype=float)
            actuals_arr = np.asarray(actuals, dtype=float)
            valid_mask = np.isfinite(predictions_arr) & np.isfinite(actuals_arr)
            if not np.any(valid_mask):
                return BattleMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
            predictions_clean = predictions_arr[valid_mask]
            actuals_clean = actuals_arr[valid_mask]
            mse = np.mean((predictions_clean - actuals_clean) ** 2)
            accuracy = np.mean(np.sign(predictions_clean) == np.sign(actuals_clean))
            prediction_std = float(np.std(predictions_clean))
            sharpe_ratio = (
                float(np.mean(predictions_clean) / prediction_std)
                if np.isfinite(prediction_std) and prediction_std > 1e-12
                else 0.0
            )
            return BattleMetrics(accuracy=accuracy, precision=accuracy,
                recall=accuracy, f1_score=accuracy, sharpe_ratio=
                sharpe_ratio, max_drawdown=0.0, win_rate=accuracy,
                execution_time=0.1, confidence_score=0.7, mse=mse)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError):
            logger.exception('[ARENA] Failed to calculate metrics')
            return BattleMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

    def _calculate_max_drawdown(self, returns: np.ndarray) ->float:
        """Розрахунок максимального просідання"""
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return float(np.min(drawdown))
        except (ValueError, TypeError, Exception) as e:
            raise RuntimeError(f"Max drawdown calculation failed: {e}") from e

    def _determine_battle_winner(self, metrics1: BattleMetrics, metrics2:
        BattleMetrics) ->str:
        """Визначення переможця бою на основі метрик"""
        score1 = self._calculate_weighted_score(metrics1)
        score2 = self._calculate_weighted_score(metrics2)
        if abs(score1 - score2) < 0.05:
            return 'draw'
        return 'model1' if score1 > score2 else 'model2'

    def _get_battle_result(self, winner: str) ->BattleResult:
        if winner == 'draw':
            return BattleResult.DRAW
        return (BattleResult.MODEL1_WIN if winner == 'model1' else
            BattleResult.MODEL2_WIN)

    def _update_model_stats(self, battle: Battle):
        if battle.model1_name in self.models:
            self.models[battle.model1_name]['battles_fought'] += 1
            if battle.winner == 'model1':
                self.models[battle.model1_name]['wins'] += 1
        if battle.model2_name in self.models:
            self.models[battle.model2_name]['battles_fought'] += 1
            if battle.winner == 'model2':
                self.models[battle.model2_name]['wins'] += 1

    def get_leaderboard(self) ->dict[str, Any]:
        leaderboard = []
        for model_name, model_info in self.models.items():
            if model_info['battles_fought'] > 0:
                leaderboard.append({'model_name': model_name, 'points':
                    model_info['wins'] * 3 + model_info['draws']})
        leaderboard.sort(key=lambda x: x['points'], reverse=True)
        return {'leaderboard': leaderboard, 'last_updated': datetime.now().
            isoformat()}

    def save_arena_state(self, filepath: str) ->bool:
        try:
            serializable = {k: {sk: sv for sk, sv in v.items() if sk !=
                'instance'} for k, v in self.models.items()}
            with open(filepath, 'w') as f:
                json.dump({'models': serializable}, f, indent=2, default=str)
            return True
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError):
            self.logger.exception('Виникла помилка під час збереження стану арени')
            return False

    def load_arena_state(self, filepath: str) ->bool:
        try:
            with open(filepath) as f:
                state = json.load(f)
            self.models = state.get('models', {})
            return True
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError):
            self.logger.exception('Виникла помилка під час завантаження стану арени')
            return False


_trading_arena: TradingModelArena | None = None


def get_trading_arena() ->TradingModelArena:
    global _trading_arena
    if _trading_arena is None:
        _trading_arena = TradingModelArena()
    return _trading_arena
