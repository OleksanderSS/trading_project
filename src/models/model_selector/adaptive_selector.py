"""
AdaptiveModelSelector: Extends SmartModelSelector with online learning and Arena integration

Features:
- **Arena Battle System integration** - Uses Arena leaderboard as source of truth
- Online learning from feedback
- Recent performance tracking
- Alternative model selection
- Exponential moving average
- Context-aware model selection using SmartModelSelector

Architecture:
- SmartModelSelector: Context analysis (volatility, trend, regime)
- Arena Battle System: Champion selection through battles
- AdaptiveModelSelector: Online learning + Arena integration

Usage:
    selector = AdaptiveModelSelector(
        fallback="lightgbm",
        arena=trading_arena,  # Optional: use Arena leaderboard
        learning_rate=0.1
    )

    # Select model (uses Arena if available)
    model_id = selector.select_best_model_adaptive("1|0|-1|1|0")

    # Provide feedback (updates Arena)
    selector.update_from_feedback(
        model_id, "1|0|-1|1|0",
        actual_return=0.05,
        predicted_return=0.04
    )
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from src.core.logging.logger import ProjectLogger
from src.models.model_selector.fingerprint_selector import SmartModelSelector as FingerprintModelSelector

if TYPE_CHECKING:
    from src.analytics.arena.arena_battle import TradingModelArena
logger = ProjectLogger.get_logger(__name__)


class AdaptiveModelSelector(FingerprintModelSelector):
    """
    Adaptive selector with online learning, Arena integration, and persistence.

    **Integration with Arena Battle System**:
    - If `arena` is provided, uses Arena leaderboard as source of truth
    - Falls back to local leaderboard if Arena not available
    - Syncs performance data with Arena battles

    **Integration with SmartModelSelector**:
    - Uses SmartModelSelector for context analysis
    - Combines context scores with Arena battle results

    New features:
    - Arena Battle System integration (optional)
    - Leaderboard persistence across runs
    - Online learning from feedback
    - Recent performance tracking
    - Alternative model selection
    - Exponential moving average for win rates

    Attributes:
        arena: Optional TradingModelArena for champion selection
        leaderboard_path: Path to persist leaderboard (fallback)
        learning_rate: Learning rate for EMA (0-1)
        arena_leaderboard: Persistent leaderboard (local or Arena)
        selection_history: History of selections
        performance_tracker: Recent performance per model
    """

    def __init__(self, fallback: str='lightgbm', arena: Optional[
        'TradingModelArena']=None, leaderboard_path: str=
        'data/leaderboard.json', learning_rate: float=0.1):
        """
        Initialize adaptive selector with optional Arena integration.

        Args:
            fallback: Fallback model when no match found
            arena: Optional TradingModelArena for champion selection
            leaderboard_path: Path to persist leaderboard (fallback if no Arena)
            learning_rate: Learning rate for EMA (0-1)
                          Higher = more reactive to recent performance
        """
        super().__init__(fallback)
        self.arena = arena
        self.leaderboard_path = Path(leaderboard_path)
        self.learning_rate = learning_rate
        if self.arena:
            self.arena_leaderboard = self._get_arena_leaderboard()
            logger.info(
                'AdaptiveModelSelector initialized with Arena integration')
        else:
            self.arena_leaderboard = self._load_leaderboard()
            logger.info(
                f'AdaptiveModelSelector initialized: leaderboard={leaderboard_path}, lr={learning_rate}'
                )
        self.selection_history: list[dict[str, Any]] = []
        self.performance_tracker: dict[str, list[float]] = {}

    def _get_arena_leaderboard(self) ->dict[str, Any]:
        """Get leaderboard from Arena Battle System."""
        if not self.arena:
            return {}
        try:
            arena_data = self.arena.get_leaderboard()
            leaderboard = arena_data.get('leaderboard', [])
            converted: dict[str, dict[str, Any]] = {}
            for entry in leaderboard:
                model_name = entry.get('model_name', '')
                points = entry.get('points', 0)
                parts = model_name.split('_')
                if len(parts) >= 3:
                    context = f'{parts[1]}_{parts[2]}'
                else:
                    context = 'default'
                if context not in converted:
                    converted[context] = {}
                converted[context][model_name] = {'points': points,
                    'win_rate': min(points / 10.0, 1.0),
                    'total_predictions': points // 3}
            logger.info(
                f'Loaded Arena leaderboard: {len(converted)} contexts, {len(leaderboard)} models'
                )
            return converted
        except Exception as e:
            logger.error(f'Failed to get Arena leaderboard: {e}')
            raise RuntimeError("Failed to get Arena leaderboard") from e

    def select_best_model_adaptive(self, context_fingerprint: str, features:
        Any=None) ->str:
        """
        Adaptive selection with recent performance check.

        Args:
            context_fingerprint: Context fingerprint string
            features: Optional features for context-aware selection

        Returns:
            Selected model ID

        Example:
            model_id = selector.select_best_model_adaptive("1|0|-1|1|0")
        """
        base_model = self.select_best_model(context_fingerprint, self.
            arena_leaderboard)
        recent_perf = self._get_recent_performance(base_model)
        if recent_perf < 0.3:
            logger.warning(
                f'Model {base_model} recent performance low: {recent_perf:.2f}'
                )
            alternative = self._get_alternative_model(context_fingerprint)
            if alternative:
                logger.info(f'Switching to alternative: {alternative}')
                base_model = alternative
        self.selection_history.append({'timestamp': datetime.now().
            isoformat(), 'context': context_fingerprint, 'selected_model':
            base_model, 'recent_performance': recent_perf})
        return base_model

    def update_from_feedback(self, model_id: str, context_fingerprint: str,
        actual_return: float, predicted_return: float) ->None:
        """
        Update leaderboard from actual results (online learning).

        **Arena Integration**: If Arena is available, updates Arena leaderboard.
        Otherwise updates local leaderboard.

        Args:
            model_id: Model that made prediction
            context_fingerprint: Context fingerprint
            actual_return: Actual return
            predicted_return: Predicted return

        Example:
            selector.update_from_feedback(
                "catboost_v1", "1|0|-1|1|0",
                actual_return=0.05,
                predicted_return=0.04
            )
        """
        error = abs(actual_return - predicted_return)
        accuracy = 1.0 - min(error / (abs(actual_return) + 1e-06), 1.0)
        if self.arena:
            self._update_arena_feedback(model_id, accuracy)
        self._update_local_leaderboard(model_id, context_fingerprint, accuracy)
        if model_id not in self.performance_tracker:
            self.performance_tracker[model_id] = []
        self.performance_tracker[model_id].append(accuracy)
        self._save_leaderboard()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f'Updated {model_id} for {context_fingerprint}: accuracy {accuracy:.3f}'
                )

    def _update_arena_feedback(self, model_id: str, accuracy: float) ->None:
        """Update Arena with feedback."""
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Arena feedback: {model_id} accuracy={accuracy:.3f}')
            if self.arena is not None and hasattr(self.arena,
                'update_model_performance'):
                self.arena.update_model_performance(model_id, accuracy)
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            logger.warning(f'Failed to update Arena feedback: {e}')
            raise

    def _update_local_leaderboard(self, model_id: str, context_fingerprint:
        str, accuracy: float) ->None:
        """Update local leaderboard with feedback."""
        if context_fingerprint not in self.arena_leaderboard:
            self.arena_leaderboard[context_fingerprint] = {}
        if model_id not in self.arena_leaderboard[context_fingerprint]:
            self.arena_leaderboard[context_fingerprint][model_id] = {'points':
                0, 'win_rate': 0.5, 'total_predictions': 0}
        model_stats = self.arena_leaderboard[context_fingerprint][model_id]
        old_win_rate = model_stats['win_rate']
        new_win_rate = old_win_rate * (1 - self.learning_rate
            ) + accuracy * self.learning_rate
        model_stats['win_rate'] = new_win_rate
        model_stats['points'] += 1 if accuracy > 0.5 else -1
        model_stats['total_predictions'] += 1

    def _get_recent_performance(self, model_id: str, window: int=10) ->float:
        """
        Get recent performance for model.

        Args:
            model_id: Model identifier
            window: Number of recent predictions to consider

        Returns:
            Average recent performance (0-1)
        """
        if model_id not in self.performance_tracker:
            return 0.5
        recent = self.performance_tracker[model_id][-window:]
        return np.mean(recent) if recent else 0.5

    def _get_alternative_model(self, context_fingerprint: str) ->str | None:
        """
        Get alternative model for context.

        Args:
            context_fingerprint: Context fingerprint

        Returns:
            Alternative model ID or None
        """
        if context_fingerprint not in self.arena_leaderboard:
            return None
        models = self.arena_leaderboard[context_fingerprint]
        sorted_models = sorted(models.items(), key=lambda x: x[1].get(
            'win_rate', 0), reverse=True)
        if len(sorted_models) > 1:
            second_best = sorted_models[1][0]
            if isinstance(second_best, str):
                return second_best
        return None

    def _load_leaderboard(self) ->dict[str, Any]:
        """Load leaderboard from disk."""
        if not self.leaderboard_path.exists():
            logger.info('No existing leaderboard. Starting fresh.')
            return {}
        try:
            with open(self.leaderboard_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                logger.info(f'Loaded leaderboard: {len(data)} contexts')
                return data
            else:
                logger.warning('Leaderboard data is not a dictionary')
                return {}
        except Exception as e:
            logger.error(f'Failed to load leaderboard: {e}')
            raise RuntimeError(f"Failed to load leaderboard from {self.leaderboard_path}") from e

    def _save_leaderboard(self) ->None:
        """Save leaderboard to disk."""
        self.leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.leaderboard_path, 'w') as f:
                json.dump(self.arena_leaderboard, f, indent=2)
        except Exception as e:
            logger.error(f'Failed to save leaderboard: {e}')

    def get_leaderboard_summary(self) ->dict[str, Any]:
        """
        Get leaderboard summary.

        **Arena Integration**: If Arena is available, includes Arena stats.

        Returns:
            Dict with summary statistics
        """
        total_contexts = len(self.arena_leaderboard)
        total_models = set()
        for context_models in self.arena_leaderboard.values():
            total_models.update(context_models.keys())
        summary = {'total_contexts': total_contexts, 'total_models': len(
            total_models), 'models': list(total_models),
            'selection_history_size': len(self.selection_history),
            'last_updated': datetime.now().isoformat(), 'arena_integrated':
            self.arena is not None}
        if self.arena:
            try:
                arena_data = self.arena.get_leaderboard()
                summary['arena_leaderboard_size'] = len(arena_data.get(
                    'leaderboard', []))
                summary['arena_last_updated'] = arena_data.get('last_updated')
            except Exception as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                logger.warning(f'Failed to get Arena stats: {e}')
                raise
        return summary

    def sync_with_arena(self) ->None:
        """
        Sync local leaderboard with Arena Battle System.

        This method:
        1. Pulls latest Arena leaderboard
        2. Merges with local leaderboard
        3. Resolves conflicts (Arena wins)
        """
        if not self.arena:
            logger.warning('No Arena available for sync')
            return
        try:
            arena_leaderboard = self._get_arena_leaderboard()
            for context, models in arena_leaderboard.items():
                if context not in self.arena_leaderboard:
                    self.arena_leaderboard[context] = {}
                for model_id, stats in models.items():
                    self.arena_leaderboard[context][model_id] = stats
            self._save_leaderboard()
            logger.info(
                f'✅ Synced with Arena: {len(arena_leaderboard)} contexts')
        except Exception as e:
            logger.error(f'Failed to sync with Arena: {e}')

    def export_history(self, filepath: str) ->None:
        """
        Export selection history.

        Args:
            filepath: Path to save history
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({'selection_history': self.selection_history,
                'performance_tracker': {k: v[-100:] for k, v in self.
                performance_tracker.items()}, 'exported_at': datetime.now()
                .isoformat()}, f, indent=2)
        logger.info(f'Exported history to {filepath}')

    def get_model_performance(self, model_id: str) ->dict[str, Any]:
        """
        Get performance statistics for model.

        Args:
            model_id: Model identifier

        Returns:
            Dict with performance stats
        """
        if model_id not in self.performance_tracker:
            return {'status': 'no_data'}
        history = self.performance_tracker[model_id]
        return {'model_id': model_id, 'total_predictions': len(history),
            'avg_accuracy': float(np.mean(history)), 'recent_accuracy':
            float(np.mean(history[-10:])) if len(history) >= 10 else float(
            np.mean(history)), 'std_accuracy': float(np.std(history)),
            'min_accuracy': float(np.min(history)), 'max_accuracy': float(
            np.max(history))}
