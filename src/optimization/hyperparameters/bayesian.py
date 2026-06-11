# src/optimization/hyperparameters/bayesian.py

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.optimization.base import BaseOptimizer

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class BayesianOptimizer(BaseOptimizer):
    """
    Інструмент для підбору гіперпараметрів моделей за допомогою байєсівської оптимізації (Optuna).

    Combines features from both implementations:
    - Inherits from BaseOptimizer (structured approach)
    - Configurable scoring and cv (flexibility)
    - Graceful Optuna handling (soft dependency)
    """

    def __init__(
        self,
        model_func: Callable,
        param_space: dict[str, Any],
        n_trials: int = 50,
        scoring: str = 'neg_mean_squared_error',
        cv: int = 3
    ):
        """
        Ініціалізує оптимізатор.

        Args:
            model_func: Функція або клас моделі для ініціалізації.
            param_space: Словник з визначенням простору параметрів.
            n_trials: Кількість спроб оптимізації.
            scoring: Метрика для оцінки (sklearn scoring).
            cv: Кількість фолдів для cross-validation.
        """
        super().__init__()

        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is not installed. Please install it to use Bayesian optimization: "
                "`pip install optuna`"
            )

        self.model_func = model_func
        self.param_space = param_space
        self.n_trials = n_trials
        self.scoring = scoring
        self.cv = cv
        self.best_params: dict[str, Any] = {}
        self.best_score = -np.inf
        self.study: optuna.Study | None = None

    @property
    def optimizer_type(self) -> str:
        return "bayesian_hyperparameters"

    def objective(self, trial: "optuna.trial.Trial", X: np.ndarray, y: np.ndarray) -> float:
        """
        Цільова функція для Optuna.

        Args:
            trial: Optuna trial object
            X: Feature matrix
            y: Target vector

        Returns:
            Mean cross-validation score
        """
        # Генеруємо параметри для поточної спроби
        params = {}
        for param_name, (param_type, *args) in self.param_space.items():
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, *args)
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, *args)
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, args[0])

        # Тренуємо модель з вибраними параметрами
        model = self.model_func(**params)

        # Оцінюємо через cross-validation
        scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=self.cv), scoring=self.scoring)
        return float(scores.mean())

    def optimize(self, data: Any, target: Any = None, **kwargs) -> dict[str, Any]:
        """
        Запускає процес байєсівської оптимізації.

        Args:
            data: Feature matrix (X)
            target: Target vector (y)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dict with best_params and best_score
        """
        if target is None:
            raise ValueError("BayesianOptimizer requires 'target' data for optimization.")

        try:
            self.logger.info(
                f"Запуск байєсівської оптимізації ({self.n_trials} спроб, "
                f"scoring={self.scoring}, cv={self.cv})..."
            )

            self.study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler()
            )

            self.study.optimize(
                lambda trial: self.objective(trial, data, target),
                n_trials=self.n_trials,
                show_progress_bar=False,
                catch=(),
            )

            self.best_params = self.study.best_params
            self.best_score = float(self.study.best_value)

            self.logger.info(
                f"Байєсівська оптимізація завершена: найкращий скор = {self.best_score:.4f}"
            )
            self.logger.info(f"Найкращі параметри: {self.best_params}")

            return {"best_params": self.best_params, "best_score": self.best_score}

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Критична помилка під час оптимізації: {e}", exc_info=True)
            raise

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Перевіряє валідність простору параметрів."""
        return bool(params and isinstance(params, dict))

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """
        Повертає історію оптимізації.

        Returns:
            List of trials with params and scores
        """
        if self.study is None:
            return []

        return [
            {
                'trial_number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'state': trial.state.name
            }
            for trial in self.study.trials
        ]
