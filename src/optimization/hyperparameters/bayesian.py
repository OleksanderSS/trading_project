# src/optimization/hyperparameters/bayesian.py

import numpy as np
from typing import Dict, Any, Optional, Callable
from sklearn.model_selection import cross_val_score
from src.core.logging.logger import ProjectLogger
from src.optimization.base import BaseOptimizer

try:
    import optuna
except ImportError as e:
    # Raise the error immediately to signal a missing critical dependency.
    # This prevents the system from silently proceeding with default parameters.
    raise ImportError("Optuna is not installed. Please install it to use Bayesian optimization: `pip install optuna`") from e

class BayesianOptimizer(BaseOptimizer):
    """
    Інструмент для підбору гіперпараметрів моделей за допомогою байєсівської оптимізації (Optuna).
    """

    def __init__(self, model_func: Callable, param_space: Dict[str, Any], n_trials: int = 50):
        """
        Ініціалізує оптимізатор.

        Args:
            model_func: Функція або клас моделі для ініціалізації.
            param_space: Словник з визначенням простору параметрів.
            n_trials: Кількість спроб оптимізації.
        """
        self.logger = ProjectLogger.get_logger("BayesianOptimizer")
        self.model_func = model_func
        self.param_space = param_space
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = -np.inf

    @property
    def optimizer_type(self) -> str:
        return "bayesian_hyperparameters"

    def objective(self, trial: "optuna.trial.Trial", X: np.ndarray, y: np.ndarray) -> float:
        """
        Цільова функція для Optuna. 
        Note: We removed the try-except block here. If any systemic error occurs during
        model training or evaluation, it will propagate up and stop the optimization study,
        making the problem immediately visible.
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
        
        # Оцінюємо через cross-validation (максимізуємо, тому Optuna максимізує це значення)
        scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
        return float(scores.mean())

    def optimize(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """
        Запускає процес байєсівської оптимізації.

        Args:
            X: Ознаки для навчання.
            y: Цільова змінна.
            **kwargs: Додаткові параметри.

        Returns:
            Словник з найкращими знайденими параметрами.
        """
        try:
            self.logger.info(f"Запуск байєсівської оптимізації ({self.n_trials} спроб)...")
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
            
            # We pass `catch=()` to ensure that any exception inside the objective function
            # is NOT caught by Optuna. This makes the entire process fail fast, 
            # which is crucial for identifying and debugging systemic issues quickly.
            study.optimize(
                lambda trial: self.objective(trial, X, y), 
                n_trials=self.n_trials, 
                show_progress_bar=False,
                catch=()
            )
            
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            self.logger.info(f"Байєсівська оптимізація завершена: найкращий скор = {self.best_score:.4f}")
            return self.best_params
            
        except Exception as e:
            self.logger.error(f"Критична помилка під час оптимізації: {e}", exc_info=True)
            # Re-raise the exception to ensure the calling process is aware of the failure.
            raise

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Перевіряє валідність простору параметрів."""
        return bool(params and isinstance(params, dict))