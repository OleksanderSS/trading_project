
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import logging

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)

class BayesianOptimizer:
    def __init__(self, model_func, param_space, n_trials=50, scoring='neg_mean_squared_error', cv=3):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for BayesianOptimizer. Please install it using 'pip install optuna'")
        self.model_func = model_func
        self.param_space = param_space
        self.n_trials = n_trials
        self.scoring = scoring
        self.cv = cv
        self.best_params = None
        self.best_score = -np.inf

    def objective(self, trial, X, y):
        """Objective function for Optuna to optimize."""
        params = {}
        for param_name, (param_type, *args) in self.param_space.items():
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, *args)
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, *args)
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, args[0])

        model = self.model_func(**params)
        
        # Using cross-validation to get a robust score
        scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=self.cv), scoring=self.scoring)
        return scores.mean()

    def optimize(self, X, y):
        """Run the optimization process."""
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        
        study.optimize(lambda trial: self.objective(trial, X, y), 
                      n_trials=self.n_trials, show_progress_bar=False)

        self.best_params = study.best_params
        self.best_score = study.best_value

        logger.info(f"Bayesian optimization finished. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
