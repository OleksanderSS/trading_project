# models/bayesian_optimizer.py

import numpy as np
from sklearn.model_selection import cross_val_score
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class BayesianOptimizer:
    def __init__(self, model_func, param_space, n_trials=50):
        self.model_func = model_func
        self.param_space = param_space
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = -np.inf
        
    def objective(self, trial, X, y):
        # Геnotруємо параметри for trial
        params = {}
        for param_name, (param_type, *args) in self.param_space.items():
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, *args)
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, *args)
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, args[0])
                
        # Тренуємо model with параметрами
        model = self.model_func(**params)
        model.fit(X, y)
        
        # Оцandнюємо череwith cross-validation
        scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
        return scores.mean()
        
    def optimize(self, X, y):
        try:
            import optuna
            
            study = optuna.create_study(direction='maximize', 
                                      sampler=optuna.samplers.TPESampler())
            
            study.optimize(lambda trial: self.objective(trial, X, y), 
                          n_trials=self.n_trials, show_progress_bar=False)
            
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            logger.info(f"[OK] Байєсandвська оптимandforцandя: score={self.best_score:.4f}")
            return self.best_params
            
        except ImportError:
            logger.warning("[WARN] Optuna notдоступна, використовую whereфолтнand параметри")
            return {}

def optimize_lgbm_params(X, y):
    """Оптимandforцandя LightGBM параметрandв"""
    from lightgbm import LGBMRegressor
    
    param_space = {
        'n_estimators': ('int', 50, 300),
        'max_depth': ('int', 3, 10),
        'learning_rate': ('float', 0.01, 0.3),
        'num_leaves': ('int', 10, 100)
    }
    
    optimizer = BayesianOptimizer(LGBMRegressor, param_space)
    return optimizer.optimize(X, y)