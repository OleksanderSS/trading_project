# HyperparameterSearcher Implementation
# Automated hyperparameter tuning with Optuna or grid search fallback

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json


from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import get_current_config

class HyperparameterSearcher:
    """
    Автоматична оптимізація гіперпараметрів моделей.
    """

    def __init__(self, project_path=None, method='optuna', n_trials=20):
        """
        Args:
            project_path: Шлях проекту для збереження результатів
            method: 'optuna' (default), 'grid', 'random', 'custom'
            n_trials: Кількість спроб для оптимізації
        """
        self.config = get_current_config()
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self.method = method
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = None
        self.trial_history = []
        self.logger = ProjectLogger.get_logger("HyperparameterSearcher")
        self.optuna_available = self._check_optuna()
        self.random_seed = self.config.get('performance.random_seed', 42)

    def _check_optuna(self):
        """Перевірити доступність Optuna"""
        try:
            import optuna
            self.logger.info("Optuna available - using Bayesian optimization")
            return True
        except ImportError:
            self.logger.warning("Optuna not available - falling back to grid/random search")
            return False

    def search_mlp_params(self, X_train, y_train, x_val, y_val, metric_func=None):
        """
        Пошук оптимальних параметрів для MLP.
        """
        param_space = {
            'hidden_size_1': [64, 128, 256],
            'hidden_size_2': [32, 64, 128],
            'dropout': [0.1, 0.2, 0.3, 0.4],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'batch_size': [16, 32, 64],
            'epochs': [50, 100, 200]
        }
        
        if self.optuna_available and self.method == 'optuna':
            return self._optimize_optuna_mlp(
                X_train, y_train, x_val, y_val,
                param_space, metric_func
            )
        else:
            return self._optimize_grid_search(
                X_train, y_train, x_val, y_val,
                param_space, metric_func,
                model_type='mlp'
            )

    def search_lstm_params(self, X_train, y_train, x_val, y_val, metric_func=None):
        """Пошук параметрів для LSTM"""
        param_space = {
            'hidden_size': [32, 64, 128],
            'n_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.0001, 0.0005, 0.001],
            'batch_size': [16, 32, 64],
            'epochs': [50, 100, 200]
        }
        
        if self.optuna_available and self.method == 'optuna':
            return self._optimize_optuna_lstm(
                X_train, y_train, x_val, y_val,
                param_space, metric_func
            )
        else:
            return self._optimize_grid_search(
                X_train, y_train, x_val, y_val,
                param_space, metric_func,
                model_type='lstm'
            )

    def _optimize_optuna_mlp(self, X_train, y_train, x_val, y_val, param_space, metric_func):
        """Optuna-based optimization для MLP"""
        try:
            import optuna
            from optuna.pruners import MedianPruner
            
            def objective(trial):
                # Suggest parameters
                hidden_size_1 = trial.suggest_categorical('hidden_size_1', param_space['hidden_size_1'])
                hidden_size_2 = trial.suggest_categorical('hidden_size_2', param_space['hidden_size_2'])
                dropout = trial.suggest_categorical('dropout', param_space['dropout'])
                learning_rate = trial.suggest_categorical('learning_rate', param_space['learning_rate'])
                
                params = {
                    'hidden_size_1': hidden_size_1,
                    'hidden_size_2': hidden_size_2,
                    'dropout': dropout,
                    'learning_rate': learning_rate,
                    'batch_size': 32,
                    'epochs': 50
                }
                
                try:
                    score = self._evaluate_mlp_params(
                        X_train, y_train, x_val, y_val,
                        params, metric_func
                    )
                    
                    self.trial_history.append({
                        'trial': trial.number,
                        'params': params,
                        'score': score
                    })
                    
                    return score
                except Exception:
                    self.logger.warning(f"Trial {trial.number} failed")
                    return float('-inf')
            
            # Запустити оптимізацію
            sampler = optuna.samplers.TPESampler(seed=self.random_seed)
            pruner = MedianPruner()
            study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            
            # Отримати результати
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            self.logger.info("Optuna optimization complete")
            self.logger.info(f"Best score: {self.best_score:.4f}")
            self.logger.info(f"Best params: {self.best_params}")
            
            return {
                'method': 'optuna',
                'model_type': 'mlp',
                'best_params': self.best_params,
                'best_score': self.best_score,
                'n_trials': self.n_trials,
                'trial_history': self.trial_history
            }
        
        except Exception as e:
            self.logger.error(f"Optuna optimization failed: {e}")
            # Fallback to grid search
            return self._optimize_grid_search(
                X_train, y_train, x_val, y_val,
                self._get_default_param_space('mlp'),
                metric_func,
                model_type='mlp'
            )

    def _optimize_optuna_lstm(self, X_train, y_train, x_val, y_val, param_space, metric_func):
        """Optuna-based optimization для LSTM"""
        try:
            import optuna
            
            def objective(trial):
                hidden_size = trial.suggest_categorical('hidden_size', param_space['hidden_size'])
                n_layers = trial.suggest_categorical('n_layers', param_space['n_layers'])
                dropout = trial.suggest_categorical('dropout', param_space['dropout'])
                learning_rate = trial.suggest_categorical('learning_rate', param_space['learning_rate'])
                
                params = {
                    'hidden_size': hidden_size,
                    'n_layers': n_layers,
                    'dropout': dropout,
                    'learning_rate': learning_rate,
                    'batch_size': 32,
                    'epochs': 50
                }
                
                try:
                    score = self._evaluate_lstm_params(
                        X_train, y_train, x_val, y_val,
                        params, metric_func
                    )
                    
                    self.trial_history.append({
                        'trial': trial.number,
                        'params': params,
                        'score': score
                    })
                    
                    return score
                except Exception:
                    return float('-inf')
            
            sampler = optuna.samplers.TPESampler(seed=self.random_seed)
            study = optuna.create_study(sampler=sampler, direction='maximize')
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            self.logger.info(f"LSTM optimization complete (best: {self.best_score:.4f})")
            
            return {
                'method': 'optuna',
                'model_type': 'lstm',
                'best_params': self.best_params,
                'best_score': self.best_score,
                'n_trials': self.n_trials
            }
        except Exception as e:
            self.logger.error(f"LSTM optimization failed: {e}")
            return self._optimize_grid_search(
                X_train, y_train, x_val, y_val,
                self._get_default_param_space('lstm'),
                metric_func,
                model_type='lstm'
            )

    def _optimize_grid_search(self, x_train, y_train, x_val, y_val, param_space, metric_func, model_type='mlp'):
        """Grid search optimization (fallback)"""
        from itertools import product
        
        self.logger.info(f"Grid search optimization for {model_type}...")
        
        keys = list(param_space.keys())
        values = list(param_space.values())
        
        best_score = float('-inf')
        best_params = None
        trial_count = 0
        
        rng = np.random.default_rng(self.random_seed)
        combinations = list(product(*values))
        if len(combinations) > self.n_trials:
            combinations = rng.choice(combinations, self.n_trials, replace=False).tolist()
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            
            try:
                if model_type == 'mlp':
                    score = self._evaluate_mlp_params(
                        x_train, y_train, x_val, y_val,
                        params, metric_func
                    )
                elif model_type == 'lstm':
                    score = self._evaluate_lstm_params(
                        x_train, y_train, x_val, y_val,
                        params, metric_func
                    )
                else:
                    score = 0.0
                
                self.trial_history.append({
                    'trial': trial_count,
                    'params': params,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                trial_count += 1
                self.logger.info(f"Trial {trial_count}: score={score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Trial {trial_count} failed: {e}")
                trial_count += 1
        
        self.best_params = best_params
        self.best_score = best_score
        
        self.logger.info(f"Grid search complete (best: {best_score:.4f})")
        
        return {
            'method': 'grid_search',
            'model_type': model_type,
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': trial_count
        }

    def _evaluate_mlp_params(self, _x_train, _y_train, _x_val, _y_val, _params, _metric_func=None):
        """Evaluate MLP with given parameters (mock implementation)"""
        rng = np.random.default_rng(self.random_seed)
        score = rng.uniform(0.5, 0.95)
        return score

    def _evaluate_lstm_params(self, _x_train, _y_train, _x_val, _y_val, _params, _metric_func=None):
        """Evaluate LSTM with given parameters (mock implementation)"""
        rng = np.random.default_rng(self.random_seed)
        score = rng.uniform(0.4, 0.9)
        return score

    def _get_default_param_space(self, model_type):
        """Get default parameter space for model"""
        if model_type == 'mlp':
            return {
                'hidden_size_1': [64, 128, 256],
                'hidden_size_2': [32, 64, 128],
                'dropout': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.005],
                'batch_size': [32, 64],
                'epochs': [50, 100]
            }
        elif model_type == 'lstm':
            return {
                'hidden_size': [64, 128],
                'n_layers': [1, 2],
                'dropout': [0.1, 0.2],
                'learning_rate': [0.001, 0.005],
                'batch_size': [32, 64],
                'epochs': [50, 100]
            }
        return {}

    def save_results(self, filepath=None):
        """Зберегти результати пошуку"""
        if filepath is None:
            filepath = self.project_path / f"hyperparameter_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results = {
            'method': self.method,
            'n_trials': self.n_trials,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'trial_history': self.trial_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Search results saved to {filepath}")
        return filepath

    def get_summary(self):
        """Отримати короткий звіт"""
        return {
            'method': self.method,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'n_trials_completed': len(self.trial_history),
            'top_5_trials': sorted(
                self.trial_history,
                key=lambda x: x['score'],
                reverse=True
            )[:5]
        }


# EXAMPLE USAGE
if __name__ == "__main__":
    ProjectLogger.setup_logging()
    logger = ProjectLogger.get_logger("HyperparameterSearchRunner")
    logger.info("=== HyperparameterSearcher Example ===")
    
    # Симуляція даних
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((500, 20))
    y_train = rng.standard_normal(500)
    x_val = rng.standard_normal((100, 20))
    y_val = rng.standard_normal(100)
    
    # Ініціалізація
    searcher = HyperparameterSearcher(method='optuna', n_trials=5)
    
    # Пошук параметрів для MLP
    logger.info("Searching for optimal MLP parameters...")
    results = searcher.search_mlp_params(X_train, y_train, x_val, y_val)
    
    logger.info(f"Best parameters: {results['best_params']}")
    logger.info(f"Best score: {results['best_score']:.4f}")
    
    # Збереження
    searcher.save_results()
    
    # Звіт
    summary = searcher.get_summary()
    logger.info(f"Summary: {summary}")
