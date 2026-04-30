# ModelEnsembleComposer Implementation
# Combines top-N trained models into ensemble for improved predictions

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.core.logging.logger import ProjectLogger

class ModelEnsembleComposer:
    """
    Комбінує топ-N тренованих моделей в ensemble для покращених прогнозів.
    """

    # Constants to avoid duplication
    ERROR_NO_MODELS = "No models in ensemble"

    def __init__(self, project_path=None, weighting_metric='r2'):
        """
        Args:
            project_path: Шлях проекту для збереження ensemble моделей
            weighting_metric: Метрика для ваг ('r2', 'rmse', 'mape', 'equal')
        """
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self.weighting_metric = weighting_metric
        self.ensemble_models = []
        self.ensemble_weights = []
        self.ensemble_config = None
        self.composition_log = []
        self.logger = ProjectLogger.get_logger("ModelEnsembleComposer")

    def add_model(self, model_name, model_metrics, model_object=None):
        """
        Додає модель до ensemble кандидатів.
        """
        if model_metrics.get('r2') is None or model_metrics.get('r2') < 0:
            self.logger.warning(f"Skipping {model_name}: negative or missing R² ({model_metrics.get('r2')})")
            return False

        self.ensemble_models.append({
            'name': model_name,
            'metrics': model_metrics,
            'object': model_object,
            'added_at': datetime.now().isoformat()
        })
        
        self.logger.info(f"Added {model_name} to ensemble (R²={model_metrics.get('r2'):.4f})")
        return True

    def compose_weighted_average(self, top_n=3):
        """
        Компонує ensemble з топ-N моделей, використовуючи weighted average.
        """
        if not self.ensemble_models:
            self.logger.error(self.ERROR_NO_MODELS)
            return None

        # Сортуємо за R²
        sorted_models = sorted(
            self.ensemble_models,
            key=lambda x: x['metrics'].get('r2', 0),
            reverse=True
        )

        top_models = sorted_models[:top_n]
        
        # Розраховуємо ваги
        if self.weighting_metric == 'r2':
            r2_scores = [max(0, m['metrics'].get('r2', 0)) for m in top_models]
            total_r2 = sum(r2_scores)
            weights = [r2 / total_r2 if total_r2 > 0 else 1/len(top_models) for r2 in r2_scores]
        
        elif self.weighting_metric == 'rmse':
            rmse_scores = [m['metrics'].get('rmse', 0.1) for m in top_models]
            inverse_rmse = [1 / (rmse + 1e-6) for rmse in rmse_scores]
            total = sum(inverse_rmse)
            weights = [w / total for w in inverse_rmse]
        
        elif self.weighting_metric == 'mape':
            mape_scores = [m['metrics'].get('mape', 1) for m in top_models]
            inverse_mape = [1 / (mape + 1e-6) for mape in mape_scores]
            total = sum(inverse_mape)
            weights = [w / total for w in inverse_mape]
        
        else:  # equal weights
            weights = [1 / len(top_models)] * len(top_models)

        self.ensemble_models = top_models
        self.ensemble_weights = weights

        config = {
            'method': 'weighted_average',
            'weighting_metric': self.weighting_metric,
            'models_count': len(top_models),
            'model_names': [m['name'] for m in top_models],
            'weights': weights,
            'model_metrics': [m['metrics'] for m in top_models],
            'composition_time': datetime.now().isoformat()
        }
        self.ensemble_config = config
        self._log_composition(config)
        
        self.logger.info(f"Ensemble composed (weighted average, top-{top_n})")
        for model, weight in zip(top_models, weights):
            self.logger.info(f" - {model['name']}: weight={weight:.4f}, R²={model['metrics'].get('r2', 0):.4f}")

        return config

    def compose_median_ensemble(self, top_n=3):
        """
        Компонує ensemble з топ-N моделей, використовуючи median (стійко до викидів).
        """
        if not self.ensemble_models:
            self.logger.error(self.ERROR_NO_MODELS)
            return None

        sorted_models = sorted(
            self.ensemble_models,
            key=lambda x: x['metrics'].get('r2', 0),
            reverse=True
        )

        top_models = sorted_models[:top_n]
        weights = [1 / len(top_models)] * len(top_models)

        self.ensemble_models = top_models
        self.ensemble_weights = weights

        config = {
            'method': 'median',
            'models_count': len(top_models),
            'model_names': [m['name'] for m in top_models],
            'weights': weights,
            'model_metrics': [m['metrics'] for m in top_models],
            'composition_time': datetime.now().isoformat()
        }
        self.ensemble_config = config
        self._log_composition(config)

        self.logger.info(f"Ensemble composed (median, top-{top_n})")
        for model in top_models:
            self.logger.info(f" - {model['name']}: R²={model['metrics'].get('r2', 0):.4f}")

        return config

    def compose_voting_ensemble(self, top_n=3):
        """
        Компонує ensemble на основі голосування (для класифікації напрямку).
        """
        if not self.ensemble_models:
            self.logger.error(self.ERROR_NO_MODELS)
            return None

        sorted_models = sorted(
            self.ensemble_models,
            key=lambda x: x['metrics'].get('r2', 0),
            reverse=True
        )

        top_models = sorted_models[:top_n]
        weights = [1 / len(top_models)] * len(top_models)

        self.ensemble_models = top_models
        self.ensemble_weights = weights

        config = {
            'method': 'voting',
            'models_count': len(top_models),
            'model_names': [m['name'] for m in top_models],
            'weights': weights,
            'model_metrics': [m['metrics'] for m in top_models],
            'composition_time': datetime.now().isoformat()
        }
        self.ensemble_config = config
        self._log_composition(config)

        self.logger.info(f"Ensemble composed (voting, top-{top_n})")
        for model in top_models:
            self.logger.info(f" - {model['name']}: R²={model['metrics'].get('r2', 0):.4f}")

        return config

    def predict_ensemble(self, y_predictions_dict):
        """
        Комбінує прогнози від кількох моделей.
        """
        if self.ensemble_config is None:
            self.logger.error("Ensemble not composed yet")
            return None

        method = self.ensemble_config['method']
        model_names = self.ensemble_config['model_names']

        predictions_matrix = []
        for model_name in model_names:
            if model_name in y_predictions_dict:
                predictions_matrix.append(y_predictions_dict[model_name])

        if not predictions_matrix:
            self.logger.error("No predictions found for ensemble models")
            return None

        predictions_array = np.array(predictions_matrix)

        if method == 'weighted_average':
            ensemble_pred = np.average(predictions_array, axis=0, weights=self.ensemble_weights)
        elif method == 'median':
            ensemble_pred = np.median(predictions_array, axis=0)
        elif method == 'voting':
            signs = np.sign(predictions_array)
            votes = np.sum(signs, axis=0)
            ensemble_pred = np.sign(votes)
        else:
            ensemble_pred = np.mean(predictions_array, axis=0)

        return ensemble_pred

    def get_ensemble_metrics(self, y_true, ensemble_pred):
        """
        Розраховує метрики для ensemble прогнозів.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_true, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
        r2 = r2_score(y_true, ensemble_pred)
        
        mask = y_true != 0
        mape = (
            np.mean(np.abs((y_true[mask] - ensemble_pred[mask]) / y_true[mask])) * 100
            if mask.sum() > 0 else 0.0
        )

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape)
        }

    def _log_composition(self, config):
        """Залогувати складання ensemble"""
        self.composition_log.append(config)

    def save_ensemble_config(self, filepath=None):
        """Зберегти конфігурацію ensemble"""
        if filepath is None:
            filepath = self.project_path / "ensemble_config.json"

        config_to_save = {
            'ensemble_config': self.ensemble_config,
            'all_compositions': self.composition_log
        }

        with open(filepath, 'w') as f:
            json.dump(config_to_save, f, indent=2)

        self.logger.info(f"Ensemble configuration saved to {filepath}")
        return filepath

    def get_config(self):
        """Отримати поточну конфігурацію ensemble"""
        return self.ensemble_config


# EXAMPLE USAGE
if __name__ == "__main__":
    ProjectLogger.setup_logging()
    logger = ProjectLogger.get_logger("EnsembleComposerExample")
    logger.info("=== Model Ensemble Composer Example ===")

    # Ініціалізація
    composer = ModelEnsembleComposer(weighting_metric='r2')

    # Додавання моделей
    models = [
        {'name': 'lstm', 'metrics': {'r2': 0.87, 'rmse': 0.025, 'mape': 8.5}},
        {'name': 'transformer', 'metrics': {'r2': 0.85, 'rmse': 0.028, 'mape': 9.2}},
        {'name': 'mlp', 'metrics': {'r2': 0.81, 'rmse': 0.032, 'mape': 11.3}},
        {'name': 'cnn', 'metrics': {'r2': 0.79, 'rmse': 0.035, 'mape': 12.5}},
        {'name': 'gru', 'metrics': {'r2': 0.83, 'rmse': 0.030, 'mape': 10.1}},
    ]

    for model in models:
        composer.add_model(model['name'], model['metrics'])

    # Складання ensemble
    logger.info("--- Weighted Average Ensemble ---")
    composer.compose_weighted_average(top_n=3)

    # Збереження конфігурації
    composer.save_ensemble_config()

    # Приклад прогнозів
    logger.info("--- Ensemble Predictions ---")
    y_test = np.array([0.05, -0.02, 0.03, 0.01, -0.01])
    y_preds = {
        'lstm': np.array([0.048, -0.018, 0.032, 0.012, -0.008]),
        'transformer': np.array([0.052, -0.025, 0.028, 0.008, -0.005]),
        'mlp': np.array([0.045, -0.010, 0.035, 0.015, -0.012])
    }

    ensemble_pred = composer.predict_ensemble(y_preds)
    logger.info(f"Ensemble predictions: {ensemble_pred}")

    metrics = composer.get_ensemble_metrics(y_test, ensemble_pred)
    logger.info(f"Ensemble metrics: {metrics}")
