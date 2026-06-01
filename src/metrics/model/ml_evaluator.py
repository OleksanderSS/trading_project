import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from src.core.logging.logger import ProjectLogger
from src.metrics.base import BaseMetricCalculator


class MLEvaluator(BaseMetricCalculator):
    """
    Клас для оцінки якості прогнозів моделей машинного навчання.
    Підтримує метрики регресії, класифікації та імовірнісні метрики.
    """

    def __init__(self):
        self.logger = ProjectLogger.get_logger('MLEvaluator')

    @property
    def category(self) ->str:
        """Повертає категорію метрик."""
        return 'ml'

    def validate_input(self, data: Any) ->bool:
        """Перевіряє придатність даних для розрахунку метрик."""
        if not isinstance(data, (np.ndarray, pd.Series, list)):
            self.logger.error(
                'Дані повинні бути у форматі numpy array, pandas Series або list.'
                )
            return False
        return True

    def calculate(self, y_true: Any, y_pred: Any, task_type: Optional[str]=
        None, **kwargs) ->Dict[str, float]:
        """
        Розраховує ML метрики на основі типу задачі.
        
        Args:
            y_true: Істинні значення.
            y_pred: Прогнозовані значення.
            task_type: Тип задачі ('classification', 'regression', 'probabilistic').
            **kwargs: Додаткові параметри (наприклад, y_prob для AUC).
        """
        if not self.validate_input(y_true) or not self.validate_input(y_pred):
            return {}
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & np.isfinite(y_true
            ) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        if len(y_true) == 0:
            self.logger.warning('Немає валідних даних для розрахунку метрик.')
            return {}
        if task_type is None:
            task_type = self._infer_task_type(y_true, y_pred)
            self.logger.info(f'Тип задачі визначено автоматично: {task_type}')
        if task_type == 'regression':
            return self.calculate_regression_metrics(y_true, y_pred)
        elif task_type == 'classification':
            return self.calculate_classification_metrics(y_true, y_pred)
        elif task_type == 'probabilistic':
            y_prob = kwargs.get('y_prob')
            return self._calculate_probabilistic_metrics(y_true, y_pred, y_prob
                )
        else:
            self.logger.error(f'Невідомий тип задачі: {task_type}')
            return {}

    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.
        ndarray) ->Dict[str, float]:
        """Розраховує метрики регресії."""
        mse = mean_squared_error(y_true, y_pred)
        return {'MAE': float(mean_absolute_error(y_true, y_pred)), 'MSE':
            float(mse), 'RMSE': float(np.sqrt(mse)), 'R2': float(r2_score(
            y_true, y_pred))}

    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred:
        np.ndarray, y_prob: Optional[np.ndarray]=None) ->Dict[str, float]:
        """Розраховує метрики класифікації."""
        metrics = {'Accuracy': float(accuracy_score(y_true, y_pred)),
            'Precision': float(precision_score(y_true, y_pred, average=
            'binary', zero_division=0)), 'Recall': float(recall_score(
            y_true, y_pred, average='binary', zero_division=0)), 'F1':
            float(f1_score(y_true, y_pred, average='binary', zero_division=0))}
        if y_prob is not None:
            try:
                metrics['ROC_AUC'] = float(roc_auc_score(y_true, y_prob))
                metrics['Log_Loss'] = float(log_loss(y_true, y_prob))
            except Exception as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                self.logger.warning(
                    f'Не вдалося розрахувати імовірнісні метрики: {e}')
                raise
        return metrics

    def _calculate_probabilistic_metrics(self, y_true: np.ndarray, y_pred:
        np.ndarray, y_prob: Optional[np.ndarray]=None) ->Dict[str, float]:
        """Розраховує імовірнісні метрики (ROC AUC, Log Loss)."""
        probs = y_prob if y_prob is not None else y_pred
        try:
            return {'ROC_AUC': float(roc_auc_score(y_true, probs)),
                'Log_Loss': float(log_loss(y_true, probs))}
        except Exception as e:
            self.logger.error(f'Помилка розрахунку імовірнісних метрик: {e}')
            return {}

    def _infer_task_type(self, y_true: np.ndarray, y_pred: np.ndarray) ->str:
        """Визначає тип задачі на основі структури даних."""
        unique_vals = np.unique(y_pred)
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
            return 'classification'
        elif np.all((y_pred >= 0) & (y_pred <= 1)) and np.issubdtype(y_pred
            .dtype, np.floating):
            return 'probabilistic'
        else:
            return 'regression'


def calculate_all_metrics(y_true: Any, y_pred: Any, task_type: Optional[str
    ]=None, **kwargs) ->Dict[str, float]:
    """Глобальна функція-обгортка для сумісності з іншими модулями."""
    evaluator = MLEvaluator()
    return evaluator.calculate(y_true, y_pred, task_type=task_type, **kwargs)


def infer_task_type(y_true: Any, y_pred: Any) ->str:
    """Глобальна функція для визначення типу задачі."""
    evaluator = MLEvaluator()
    return evaluator._infer_task_type(np.asarray(y_true), np.asarray(y_pred))
