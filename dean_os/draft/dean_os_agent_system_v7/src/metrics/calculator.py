from typing import Any

import numpy as np
import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.metrics.financial.portfolio_metrics import PortfolioMetricsCalculator
from src.metrics.model.ml_evaluator import MLEvaluator


class MetricsCalculator:
    """
    Єдиний двигун метрик для розрахунку показників моделей та фінансових результатів.
    Об'єднує функціонал MLEvaluator та PortfolioMetricsCalculator.
    """

    @staticmethod
    def _get_config_value(config: Any, key: str, default: Any) -> Any:
        for accessor_name in ("get_setting", "get"):
            accessor = getattr(config, accessor_name, None)
            if callable(accessor):
                return accessor(key, default)
        return default

    def __init__(self, config_manager: Any | None = None):
        """
        Ініціалізація калькулятора метрик.

        Args:
            config_manager: Менеджер конфігурацій для налаштування порогів та параметрів.
        """
        self.config = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger("MetricsCalculator")

        # Ініціалізація спеціалізованих калькуляторів
        self.ml_evaluator = MLEvaluator()  # MLEvaluator не приймає параметри
        self.portfolio_metrics = PortfolioMetricsCalculator(config_manager=config_manager)  # Передаємо оригінальний config_manager

        # Отримання порогів для оцінки з конфігурації
        self.grade_thresholds = self._get_config_value(self.config,
            'metrics.grade_thresholds', {})
        self.accuracy_threshold = self.grade_thresholds.get('high_performance_accuracy', 0.6)
        self.sharpe_threshold = self.grade_thresholds.get('stable_profit_sharpe', 1.0)

        self.logger.info("MetricsCalculator successfully initialized.")

    def get_ml_metrics(self,
                       y_true: np.ndarray | pd.Series,
                       y_pred: np.ndarray | pd.Series,
                       y_prob: np.ndarray | pd.Series | None = None,
                       **kwargs) -> dict[str, Any]:
        """
        Розраховує метрики машинного навчання (точність, повнота, F1 тощо).
        """
        self.logger.info("Розрахунок ML метрик...")
        return self.ml_evaluator.calculate(y_true=y_true, y_pred=y_pred, y_prob=y_prob, **kwargs)

    def get_portfolio_metrics(self,
                             equity_curve: np.ndarray | pd.Series,
                             **kwargs) -> dict[str, Any]:
        """
        Розраховує фінансові метрики портфеля (Sharpe, Drawdown, PnL).
        """
        self.logger.info("Розрахунок фінансових метрик портфеля...")

        if isinstance(equity_curve, np.ndarray):
            equity_curve = pd.Series(equity_curve)

        return self.portfolio_metrics.calculate(equity_curve=equity_curve, **kwargs)

    def calculate(self,
                  y_true: np.ndarray | pd.Series | None = None,
                  y_pred: np.ndarray | pd.Series | None = None,
                  equity_curve: np.ndarray | pd.Series | None = None,
                  **kwargs) -> dict[str, Any]:
        """
        Standardized calculation method returning a unified dictionary.
        """
        results = {
            "ml": {},
            "portfolio": {}
        }

        if y_true is not None and y_pred is not None:
            y_prob = kwargs.pop('y_prob', None)
            results["ml"] = self.get_ml_metrics(y_true, y_pred, y_prob=y_prob, **kwargs)

        if equity_curve is not None:
            results["portfolio"] = self.get_portfolio_metrics(equity_curve, **kwargs)

        return results

    def get_full_report(self,
                        y_true: np.ndarray | pd.Series | None = None,
                        y_pred: np.ndarray | pd.Series | None = None,
                        equity_curve: np.ndarray | pd.Series | None = None,
                        **kwargs) -> dict[str, Any]:
        """
        Генерує повний звіт, що поєднує ML та фінансові метрики.
        """
        report = self.calculate(y_true=y_true, y_pred=y_pred, equity_curve=equity_curve, **kwargs)
        report["summary"] = {}

        # Формування підсумкового статусу на основі порогів з конфігурації
        if report["ml"] or report["portfolio"]:
            report["summary"]["status"] = "success"
            # Перевірка на основі конфігурованих порогів
            if "Accuracy" in report["ml"] and report["ml"]["Accuracy"] > self.accuracy_threshold:
                report["summary"]["grade"] = "high_performance"
            elif "sharpe_ratio" in report["portfolio"] and report["portfolio"]["sharpe_ratio"] > self.sharpe_threshold:
                report["summary"]["grade"] = "stable_profit"
            else:
                report["summary"]["grade"] = "needs_review"
        else:
            report["summary"]["status"] = "no_data"

        self.logger.info("Повний звіт сформовано.")
        return report
