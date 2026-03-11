"""
Пакет оптимізації (src/optimization).
Відповідає за математичне моделювання розподілу активів та знаходження оптимальних параметрів системи.

Optimization package.
Handles portfolio allocation and hyperparameter optimization.
"""

from .portfolio.optimizer import PortfolioOptimizer
from .hyperparameters.bayesian import BayesianOptimizer
from .factory import OptimizationFactory

__all__ = [
    "PortfolioOptimizer",
    "BayesianOptimizer",
    "OptimizationFactory"
]