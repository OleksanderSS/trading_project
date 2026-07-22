"""
Пакет оптимізації (src/optimization).
Відповідає за математичне моделювання розподілу активів та знаходження оптимальних параметрів системи.

Optimization package.
Handles portfolio allocation and hyperparameter optimization.
"""

from .factory import OptimizationFactory
from .hyperparameters.bayesian import BayesianOptimizer
from .portfolio.optimizer import PortfolioOptimizer

__all__ = [
    "PortfolioOptimizer",
    "BayesianOptimizer",
    "OptimizationFactory"
]
