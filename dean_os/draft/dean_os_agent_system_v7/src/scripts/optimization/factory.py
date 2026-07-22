# src/optimization/factory.py

from src.core.logging.logger import ProjectLogger
from src.scripts.optimization.base import BaseOptimizer
from src.scripts.optimization.hyperparameters.bayesian import BayesianOptimizer
from src.scripts.optimization.portfolio.optimizer import PortfolioOptimizer


class OptimizationFactory:
    """
    Central point for obtaining optimization tools based on the requested type.
    This factory simplifies the instantiation of different optimization engines
    used across the trading system.
    """

    logger = ProjectLogger.get_logger("OptimizationFactory")

    @staticmethod
    def get_optimizer(optimizer_type: str, **kwargs) -> BaseOptimizer:
        """
        Instantiates and returns an optimizer instance based on the provided type.

        Args:
            optimizer_type (str): The type of optimizer ('portfolio', 'hyperparameters', or 'bayesian').
            **kwargs: Configuration parameters to pass to the specific optimizer constructor.

        Returns:
            BaseOptimizer: An instance of a class derived from BaseOptimizer.

        Raises:
            ValueError: If the requested optimizer_type is not supported.
        """
        optimizer_type = optimizer_type.lower()

        OptimizationFactory.logger.info(f"Creating optimizer of type: {optimizer_type}")

        if optimizer_type == 'portfolio':
            return PortfolioOptimizer(**kwargs)

        elif optimizer_type in ['hyperparameters', 'bayesian']:
            # BayesianOptimizer might require model_func and param_space in kwargs
            return BayesianOptimizer(**kwargs)

        else:
            error_msg = f"Unsupported optimizer type: {optimizer_type}"
            OptimizationFactory.logger.error(error_msg)
            raise ValueError(error_msg)
