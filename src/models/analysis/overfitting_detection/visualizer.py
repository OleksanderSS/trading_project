from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class OverfittingVisualizer:
    """Provides visualizations for overfitting analysis."""

    def __init__(self, config: Any):
        self.config = config

    def plot_learning_curve(self, learning_curve_data: dict[str, Any], save_path: Path) -> None:
        """Plot and save the learning curve."""
        try:
            if not learning_curve_data:
                return

            plt.figure(figsize=(10, 6))
            train_sizes = learning_curve_data['train_sizes']
            train_scores_mean = learning_curve_data['train_scores_mean']
            test_scores_mean = learning_curve_data['test_scores_mean']

            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

            plt.title("Learning Curve")
            plt.xlabel("Training Examples")
            plt.ylabel("Score")
            plt.legend(loc="best")
            plt.grid(True)

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error plotting learning curve: {e}", exc_info=True)

    def plot_cv_distribution(self, cv_results: dict[str, Any], save_path: Path) -> None:
        """Plot cross-validation score distribution."""
        try:
            if not cv_results:
                return

            plt.figure(figsize=(10, 6))
            sns.boxplot(y=cv_results['scores'])
            plt.title("Cross-Validation Score Distribution")
            plt.ylabel("Score")

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error plotting CV distribution: {e}", exc_info=True)
