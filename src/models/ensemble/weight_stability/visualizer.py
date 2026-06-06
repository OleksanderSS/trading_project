import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
from typing import Any

import matplotlib.pyplot as plt

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("WeightStabilityVisualizer")

class WeightStabilityVisualizer:
    """Provides visualization for weight stability metrics."""

    def __init__(self, config: Any):
        self.logger = logger
        self.config = config

    def plot_stability_metrics(self,
                             weight_history: list[dict[str, Any]],
                             weight_changes: list[dict[str, float]],
                             current_models: list[str],
                             save_path: str | None = None) -> None:
        """Plot stability metrics over time."""
        try:
            if len(weight_history) < 2:
                return

            timestamps = [record['timestamp'] for record in weight_history]
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Weight Stability Metrics Over Time')

            for model in current_models:
                weights = [record.get('weights', {}).get(model, np.nan) for record in weight_history]
                axes[0, 0].plot(timestamps, weights, marker='o', label=model)
            axes[0, 0].set_title('Model Weights')
            axes[0, 0].set_ylabel('Weight')
            axes[0, 0].legend(loc='best')

            change_timestamps = timestamps[-len(weight_changes):] if weight_changes else []
            for model in current_models:
                changes = [abs(change.get(model, 0.0)) for change in weight_changes]
                if changes:
                    axes[0, 1].plot(change_timestamps, changes, marker='o', label=model)
            axes[0, 1].set_title('Absolute Weight Changes')
            axes[0, 1].set_ylabel('Abs Change')
            axes[0, 1].legend(loc='best')

            avg_changes = [
                np.mean([abs(change.get(model, 0.0)) for change in weight_changes])
                for model in current_models
            ]
            axes[1, 0].bar(current_models, avg_changes)
            axes[1, 0].set_title('Average Absolute Change')
            axes[1, 0].set_ylabel('Mean Abs Change')

            stability_scores = [
                max(0.0, 1.0 - sum(abs(value) for value in change.values()))
                for change in weight_changes
            ]
            if stability_scores:
                axes[1, 1].plot(change_timestamps, stability_scores, marker='o')
            axes[1, 1].set_title('Stability Score Proxy')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_ylim(0, 1)

            for axis in axes.flatten():
                axis.tick_params(axis='x', rotation=30)

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Stability metrics plot saved to {save_path}")
            else:
                plt.show()
            plt.close()
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error plotting stability metrics: {e}")
