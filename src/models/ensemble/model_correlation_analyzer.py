#!/usr/bin/env python3
"""
Model Correlation Analyzer - Facade for Modular Model Correlation Analysis.
Maintains backward compatibility with the original ModelCorrelationAnalyzer.
"""

from typing import Any

import pandas as pd

from .correlation.correlation_engine import get_correlation_engine
from .correlation.correlation_visualizer import get_correlation_visualizer


class ModelCorrelationAnalyzer:
    """
    Facade for ModelCorrelationAnalyzer.
    Delegates to modular components in the 'correlation' subdirectory.
    """

    def __init__(self,
                 correlation_method: str = "pearson",
                 diversity_threshold: float = 0.7):
        """
        Initialize Model Correlation Analyzer.

        Args:
            correlation_method: Method for correlation calculation
            diversity_threshold: Threshold for diversity filtering
        """
        self.correlation_method = correlation_method
        self.diversity_threshold = diversity_threshold

        # Initialize components
        self.engine = get_correlation_engine(correlation_method, diversity_threshold)
        self.visualizer = get_correlation_visualizer()

        # Analysis history for backward compatibility
        self.analysis_history = []

    def analyze_correlation(self,
                           models: dict[str, Any],
                           X: pd.DataFrame,
                           y: pd.Series,
                           sample_size: int | None = None) -> dict[str, Any]:
        """Analyze prediction correlation between models."""
        results = self.engine.analyze_correlation(models, X, y, sample_size)
        if 'error' not in results:
            self.analysis_history.append(results)
        return results

    def select_diverse_subset(self,
                             models: dict[str, Any],
                             X: pd.DataFrame,
                             y: pd.Series,
                             max_models: int = 5,
                             diversity_threshold: float | None = None) -> list[str]:
        """Select diverse subset of models for ensemble."""
        return self.engine.select_diverse_subset(models, X, y, max_models, diversity_threshold)

    def adjust_weights_by_correlation(self,
                                    base_weights: dict[str, float],
                                    correlation_matrix: dict[str, dict[str, float]]) -> dict[str, float]:
        """Adjust ensemble weights based on model correlation."""
        return self.engine.adjust_weights_by_correlation(base_weights, correlation_matrix)

    def plot_correlation_matrix(self,
                               correlation_matrix: dict[str, dict[str, float]],
                               save_path: str | None = None) -> None:
        """Plot correlation matrix heatmap."""
        self.visualizer.plot_correlation_matrix(correlation_matrix, self.correlation_method, save_path)

    def get_analysis_summary(self, days: int = 30) -> dict[str, Any]:
        """Get summary of correlation analysis over time period."""
        return self.engine.get_analysis_summary(self.analysis_history, days)

# Factory function for easy instantiation
def get_model_correlation_analyzer(correlation_method: str = "pearson",
                                 diversity_threshold: float = 0.7) -> ModelCorrelationAnalyzer:
    """Factory function to get ModelCorrelationAnalyzer instance."""
    return ModelCorrelationAnalyzer(correlation_method, diversity_threshold)

# Convenience function for quick analysis
def analyze_model_correlation_quick(models: dict[str, Any],
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 correlation_method: str = "pearson") -> dict[str, Any]:
    """Quick model correlation analysis."""
    analyzer = get_model_correlation_analyzer(correlation_method)
    return analyzer.analyze_correlation(models, X, y)
