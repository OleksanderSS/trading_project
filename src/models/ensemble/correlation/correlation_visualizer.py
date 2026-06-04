#!/usr/bin/env python3
"""
Correlation Visualizer - Visualization for model correlation analysis
Handles visualization of correlation matrices and analysis results.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("CorrelationVisualizer")


class CorrelationVisualizer:
    """
    Correlation visualizer for model correlation analysis.

    Handles:
    - Correlation matrix heatmap plotting
    - Diversity metrics visualization
    - Redundancy analysis visualization
    - Trend analysis visualization
    """

    def __init__(self, output_dir: Path | None = None):
        """
        Initialize Correlation Visualizer.

        Args:
            output_dir: Directory for saving visualizations
        """
        self.logger = logger
        self.output_dir = output_dir or Path('reports/correlation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("✅ CorrelationVisualizer initialized")

    def plot_correlation_matrix(self, correlation_matrix: dict[str, dict[str, float]],
                               correlation_method: str = "pearson",
                               save_path: str | None = None) -> str:
        """
        Plot correlation matrix heatmap.

        Args:
            correlation_matrix: Correlation matrix dictionary
            correlation_method: Correlation method used
            save_path: Optional path to save the plot

        Returns:
            Path to saved plot
        """
        try:
            # Convert to DataFrame
            model_names = list(correlation_matrix.keys())
            corr_df = pd.DataFrame(correlation_matrix, index=model_names, columns=model_names)

            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})

            plt.title(f'Model Correlation Matrix ({correlation_method.capitalize()})')
            plt.xlabel('Models')
            plt.ylabel('Models')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Correlation matrix plot saved to {save_path}")
            else:
                default_path = self.output_dir / 'correlation_matrix.png'
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Correlation matrix plot saved to {default_path}")
                save_path = str(default_path)

            plt.close()
            return save_path

        except Exception as e:
            self.logger.error(f"Error plotting correlation matrix: {e}")
            return ""

    def plot_diversity_metrics(self, diversity_metrics: dict[str, float],
                               save_path: str | None = None) -> str:
        """
        Plot diversity metrics bar chart.

        Args:
            diversity_metrics: Dictionary of diversity metrics
            save_path: Optional path to save the plot

        Returns:
            Path to saved plot
        """
        try:
            # Filter out overall_diversity for separate display
            metrics_to_plot = {k: v for k, v in diversity_metrics.items() if k != 'overall_diversity'}

            if not metrics_to_plot:
                return ""

            # Create bar chart
            plt.figure(figsize=(10, 6))
            plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
            plt.title('Diversity Metrics')
            plt.ylabel('Value')
            plt.xlabel('Metric')
            plt.xticks(rotation=45)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Diversity metrics plot saved to {save_path}")
            else:
                default_path = self.output_dir / 'diversity_metrics.png'
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Diversity metrics plot saved to {default_path}")
                save_path = str(default_path)

            plt.close()
            return save_path

        except Exception as e:
            self.logger.error(f"Error plotting diversity metrics: {e}")
            return ""

    def plot_redundant_pairs(self, redundant_pairs: list,
                            save_path: str | None = None) -> str:
        """
        Plot redundant pairs visualization.

        Args:
            redundant_pairs: List of redundant pair dictionaries
            save_path: Optional path to save the plot

        Returns:
            Path to saved plot
        """
        try:
            if not redundant_pairs:
                self.logger.info("No redundant pairs to plot")
                return ""

            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 6))

            pair_names = [f"{p['model1']}-{p['model2']}" for p in redundant_pairs]
            correlations = [p['correlation'] for p in redundant_pairs]
            colors = ['red' if p['redundancy_level'] == 'high' else 'orange' for p in redundant_pairs]

            ax.barh(pair_names, correlations, color=colors)
            ax.set_xlabel('Correlation')
            ax.set_title('Redundant Model Pairs')
            ax.axvline(x=0.7, color='green', linestyle='--', label='Threshold')
            ax.legend()
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Redundant pairs plot saved to {save_path}")
            else:
                default_path = self.output_dir / 'redundant_pairs.png'
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Redundant pairs plot saved to {default_path}")
                save_path = str(default_path)

            plt.close()
            return save_path

        except Exception as e:
            self.logger.error(f"Error plotting redundant pairs: {e}")
            return ""

    def plot_diversity_trend(self, analysis_history: list,
                            save_path: str | None = None) -> str:
        """
        Plot diversity score trend over time.

        Args:
            analysis_history: List of analysis results
            save_path: Optional path to save the plot

        Returns:
            Path to saved plot
        """
        try:
            if len(analysis_history) < 2:
                self.logger.info("Not enough data to plot trend")
                return ""

            # Extract diversity scores over time
            timestamps = [a['timestamp'] for a in analysis_history]
            diversity_scores = [
                a.get('diversity_metrics', {}).get('overall_diversity', 0)
                for a in analysis_history
            ]

            # Create line plot
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, diversity_scores, marker='o', linewidth=2)
            plt.title('Diversity Score Trend Over Time')
            plt.xlabel('Time')
            plt.ylabel('Overall Diversity Score')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Diversity trend plot saved to {save_path}")
            else:
                default_path = self.output_dir / 'diversity_trend.png'
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Diversity trend plot saved to {default_path}")
                save_path = str(default_path)

            plt.close()
            return save_path

        except Exception as e:
            self.logger.error(f"Error plotting diversity trend: {e}")
            return ""

    def create_analysis_report(self, analysis_results: dict[str, Any],
                             save_path: str | None = None) -> str:
        """
        Create comprehensive analysis report with visualizations.

        Args:
            analysis_results: Dictionary with analysis results
            save_path: Optional path to save the report

        Returns:
            Path to saved report
        """
        try:
            report_paths = []

            # Plot correlation matrix
            if 'correlation_matrix' in analysis_results:
                corr_path = self.plot_correlation_matrix(
                    analysis_results['correlation_matrix'],
                    analysis_results.get('correlation_method', 'pearson')
                )
                if corr_path:
                    report_paths.append(corr_path)

            # Plot diversity metrics
            if 'diversity_metrics' in analysis_results:
                div_path = self.plot_diversity_metrics(analysis_results['diversity_metrics'])
                if div_path:
                    report_paths.append(div_path)

            # Plot redundant pairs
            if 'redundant_pairs' in analysis_results:
                red_path = self.plot_redundant_pairs(analysis_results['redundant_pairs'])
                if red_path:
                    report_paths.append(red_path)

            self.logger.info(f"Analysis report created with {len(report_paths)} visualizations")
            return str(report_paths) if report_paths else ""

        except Exception as e:
            self.logger.error(f"Error creating analysis report: {e}")
            return ""


# Factory function
def get_correlation_visualizer(output_dir: Path | None = None) -> CorrelationVisualizer:
    """Factory function to get CorrelationVisualizer instance."""
    return CorrelationVisualizer(output_dir)
