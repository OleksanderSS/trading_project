# src/core/reporting/visualization.py

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.core.file_management.file_manager import FileManager
from src.core.logging.logger import ProjectLogger

# Initialize logger for the module
logger = ProjectLogger.get_logger("Visualizer")

# Set a consistent plot style
sns.set_theme(style="whitegrid")

class Visualizer:
    """
    Handles the creation and saving of various plots and charts for data analysis.
    """

    def __init__(self, file_manager: FileManager, output_dir: str = "reports/charts"):
        self.fm = file_manager
        self.output_dir = Path(output_dir)
        self.fm.ensure_directory(self.output_dir)
        logger.info(f"Visualizer initialized. Charts will be saved to '{self.output_dir}'.")

    def _save_plot(self, fig: plt.Figure, filename: str) -> Path | None:
        """Saves a matplotlib figure to the specified file."""
        try:
            full_path = self.output_dir / filename
            fig.savefig(full_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            logger.info(f"Successfully saved plot: {full_path}")
            return full_path
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Failed to save plot {filename}: {e}")
            plt.close(fig)
            raise RuntimeError(f"Failed to save plot {filename}") from e

    def plot_price_history(self, df: pd.DataFrame, price_col: str, title: str, filename: str) -> Path | None:
        """
        Plots the historical price data.

        Args:
            df: DataFrame with a datetime index and a price column.
            price_col: The name of the column containing the price data.
            title: The title for the chart.
            filename: The name of the file to save the chart.

        Returns:
            The path to the saved chart, or None on failure.
        """
        if price_col not in df.columns:
            logger.error(f"Price column '{price_col}' not found in DataFrame.")
            return None

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df.index, df[price_col], label=f'{price_col} Price')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.tight_layout()

        return self._save_plot(fig, filename)

    def plot_distribution(self, data: pd.Series, title: str, filename: str, bins: int = 30) -> Path | None:
        """
        Plots the distribution of a data series.

        Args:
            data: A pandas Series of numerical data.
            title: The title for the chart.
            filename: The name of the file to save the chart.
            bins: The number of bins for the histogram.

        Returns:
            The path to the saved chart, or None on failure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data, kde=True, ax=ax, bins=bins)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        return self._save_plot(fig, filename)

    def plot_correlation_matrix(self, df: pd.DataFrame, title: str, filename: str) -> Path | None:
        """
        Plots a heatmap of the correlation matrix for a DataFrame.

        Args:
            df: DataFrame with numerical columns to correlate.
            title: The title for the chart.
            filename: The name of the file to save the chart.

        Returns:
            The path to the saved chart, or None on failure.
        """
        corr_matrix = df.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title(title, fontsize=16)
        fig.tight_layout()

        return self._save_plot(fig, filename)
