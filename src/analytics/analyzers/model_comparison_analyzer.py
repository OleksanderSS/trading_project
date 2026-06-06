"""
Model Comparison Analyzer
Analyzes and compares the performance of various machine learning architectures.
Facilitates champion model selection by contrasting heavy vs. light model results.
"""
from typing import Any

import pandas as pd

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger

from ..interfaces import IAnalyzer

logger = ProjectLogger.get_logger(__name__)

class ModelComparisonAnalyzer(IAnalyzer):
    """
    Performs comparative analysis on model result DataFrames to identify
    optimal architectures and assess performance stability across different cycles.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initializes the ModelComparisonAnalyzer.

        Args:
            config: Optional configuration override for heavy model categorization.
        """
        # Default identifiers for heavy/deep learning models
        self.HEAVY_MODELS = ["gru", "tabnet", "transformer", "cnn", "lstm", "autoencoder"]  # audit-ignore: AUTOENCODER_ROUTING_REVIEW
        self.config = config or {}
        self.configured_heavy_models = self.config.get('heavy_models', self.HEAVY_MODELS)
        logger.info("ModelComparisonAnalyzer initialized for comparative benchmarking.")

    def analyze(self, data: dict[str, pd.DataFrame], **kwargs: Any) -> dict[str, Any]:
        """
        Main interface for UnifiedAnalyticsEngine.

        Args:
            data: Dictionary containing a 'results' DataFrame with performance metrics.
            **kwargs: Control flags for specific analytical routines.
        """
        results_df = data.get('results')
        if not isinstance(results_df, pd.DataFrame) or results_df.empty:
            raise DataProcessingError("Input 'results' DataFrame is empty or invalid.")

        # Categorize models as 'heavy' or 'light' if metadata is missing
        if 'model_type' not in results_df.columns:
            results_df['model_type'] = results_df['model'].apply(
                lambda x: 'heavy' if str(x).lower() in [m.lower() for m in self.configured_heavy_models] else 'light'
            )

        analysis_payload: dict[str, Any] = {}

        # Architecture Benchmarking Routine
        if kwargs.get("run_architecture_comparison", True):
            analysis_payload['architecture_comparison'] = self._compare_architectures(results_df)

        # Segmented Leaderboard Routine
        if kwargs.get("run_best_model_finder", True):
            analysis_payload['best_models_by_type'] = self._get_best_models_by_type(results_df)

        # High-level Aggregation Routine
        if kwargs.get("run_overall_summary", True):
            analysis_payload['overall_summary'] = self._summarize_by_type(results_df)

        return analysis_payload

    def _compare_architectures(self, results_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Evaluates reliability and stability across competing model architectures."""
        if 'accuracy' not in results_df.columns:
            raise DataProcessingError("'accuracy' column missing from results.")

        arch_stats = []
        for arch, group in results_df.groupby('model'):
            metrics = {
                'architecture': arch,
                'category': group['model_type'].iloc[0],
                'mean_accuracy': float(group['accuracy'].mean()),
                'stability_std': float(group['accuracy'].std()) if len(group) > 1 else 0.0,
                'sample_count': int(len(group)),
                'peak_performance': float(group['accuracy'].max())
            }
            # Reliability Score: Performance normalized by inverse volatility (stability)
            metrics['reliability_score'] = metrics['mean_accuracy'] * (1 - metrics.get('stability_std', 0.0))
            arch_stats.append(metrics)

        return sorted(arch_stats, key=lambda x: x['reliability_score'], reverse=True)

    def _get_best_models_by_type(self, results_df: pd.DataFrame) -> dict[str, Any]:
        """Identifies champions for each category within asset/timeframe cohorts."""
        if 'accuracy' not in results_df.columns:
            return {}

        leaders: dict[str, dict[str, Any]] = {'light': {}, 'heavy': {}}

        # Determine champion indices via group maximization
        try:
            leader_indices = results_df.loc[results_df.groupby(['ticker', 'timeframe', 'model_type'])['accuracy'].idxmax()]

            for _, row in leader_indices.iterrows():
                category = row['model_type']
                group_key = f"{row['ticker']}_{row['timeframe']}"
                leaders[category][group_key] = {
                    'model': row['model'],
                    'accuracy': float(row['accuracy']),
                }
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Failed to calculate best models by type: {e}")
            raise DataProcessingError(f"Failed to calculate best models by type: {e}") from e

        return leaders

    def _summarize_by_type(self, results_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Provides a high-level statistical summary of the light vs. heavy model split."""
        if 'accuracy' not in results_df.columns:
            return []

        summary = []
        for model_type, group in results_df.groupby('model_type'):
            summary.append({
                'model_category': model_type,
                'avg_accuracy': float(group['accuracy'].mean()),
                'max_accuracy': float(group['accuracy'].max()),
                'min_accuracy': float(group['accuracy'].min()),
                'unique_models': int(group['model'].nunique()),
                'total_records': int(len(group)),
                'ticker_coverage': int(group['ticker'].nunique()),
                'timeframe_coverage': int(group['timeframe'].nunique())
            })

        return summary

    def compare_models(self, training_results: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """
        Contrasts live training results to select the final 'Champion' for production.
        """
        logger.info("Conducting model cross-comparison to determine production champion...")

        tickers_results = training_results.get('tickers_results', {})

        if not tickers_results:
            raise DataProcessingError("Champion selection aborted: No valid training results provided.")

        model_cohort = self._build_model_cohort(tickers_results)

        if not model_cohort:
            raise DataProcessingError("No successful model instances found in the provided cohort.")

        best_heavy, best_light = self._identify_cluster_leaders(model_cohort)
        champion, reason = self._arbitrate_champion(best_heavy, best_light)

        logger.info(f"DETERMINED CHAMPION: {champion} | Methodology: {reason}")

        return {
            'champion_model': champion,
            'selection_reason': reason,
            'best_heavy': best_heavy,
            'best_light': best_light,
            'cohort_data': model_cohort
        }

    def _build_model_cohort(self, tickers_results: dict[str, Any]) -> list[dict[str, Any]]:
        """Build cohort of successful models from ticker results."""
        model_cohort = []
        for ticker, ticker_data in tickers_results.items():
            if ticker_data.get('status') != 'success':
                continue

            winner = ticker_data.get('winner', 'unknown')
            metrics = ticker_data.get('metrics', {})

            model_type = self._classify_model_type(winner)
            accuracy = self._extract_performance_metric(metrics)

            model_cohort.append({
                'model_name': winner,
                'model_type': model_type,
                'performance_score': accuracy,
                'full_metrics': metrics,
                'source_ticker': ticker
            })

        return model_cohort

    def _classify_model_type(self, winner: str) -> str:
        """Classify model as heavy or light based on configuration."""
        return 'heavy' if str(winner).lower() in [m.lower() for m in self.configured_heavy_models] else 'light'

    def _extract_performance_metric(self, metrics: dict[str, Any]) -> float:
        """Extract primary performance metric from metrics dictionary."""
        return float(metrics.get('accuracy', metrics.get('test_accuracy', metrics.get('r2', 0.0))))

    def _identify_cluster_leaders(self, model_cohort: list[dict[str, Any]]) -> tuple:
        """Identify best performing models in each cluster."""
        heavy_leaders = [m for m in model_cohort if m['model_type'] == 'heavy']
        light_leaders = [m for m in model_cohort if m['model_type'] == 'light']

        best_heavy = max(heavy_leaders, key=lambda x: x['performance_score']) if heavy_leaders else None
        best_light = max(light_leaders, key=lambda x: x['performance_score']) if light_leaders else None

        logger.info(f"Cohort Population: Heavy={len(heavy_leaders)}, Light={len(light_leaders)}")
        if best_heavy:
            logger.info(f"Heavy Cluster Leader: {best_heavy['model_name']} (score: {best_heavy['performance_score']:.4f})")
        if best_light:
            logger.info(f"Light Cluster Leader: {best_light['model_name']} (score: {best_light['performance_score']:.4f})")

        return best_heavy, best_light

    def _arbitrate_champion(self, best_heavy: dict[str, Any] | None, best_light: dict[str, Any] | None) -> tuple[str, str]:
        """Arbitrate final champion from cluster leaders."""
        if best_heavy and best_light:
            if best_heavy['performance_score'] >= best_light['performance_score']:
                champion = best_heavy['model_name']
                reason = f"Structural advantage: Heavy ({best_heavy['performance_score']:.4f}) outpaced Light ({best_light['performance_score']:.4f})"
            else:
                champion = best_light['model_name']
                reason = f"Efficiency advantage: Light ({best_light['performance_score']:.4f}) outpaced Heavy ({best_heavy['performance_score']:.4f})"
        elif best_heavy:
            champion = best_heavy['model_name']
            reason = "Defaulted to heavy cluster leader (no light alternatives)"
        elif best_light:
            champion = best_light['model_name']
            reason = "Defaulted to light cluster leader (no heavy alternatives)"
        else:
            raise DataProcessingError("Arbitration failed: zero model population")

        return champion, reason
