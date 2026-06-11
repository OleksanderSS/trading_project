import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ModelAnalyzer")

class ModelAnalyzer:
    """
    Analyzes model training results from the ML Arena (Stage 4).
    Provides comparisons between different architectures and architecture types (Light vs Heavy).
    """

    def __init__(self, training_results: dict[str, Any]):
        """
        Initializes the analyzer with a dictionary of training results.

        Args:
            training_results: Dictionary where keys are ticker_target identifiers
                             and values contain lists of candidate model metrics.
        """
        self.results = training_results
        self.summary: dict[str, Any] = {}
        self.report_dir = Path("reports/models")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_training_summary(self) -> dict[str, Any]:
        """
        Aggregates results to identify champions and compare performance metrics.
        """
        logger.info("Generating training summary report...")

        total_models = 0
        champions = []
        light_metrics: list[float] = []
        heavy_metrics: list[float] = []

        for key, data in self.results.items():
            candidates = data.get("candidates", [])
            total_models += len(candidates)

            if not candidates:
                continue

            # Determine metric type
            metric_info = self._determine_metric_type(key)

            # Find champion and collect metrics
            champion = self._find_champion(candidates, metric_info)
            champions.append(self._create_champion_info(key, champion, metric_info))

            # Collect architecture metrics
            self._collect_architecture_metrics(candidates, metric_info, light_metrics, heavy_metrics)

        # Calculate summary statistics
        architecture_comparison = self._calculate_architecture_comparison(light_metrics, heavy_metrics)

        self.summary = {
            "timestamp": datetime.now().isoformat(),
            "total_ticker_targets": len(self.results),
            "total_models_trained": total_models,
            "champions": champions,
            "architecture_comparison": architecture_comparison,
            "best_overall_model": champions[0] if champions else None
        }

        logger.info(f"Summary generated. Heavy models are {architecture_comparison['heavy_improvement_pct']}% better than Light.")
        return self.summary

    def _determine_metric_type(self, key: str) -> dict[str, Any]:
        """Determine if classification or regression and select appropriate metric"""
        is_clf = "direction" in key.lower()
        main_metric = "f1" if is_clf else "rmse"
        is_higher_better = is_clf

        return {
            "is_classification": is_clf,
            "main_metric": main_metric,
            "is_higher_better": is_higher_better
        }

    def _find_champion(self, candidates: list[dict], metric_info: dict[str, Any]) -> dict:
        """Find the champion model from candidates"""
        main_metric = metric_info["main_metric"]
        is_higher_better = metric_info["is_higher_better"]

        def sort_key(candidate, metric=main_metric, higher_better=is_higher_better):
            default_value = 0 if higher_better else float('inf')
            return candidate['metrics'].get(metric, default_value)

        sorted_candidates = sorted(candidates, key=sort_key, reverse=is_higher_better)
        return sorted_candidates[0]

    def _create_champion_info(self, key: str, champion: dict, metric_info: dict[str, Any]) -> dict[str, Any]:
        """Create champion information dictionary"""
        main_metric = metric_info["main_metric"]

        return {
            "id": key,
            "model_name": champion["name"],
            "archetype": champion.get("archetype", "unknown"),
            "metric": main_metric,
            "value": champion["metrics"].get(main_metric)
        }

    def _collect_architecture_metrics(self, candidates: list[dict], metric_info: dict[str, Any],
                                    light_metrics: list[float], heavy_metrics: list[float]) -> None:
        """Collect metrics for light vs heavy architecture comparison"""
        main_metric = metric_info["main_metric"]

        for candidate in candidates:
            metric_value = candidate['metrics'].get(main_metric)
            if metric_value is None:
                continue

            arch_type = candidate.get("type")
            if arch_type == "light":
                light_metrics.append(metric_value)
            elif arch_type == "heavy":
                heavy_metrics.append(metric_value)

    def _calculate_architecture_comparison(self, light_metrics: list[float], heavy_metrics: list[float]) -> dict[str, float]:
        """Calculate architecture performance comparison"""
        from src.utils.math_safe import safe_div
        avg_light = safe_div(sum(light_metrics), len(light_metrics))
        avg_heavy = safe_div(sum(heavy_metrics), len(heavy_metrics))

        heavy_improvement_pct = 0.0
        if avg_light > 0:
            heavy_improvement_pct = float(round(((avg_heavy - avg_light) / (avg_light + 1e-6)) * 100, 2))

        return {
            "avg_light_performance": round(avg_light, 4),
            "avg_heavy_performance": round(avg_heavy, 4),
            "heavy_improvement_pct": heavy_improvement_pct
        }

    def save_report(self, report_name: str = "training_summary") -> str:
        """
        Saves the generated summary to a JSON file.
        """
        if not self.summary:
            self.generate_training_summary()

        file_path = self.report_dir / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.summary, f, indent=4, ensure_ascii=False)
            logger.info(f"Training report saved successfully to {file_path}")
            return str(file_path)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Failed to save training report: {e}")
            return ""

    def get_light_vs_heavy_stats(self) -> dict[str, float]:
        """
        Quick helper to get comparison statistics.
        """
        if not self.summary:
            self.generate_training_summary()
        return dict(self.summary.get("architecture_comparison", {}))
