import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ModelAnalyzer")

class ModelAnalyzer:
    """
    Analyzes model training results from the ML Arena (Stage 4).
    Provides comparisons between different architectures and architecture types (Light vs Heavy).
    """

    def __init__(self, training_results: Dict[str, Any]):
        """
        Initializes the analyzer with a dictionary of training results.
        
        Args:
            training_results: Dictionary where keys are ticker_target identifiers 
                             and values contain lists of candidate model metrics.
        """
        self.results = training_results
        self.summary: Dict[str, Any] = {}
        self.report_dir = Path("reports/models")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_training_summary(self) -> Dict[str, Any]:
        """
        Aggregates results to identify champions and compare performance metrics.
        """
        logger.info("Generating training summary report...")
        
        total_models = 0
        champions = []
        light_metrics: List[float] = []
        heavy_metrics: List[float] = []
        
        for key, data in self.results.items():
            candidates = data.get("candidates", [])
            total_models += len(candidates)
            
            if not candidates:
                continue

            # Determine if classification or regression for metric selection
            is_clf = "direction" in key.lower()
            main_metric = "f1" if is_clf else "rmse"
            is_higher_better = is_clf # F1 higher is better, RMSE lower is better

            # Sort to find the ticker champion
            sorted_cand = sorted(
                candidates, 
                key=lambda x: x['metrics'].get(main_metric, 0 if is_higher_better else float('inf')), 
                reverse=is_higher_better
            )
            
            champion = sorted_cand[0]
            champions.append({
                "id": key,
                "model_name": champion["name"],
                "archetype": champion.get("archetype", "unknown"),
                "metric": main_metric,
                "value": champion["metrics"].get(main_metric)
            })

            # Performance comparison: Light vs Heavy
            for cand in candidates:
                m_val = cand['metrics'].get(main_metric)
                if m_val is None: continue
                
                if cand.get("type") == "light":
                    light_metrics.append(m_val)
                elif cand.get("type") == "heavy":
                    heavy_metrics.append(m_val)

        avg_light = sum(light_metrics) / len(light_metrics) if light_metrics else 0
        avg_heavy = sum(heavy_metrics) / len(heavy_metrics) if heavy_metrics else 0

        self.summary = {
            "timestamp": datetime.now().isoformat(),
            "total_ticker_targets": len(self.results),
            "total_models_trained": total_models,
            "champions": champions,
            "architecture_comparison": {
                "avg_light_performance": round(avg_light, 4),
                "avg_heavy_performance": round(avg_heavy, 4),
                "heavy_improvement_pct": round(((avg_heavy - avg_light) / (avg_light + 1e-6)) * 100, 2) if avg_light > 0 else 0
            },
            "best_overall_model": champions[0] if champions else None
        }
        
        logger.info(f"Summary generated. Heavy models are {self.summary['architecture_comparison']['heavy_improvement_pct']}% better than Light.")
        return self.summary

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
        except Exception as e:
            logger.error(f"Failed to save training report: {e}")
            return ""

    def get_light_vs_heavy_stats(self) -> Dict[str, float]:
        """
        Quick helper to get comparison statistics.
        """
        if not self.summary:
            self.generate_training_summary()
        return self.summary.get("architecture_comparison", {})