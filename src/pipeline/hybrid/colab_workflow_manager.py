# audit-ignore: ARCHITECTURAL_USAGE
# src/pipeline/hybrid/colab_workflow_manager.py
"""
Colab Workflow Manager for Hybrid Orchestrator.

Manages Colab-specific workflow: instructions, fallback features, execution paths.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger


class ColabWorkflowManager:
    """
    Manages Colab workflow operations.

    Handles Colab preparation, fallback features, and user instructions.
    """

    def __init__(self, output_dir: Path, batch_name: str, light_models: list[str]):
        self.output_dir = output_dir
        self.batch_name = batch_name
        self.light_models = light_models
        self.logger = ProjectLogger.get_logger(__name__)

    async def _handle_skip_colab_path(self, b_info: dict[str, Any], n_f: pd.DataFrame,
                                     tickers: list[str] | None,
                                     timeframes: list[str] | None,
                                     final_stages_runner) -> dict[str, Any]:
        """Handle skip Colab path."""
        self._create_fallback_selected_features(b_info, n_f)
        final_results = await final_stages_runner(None, None, None, None, tickers, timeframes, self.batch_name)
        return {'status': 'completed_without_colab', 'final_results': final_results}

    def _handle_colab_path(self, b_info: dict[str, Any]) -> dict[str, Any]:
        """Handle Colab training path."""
        instr = self._generate_colab_instructions(b_info)
        self.logger.info(f"🚨 PAUSED: Colab training required.\n{instr}")
        return {'status': 'paused_for_colab', 'colab_batch': b_info, 'colab_instructions': instr}

    def _generate_colab_instructions(self, batch_info: dict[str, Any]) -> str:
        """Generates instructions for running in Colab."""
        name = batch_info['batch_name']
        return f"""
COLAB INSTRUCTIONS:
1. Transfer the batch folder '{name}' to your Google Drive.
2. Run the Colab notebook and mount your drive.
3. Perform feature selection and heavy model training.
4. Once finished, run: python run_hybrid_pipeline.py --mode continue --batch-name {name}
"""

    def _create_fallback_selected_features(self, batch_info: dict[str, Any], features_df: pd.DataFrame) -> None:
        """Creates fallback feature selection files (all features)."""
        b_dir = Path(batch_info['batch_dir'])
        b_dir.mkdir(parents=True, exist_ok=True)
        feats = [c for c in features_df.columns if not c.startswith('target_')]
        for m in self.light_models:
            with open(b_dir / f"selected_features_{m}.json", 'w', encoding='utf-8') as f:
                json.dump({'model_name': m, 'selected_features': feats, 'selection_method': 'fallback'}, f, indent=2, default=str)
