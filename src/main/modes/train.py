#!/usr/bin/env python3
"""
Train mode - training models based on the new PipelineOrchestrator architecture.
"""

import asyncio
import inspect
from typing import Any

from src.core.logging.logger import ProjectLogger
from src.main.modes.base import BaseMode
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator


class TrainMode(BaseMode):
    """Mode for running the model training pipeline."""

    def __init__(self, config_manager, brain: Any | None = None):
        super().__init__(config_manager)
        self.brain = brain
        self.logger = ProjectLogger.get_logger(__name__)

    def run(self, tickers: list[str] | None = None, timeframes: list[str] | None = None, **kwargs) -> dict[str, Any]:
        """
        Launches the pipeline responsible for training, validating, and saving models.
        """
        self.logger.info("--- Starting Model Training Mode ---")
        try:
            # 1. INITIALIZE THE ORCHESTRATOR
            orchestrator = PipelineOrchestrator(self.config_manager, brain=self.brain)

            # 2. RUN THE PIPELINE
            self.logger.info("Running the training pipeline...")

            initial_data = {
                "tickers": tickers,
                "timeframes": timeframes,
                **kwargs
            }

            final_results = orchestrator.run(**initial_data)
            if inspect.isawaitable(final_results):
                final_results = asyncio.run(final_results)

            if not final_results:
                raise RuntimeError("Training pipeline did not return any results.")

            # Here you could add analysis of the results if needed
            self.logger.info("--- Model Training Completed Successfully ---")

            return {
                'status': 'success',
                'summary': "Training finished.",
                'results': final_results
            }

        except Exception as e:
            self.logger.exception(f"A critical error occurred during the training process: {e}")
            return {'status': 'failed', 'error': str(e)}

