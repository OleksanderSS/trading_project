#!/usr/bin/env python3
"""
Predict mode - inference mode for generating real-time signals.
Uses the PipelineOrchestrator to run only the prediction and signal generation stages.
"""

import asyncio
import inspect
from typing import Any

from src.config.unified_config_manager import UnifiedConfigManager
from src.main.modes.base import BaseMode
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator


class PredictMode(BaseMode):
    """
    PredictMode handles the inference lifecycle.
    It orchestrates the pipeline to process data and generate trading signals
    based on previously trained models.
    """

    def __init__(self, config_manager: UnifiedConfigManager):
        """
        Initializes the PredictMode with the global configuration.

        Args:
            config_manager: Instance of UnifiedConfigManager for settings.
        """
        super().__init__(config_manager)
        self.logger.info("PredictMode initialized for signal generation.")

    def run(self, tickers: list[str] | None = None, timeframes: list[str] | None = None, **kwargs) -> dict[str, Any]:
        """
        Runs the inference pipeline to generate real-time signals.

        Args:
            tickers: Optional list of tickers to predict for.
            timeframes: Optional list of timeframes to analyze.
            **kwargs: Additional parameters passed to the pipeline.

        Returns:
            dict[str, Any]: The results of the prediction pipeline including signals.
        """
        self.logger.info(f"Starting Prediction Mode (Inference) for tickers: {tickers or 'Default'}")

        try:
            # 1. Initialize the Pipeline Orchestrator
            # Note: The brain can be passed via kwargs if 'intelligent' mode is used
            brain = kwargs.get('brain')
            pipeline = PipelineOrchestrator(self.config_manager, brain=brain)

            # 2. Run the pipeline in 'predict' mode
            # This typically executes Stage 0 (Setup), Stage 1 (Collection), Stage 2 (Processing),
            # Stage 3 (Features), and then jumps to Stage 5 (Prediction) and Stage 6 (Signals).
            results = pipeline.run(
                tickers=tickers,
                timeframes=timeframes,
                run_mode='predict'
            )
            if inspect.isawaitable(results):
                # Check if there's already a running event loop
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in an async context, we can't use asyncio.run()
                    # Return the coroutine to be awaited by the caller
                    return {
                        'status': 'async_required',
                        'coroutine': results,
                        'message': 'Mode requires async execution - coroutine returned for awaiting'
                    }
                except RuntimeError:
                    # No running loop, safe to use asyncio.run()
                    results = asyncio.run(results)

            if not results:
                self.logger.warning("Pipeline completed but returned no results.")
                return {'status': 'empty_results', 'mode': 'predict'}

            self.logger.info("Inference pipeline completed successfully.")
            return {
                'status': 'success',
                'mode': 'predict',
                'results': results
            }

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Critical error during prediction execution: {e}", exc_info=True)
            return {
                'status': 'failed',
                'mode': 'predict',
                'error': str(e)
            }
