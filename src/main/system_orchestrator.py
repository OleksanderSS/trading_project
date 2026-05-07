"""
Central coordination node for managing the operating modes of the trading system.
Ensures resource initialization and launches the appropriate scenarios (modes).
"""

import asyncio
import inspect
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from src.config.unified_config_manager import UnifiedConfigManager, get_current_config
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager
from src.main.modes.backtest import BacktestMode
from src.main.modes.predict import PredictMode
from src.main.modes.train import TrainMode
from src.main.modes.training_data_pipeline import run_pipeline as run_training_data_pipeline
from src.models.dean.dean_bootstrap_system import get_dean_system
from src.pipeline.hybrid_orchestrator import HybridOrchestrator


@dataclass
class ExecutionConfig:
    """Configuration for mode execution to reduce argument count."""
    mode: str
    tickers: list[str] | None = None
    timeframes: list[str] | None = None
    parallel: bool = False

class SystemOrchestrator:
    """
    A central dispatcher for managing the system's operational modes.
    It initializes resources and runs scenarios like training, prediction, etc.
    """

    def __init__(self, config_manager: UnifiedConfigManager | None = None):
        """
        Initializes the orchestrator with a configuration manager.
        """
        self.config_manager = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self.logger.info("SystemOrchestrator initialized successfully.")

    async def run_mode(self, mode: str, tickers: list[str] | None = None, timeframes: list[str] | None = None, **kwargs) -> dict[str, Any]:
        """
        Runs the system in a selected mode, delegating execution to the corresponding mode class.
        """
        self.logger.info(f"--- Starting execution of mode: '{mode}' ---")

        try:
            parallel = self.config_manager.get_config('execution.parallel_tickers', False)

            config = ExecutionConfig(mode=mode, tickers=tickers, timeframes=timeframes, parallel=parallel)
            return await self._execute_mode(config, **kwargs)

        except Exception as e:
            return self._handle_mode_error(mode, e)
        finally:
            self.logger.info(f"--- Finished execution of mode: '{mode}' ---")

    async def _execute_mode(self, config: ExecutionConfig, **kwargs) -> dict[str, Any]:
        """Execute the specified mode."""
        if config.mode in ['train', 'predict', 'backtest']:
            mode_class = self._get_mode_class(config.mode)
            return await self._dispatch(mode_class, config, **kwargs)

        elif config.mode == 'hybrid':
            return await self._run_hybrid_mode(config.tickers, config.timeframes, **kwargs)

        elif config.mode == 'training_data_pipeline':
            return await self._run_training_data_pipeline()

        elif config.mode in ['web-ui', 'dashboard']:
            return self._run_web_ui()

        elif config.mode == 'intelligent':
            return await self._run_intelligent_mode(config.tickers, config.timeframes, config.parallel, **kwargs)

        elif config.mode == 'monster_test':
            return await self._run_monster_test(config.tickers, config.timeframes, config.parallel, **kwargs)

        else:
            return self._handle_unknown_mode(config.mode)

    def _get_mode_class(self, mode: str) -> Any:
        """Get the mode class for the specified mode."""
        mode_classes = {
            'train': TrainMode,
            'predict': PredictMode,
            'backtest': BacktestMode
        }
        return mode_classes.get(mode)

    async def _run_training_data_pipeline(self) -> dict[str, Any]:
        """Run the training data pipeline."""
        db_manager = DataManager(self.config_manager)
        await run_training_data_pipeline(config_manager=self.config_manager, db_manager=db_manager)
        return {"status": "success", "message": "Training data pipeline completed successfully."}

    def _handle_unknown_mode(self, mode: str) -> dict[str, Any]:
        """Handle unknown mode error."""
        error_msg = f"Unknown operational mode: {mode}"
        self.logger.error(error_msg)
        return {"status": "error", "message": error_msg}

    def _handle_mode_error(self, mode: str, error: Exception) -> dict[str, Any]:
        """Handle mode execution error."""
        self.logger.critical(f"A critical error occurred while executing mode '{mode}': {error}", exc_info=True)
        return {"status": "critical_failure", "error": str(error)}

    async def _dispatch(self, mode_class: Any, config: ExecutionConfig, **kwargs) -> dict[str, Any]:
        """
        Creates and runs an instance of a mode, supporting parallelization across tickers.
        """
        results = {"status": "completed", "tickers_processed": []}

        if self._should_run_parallel(config.tickers, config.parallel):
            return self._run_parallel_execution(mode_class, config, results=results, **kwargs)
        else:
            return await self._run_sequential_execution(mode_class, config, results=results, **kwargs)

        return results

    def _should_run_parallel(self, tickers: list[str] | None, parallel: bool) -> bool:
        """Check if execution should run in parallel."""
        return tickers is not None and parallel and len(tickers) > 1

    def _run_parallel_execution(self, mode_class: Any, config: ExecutionConfig, results: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Run mode execution in parallel across tickers."""
        self.logger.info(f"Running {mode_class.__name__} in parallel for {len(config.tickers or [])} tickers.")
        max_workers = self.config_manager.get_config('execution.max_workers', os.cpu_count())

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_single_instance_sync, mode_class, [ticker], config.timeframes, **kwargs): ticker
                for ticker in (config.tickers or [])
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    future.result()
                    results["tickers_processed"].append(ticker)
                    self.logger.info(f"Ticker {ticker} processed successfully.")
                except Exception as e:
                    self.logger.error(f"Error processing ticker {ticker}: {e}")

        return results

    async def _run_sequential_execution(self, mode_class: Any, config: ExecutionConfig, results: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Run mode execution sequentially."""
        result = await self._run_single_instance(mode_class, config.tickers, config.timeframes, **kwargs)
        if result is not None:
            results["tickers_processed"] = config.tickers if config.tickers else ["all_configured"]

        return results

    def _run_single_instance_sync(self, mode_class: Any, tickers: list[str] | None, timeframes: list[str] | None, **kwargs):
        """Sync helper for parallel execution in ProcessPoolExecutor."""
        instance = mode_class(self.config_manager)
        result = instance.run(tickers=tickers or [], timeframes=timeframes or [], **kwargs)
        if inspect.isawaitable(result):
            return result  # Return awaitable as-is for ProcessPoolExecutor
        return result

    async def _run_single_instance(self, mode_class: Any, tickers: list[str] | None, timeframes: list[str] | None, **kwargs):
        """Initializes and runs a single mode instance in the current event loop."""
        instance = mode_class(self.config_manager)
        result = instance.run(tickers=tickers or [], timeframes=timeframes or [], **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _run_hybrid_mode(self, tickers: list[str] | None, timeframes: list[str] | None, **kwargs) -> dict[str, Any]:
        """Runs the hybrid pipeline via HybridOrchestrator."""
        from src.pipeline.hybrid_orchestrator import HybridPipelineRequest

        self.logger.info("🚀 Running hybrid pipeline mode...")
        batch_name = kwargs.pop('batch_name', 'main_database')
        orchestrator = HybridOrchestrator(self.config_manager, batch_name=batch_name)

        request = HybridPipelineRequest(
            tickers=tickers,
            timeframes=timeframes,
            accumulate=kwargs.pop('accumulate', True),
            force_training=kwargs.pop('force_training', False),
            skip_colab=kwargs.pop('skip_colab', False),
            force_feature_selection=kwargs.pop('force_feature_selection', False)
        )
        return await orchestrator.run_full_hybrid_pipeline(request)

    def _run_web_ui(self) -> dict[str, Any]:
        """Launches the Streamlit Dashboard."""
        self.logger.info("Launching Streamlit Dashboard...")
        dashboard_path = os.path.join("src", "dashboard", "main_app.py")
        try:
            subprocess.run(["streamlit", "run", dashboard_path], check=True)
            return {"status": "success", "message": "Dashboard launched"}
        except Exception as e:
            self.logger.error(f"Failed to launch Web-UI: {e}")
            return {"status": "error", "message": str(e)}

    async def _run_intelligent_mode(self, tickers: list[str] | None, timeframes: list[str] | None, parallel: bool, **kwargs) -> dict[str, Any]:
        """Runs INTELLIGENT mode with self-diagnosis."""
        self.logger.info("🧠 Running INTELLIGENT mode...")
        dean_brain = get_dean_system()
        mode_type: type[PredictMode] | type[TrainMode] = PredictMode

        try:
            if hasattr(dean_brain, 'experience_diary') and dean_brain.experience_diary.needs_retraining(tickers or []):
                mode_type = TrainMode
                self.logger.info("🧠 ExperienceDiary recommends retraining models.")
        except Exception as e:
            self.logger.warning(f"Could not get advice from ExperienceDiary: {e}")

        config = ExecutionConfig(mode="intelligent", tickers=tickers, timeframes=timeframes, parallel=parallel)
        return await self._dispatch(mode_type, config, brain=dean_brain, **kwargs)

    async def _run_monster_test(self, tickers: list[str] | None, timeframes: list[str] | None, parallel: bool, **kwargs) -> dict[str, Any]:
        """
        Runs the stress test (Monster Test).
        """
        self.logger.info("👹 Running MONSTER TEST...")
        test_tickers = tickers or self.config_manager.get_config('monster_test.tickers', ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"])
        config = ExecutionConfig(mode="monster_test", tickers=test_tickers, timeframes=timeframes, parallel=parallel)
        return await self._dispatch(TrainMode, config, **kwargs)
