"""
Central coordination node for managing the operating modes of the trading system.
Ensures resource initialization and launches the appropriate scenarios (modes).
"""

import os
import asyncio
import inspect
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Dict, Any

from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import UnifiedConfigManager, get_current_config
from src.data.management.data_manager import DataManager # <<< IMPORT
from src.pipeline.hybrid_orchestrator import HybridOrchestrator
from src.main.modes.train import TrainMode
from src.main.modes.backtest import BacktestMode
from src.main.modes.predict import PredictMode
from src.main.modes.training_data_pipeline import run_pipeline as run_training_data_pipeline
from src.models.dean.dean_bootstrap_system import get_dean_system

class SystemOrchestrator:
    """
    A central dispatcher for managing the system's operational modes.
    It initializes resources and runs scenarios like training, prediction, etc.
    """

    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None):
        """
        Initializes the orchestrator with a configuration manager.
        """
        self.config_manager = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self.logger.info("SystemOrchestrator initialized successfully.")

    async def run_mode(self, mode: str, tickers: Optional[List[str]] = None, timeframes: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Runs the system in a selected mode, delegating execution to the corresponding mode class.
        """
        self.logger.info(f"--- Starting execution of mode: '{mode}' ---")
        
        try:
            parallel = self.config_manager.get_config('execution.parallel_tickers', False)
            
            if mode == 'train':
                return self._dispatch(TrainMode, tickers, timeframes, parallel, **kwargs)
            
            elif mode == 'predict':
                return self._dispatch(PredictMode, tickers, timeframes, parallel, **kwargs)
            
            elif mode == 'backtest':
                return self._dispatch(BacktestMode, tickers, timeframes, parallel, **kwargs)

            elif mode == 'hybrid':
                return await self._run_hybrid_mode(tickers, timeframes, **kwargs)

            elif mode == 'training_data_pipeline':
                # FIX: Instantiate DataManager and pass it to the pipeline
                db_manager = DataManager(self.config_manager)
                await run_training_data_pipeline(config_manager=self.config_manager, db_manager=db_manager)
                return {"status": "success", "message": "Training data pipeline completed successfully."}

            elif mode in ['web-ui', 'dashboard']:
                return self._run_web_ui()
            
            elif mode == 'intelligent':
                return self._run_intelligent_mode(tickers, timeframes, parallel, **kwargs)
            
            elif mode == 'monster_test':
                return self._run_monster_test(tickers, timeframes, parallel, **kwargs)
            
            else:
                error_msg = f"Unknown operational mode: {mode}"
                self.logger.error(error_msg)
                return {"status": "error", "message": error_msg}

        except Exception as e:
            self.logger.critical(f"A critical error occurred while executing mode '{mode}': {e}", exc_info=True)
            return {"status": "critical_failure", "error": str(e)}
        finally:
            self.logger.info(f"--- Finished execution of mode: '{mode}' ---")

    def _dispatch(self, mode_class: Any, tickers: Optional[List[str]], timeframes: Optional[List[str]], parallel: bool, **kwargs) -> Dict[str, Any]:
        """
        Creates and runs an instance of a mode, supporting parallelization across tickers.
        """
        results = {"status": "completed", "tickers_processed": []}
        
        if tickers and parallel and len(tickers) > 1:
            self.logger.info(f"Running {mode_class.__name__} in parallel for {len(tickers)} tickers.")
            max_workers = self.config_manager.get_config('execution.max_workers', os.cpu_count())
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._run_single_instance_sync, mode_class, [ticker], timeframes, **kwargs): ticker 
                    for ticker in tickers
                }
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        future.result()
                        results["tickers_processed"].append(ticker)
                        self.logger.info(f"Ticker {ticker} processed successfully.")
                    except Exception as e:
                        self.logger.error(f"Error processing ticker {ticker}: {e}")
        else:
            # Run within the current event loop if the mode returns a coroutine.
            result = asyncio.run(self._run_single_instance(mode_class, tickers, timeframes, **kwargs))
            if result is not None:
                results["tickers_processed"] = tickers if tickers else ["all_configured"]
        
        return results

    def _run_single_instance_sync(self, mode_class: Any, tickers: Optional[List[str]], timeframes: Optional[List[str]], **kwargs):
        """Sync helper for parallel execution in ProcessPoolExecutor."""
        instance = mode_class(self.config_manager)
        result = instance.run(tickers=tickers, timeframes=timeframes, **kwargs)
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result

    async def _run_single_instance(self, mode_class: Any, tickers: Optional[List[str]], timeframes: Optional[List[str]], **kwargs):
        """Initializes and runs a single mode instance in the current event loop."""
        instance = mode_class(self.config_manager)
        result = instance.run(tickers=tickers, timeframes=timeframes, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _run_hybrid_mode(self, tickers: Optional[List[str]], timeframes: Optional[List[str]], **kwargs) -> Dict[str, Any]:
        """Runs the hybrid pipeline via HybridOrchestrator."""
        self.logger.info("🚀 Running hybrid pipeline mode...")
        batch_name = kwargs.pop('batch_name', 'main_database')
        orchestrator = HybridOrchestrator(self.config_manager, batch_name=batch_name)
        return await orchestrator.run_full_hybrid_pipeline(
            tickers=tickers,
            timeframes=timeframes,
            run_colab=kwargs.pop('run_colab', False),
            accumulate=kwargs.pop('accumulate', True),
            force_training=kwargs.pop('force_training', False),
            skip_colab=kwargs.pop('skip_colab', False),
            force_feature_selection=kwargs.pop('force_feature_selection', False)
        )

    def _run_web_ui(self) -> Dict[str, Any]:
        """Launches the Streamlit Dashboard."""
        self.logger.info("Launching Streamlit Dashboard...")
        dashboard_path = os.path.join("src", "dashboard", "main_app.py")
        try:
            subprocess.run(["streamlit", "run", dashboard_path], check=True)
            return {"status": "success", "message": "Dashboard launched"}
        except Exception as e:
            self.logger.error(f"Failed to launch Web-UI: {e}")
            return {"status": "error", "message": str(e)}

    def _run_intelligent_mode(self, tickers: Optional[List[str]], timeframes: Optional[List[str]], parallel: bool, **kwargs) -> Dict[str, Any]:
        """Runs INTELLIGENT mode with self-diagnosis."""
        self.logger.info("🧠 Running INTELLIGENT mode...")
        dean_brain = get_dean_system()
        mode_type = PredictMode
        
        try:
            if hasattr(dean_brain, 'experience_diary') and dean_brain.experience_diary.needs_retraining(tickers):
                mode_type = TrainMode
                self.logger.info("🧠 ExperienceDiary recommends retraining models.")
        except Exception as e:
            self.logger.warning(f"Could not get advice from ExperienceDiary: {e}")

        return self._dispatch(mode_type, tickers, timeframes, parallel, brain=dean_brain, **kwargs)

    def _run_monster_test(self, tickers: Optional[List[str]], timeframes: Optional[List[str]], parallel: bool, **kwargs) -> Dict[str, Any]:
        """
        Runs the stress test (Monster Test).
        """
        self.logger.info("👹 Running MONSTER TEST...")
        test_tickers = tickers or self.config_manager.get_config('monster_test.tickers', ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"])
        return self._dispatch(TrainMode, test_tickers, timeframes, parallel, **kwargs)
