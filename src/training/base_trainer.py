"""
BaseTrainer: Abstract base class for training orchestration

This module provides the template method pattern for training execution,
eliminating duplication between BatchTrainer and ProgressiveTrainer.
All trainer implementations share common workflow:
1. Prepare ticker groups (different strategies for batch vs progressive)
2. Train each group (parallel vs sequential)
3. Generate results summary
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from datetime import datetime

from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import UnifiedConfigManager


class TrainingException(Exception):
    """Base exception for training errors"""
    pass


class TrainingConfigException(TrainingException):
    """Exception for training configuration issues"""
    pass


class TrainerConfig:
    """Base configuration for all trainers"""
    def __init__(self, batch_size: int = 10, max_memory_gb: float = 12.0):
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb


class BaseTrainer(ABC):
    """
    Abstract base class for training orchestration.
    
    Implements template method pattern for common training workflow:
    - Prepare ticker groups
    - Train each group
    - Aggregate results and generate summary
    
    Subclasses must implement:
    - _prepare_ticker_groups(): Define how to group tickers
    - _train_ticker_group(): Define how to train a group
    """
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        """
        Initialize BaseTrainer.
        
        Args:
            config: TrainerConfig instance with batch_size and max_memory_gb
        """
        self.config = config or TrainerConfig()
        self.config_manager = UnifiedConfigManager()
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        
        # Initialize output directory
        try:
            models_path = self.config_manager.get_models_path()
            self.output_dir = Path(models_path)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory: {self.output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize output directory: {e}")
            raise TrainingConfigException(f"Cannot initialize output directory: {e}")
    
    def execute_training(self, plan: Dict[str, Any], data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute complete training workflow (Template Method).
        
        This is the common orchestration logic shared by all trainers.
        Subclasses override _prepare_ticker_groups() and _train_ticker_group()
        to define their specific behavior.
        
        Args:
            plan: Training plan with tickers, strategy, etc.
            data_context: Prepared data for training
        
        Returns:
            Dictionary with status, results, and summary
        """
        if not plan or not isinstance(plan, dict):
            raise TrainingException("Invalid training plan")
        
        tickers = plan.get('tickers', [])
        if not tickers:
            self.logger.warning("No tickers provided in training plan")
            return {"status": "failed", "reason": "no_tickers"}
        
        try:
            self.logger.info(
                f"Starting {self.__class__.__name__} training for {len(tickers)} tickers. "
                f"Strategy: {plan.get('strategy', 'unknown')}"
            )
            
            # Step 1: Prepare ticker groups (batch vs progressive logic)
            ticker_groups = self._prepare_ticker_groups(plan)
            self.logger.debug(f"Created {len(ticker_groups)} ticker groups")
            
            # Step 2: Train each group
            results = {}
            for group_idx, ticker_group in enumerate(ticker_groups, 1):
                self.logger.info(f"Training group {group_idx}/{len(ticker_groups)} ({len(ticker_group)} tickers)")
                group_results = self._train_ticker_group(ticker_group, data_context)
                results.update(group_results)
            
            # Step 3: Generate summary
            summary = self._generate_summary(results)
            
            self.logger.info(
                f"✅ Training complete. Success rate: {summary['success_rate']:.1%} "
                f"({summary['successful']}/{summary['total_tickers']})"
            )
            
            return {
                "status": "success",
                "tickers_results": results,
                "training_summary": summary
            }
        
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "reason": str(e),
                "tickers_results": {},
                "training_summary": {}
            }
    
    @abstractmethod
    def _prepare_ticker_groups(self, plan: Dict[str, Any]) -> List[List[str]]:
        """
        Prepare ticker groups for training.
        
        Subclasses implement their specific grouping strategy:
        - BatchTrainer: All tickers in one group
        - ProgressiveTrainer: Adaptive batches with growth factor
        
        Args:
            plan: Training plan containing tickers
        
        Returns:
            List of ticker groups: [[ticker1, ticker2], [ticker3, ...], ...]
        """
        pass
    
    @abstractmethod
    def _train_ticker_group(self, ticker_group: List[str], data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a group of tickers.
        
        Subclasses implement their specific training strategy:
        - BatchTrainer: Parallel training using Parallel/delayed
        - ProgressiveTrainer: Sequential training with adaptation
        
        Args:
            ticker_group: List of tickers to train
            data_context: Prepared data for training
        
        Returns:
            Dictionary: {ticker: training_result, ...}
        """
        pass
    
    def _train_ticker_suite(self, ticker: str, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a single ticker with all configured models.
        
        This is the core training logic that's common to all trainers.
        Should be called by subclasses from _train_ticker_group().
        
        Args:
            ticker: Ticker symbol to train
            data_context: Prepared data for this ticker
        
        Returns:
            Training result for this ticker
        """
        # This method is typically overridden or implemented in subclasses
        # to provide specific model training logic
        raise NotImplementedError("Subclasses must implement _train_ticker_suite or override _train_ticker_group")
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate training summary statistics.
        
        Common summary generation logic shared by all trainers.
        
        Args:
            results: Dictionary of training results by ticker
        
        Returns:
            Summary dictionary with statistics
        """
        total_tickers = len(results)
        successful_tickers = sum(1 for r in results.values() if r.get('status') == 'success')
        failed_tickers = total_tickers - successful_tickers
        
        # Calculate average score if available
        scores = []
        for result in results.values():
            if 'best_score' in result:
                scores.append(result['best_score'])
        
        avg_score = np.mean(scores) if scores else None
        
        return {
            "total_tickers": total_tickers,
            "successful_tickers": successful_tickers,
            "failed_tickers": failed_tickers,
            "success_rate": successful_tickers / total_tickers if total_tickers > 0 else 0,
            "average_score": float(avg_score) if avg_score is not None else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def _validate_data_context(self, data_context: Dict[str, Any]) -> bool:
        """
        Validate that data_context has required fields.
        
        Args:
            data_context: Data context to validate
        
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['X_train', 'y_train', 'X_test', 'y_test', 'target_name']
        for key in required_keys:
            if key not in data_context or data_context[key] is None:
                self.logger.warning(f"Data context missing required key: {key}")
                return False
        return True
