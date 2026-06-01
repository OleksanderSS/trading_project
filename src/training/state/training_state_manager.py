#!/usr/bin/env python3
"""
Training State Manager - Training State Persistence
Handles training state management and persistence.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("TrainingStateManager")


@dataclass
class TrainingState:
    """Persistence object for tracking training progress and lifecycle."""
    processed_tickers: set[str] = field(default_factory=set)
    successful_tickers: set[str] = field(default_factory=set)
    failed_tickers: set[str] = field(default_factory=set)
    current_batch_size: int = 5
    total_batches_processed: int = 0
    start_time: float = field(default_factory=time.time)
    last_checkpoint: float = field(default_factory=time.time)


class TrainingStateManager:
    """
    Training state manager.
    
    Handles:
    - State initialization
    - State updates
    - Checkpoint saving
    - Checkpoint loading
    """
    
    def __init__(self, checkpoints_dir: Path):
        """
        Initialize Training State Manager.
        
        Args:
            checkpoints_dir: Directory for storing checkpoints
        """
        self.logger = logger
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.state = TrainingState()
        self.logger.info("✅ TrainingStateManager initialized")
    
    def initialize_state(self, initial_batch_size: int = 5) -> None:
        """Initialize training state."""
        self.state = TrainingState(current_batch_size=initial_batch_size)
        self.logger.info("Training state initialized")
    
    def update_processed_tickers(self, tickers: list[str], status: str) -> None:
        """Update processed tickers based on status."""
        self.state.processed_tickers.update(tickers)
        
        if status == "completed":
            self.state.successful_tickers.update(tickers)
        else:
            self.state.failed_tickers.update(tickers)
        
        self.state.total_batches_processed += 1
        self.state.last_checkpoint = time.time()
    
    def increment_batch_count(self) -> None:
        """Increment batch count."""
        self.state.total_batches_processed += 1
        self.state.last_checkpoint = time.time()
    
    def adjust_batch_size(self, new_size: int) -> None:
        """Adjust current batch size."""
        self.state.current_batch_size = new_size
        self.logger.info(f"Batch size adjusted to {new_size}")
    
    def should_skip_ticker(self, ticker: str) -> bool:
        """Determine if ticker should be skipped."""
        if ticker in self.state.processed_tickers:
            return True
        return False
    
    def save_checkpoint(self, batch_id: int, analytics: dict[str, list[Any]]) -> None:
        """Save training state checkpoint."""
        checkpoint = {
            "batch_id": batch_id,
            "state": {
                "processed_tickers": list(self.state.processed_tickers),
                "successful_tickers": list(self.state.successful_tickers),
                "failed_tickers": list(self.state.failed_tickers),
                "current_batch_size": self.state.current_batch_size,
                "total_batches_processed": self.state.total_batches_processed
            },
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        }
        
        filepath = self.checkpoints_dir / f"checkpoint_batch_{batch_id}.json"
        try:
            with open(filepath, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            self.logger.debug(f"Checkpoint saved: {filepath.name}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_file: str, path_validator) -> bool:
        """Load training state from checkpoint."""
        try:
            # Secure path validation
            checkpoint_path = path_validator.sanitize_path_input(
                checkpoint_file, 
                base_dir=str(self.checkpoints_dir)
            )
            
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
            
            state_data = checkpoint["state"]
            self.state.processed_tickers = set(state_data["processed_tickers"])
            self.state.successful_tickers = set(state_data["successful_tickers"])
            self.state.failed_tickers = set(state_data["failed_tickers"])
            self.state.current_batch_size = state_data["current_batch_size"]
            self.state.total_batches_processed = state_data["total_batches_processed"]
            
            self.logger.info(f"Recovered from checkpoint: {checkpoint_file}")
            return True
            
        except ValueError as e:
            self.logger.error(f"Security validation failed for {checkpoint_file}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"State recovery failed for {checkpoint_file}: {e}")
            return False
    
    def get_state_summary(self) -> dict[str, Any]:
        """Get state summary."""
        return {
            "processed_tickers": list(self.state.processed_tickers),
            "successful_tickers": list(self.state.successful_tickers),
            "failed_tickers": list(self.state.failed_tickers),
            "current_batch_size": self.state.current_batch_size,
            "total_batches_processed": self.state.total_batches_processed,
            "elapsed_time": time.time() - self.state.start_time
        }
