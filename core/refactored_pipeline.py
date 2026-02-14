#!/usr/bin/env python3
"""
Refactored Pipeline - Clean Architecture Implementation

This module provides a clean, modular implementation of trading pipeline
with proper separation of concerns and error handling.
"""

import logging
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from utils.logger_fixed import ProjectLogger
from config.config import TICKERS, TIME_FRAMES
from core.pipeline_final import FinalPipeline

logger = ProjectLogger.get_logger("RefactoredPipeline")

class RefactoredPipeline:
    """
    Clean implementation of trading pipeline with modular architecture.
    This class serves as a wrapper around FinalPipeline with additional
    error handling and monitoring capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize refactored pipeline.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.pipeline = FinalPipeline()
        self.logger = ProjectLogger.get_logger("RefactoredPipeline")
        
        # Load configuration if provided
        self.config = {}
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        logger.info("RefactoredPipeline initialized successfully")
    
    def run_complete_pipeline(self, 
                            target_config: Optional[Dict] = None,
                            model_config: Optional[Dict] = None,
                            force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run complete trading pipeline with enhanced error handling.
        
        Args:
            target_config: Target configuration for feature selection
            model_config: Model configuration for training
            force_refresh: Force refresh all caches
            
        Returns:
            Dictionary with pipeline results and metadata
        """
        logger.info("Starting RefactoredPipeline complete run...")
        start_time = datetime.now()
        
        try:
            # Merge configurations
            final_target_config = self._merge_configs(
                self.config.get("targets", {}), 
                target_config or {}
            )
            final_model_config = self._merge_configs(
                self.config.get("models", {}), 
                model_config or {}
            )
            
            # Run pipeline
            results = self.pipeline.run_complete_pipeline(
                target_config=final_target_config,
                model_config=final_model_config,
                force_refresh=force_refresh
            )
            
            # Add refactored pipeline metadata
            results["refactored_metadata"] = {
                "pipeline_version": "refactored_v1",
                "config_used": bool(self.config),
                "config_path": self.config_path,
                "start_time": start_time,
                "end_time": datetime.now(),
                "status": "success"
            }
            
            logger.info("RefactoredPipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"RefactoredPipeline failed: {e}")
            return {
                "refactored_metadata": {
                    "pipeline_version": "refactored_v1",
                    "config_used": bool(self.config),
                    "config_path": self.config_path,
                    "start_time": start_time,
                    "end_time": datetime.now(),
                    "status": "failed",
                    "error": str(e)
                },
                "error": str(e)
            }
    
    def run_stage_only(self, stage: int, **kwargs) -> Dict[str, Any]:
        """
        Run a specific stage with enhanced error handling.
        
        Args:
            stage: Stage number to run (1-5)
            **kwargs: Additional parameters for stage
            
        Returns:
            Dictionary with stage results
        """
        logger.info(f"Running RefactoredPipeline stage {stage}...")
        
        try:
            # Merge configurations
            if "target_config" in kwargs:
                kwargs["target_config"] = self._merge_configs(
                    self.config.get("targets", {}), 
                    kwargs["target_config"]
                )
            if "model_config" in kwargs:
                kwargs["model_config"] = self._merge_configs(
                    self.config.get("models", {}), 
                    kwargs["model_config"]
                )
            
            result = self.pipeline.run_stage_only(stage, **kwargs)
            result["refactored_metadata"] = {
                "stage": stage,
                "pipeline_version": "refactored_v1",
                "timestamp": datetime.now(),
                "status": "success"
            }
            
            logger.info(f"Stage {stage} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Stage {stage} failed: {e}")
            return {
                "refactored_metadata": {
                    "stage": stage,
                    "pipeline_version": "refactored_v1",
                    "timestamp": datetime.now(),
                    "status": "failed",
                    "error": str(e)
                },
                "error": str(e)
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline status.
        
        Returns:
            Dictionary with pipeline status information
        """
        try:
            base_status = self.pipeline.get_pipeline_status()
            
            # Add refactored pipeline specific info
            base_status["refactored_info"] = {
                "pipeline_version": "refactored_v1",
                "config_loaded": bool(self.config),
                "config_path": self.config_path,
                "available_tickers": list(TICKERS.keys()),
                "available_timeframes": TIME_FRAMES,
                "last_check": datetime.now().isoformat()
            }
            
            return base_status
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return {
                "refactored_info": {
                    "pipeline_version": "refactored_v1",
                    "config_loaded": bool(self.config),
                    "config_path": self.config_path,
                    "last_check": datetime.now().isoformat(),
                    "status": "error",
                    "error": str(e)
                }
            }
    
    def _merge_configs(self, base_config: Dict, override_config: Dict) -> Dict:
        """
        Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        merged.update(override_config)
        return merged
