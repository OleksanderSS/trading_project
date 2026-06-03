#!/usr/bin/env python3
"""
Model Registry - Centralized Model Management
Handles model registration, metadata storage, and retrieval.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from src.core.logging.logger import ProjectLogger
from src.core.exceptions import DataProcessingError

logger = ProjectLogger.get_logger("ModelRegistry")


class ModelRegistry:
    """
    Centralized registry for model management.
    
    Handles:
    - Model registration and storage
    - Metadata management
    - Model retrieval
    - Storage persistence
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize Model Registry.
        
        Args:
            storage_path: Path for storing model metadata
        """
        self.logger = logger
        self.storage_path = storage_path or Path('data/models/registry')
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}
        
        self.logger.info("✅ ModelRegistry initialized")
    
    def register_model(self, model: Any, model_name: str) -> None:
        """
        Register model in the registry.
        
        Args:
            model: Model to register
            model_name: Name of the model
        """
        try:
            self.models[model_name] = model
            self.model_metadata[model_name] = {
                'registered_at': datetime.now(),
                'model_type': type(model).__name__,
                'last_analysis': None,
                'analysis_count': 0
            }
            self.logger.info(f"✅ Model registered: {model_name}")
        except Exception as e:
            self.logger.error(f"Error registering model: {model_name}: {e}", exc_info=True)
            raise DataProcessingError(f"Error registering model {model_name}: {e}") from e
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """
        Get model by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model if found, None otherwise
        """
        return self.models.get(model_name)
    
    def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get model metadata.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Metadata dictionary if found, None otherwise
        """
        return self.model_metadata.get(model_name)
    
    def update_metadata(self, model_name: str, metadata: Dict[str, Any]) -> None:
        """
        Update model metadata.
        
        Args:
            model_name: Name of the model
            metadata: Metadata to update
        """
        if model_name in self.model_metadata:
            self.model_metadata[model_name].update(metadata)
    
    def list_models(self) -> list[str]:
        """
        List all registered model names.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def remove_model(self, model_name: str) -> None:
        """
        Remove model from registry.
        
        Args:
            model_name: Name of the model to remove
        """
        if model_name in self.models:
            del self.models[model_name]
        if model_name in self.model_metadata:
            del self.model_metadata[model_name]
        self.logger.info(f"🗑️ Model removed: {model_name}")
    
    def save_metadata(self, model_name: str) -> None:
        """
        Save model metadata to file.
        
        Args:
            model_name: Name of the model
        """
        try:
            if model_name in self.model_metadata:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"metadata_{model_name}_{timestamp}.json"
                filepath = self.storage_path / filename
                
                with open(filepath, 'w') as f:
                    json.dump(self.model_metadata[model_name], f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving metadata for {model_name}: {e}", exc_info=True)
            raise DataProcessingError(f"Error saving metadata for {model_name}: {e}") from e
    
    def load_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load model metadata from file.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Metadata dictionary if found, None otherwise
        """
        try:
            metadata_files = list(self.storage_path.glob(f"metadata_{model_name}_*.json"))
            if metadata_files:
                latest_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Error loading metadata for {model_name}: {e}", exc_info=True)
            raise DataProcessingError(f"Error loading metadata for {model_name}: {e}") from e
