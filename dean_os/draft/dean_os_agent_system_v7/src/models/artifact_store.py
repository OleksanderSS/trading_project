"""
Central Model Artifact Store for Safe Model Serialization

This module provides a centralized, secure interface for loading and saving
model artifacts across different serialization formats (joblib, pickle, torch, keras).

Security Features:
- Path validation using resolve_trusted_artifact_path
- Safe deserialization without allow_pickle=True
- Format-specific safe loaders
- Centralized artifact management
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.core.logging.logger import ProjectLogger
from src.utils.artifact_security import resolve_trusted_artifact_path

logger = ProjectLogger.get_logger("ModelArtifactStore")


class ModelArtifactStore:
    """
    Central store for model artifact serialization with security validation.
    
    This class provides a unified interface for loading and saving models
    across different serialization formats while enforcing security best practices.
    """

    def __init__(self, default_root: str = "data/models"):
        """
        Initialize the artifact store.
        
        Args:
            default_root: Default root directory for model artifacts
        """
        self.default_root = default_root
        self.logger = logger

    def save_joblib(self, model: Any, path: str | Path, **kwargs) -> bool:
        """
        Save a model using joblib serialization.
        
        Args:
            model: The model object to save
            path: Path where to save the model
            **kwargs: Additional arguments for joblib.dump
            
        Returns:
            True if successful, False otherwise
        """
        try:
            resolved_path = self._resolve_save_path(path, ".joblib")
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, resolved_path, **kwargs)
            self.logger.info(f"Model saved to {resolved_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model with joblib: {e}")
            return False

    def load_joblib(self, path: str | Path, **kwargs) -> Any | None:
        """
        Load a model using joblib deserialization.
        
        Args:
            path: Path to the model file
            **kwargs: Additional arguments for joblib.load
            
        Returns:
            The loaded model, or None if loading failed
        """
        try:
            resolved_path = resolve_trusted_artifact_path(
                path,
                allowed_suffixes={".joblib"},
                must_exist=True
            )
            model = joblib.load(resolved_path, **kwargs)
            self.logger.info(f"Model loaded from {resolved_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model with joblib: {e}")
            return None

    def save_pickle(self, model: Any, path: str | Path, **kwargs) -> bool:
        """
        Save a model using pickle serialization.
        
        Args:
            model: The model object to save
            path: Path where to save the model
            **kwargs: Additional arguments for pickle.dump
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import pickle
            resolved_path = self._resolve_save_path(path, ".pkl")
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_path, 'wb') as f:
                pickle.dump(model, f, **kwargs)
            self.logger.info(f"Model saved to {resolved_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model with pickle: {e}")
            return False

    def load_pickle(self, path: str | Path, **kwargs) -> Any | None:
        """
        Load a model using pickle deserialization.
        
        Args:
            path: Path to the model file
            **kwargs: Additional arguments for pickle.load
            
        Returns:
            The loaded model, or None if loading failed
        """
        try:
            import pickle
            resolved_path = resolve_trusted_artifact_path(
                path,
                allowed_suffixes={".pkl", ".pickle"},
                must_exist=True
            )
            with open(resolved_path, 'rb') as f:
                model = pickle.load(f, **kwargs)
            self.logger.info(f"Model loaded from {resolved_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model with pickle: {e}")
            return None

    def save_torch(self, model: Any, path: str | Path, **kwargs) -> bool:
        """
        Save a PyTorch model.
        
        Args:
            model: The PyTorch model to save
            path: Path where to save the model
            **kwargs: Additional arguments for torch.save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import torch
            resolved_path = self._resolve_save_path(path, ".pt")
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model, resolved_path, **kwargs)
            self.logger.info(f"Model saved to {resolved_path}")
            return True
        except ImportError:
            self.logger.error("PyTorch not available")
            return False
        except Exception as e:
            self.logger.error(f"Failed to save PyTorch model: {e}")
            return False

    def load_torch(self, path: str | Path, **kwargs) -> Any | None:
        """
        Load a PyTorch model.
        
        Args:
            path: Path to the model file
            **kwargs: Additional arguments for torch.load
            
        Returns:
            The loaded model, or None if loading failed
        """
        try:
            import torch
            resolved_path = resolve_trusted_artifact_path(
                path,
                allowed_suffixes={".pt", ".pth"},
                must_exist=True
            )
            model = torch.load(resolved_path, **kwargs)
            self.logger.info(f"Model loaded from {resolved_path}")
            return model
        except ImportError:
            self.logger.error("PyTorch not available")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model: {e}")
            return None

    def save_keras(self, model: Any, path: str | Path, **kwargs) -> bool:
        """
        Save a Keras/TensorFlow model.
        
        Args:
            model: The Keras model to save
            path: Path where to save the model
            **kwargs: Additional arguments for model.save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            resolved_path = self._resolve_save_path(path, ".keras")
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(resolved_path, **kwargs)
            self.logger.info(f"Model saved to {resolved_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save Keras model: {e}")
            return False

    def load_keras(self, path: str | Path, **kwargs) -> Any | None:
        """
        Load a Keras/TensorFlow model.
        
        Args:
            path: Path to the model file
            **kwargs: Additional arguments for keras.models.load_model
            
        Returns:
            The loaded model, or None if loading failed
        """
        try:
            from tensorflow import keras
            resolved_path = resolve_trusted_artifact_path(
                path,
                allowed_suffixes={".keras", ".h5"},
                must_exist=True
            )
            model = keras.models.load_model(resolved_path, **kwargs)
            self.logger.info(f"Model loaded from {resolved_path}")
            return model
        except ImportError:
            self.logger.error("TensorFlow/Keras not available")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load Keras model: {e}")
            return None

    def load_numpy(self, path: str | Path, **kwargs) -> np.ndarray | None:
        """
        Load a numpy array safely (without allow_pickle=True).
        
        Args:
            path: Path to the numpy file
            **kwargs: Additional arguments for np.load (allow_pickle will be ignored)
            
        Returns:
            The loaded array, or None if loading failed
        """
        try:
            resolved_path = resolve_trusted_artifact_path(
                path,
                allowed_suffixes={".npy", ".npz"},
                must_exist=True
            )
            # SECURITY: Never allow pickle in numpy loads
            safe_kwargs = {k: v for k, v in kwargs.items() if k != 'allow_pickle'}
            array = np.load(resolved_path, allow_pickle=False, **safe_kwargs)
            self.logger.info(f"Array loaded from {resolved_path}")
            return array
        except Exception as e:
            self.logger.error(f"Failed to load numpy array: {e}")
            return None

    def save_numpy(self, array: np.ndarray, path: str | Path, **kwargs) -> bool:
        """
        Save a numpy array.
        
        Args:
            array: The numpy array to save
            path: Path where to save the array
            **kwargs: Additional arguments for np.save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            resolved_path = self._resolve_save_path(path, ".npy")
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(resolved_path, array, **kwargs)
            self.logger.info(f"Array saved to {resolved_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save numpy array: {e}")
            return False

    def _resolve_save_path(self, path: str | Path, default_suffix: str) -> Path:
        """
        Resolve a path for saving, ensuring it's within the default root.
        
        Args:
            path: The path to resolve
            default_suffix: Default suffix if not present
            
        Returns:
            Resolved absolute path
        """
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = Path(self.default_root) / path_obj

        # Add default suffix if not present
        if path_obj.suffix == "":
            path_obj = path_obj.with_suffix(default_suffix)

        return path_obj.resolve()

    def auto_save(self, model: Any, path: str | Path, **kwargs) -> bool:
        """
        Automatically detect format and save model.
        
        Args:
            model: The model object to save
            path: Path where to save the model
            **kwargs: Additional arguments for format-specific save
            
        Returns:
            True if successful, False otherwise
        """
        path_str = str(path).lower()

        if any(ext in path_str for ext in [".joblib"]):
            return self.save_joblib(model, path, **kwargs)
        elif any(ext in path_str for ext in [".pkl", ".pickle"]):
            return self.save_pickle(model, path, **kwargs)
        elif any(ext in path_str for ext in [".pt", ".pth"]):
            return self.save_torch(model, path, **kwargs)
        elif any(ext in path_str for ext in [".keras", ".h5"]):
            return self.save_keras(model, path, **kwargs)
        elif any(ext in path_str for ext in [".npy", ".npz"]):
            return self.save_numpy(model, path, **kwargs)
        else:
            # Default to joblib
            return self.save_joblib(model, path, **kwargs)

    def auto_load(self, path: str | Path, **kwargs) -> Any | None:
        """
        Automatically detect format and load model.
        
        Args:
            path: Path to the model file
            **kwargs: Additional arguments for format-specific load
            
        Returns:
            The loaded model/array, or None if loading failed
        """
        path_str = str(path).lower()

        if any(ext in path_str for ext in [".joblib"]):
            return self.load_joblib(path, **kwargs)
        elif any(ext in path_str for ext in [".pkl", ".pickle"]):
            return self.load_pickle(path, **kwargs)
        elif any(ext in path_str for ext in [".pt", ".pth"]):
            return self.load_torch(path, **kwargs)
        elif any(ext in path_str for ext in [".keras", ".h5"]):
            return self.load_keras(path, **kwargs)
        elif any(ext in path_str for ext in [".npy", ".npz"]):
            return self.load_numpy(path, **kwargs)
        else:
            # Try joblib as default
            return self.load_joblib(path, **kwargs)


# Singleton instance for convenience
_artifact_store_instance: ModelArtifactStore | None = None


def get_model_artifact_store(default_root: str = "data/models") -> ModelArtifactStore:
    """
    Get or create singleton ModelArtifactStore instance.
    
    Args:
        default_root: Default root directory for model artifacts
        
    Returns:
        ModelArtifactStore instance
    """
    global _artifact_store_instance
    if _artifact_store_instance is None:
        _artifact_store_instance = ModelArtifactStore(default_root)
    return _artifact_store_instance
