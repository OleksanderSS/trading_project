# src/models/neural/base_neural.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from abc import abstractmethod
from typing import Dict, Any, Optional, Tuple

from src.models.interfaces import BaseModel
from src.core.logging.logger import ProjectLogger

class BaseNeuralModel(BaseModel):
    """
    Абстрактна база для глибокого навчання (TensorFlow/Keras).
    Забезпечує уніфіковану нормалізацію, серіалізацію та відтворюваність.
    """

    def __init__(self, model_type: str, task_type: str = "regression", random_state: int = 42):
        super().__init__(model_type, task_type)
        self.random_state = random_state
        self.model: Optional[tf.keras.Model] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self._set_seed()

    def _set_seed(self):
        """Встановлює random seed для відтворюваності результатів."""
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        os.environ['PYTHONHASHSEED'] = str(self.random_state)

    def _normalize_data(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Z-score нормалізація даних.
        
        Args:
            x: Вхідний масив даних.
            fit: Якщо True, обчислює параметри (mean, std) на основі x.
        """
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        if fit:
            self.scaler_mean = np.mean(x_clean, axis=0)
            self.scaler_std = np.std(x_clean, axis=0)
            # Запобігаємо діленню на нуль
            self.scaler_std[self.scaler_std == 0] = 1.0
            self.logger.debug(f"Normalization params fitted for {self.model_type}")

        if self.scaler_mean is not None and self.scaler_std is not None:
            return (x_clean - self.scaler_mean) / self.scaler_std
        
        return x_clean

    @abstractmethod
    def _build_architecture(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Визначає архітектуру нейромережі. Має бути реалізовано в нащадках."""
        pass

    def train(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """
        Уніфікований цикл навчання для нейромереж.
        """
        try:
            # Extract neural-specific parameters from kwargs
            epochs = kwargs.get('epochs', 50)
            batch_size = kwargs.get('batch_size', 32)
            validation_split = kwargs.get('validation_split', 0.2)
            
            # Перетворення в numpy з правильними типами даних
            X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            y_np = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)
            
            # Перетворення в числові типи для Keras
            X_np = X_np.astype(np.float32)
            y_np = y_np.astype(np.float32)

            # 1. Підготовка та нормалізація
            x_norm = self._normalize_data(X_np, fit=True)
            
            # 2. Побудова моделі, якщо ще не створена
            if self.model is None:
                input_shape = x_norm.shape[1:]
                self.model = self._build_architecture(input_shape)
                self.logger.info(f"Model architecture built for {self.model_type}. Input shape: {input_shape}")

            # 3. Навчання
            history = self.model.fit(
                x_norm, y_np,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=kwargs.get('verbose', 0),
                callbacks=kwargs.get('callbacks', [])
            )

            self.is_trained = True
            self.logger.info(f"Training completed for {self.name}. Last loss: {history.history['loss'][-1]:.4f}")
            
            return dict(history.history) if hasattr(history, 'history') else {}

        except Exception as e:
            self.logger.error(f"Failed to train {self.model_type}: {str(e)}", exc_info=True)
            raise

    def predict(self, x: Any) -> np.ndarray:
        """Makes predictions.на нормалізованих даних."""
        if not self.is_trained or self.model is None:
            raise RuntimeError(f"Model {self.model_type} is not trained.")

        x_np = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
        x_np = x_np.astype(np.float32)  # Перетворення в числовий тип
        x_norm = self._normalize_data(x_np, fit=False)
        
        preds = self.model.predict(x_norm, verbose=0)
        
        # Для classифікації повертаємо індекс classу, For regression - значення
        if self.task_type == "classification" and preds.shape[-1] > 1:
            return np.argmax(preds, axis=1)
        return preds.flatten()

    def save_model(self, path: str) -> bool:
        """Зберігає модель та параметри нормалізації."""
        if self.model is None:
            return False
        try:
            # Збереження Keras моделі
            model_path = f"{path}.h5"
            self.model.save(model_path)
            
            # Збереження метаданих (нормалізація)
            meta_path = f"{path}_meta.npy"
            np.save(meta_path, {
                'mean': self.scaler_mean, 
                'std': self.scaler_std,
                'task_type': self.task_type
            })
            
            self.logger.info(f"Model and metadata saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving neural model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Завантажує модель та параметри нормалізації."""
        try:
            self.model = tf.keras.models.load_model(f"{path}.h5")
            
            meta_path = f"{path}_meta.npy"
            if os.path.exists(meta_path):
                meta = np.load(meta_path, allow_pickle=True).item()
                self.scaler_mean = meta['mean']
                self.scaler_std = meta['std']
                self.task_type = meta.get('task_type', self.task_type)
            
            self.is_trained = True
            self.logger.info(f"Model and metadata loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading neural model: {e}")
            return False

    @property
    def name(self) -> str:
        return f"neural_{self.model_type}"