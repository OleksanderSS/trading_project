# src/models/neural/gru_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, Any, Tuple

from src.models.neural.base_neural import BaseNeuralModel
from src.core.logging.logger import ProjectLogger

class GRUModel(BaseNeuralModel):
    """
    GRU (Gated Recurrent Unit) model for time series forecasting.
    This model is suitable for sequence prediction tasks and adheres to the BaseNeuralModel interface.
    """

    def __init__(self, task_type: str = "regression", epochs: int = 50, batch_size: int = 32, random_state: int = 42):
        super().__init__(model_type="gru", task_type=task_type, random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.logger = ProjectLogger.get_logger("GRUModel")

    @property
    def name(self) -> str:
        """Returns the unique name of the model."""
        return "gru_tf"

    def _build_architecture(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """
        Defines the GRU architecture using TensorFlow/Keras.
        The input_shape is expected to be (timesteps, n_features).
        """
        if len(input_shape) != 2:
            raise ValueError(f"Expected input_shape to be a tuple of length 2 (timesteps, n_features), but got {input_shape}")

        timesteps, n_features = input_shape

        inputs = layers.Input(shape=(timesteps, n_features))
        
        x = layers.GRU(64, activation='relu', return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.GRU(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer depends on the task type
        if self.task_type == "classification":
            # Assuming binary classification for now
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:  # Regression
            outputs = layers.Dense(1, activation='linear')(x)
            loss = 'mse'

        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss=loss)

        self.logger.info(f"GRU architecture built for '{self.task_type}' task.")
        return model

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Trains the GRU model.
        """
        self.logger.info(f"Starting training for {self.name}...")

        # Reshape data if it's 2D
        if len(X.shape) == 2:
            # Assuming each row is a timestep, and we have one feature
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return super().train(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions with the trained GRU model.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction.")

        if len(X.shape) == 2:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        X_norm = self._normalize_data(X, fit=False)
        predictions = self.model.predict(X_norm, verbose=0)

        self.logger.info(f"Made predictions for {X.shape[0]} samples.")
        return predictions.flatten() # Flatten to return a 1D array of predictions
