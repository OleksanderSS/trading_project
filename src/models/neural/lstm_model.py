# src/models/neural/lstm_model.py

from typing import Any

import numpy as np

from src.core.logging.logger import ProjectLogger
from src.models.neural.base_neural import BaseNeuralModel, _get_tf


class LSTMModel(BaseNeuralModel):
    """
    LSTM (Long Short-Term Memory) model for time series forecasting.
    This model is designed for sequence prediction tasks and follows the BaseNeuralModel interface.
    """

    def __init__(self, task_type: str = "regression", epochs: int = 50, batch_size: int = 32, random_state: int = 42):
        super().__init__(model_type="lstm", task_type=task_type, random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.logger = ProjectLogger.get_logger("LSTMModel")

    @property
    def name(self) -> str:
        """Returns the unique name of the model."""
        return "lstm_tf"

    def _build_architecture(self, input_shape: tuple[int, ...]) -> Any:
        """
        Defines the LSTM architecture using TensorFlow/Keras.
        The input_shape is expected to be (timesteps, n_features).
        """
        tf = _get_tf()
        layers = tf.keras.layers
        models = tf.keras
        if len(input_shape) != 2:
            raise ValueError(f"Expected input_shape to be a tuple of length 2 (timesteps, n_features), but got {input_shape}")

        timesteps, n_features = input_shape

        inputs = layers.Input(shape=(timesteps, n_features))

        # Using CuDNNLSTM for GPU acceleration if available
        lstm_layer = layers.LSTM(64, activation='relu', return_sequences=True)
        x = lstm_layer(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        # Output layer varies based on the task type
        if self.task_type == "classification":
            # Binary classification assumed
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            # Regression task
            outputs = layers.Dense(1, activation='linear')(x)
            loss = 'mse'

        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss=loss)

        self.logger.info(f"LSTM architecture built for '{self.task_type}' task.")
        return model

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict[str, Any]:
        """
        Trains the LSTM model.
        """
        self.logger.info(f"Starting training for {self.name}...")

        # Reshape data if it's 2D
        if len(X.shape) == 2:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Перетворення в числові типи для Keras
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        return super().train(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions with the trained LSTM model.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction.")

        if len(X.shape) == 2:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Перетворення в числовий тип для Keras
        X = X.astype(np.float32)
        x_norm = self._normalize_data(X, fit=False)
        predictions = self.model.predict(x_norm, verbose=0)

        self.logger.info(f"Made predictions for {X.shape[0]} samples.")
        return predictions.flatten()  # Return a 1D array
