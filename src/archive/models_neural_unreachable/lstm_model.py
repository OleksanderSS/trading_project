# src/models/neural/lstm_model.py

from typing import Any

import numpy as np

from src.core.logging.logger import ProjectLogger
from src.models.neural.base_neural import BaseNeuralModel, _get_tf
from src.models.neural.sequence_builder import SequenceBuilder


class LSTMModel(BaseNeuralModel):
    """
    LSTM (Long Short-Term Memory) model for time series forecasting.
    This model is designed for sequence prediction tasks and follows the BaseNeuralModel interface.
    """

    def __init__(self, task_type: str = "regression", epochs: int = 50, batch_size: int = 32, random_state: int = 42, sequence_builder: SequenceBuilder | None = None):
        super().__init__(model_type="lstm", task_type=task_type, random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_builder = sequence_builder or SequenceBuilder(strategy='sliding_window')
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
        
        Args:
            X: Input data (2D or 3D). If 2D, will be converted to sequences using sequence_builder.
            y: Target data
            **kwargs: Additional arguments including window_size, step_size for sequence building
            
        Returns:
            Training metrics
        """
        self.logger.info(f"Starting training for {self.name}...")

        # Build sequences if input is 2D
        if len(X.shape) == 2:
            window_size = kwargs.get('window_size', 10)
            step_size = kwargs.get('step_size', 1)
            X = self.sequence_builder.build_sequences(X, window_size=window_size, step_size=step_size)
            # Adjust y to match sequence length
            if len(y) > len(X):
                y = y[-len(X):]
            elif len(y) < len(X):
                raise ValueError(f"Target length ({len(y)}) is insufficient for {len(X)} sequences")

        # Перетворення в числові типи для Keras
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        return super().train(X, y, **kwargs)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Makes predictions with the trained LSTM model.
        
        Args:
            X: Input data (2D or 3D). If 2D, will be converted to sequences using sequence_builder.
            **kwargs: Additional arguments including window_size, step_size for sequence building
            
        Returns:
            Predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction.")

        # Build sequences if input is 2D
        if len(X.shape) == 2:
            window_size = kwargs.get('window_size', 10)
            step_size = kwargs.get('step_size', 1)
            X = self.sequence_builder.build_sequences(X, window_size=window_size, step_size=step_size)

        # Перетворення в числовий тип для Keras
        X = X.astype(np.float32)
        x_norm = self._normalize_data(X, fit=False)
        predictions = self.model.predict(x_norm, verbose=0)

        self.logger.info(f"Made predictions for {X.shape[0]} samples.")
        return predictions.flatten()  # Return a 1D array
