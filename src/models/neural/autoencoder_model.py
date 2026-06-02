# src/models/neural/autoencoder_model.py  # audit-ignore: AUTOENCODER_ROUTING_REVIEW

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, Any, Tuple

from src.models.neural.base_neural import BaseNeuralModel
from src.core.logging.logger import ProjectLogger

class AutoencoderModel(BaseNeuralModel):
    """
    Conv1D Autoencoder model for anomaly detection in time series data.
    This model learns a compressed representation of the input data and uses the
    reconstruction error to identify anomalies.
    It adheres to the BaseNeuralModel interface.
    """

    def __init__(self, task_type: str = "reconstruction", epochs: int = 50, batch_size: int = 32, random_state: int = 42):
        # Autoencoders are for 'reconstruction', not standard classification/regression.
        super().__init__(model_type="autoencoder", task_type=task_type, random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.logger = ProjectLogger.get_logger("AutoencoderModel")
        # The actual model will be built in the 'train' method via _build_architecture

    @property
    def name(self) -> str:
        """Returns the unique name of the model."""
        return "autoencoder_conv1d"  # audit-ignore: AUTOENCODER_ROUTING_REVIEW

    def _build_architecture(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """
        Defines the Conv1D autoencoder architecture.
        The input_shape is expected to be (timesteps, n_features).
        """
        if len(input_shape) != 2:
            raise ValueError(f"Expected input_shape to be a tuple of length 2 (timesteps, n_features), but got {input_shape}")

        timesteps, n_features = input_shape

        inputs = layers.Input(shape=(timesteps, n_features))

        # --- Encoder ---
        x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(inputs)
        x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
        x = layers.Conv1D(16, kernel_size=3, activation="relu", padding="same")(x)
        encoded = layers.MaxPooling1D(pool_size=2, padding="same")(x)

        # --- Decoder ---
        x = layers.Conv1D(16, kernel_size=3, activation="relu", padding="same")(encoded)
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.UpSampling1D(size=2)(x)
        # Ensure the output has the same number of features as the input
        decoded = layers.Conv1D(n_features, kernel_size=3, activation="linear", padding="same")(x)

        output_len = decoded.shape[1]
        if output_len is not None and output_len > timesteps:
             decoded = layers.Cropping1D(cropping=(0, output_len - timesteps))(decoded)

        model = models.Model(inputs, decoded)
        model.compile(optimizer="adam", loss="mse")

        self.logger.info(f"Autoencoder architecture built: Input({timesteps}, {n_features}) -> Output({decoded.shape[1]}, {decoded.shape[2]})")
        return model

    def train(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        Trains the Autoencoder model. 'y' is ignored, as the model learns to reconstruct 'X'.
        """
        self.logger.info(f"Starting training for {self.name}...")
        
        if len(X.shape) == 2:
            X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        # For an autoencoder, the input is also the output (we are reconstructing X)
        return super().train(X, X, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the reconstruction error for the input data.
        A higher error suggests an anomaly.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction.")

        if len(X.shape) == 2:
            X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
            
        X_norm = self._normalize_data(X, fit=False)
        X_reconstructed_norm = self.model.predict(X_norm, verbose=0)
        
        reconstruction_error = np.mean(np.square(X_norm - X_reconstructed_norm), axis=(1, 2))
        
        self.logger.info(f"Calculated reconstruction error for {X.shape[0]} samples.")
        return reconstruction_error
        
    def get_anomaly_labels(self, X: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Classifies data points as normal (0) or anomalous (1) based on a threshold.
        """
        reconstruction_errors = self.predict(X)
        return (reconstruction_errors > threshold).astype(int)
