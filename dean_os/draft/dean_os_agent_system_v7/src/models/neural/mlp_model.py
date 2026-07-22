# src/models/neural/mlp_model.py

from typing import Any

from src.core.logging.logger import ProjectLogger
from src.models.neural.base_neural import BaseNeuralModel, _get_tf


class MLPModel(BaseNeuralModel):
    """
    MLP (Multi-Layer Perceptron) model for standard classification and regression.
    Built on TensorFlow/Keras and conforms to the BaseNeuralModel interface.
    """

    def __init__(self, task_type: str = "regression", epochs: int = 50, batch_size: int = 32, random_state: int = 42):
        super().__init__(model_type="mlp", task_type=task_type, random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.logger = ProjectLogger.get_logger("MLPModel")

    @property
    def name(self) -> str:
        return "mlp_tf"

    def _build_architecture(self, input_shape: tuple[int, ...]) -> Any:
        """
        Defines the MLP architecture.
        The input_shape is expected to be a tuple with a single element (n_features,).
        """
        tf = _get_tf()
        layers = tf.keras.layers
        models = tf.keras
        if len(input_shape) != 1:
            raise ValueError(f"Expected input_shape to be a tuple of length 1, but got {input_shape}")

        n_features = input_shape[0]

        inputs = layers.Input(shape=(n_features,))

        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        # Output layer configuration depends on the task
        if self.task_type == "classification":
            # Binary classification is assumed
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            # Regression
            outputs = layers.Dense(1, activation='linear')(x)
            loss = 'mse'

        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss=loss)

        self.logger.info(f"MLP architecture built for '{self.task_type}' task.")
        return model
