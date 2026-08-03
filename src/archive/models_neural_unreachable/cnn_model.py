# src/models/neural/cnn_model.py

from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.models.neural.base_neural import BaseNeuralModel, _get_tf


class CNNModel(BaseNeuralModel):
    """
    Уніфікована модель згорткової нейронної мережі (CNN) для фінансових часових рядів.
    """

    def __init__(self, task_type: str = "regression", epochs: int = 40, batch_size: int = 32, random_state: int = 42):
        super().__init__(model_type="cnn", task_type=task_type)
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.logger = ProjectLogger.get_logger("CNNModel")

    @property
    def name(self) -> str:
        """Повертає ім'я моделі."""
        return "cnn"

    def _build_architecture(self, input_shape: tuple) -> Any:
        """
        Визначає архітектуру CNN для обробки послідовностей даних.
        """
        tf = _get_tf()
        Sequential = tf.keras.Sequential
        Input = tf.keras.layers.Input
        Conv1D = tf.keras.layers.Conv1D
        Dense = tf.keras.layers.Dense
        Dropout = tf.keras.layers.Dropout
        Flatten = tf.keras.layers.Flatten
        MaxPooling1D = tf.keras.layers.MaxPooling1D
        timesteps, n_features = input_shape

        layers = [
            Input(shape=(timesteps, n_features)),
            Conv1D(filters=64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.2)
        ]

        if self.task_type == "regression":
            layers.append(Dense(1, activation="linear"))
            loss = "mse"
            metrics = ["mae"]
        else:
            # Для classифікації використовуємо 2 виходи (Sparse Categorical Crossentropy)
            layers.append(Dense(2, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
            metrics = ["accuracy"]

        model = Sequential(layers)
        model.compile(optimizer="adam", loss=loss, metrics=metrics)
        return model

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict[str, Any]:
        """
        Навчає модель CNN на вхідних даних.
        """
        try:
            # Фіксація seed для відтворюваності
            np.random.seed(self.random_state)
            _get_tf().random.set_seed(self.random_state)

            # Підготовка та нормалізація даних
            # Перетворення в numpy з правильними типами
            x_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            y_array = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)

            # Перетворення в числові типи для Keras
            x_array = x_array.astype(np.float32)
            y_array = y_array.astype(np.float32)

            x_norm = self._normalize_data(x_array, fit=True)
            # y не нормалізуємо, просто перетворюємо в float32
            y_norm = y_array

            # Визначення вхідної форми
            if len(x_norm.shape) == 2:
                # Якщо дані плоскі, перетворюємо в 3D (samples, timesteps=1, features)
                x_norm = x_norm.reshape((x_norm.shape[0], 1, x_norm.shape[1]))

            input_shape = (x_norm.shape[1], x_norm.shape[2])
            self.model = self._build_architecture(input_shape)

            # Навчання
            self.logger.info(f"Початок навчання CNN ({self.task_type}). Форма: {x_norm.shape}")
            self.model.fit(
                x_norm,
                y_norm,
                epochs=kwargs.get('epochs', self.epochs),
                batch_size=kwargs.get('batch_size', self.batch_size),
                verbose=0
            )

            self.is_trained = True
            self.logger.info("[OK] CNN успішно натренований.")

            return self.get_model_info()

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"[ERROR] Помилка під час навчання CNN: {e}", exc_info=True)
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Генерує прогнози на основі вхідних даних.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Модель повинна бути навчена перед виконанням прогнозів.")

        try:
            # Перетворення в numpy з правильними типами
            x_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            x_array = x_array.astype(np.float32)

            x_norm = self._normalize_data(x_array, fit=False)

            if len(x_norm.shape) == 2:
                x_norm = x_norm.reshape((x_norm.shape[0], 1, x_norm.shape[1]))

            predictions = self.model.predict(x_norm, verbose=0)

            if self.task_type == "classification":
                # Повертаємо індекс classу з найвищою ймовірністю
                return np.argmax(predictions, axis=1)

            return predictions.flatten()

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"[ERROR] Помилка під час виконання прогнозу CNN: {e}")
            raise

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Повертає ймовірності classів для задач classифікації.
        """
        if self.task_type != "classification":
            raise ValueError("Метод predict_proba доступний тільки для задач classифікації.")

        if not self.is_trained or self.model is None:
            raise ValueError("Модель повинна бути навчена.")

        # Перетворення в numpy з правильними типами
        x_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        x_array = x_array.astype(np.float32)

        x_norm = self._normalize_data(x_array, fit=False)
        if len(x_norm.shape) == 2:
            x_norm = x_norm.reshape((x_norm.shape[0], 1, x_norm.shape[1]))

        return self.model.predict(x_norm, verbose=0)
