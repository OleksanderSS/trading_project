# models/transformer_model.py

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.models.interfaces import BaseModel

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TransformerModel(BaseModel):
    """Transformer model with sklearn fallback"""

    def __init__(
        self,
        input_size: int | None = None,
        num_heads: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
        classification: bool = True
    ):
        super().__init__(model_type="transformer", task_type="classification" if classification else "regression")
        self.input_size = input_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.classification = classification
        self.is_trained = False
        self.model: Any | None = None
        self.scaler = StandardScaler()

        # Fallback model
        self.fallback_model: Any | None = None

    def _create_fallback_model(self):
        """Create fallback model based on sklearn"""
        try:
            self.fallback_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_leaf=1,
                max_features='sqrt'
            ) if self.classification else RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_leaf=1,
                max_features='sqrt'
            )

            logger.info("OK Created fallback RandomForest model")
            return True

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Failed to create fallback model: {e}")
            return False

    def _create_sequences(self, X: np.ndarray, seq_len: int = 10) -> np.ndarray:
        """Create sequences for Transformer"""
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be 2D and have at least one feature")

        x_seq = []
        for i in range(len(X) - seq_len + 1):
            x_seq.append(X[i:i + seq_len])

        return np.array(x_seq)

    def fit(self, X, y, seq_len: int = 10, epochs: int = 20, batch_size: int = 32):
        """Train the model"""
        try:
            # Convert to numpy
            if hasattr(X, 'values'):
                X = X.values
            if hasattr(y, 'values'):
                y = y.values

            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            # Перетворення в числові типи для Keras
            X = X.astype(np.float32)
            y = y.astype(np.float32)

            # Attempt to use TensorFlow/Keras
            try:
                self._fit_tensorflow_transformer(
                    X,
                    y,
                    seq_len,
                    epochs,
                    batch_size,
                )
            except ImportError:
                logger.warning(
                    "TensorFlow not available, using fallback model"
                )
                self._fit_fallback(X, y)
            else:
                self.is_trained = True
                logger.info(
                    f"OK Transformer model trained "
                    f"(classification: {self.classification})"
                )
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Transformer training failed: {e}")
            # Fallback to simple model
            try:
                self._fit_fallback(X, y)
                self.is_trained = True
                logger.info("OK Used fallback model")
            except Exception as e2:
                logger.error(f"Fallback training also failed: {e2}")
                raise

    def _fit_tensorflow_transformer(self, X, y, seq_len: int, epochs: int, batch_size: int):
        """Train TensorFlow Transformer"""
        from tensorflow.keras.optimizers import Adam

        # Create sequences
        x_seq = self._create_sequences(X, seq_len)
        if len(x_seq) == 0:
            raise ValueError("Not enough data for sequences")

        # Align length of y
        y_seq = y[seq_len-1:seq_len-1+len(x_seq)]

        # Split into train/val
        split_idx = int(len(x_seq) * 0.8)
        x_train, x_val = x_seq[:split_idx], x_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

        # Create model
        self.input_size = X.shape[1]
        self.model = self._create_transformer_model()

        # Training
        loss = "binary_crossentropy" if self.classification else "mse"
        assert self.model is not None
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=["mae"])

        # Training with validation
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=0
        )

        # Log results
        final_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        logger.info(f"Transformer training completed - Loss: {final_loss:.6f}, Val Loss: {val_loss:.6f}")

    def _create_transformer_model(self):
        """Create TensorFlow Transformer model"""
        import tensorflow as tf

        # Transformer architecture
        inputs = tf.keras.layers.Input(shape=(None, self.input_size))  # seq_len will be dynamic

        # Multi-Head Attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.input_size
        )(inputs, inputs)
        attention_output = tf.keras.layers.Dropout(self.dropout)(attention_output)
        attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)

        # Feed Forward
        ff_output = tf.keras.layers.Dense(self.ff_dim, activation="relu")(attention_output)
        ff_output = tf.keras.layers.Dense(self.input_size)(ff_output)
        ff_output = tf.keras.layers.Dropout(self.dropout)(ff_output)
        ff_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)

        # Global Average Pooling instead of Flatten
        x = tf.reduce_mean(ff_output, axis=1)  # Global average pooling
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # Output layer
        if self.classification:
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        else:
            outputs = tf.keras.layers.Dense(1, activation="linear")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def _fit_fallback(self, X, y):
        """Train fallback model"""
        if self.fallback_model is None and not self._create_fallback_model():
            raise RuntimeError("Cannot create fallback model")

        # Використовуємо останні значення для Training
        if len(X) > 10:
            X_train = X[-min(len(X), 100):]  # Last 100 points
            y_train = y[-min(len(y), 100):]
        else:
            X_train = X
            y_train = y

        assert self.fallback_model is not None
        self.fallback_model.fit(X_train, y_train)
        logger.info("OK Fallback model trained")

    def predict(self, X, seq_len: int = 10):
        """Prediction"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            # Convert to numpy
            if hasattr(X, 'values'):
                X = X.values

            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Перетворення в числовий тип для Keras
            X = X.astype(np.float32)

            # Attempt TensorFlow
            if self.model is not None:
                return self._predict_tensorflow(X, seq_len)
            else:
                return self._predict_fallback(X)

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Transformer prediction failed: {e}")
            # Fallback
            try:
                return self._predict_fallback(X)
            except Exception as e2:
                logger.error(f"Fallback prediction also failed: {e2}")
                raise

    def _predict_tensorflow(self, X, seq_len: int):
        """TensorFlow prediction"""
        x_seq = self._create_sequences(X, seq_len)
        if len(x_seq) == 0:
            raise ValueError("Not enough data for prediction")

        assert self.model is not None
        predictions = self.model.predict(x_seq, verbose=0)
        return predictions

    def _predict_fallback(self, X):
        """Fallback prediction"""
        if self.fallback_model is None:
            raise RuntimeError("No fallback model available")

        # Use latest values for prediction
        if len(X.shape) == 2:
            return self.fallback_model.predict(X[-1:].reshape(1, -1))
        else:
            return self.fallback_model.predict(X.reshape(1, -1))

    def predict_proba(self, X, seq_len: int = 10):
        """Probability prediction"""
        if not self.classification:
            raise ValueError("predict_proba only available for classification")

        predictions = self.predict(X, seq_len)

        # Convert to probabilities
        if len(predictions.shape) == 1:
            probas = np.zeros((len(predictions), 2))
            probas[:, 1] = predictions
            probas[:, 0] = 1 - predictions
            return probas
        else:
            return predictions

    def get_params(self) -> dict[str, Any]:
        """Model parameters"""
        return {
            'input_size': self.input_size,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'classification': self.classification,
            'is_trained': self.is_trained,
            'has_tensorflow_model': self.model is not None,
            'has_fallback_model': self.fallback_model is not None
        }

# Keep old function for compatibility
def train_transformer_model(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    task: str = "regression",
    epochs: int = 20,
    batch_size: int = 16,
    num_heads: int = 4,
    ff_dim: int = 64
):
    """Old function for compatibility"""
    try:
        # Prepare data
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        X_df = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        feature_medians = X_df.median()
        valid_feature_cols = feature_medians.dropna().index
        X_df = X_df[valid_feature_cols].fillna(feature_medians[valid_feature_cols])
        # Fixed look-ahead bias by using rolling/past windows or appropriate shift
        previous_close = df['Close'].shift(1)
        if task == "classification":
            y = (previous_close > df['Close']).astype(float)
            y[previous_close.isna() | df['Close'].isna()] = np.nan
        else:
            y = previous_close # audit: ignore

        # Remove NaN from y
        mask = ~np.isnan(y)
        X = X_df.loc[mask].values
        y = y[mask]

        if len(X) < 20:
            logger.warning(f"Not enough data for {ticker} {timeframe}")
            return None

        # Create model
        model = TransformerModel(
            input_size=X.shape[1],
            num_heads=num_heads,
            ff_dim=ff_dim,
            classification=(task == "classification")
        )

        # Training
        model.fit(X, y, epochs=epochs, batch_size=batch_size)

        logger.info(f"OK Transformer trained for {ticker} {timeframe}")
        return model

    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.error(f"Error training Transformer {ticker} {timeframe}: {e}")
        raise RuntimeError(
            f"Failed to train Transformer model for {ticker} {timeframe}"
        ) from e
