# models/autoencoder_model.py

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import logging
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AutoencoderModel:
    """Autoencoder модель для anomaly detection з fallback на IsolationForest"""
    
    def __init__(self, n_components: int = 10, contamination: float = 0.1, 
                 reconstruction_threshold: float = 0.1):
        self.n_components = n_components
        self.contamination = contamination
        self.reconstruction_threshold = reconstruction_threshold
        self.is_trained = False
        self.model = None
        
        # Fallback модель
        self.fallback_model = None
        
    def _create_fallback_model(self):
        """Створити fallback модель на основі IsolationForest"""
        try:
            self.fallback_model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            
            logger.info("OK Created IsolationForest fallback model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create fallback model: {e}")
            return False
    
    def fit(self, X, y=None):
        """Тренування моделі"""
        try:
            # Конвертація в numpy
            if hasattr(X, 'values'):
                X = X.values
                
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Спроба використати TensorFlow/Keras
            try:
                self._fit_tensorflow_autoencoder(X)
            except ImportError:
                logger.warning("TensorFlow not available, using fallback model")
                self._fit_fallback(X)
                
            self.is_trained = True
            logger.info("OK Autoencoder model trained")
            
        except Exception as e:
            logger.error(f"Autoencoder training failed: {e}")
            # Fallback до простої моделі
            try:
                self._fit_fallback(X)
                self.is_trained = True
                logger.info("OK Used fallback model")
            except Exception as e2:
                logger.error(f"Fallback training also failed: {e2}")
                raise
    
    def _fit_tensorflow_autoencoder(self, X):
        """Тренування TensorFlow Autoencoder"""
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        # Створення моделі
        n_features = X.shape[1]
        
        # Encoder
        input_layer = layers.Input(shape=(n_features,))
        encoded = layers.Dense(self.n_components, activation="relu")(input_layer)
        
        # Decoder
        decoded = layers.Dense(n_features, activation="linear")(encoded)
        
        # Autoencoder
        self.model = models.Model(input_layer, decoded)
        self.model.compile(optimizer="adam", loss="mse")
        
        # Тренування
        self.model.fit(X, X, epochs=50, batch_size=32, verbose=0)
        logger.info("OK TensorFlow Autoencoder trained successfully")
    
    def _fit_fallback(self, X):
        """Тренування fallback моделі"""
        if self.fallback_model is None:
            if not self._create_fallback_model():
                raise RuntimeError("Cannot create fallback model")
        
        # Використовуємо останні значення для тренування
        if len(X) > 10:
            X_train = X[-min(len(X), 1000):]  # Останні 1000 точок
        else:
            X_train = X
        
        self.fallback_model.fit(X_train)
        logger.info("OK Fallback model trained")
    
    def predict(self, X):
        """Прогнозування (reconstruction error)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Конвертація в numpy
            if hasattr(X, 'values'):
                X = X.values
                
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Спроба TensorFlow
            if self.model is not None:
                return self._predict_tensorflow(X)
            else:
                return self._predict_fallback(X)
                
        except Exception as e:
            logger.error(f"Autoencoder prediction failed: {e}")
            # Fallback
            try:
                return self._predict_fallback(X)
            except Exception as e2:
                logger.error(f"Fallback prediction also failed: {e2}")
                raise
    
    def _predict_tensorflow(self, X):
        """Прогнозування TensorFlow"""
        # Reconstruction error
        X_reconstructed = self.model.predict(X, verbose=0)
        reconstruction_error = np.mean(np.square(X - X_reconstructed), axis=1)
        return reconstruction_error
    
    def _predict_fallback(self, X):
        """Прогнозування fallback"""
        if self.fallback_model is None:
            raise RuntimeError("No fallback model available")
        
        # IsolationForest повертає аномалії (-1) та нормальні дані (1)
        # Конвертуємо в reconstruction error
        anomaly_scores = self.fallback_model.decision_function(X)
        # Нормалізуємо в 0-1 діапазон
        reconstruction_error = 1 - (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        return reconstruction_error
    
    def get_anomaly_labels(self, X, threshold=None):
        """Отримати мітки аномалій"""
        if threshold is None:
            threshold = self.reconstruction_threshold
        
        reconstruction_errors = self.predict(X)
        anomaly_labels = (reconstruction_errors > threshold).astype(int)
        return anomaly_labels
    
    def get_params(self) -> Dict[str, Any]:
        """Параметри моделі"""
        return {
            'n_components': self.n_components,
            'contamination': self.contamination,
            'reconstruction_threshold': self.reconstruction_threshold,
            'is_trained': self.is_trained,
            'has_tensorflow_model': self.model is not None,
            'has_fallback_model': self.fallback_model is not None
        }

# Зберігаємо стару функцію для сумісності
def train_autoencoder_model(X_train, y_train=None, task="reconstruction"):
    """Стара функція для сумісності"""
    try:
        model = AutoencoderModel()
        model.fit(X_train, y_train)
        logger.info("OK Autoencoder model trained (compatibility function)")
        return model
        
    except Exception as e:
        logger.error(f"Error training Autoencoder model: {e}")
        return None

# models/autoencoder_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from models.deep_predict import predict_autoencoder

def train_autoencoder_model(X_train, y_train=None, task="reconstruction"):
    """
    Autoencoder for 3D data (samples, window_size, n_features).
    Використовує Conv1D for роботи with часовими рядами.
    """

    window_size = X_train.shape[1]
    n_features = X_train.shape[2]

    inputs = layers.Input(shape=(window_size, n_features))

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
    decoded = layers.Conv1D(n_features, kernel_size=3, activation="linear", padding="same")(x)

    # --- Вирandвнювання довжини виходу ---
    out_len = decoded.shape[1]
    if out_len is not None:
        if out_len > window_size:
            decoded = layers.Cropping1D(cropping=(0, out_len - window_size))(decoded)
        elif out_len < window_size:
            decoded = layers.ZeroPadding1D(padding=(0, window_size - out_len))(decoded)

    model = models.Model(inputs, decoded)
    model.compile(optimizer="adam", loss="mse")

    # Навчаємо реконструкцandю
    if len(X_train) < 50:
        model.fit(X_train, X_train, epochs=1, batch_size=8, verbose=0)
    else:
        model.fit(X_train, X_train, epochs=10, batch_size=16, verbose=0)

    # Обгортка for сумandсностand
    class AERegressorWrapper:
        def __init__(self, ae):
            self.ae = ae
        def predict(self, X, return_proba=False):
            return predict_autoencoder(self.ae, X, return_proba=return_proba)
        @property
        def feature_importances_(self):
            return None

    return AERegressorWrapper(model)