# models/transformer_model.py

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TransformerModel:
    """Transformer модель з fallback на sklearn"""
    
    def __init__(self, input_size: int = None, num_heads: int = 4, ff_dim: int = 64, 
                 dropout: float = 0.1, classification: bool = True):
        self.input_size = input_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.classification = classification
        self.is_trained = False
        self.model = None
        self.scaler = StandardScaler()
        
        # Fallback модель
        self.fallback_model = None
        
    def _create_fallback_model(self):
        """Створити fallback модель на основі sklearn"""
        try:
            self.fallback_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            ) if self.classification else RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            logger.info("OK Created fallback RandomForest model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create fallback model: {e}")
            return False
    
    def _create_sequences(self, X: np.ndarray, seq_len: int = 10) -> np.ndarray:
        """Створити послідовності для Transformer"""
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X має бути 2D і мати хоча б одну ознаку")
        
        X_seq = []
        for i in range(len(X) - seq_len + 1):
            X_seq.append(X[i:i + seq_len])
        
        return np.array(X_seq)
    
    def fit(self, X, y, seq_len: int = 10, epochs: int = 20, batch_size: int = 32):
        """Тренування моделі"""
        try:
            # Конвертація в numpy
            if hasattr(X, 'values'):
                X = X.values
            if hasattr(y, 'values'):
                y = y.values
                
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Спроба використати TensorFlow/Keras
            try:
                self._fit_tensorflow_transformer(X, y, seq_len, epochs, batch_size)
            except ImportError:
                logger.warning("TensorFlow not available, using fallback model")
                self._fit_fallback(X, y)
                
            self.is_trained = True
            logger.info(f"OK Transformer model trained (classification: {self.classification})")
            
        except Exception as e:
            logger.error(f"Transformer training failed: {e}")
            # Fallback до простої моделі
            try:
                self._fit_fallback(X, y)
                self.is_trained = True
                logger.info("OK Used fallback model")
            except Exception as e2:
                logger.error(f"Fallback training also failed: {e2}")
                raise
    
    def _fit_tensorflow_transformer(self, X, y, seq_len: int, epochs: int, batch_size: int):
        """Тренування TensorFlow Transformer"""
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
        from tensorflow.keras.layers import MultiHeadAttention, Flatten
        from tensorflow.keras.optimizers import Adam
        
        # Створення послідовностей
        X_seq = self._create_sequences(X, seq_len)
        if len(X_seq) == 0:
            raise ValueError("Not enough data for sequences")
        
        # Вирівнювання довжини y
        y_seq = y[seq_len-1:seq_len-1+len(X_seq)]
        
        # Розподіл на train/val
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Створення моделі
        self.input_size = X.shape[1]
        self.model = self._create_transformer_model()
        
        # Тренування
        loss = "binary_crossentropy" if self.classification else "mse"
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=["mae"])
        
        # Тренування з валідацією
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=0
        )
        
        # Логування результатів
        final_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        logger.info(f"Transformer training completed - Loss: {final_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    def _create_transformer_model(self):
        """Створити TensorFlow Transformer модель"""
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
        from tensorflow.keras.layers import MultiHeadAttention, Flatten
        
        # Архітектура Transformer
        inputs = Input(shape=(None, self.input_size))  # seq_len буде динамічним
        
        # Multi-Head Attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.input_size
        )(inputs, inputs)
        attention_output = Dropout(self.dropout)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)
        
        # Feed Forward
        ff_output = Dense(self.ff_dim, activation="relu")(attention_output)
        ff_output = Dense(self.input_size)(ff_output)
        ff_output = Dropout(self.dropout)(ff_output)
        ff_output = LayerNormalization(epsilon=1e-6)(ff_output + attention_output)
        
        # Global Average Pooling замість Flatten
        x = tf.reduce_mean(ff_output, axis=1)  # Global average pooling
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.3)(x)
        
        # Вихідний шар
        if self.classification:
            outputs = Dense(1, activation="sigmoid")(x)
        else:
            outputs = Dense(1, activation="linear")(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _fit_fallback(self, X, y):
        """Тренування fallback моделі"""
        if self.fallback_model is None:
            if not self._create_fallback_model():
                raise RuntimeError("Cannot create fallback model")
        
        # Використовуємо останні значення для тренування
        if len(X) > 10:
            X_train = X[-min(len(X), 100):]  # Останні 100 точок
            y_train = y[-min(len(y), 100):]
        else:
            X_train = X
            y_train = y
        
        self.fallback_model.fit(X_train, y_train)
        logger.info("OK Fallback model trained")
    
    def predict(self, X, seq_len: int = 10):
        """Прогнозування"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Конвертація в numpy
            if hasattr(X, 'values'):
                X = X.values
                
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Спроба TensorFlow
            if self.model is not None:
                return self._predict_tensorflow(X, seq_len)
            else:
                return self._predict_fallback(X)
                
        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}")
            # Fallback
            try:
                return self._predict_fallback(X)
            except Exception as e2:
                logger.error(f"Fallback prediction also failed: {e2}")
                raise
    
    def _predict_tensorflow(self, X, seq_len: int):
        """Прогнозування TensorFlow"""
        import tensorflow as tf
        
        X_seq = self._create_sequences(X, seq_len)
        if len(X_seq) == 0:
            raise ValueError("Not enough data for prediction")
        
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions
    
    def _predict_fallback(self, X):
        """Прогнозування fallback"""
        if self.fallback_model is None:
            raise RuntimeError("No fallback model available")
        
        # Використовуємо останні значення для прогнозу
        if len(X.shape) == 2:
            return self.fallback_model.predict(X[-1:].reshape(1, -1))
        else:
            return self.fallback_model.predict(X.reshape(1, -1))
    
    def predict_proba(self, X, seq_len: int = 10):
        """Прогнозування ймовірностей"""
        if not self.classification:
            raise ValueError("predict_proba only available for classification")
        
        predictions = self.predict(X, seq_len)
        
        # Конвертація в ймовірності
        if len(predictions.shape) == 1:
            probas = np.zeros((len(predictions), 2))
            probas[:, 1] = predictions
            probas[:, 0] = 1 - predictions
            return probas
        else:
            return predictions
    
    def get_params(self) -> Dict[str, Any]:
        """Параметри моделі"""
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

# Зберігаємо стару функцію для сумісності
def train_transformer_model(df: pd.DataFrame, ticker: str, timeframe: str, 
                           task: str = "regression", epochs: int = 20, batch_size: int = 16,
                           num_heads: int = 4, ff_dim: int = 64):
    """Стара функція для сумісності"""
    try:
        # Підготовка data
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        X = df[feature_cols].fillna(0).values
        y = (df['Close'].shift(-1) > df['Close']).astype(int) if task == "classification" else df['Close'].shift(-1)
        
        # Видаляємо NaN з y
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 20:
            logger.warning(f"Not enough data for {ticker} {timeframe}")
            return None
        
        # Створення моделі
        model = TransformerModel(
            input_size=X.shape[1],
            num_heads=num_heads,
            ff_dim=ff_dim,
            classification=(task == "classification")
        )
        
        # Тренування
        model.fit(X, y, epochs=epochs, batch_size=batch_size)
        
        logger.info(f"OK Transformer trained for {ticker} {timeframe}")
        return model
        
    except Exception as e:
        logger.error(f"Error training Transformer {ticker} {timeframe}: {e}")
        return None