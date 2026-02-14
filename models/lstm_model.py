# models/lstm_model.py

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM модель для часових рядів з fallback на sklearn"""
    
    def __init__(self, input_size: int = None, hidden_size: int = 64, output_size: int = 1,
                 num_layers: int = 1, dropout: float = 0.2, classification: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
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
            if self.classification:
                from sklearn.ensemble import RandomForestClassifier
                self.fallback_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                self.fallback_model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10
                )
            
            logger.info("OK Created fallback RandomForest model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create fallback model: {e}")
            return False
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None, seq_len: int = 10) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Створити послідовності для LSTM"""
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X має бути 2D і мати хоча б одну ознаку")
        
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i + seq_len])
            if y is not None:
                y_seq.append(y[i + seq_len])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
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
            
            # Спроба використати PyTorch LSTM
            try:
                self._fit_pytorch_lstm(X, y, seq_len, epochs, batch_size)
            except ImportError:
                logger.warning("PyTorch not available, using fallback model")
                self._fit_fallback(X, y)
                
            self.is_trained = True
            logger.info(f"OK LSTM model trained (classification: {self.classification})")
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            # Fallback до простої моделі
            try:
                self._fit_fallback(X, y)
                self.is_trained = True
                logger.info("OK Used fallback model")
            except Exception as e2:
                logger.error(f"Fallback training also failed: {e2}")
                raise
    
    def _fit_pytorch_lstm(self, X, y, seq_len: int, epochs: int, batch_size: int):
        """Тренування PyTorch LSTM"""
        import torch
        import torch.nn as nn
        
        # Створення послідовностей
        X_seq, y_seq = self._create_sequences(X, y, seq_len)
        if len(X_seq) == 0:
            raise ValueError("Not enough data for sequences")
        
        # Розподіл на train/val
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Створення моделі
        self.input_size = X.shape[1]
        self.model = self._create_lstm_model()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Loss та optimizer
        criterion = nn.BCELoss() if self.classification else nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Тренування
        for epoch in range(epochs):
            self.model.train()
            
            # Batch training
            for i in range(0, len(X_train), batch_size):
                X_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32).to(device)
                y_batch = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32).reshape(-1, 1).to(device)
                
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
            
            # Validation
            if epoch % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
                    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)
                    val_loss = criterion(self.model(X_val_tensor), y_val_tensor).item()
                    logger.info(f"LSTM Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.6f}")
    
    def _create_lstm_model(self):
        """Створити PyTorch LSTM модель"""
        import torch.nn as nn
        
        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, classification):
                super().__init__()
                self.classification = classification
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                                   batch_first=True, dropout=dropout if num_layers > 1 else 0)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, output_size)
                if classification:
                    self.activation = nn.Sigmoid()
                else:
                    self.activation = nn.Identity()
            
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                return self.activation(out)
        
        return LSTMNet(self.input_size, self.hidden_size, self.output_size, 
                      self.num_layers, self.dropout, self.classification)
    
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
            
            # Спроба PyTorch
            if self.model is not None:
                return self._predict_pytorch(X, seq_len)
            else:
                return self._predict_fallback(X)
                
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            # Fallback
            try:
                return self._predict_fallback(X)
            except Exception as e2:
                logger.error(f"Fallback prediction also failed: {e2}")
                raise
    
    def _predict_pytorch(self, X, seq_len: int):
        """Прогнозування PyTorch"""
        import torch
        
        X_seq, _ = self._create_sequences(X, None, seq_len)
        if len(X_seq) == 0:
            raise ValueError("Not enough data for prediction")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy()
    
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
        
        # Конвертація в ймовірності (проста емпірична формула)
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
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'classification': self.classification,
            'is_trained': self.is_trained,
            'has_pytorch_model': self.model is not None,
            'has_fallback_model': self.fallback_model is not None
        }