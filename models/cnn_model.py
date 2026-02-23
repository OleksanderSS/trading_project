# models/cnn_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from utils.logger_fixed import ProjectLogger
from models.deep_predict import predict_cnn
from models.data_preparation import prepare_data_for_models

logger = ProjectLogger.get_logger(__name__)

def train_cnn_model(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    task: str = "regression",
    epochs: int = 40,
    batch_size: int = 32,
    random_state: int = 42
):
    """
    CNN with унandфandкованою пandдготовкою data
    """
    try:
        # Пandдготовка data
        data = prepare_data_for_models(df, ticker, timeframe, seq_len=10)
        if not data:
            logger.error(f"[ERROR] Не вдалося пandдготувати данand for {ticker} {timeframe}")
            return None
        
        # Перевandрка валandдностand
        if not validate_data_shapes(data):
            logger.error(f"[ERROR] Невалandднand роwithмandри data for {ticker} {timeframe}")
            return None
        
        heavy_data = data['heavy_models']
        X_train = heavy_data['X_train']
        y_train = heavy_data['y_train']
        
        # Фandксуємо seed for вandдтворюваностand
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        # Перетворюємо у numpy and forмandняємо NaN
        Xn = np.asarray(X_train, dtype=np.float32)
        Xn = np.nan_to_num(Xn)
        
        # Нормалandforцandя data for CNN
        Xn_mean = np.mean(Xn, axis=0)
        Xn_std = np.std(Xn, axis=0)
        Xn_std[Xn_std == 0] = 1  # Запобandгаємо дandленню на нуль
        Xn = (Xn - Xn_mean) / Xn_std
        
        # Нормалandforцandя y_train
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        if y_std > 0:
            y_train_norm = (y_train - y_mean) / y_std
        else:
            y_train_norm = y_train
        
        timesteps = Xn.shape[1]    # довжина вandкна
        n_features = Xn.shape[2]   # кandлькandсть фandч
        
        # Архandтектура CNN
        model = Sequential([
            Input(shape=(timesteps, n_features)),
            Conv1D(filters=64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="linear") if task == "regression" else Dense(2, activation="softmax"),
        ])

        # Компandляцandя моwhereлand
        if task == "regression":
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        else:
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # Навчання
        model.fit(Xn, y_train_norm, epochs=epochs, batch_size=batch_size, verbose=0)
        
        logger.info(f"[OK] CNN натренований for {ticker} {timeframe}")
        logger.info(f"   Форма data: {Xn.shape} -> {y_train_norm.shape}")

        # Обгортка for прогноwithandв
        class CNNWrapper:
            def __init__(self, model, data_info):
                self.model = model
                self.data_info = data_info

            def predict(self, X, return_proba=False):
                """Унandфandкований predict череwith deep_predict"""
                return predict_cnn(self.model, X, return_proba=return_proba)

        return CNNWrapper(model, data)
        
    except Exception as e:
        logger.error(f"[ERROR] Error тренування CNN {ticker} {timeframe}: {e}")
        return None

def validate_data_shapes(data: dict) -> bool:
    """Перевandряє валandднandсть роwithмandрandв data"""
    if not data or 'heavy_models' not in data:
        return False
    
    heavy = data['heavy_models']
    X_train = heavy.get('X_train')
    y_train = heavy.get('y_train')
    
    if X_train is None or y_train is None:
        return False
    
    if len(X_train) != len(y_train):
        logger.warning(f"[WARN] Роwithмandри not спandвпадають: X={len(X_train)}, y={len(y_train)}")
        return False
    
    if len(X_train) < 10:
        logger.warning(f"[WARN] Замало data: {len(X_train)}")
        return False
    
    return True