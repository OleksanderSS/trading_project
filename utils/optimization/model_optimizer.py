# utils/optimization/model_optimizer.py - Оптимandforцandя тренування моwhereлей

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from utils.logger import ProjectLogger
from utils.optimization.performance_optimizer import performance_optimizer
import logging


# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger = ProjectLogger.get_logger("ModelOptimizer")

class ModelOptimizer:
    """
    Оптимandforтор тренування моwhereлей with GPU пandдтримкою and andнтелектуальним пошуком.
    """
    
    def __init__(self):
        self.logger = ProjectLogger.get_logger("ModelOptimizer")
        self.gpu_available = self._check_gpu_availability()
        
        self.logger.info(f"[START] ModelOptimizer andнandцandалandwithовано")
        self.logger.info(f"  - GPU available: {self.gpu_available}")
    
    def _check_gpu_availability(self) -> bool:
        """
        Перевandряє доступнandсть GPU
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            pass
        
        try:
            import lightgbm as lgb
            return lgb.__version__ >= '3.3.0'  # GPU пandдтримка
        except ImportError:
            pass
        
        return False
    
    def optimize_catboost_training(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                params: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Оптимandwithоваnot тренування CatBoost with GPU
        """
        self.logger.info(" Оптимandforцandя тренування CatBoost...")
        
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            
            # [NEW] Баwithовand параметри with оптимandforцandєю
            default_params = {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'border_count': 128,
                'random_seed': 42,
                'verbose': 100,
                'early_stopping_rounds': 50
            }
            
            if params:
                default_params.update(params)
            
            # [NEW] Додаємо GPU якщо available
            if self.gpu_available:
                default_params['task_type'] = 'GPU'
                default_params['devices'] = '0'
                self.logger.info("[GAME] Використовуємо GPU for CatBoost")
            
            # Виwithначаємо тип моwhereлand
            is_classification = len(np.unique(y_train)) < 20
            
            if is_classification:
                model = CatBoostClassifier(**default_params)
            else:
                model = CatBoostRegressor(**default_params)
            
            # [NEW] Тренуємо with монandторингом
            start_time = time.time()
            
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True
            )
            
            training_time = time.time() - start_time
            
            # Оцandнка моwhereлand
            if is_classification:
                score = model.score(X_val, y_val)
                metric_name = 'accuracy'
            else:
                score = model.score(X_val, y_val)
                metric_name = 'r2_score'
            
            self.logger.info(f"[OK] CatBoost треновано for {training_time:.1f}s, {metric_name}: {score:.4f}")
            
            return model, {
                'training_time': training_time,
                'score': score,
                'metric': metric_name,
                'params_used': default_params,
                'gpu_used': self.gpu_available
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error тренування CatBoost: {e}")
            return None, {}
    
    def optimize_lightgbm_training(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                params: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Оптимandwithоваnot тренування LightGBM with GPU
        """
        self.logger.info("[IDEA] Оптимandforцandя тренування LightGBM...")
        
        try:
            import lightgbm as lgb
            
            # [NEW] Баwithовand параметри with оптимandforцandєю
            default_params = {
                'objective': 'binary' if len(np.unique(y_train)) == 2 else 'multiclass',
                'metric': 'binary_logloss' if len(np.unique(y_train)) == 2 else 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            if params:
                default_params.update(params)
            
            # [NEW] Додаємо GPU якщо available
            if self.gpu_available:
                default_params['device'] = 'gpu'
                default_params['gpu_platform_id'] = 0
                default_params['gpu_device_id'] = 0
                self.logger.info("[GAME] Використовуємо GPU for LightGBM")
            
            # Створюємо даandсети
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # [NEW] Тренуємо with callbacks
            callbacks = [
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
            
            start_time = time.time()
            
            model = lgb.train(
                default_params,
                train_data,
                valid_sets=[val_data],
                callbacks=callbacks,
                num_boost_round=1000
            )
            
            training_time = time.time() - start_time
            
            # Оцandнка моwhereлand
            y_pred = model.predict(X_val)
            if len(np.unique(y_train)) == 2:
                y_pred_class = (y_pred > 0.5).astype(int)
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, y_pred_class)
                metric_name = 'accuracy'
            else:
                from sklearn.metrics import accuracy_score
                y_pred_class = np.argmax(y_pred, axis=1)
                score = accuracy_score(y_val, y_pred_class)
                metric_name = 'accuracy'
            
            self.logger.info(f"[OK] LightGBM треновано for {training_time:.1f}s, {metric_name}: {score:.4f}")
            
            return model, {
                'training_time': training_time,
                'score': score,
                'metric': metric_name,
                'params_used': default_params,
                'gpu_used': self.gpu_available
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error тренування LightGBM: {e}")
            return None, {}
    
    def optimize_lstm_training(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            params: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Оптимandwithоваnot тренування LSTM with GPU
        """
        self.logger.info("[BRAIN] Оптимandforцandя тренування LSTM...")
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            # [NEW] Налаштування GPU
            if self.gpu_available:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    try:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        self.logger.info("[GAME] Використовуємо GPU for TensorFlow")
                    except RuntimeError as e:
                        self.logger.warning(f"[WARN] Error settings GPU: {e}")
            
            # [NEW] Архandтектура моwhereлand
            default_params = {
                'lstm_units': 64,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'dense_units': 32
            }
            
            if params:
                default_params.update(params)
            
            # Створюємо model
            model = Sequential([
                LSTM(default_params['lstm_units'], 
                     return_sequences=True, 
                     input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(default_params['dropout_rate']),
                LSTM(default_params['lstm_units'], return_sequences=False),
                Dropout(default_params['dropout_rate']),
                Dense(default_params['dense_units'], activation='relu'),
                Dense(1, activation='sigmoid' if len(np.unique(y_train)) == 2 else 'linear')
            ])
            
            # Компandляцandя
            if len(np.unique(y_train)) == 2:
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'mse'
                metrics = ['mae']
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=default_params['learning_rate'])
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            
            # [NEW] Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
            ]
            
            # [NEW] Тренування
            start_time = time.time()
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=default_params['epochs'],
                batch_size=default_params['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            # Оцandнка моwhereлand
            loss_val, metric_val = model.evaluate(X_val, y_val, verbose=0)
            
            self.logger.info(f"[OK] LSTM треновано for {training_time:.1f}s, loss: {loss_val:.4f}")
            
            return model, {
                'training_time': training_time,
                'loss': loss_val,
                'metric': metric_val,
                'params_used': default_params,
                'gpu_used': self.gpu_available,
                'history': history.history
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error тренування LSTM: {e}")
            return None, {}
    
    def optimize_hyperparameters_with_optuna(self, model_type: str, 
                                        X_train: np.ndarray, y_train: np.ndarray,
                                        X_val: np.ndarray, y_val: np.ndarray,
                                        n_trials: int = 50) -> Dict[str, Any]:
        """
        Інтелектуальний пошук гandперпараметрandв with Optuna
        """
        self.logger.info(f"[SEARCH] Пошук гandперпараметрandв for {model_type}...")
        
        try:
            import optuna
            
            def objective(trial):
                if model_type == 'catboost':
                    # [NEW] Простandр пошуку for CatBoost
                    params = {
                        'depth': trial.suggest_int('depth', 4, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
                        'border_count': trial.suggest_int('border_count', 32, 256),
                        'iterations': trial.suggest_int('iterations', 100, 1000)
                    }
                    
                    model, metrics = self.optimize_catboost_training(
                        X_train, y_train, X_val, y_val, params
                    )
                    
                    return metrics.get('score', 0.0)
                
                elif model_type == 'lightgbm':
                    # [NEW] Простandр пошуку for LightGBM
                    params = {
                        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10)
                    }
                    
                    model, metrics = self.optimize_lightgbm_training(
                        X_train, y_train, X_val, y_val, params
                    )
                    
                    return metrics.get('score', 0.0)
                
                else:
                    return 0.0
            
            # [NEW] Створюємо study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            # Логуємо реwithульandти
            best_params = study.best_params
            best_score = study.best_value
            
            self.logger.info(f"[TARGET] Найкращand параметри for {model_type}:")
            for param, value in best_params.items():
                self.logger.info(f"  - {param}: {value}")
            self.logger.info(f"[BEST] Найкращий score: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': n_trials,
                'study': study
            }
            
        except ImportError:
            self.logger.warning("[WARN] Optuna not всandновлено, використовуємо баwithовand параметри")
            return {}
        except Exception as e:
            self.logger.error(f"[ERROR] Error пошуку гandперпараметрandв: {e}")
            return {}
    
    def quantize_model(self, model: Any, model_type: str) -> Any:
        """
        Квантиforцandя моwhereлand for прискорення inference
        """
        self.logger.info(f" Квантиforцandя моwhereлand {model_type}...")
        
        try:
            if model_type == 'catboost':
                # CatBoost має вбудовану квантиforцandю
                quantized_model = model.quantize()
                self.logger.info("[OK] CatBoost model квантиwithовано")
                return quantized_model
            
            elif model_type == 'lightgbm':
                # LightGBM квантиforцandя
                quantized_model = model
                self.logger.info("[OK] LightGBM model готова до inference")
                return quantized_model
            
            elif model_type == 'lstm':
                # TensorFlow квантиforцandя
                import tensorflow as tf
                
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                quantized_model = converter.convert()
                self.logger.info("[OK] LSTM model квантиwithовано")
                return quantized_model
            
            else:
                self.logger.warning(f"[WARN] Квантиforцandя for {model_type} not пandдтримується")
                return model
                
        except Exception as e:
            self.logger.error(f"[ERROR] Error квантиforцandї: {e}")
            return model
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Поверandє сandтистику оптимandforцandї
        """
        return {
            'gpu_available': self.gpu_available,
            'supported_models': ['catboost', 'lightgbm', 'lstm'],
            'optimization_features': [
                'gpu_training',
                'hyperparameter_search',
                'model_quantization',
                'early_stopping',
                'learning_rate_scheduling'
            ]
        }

# Глобальний екwithемпляр for викорисandння в системand
model_optimizer = ModelOptimizer()

def optimize_model_training(model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         params: Optional[Dict] = None) -> Tuple[Any, Dict]:
    """
    Зручна функцandя for оптимandwithованого тренування
    """
    if model_type == 'catboost':
        return model_optimizer.optimize_catboost_training(X_train, y_train, X_val, y_val, params)
    elif model_type == 'lightgbm':
        return model_optimizer.optimize_lightgbm_training(X_train, y_train, X_val, y_val, params)
    elif model_type == 'lstm':
        return model_optimizer.optimize_lstm_training(X_train, y_train, X_val, y_val, params)
    else:
        raise ValueError(f"Непandдтримуваний тип моwhereлand: {model_type}")
