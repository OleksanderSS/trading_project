# models/models_train.py

import os
import joblib
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from utils.logger import ProjectLogger
from models.lstm_model import LSTMModel  # train_lstm_pipeline not available
# from models.mlp_model import MLPRegressor, train_mlp  # Temporarily disabled
# from models.tree_models import (
#     train_random_forest,
#     train_catboost,
#     train_xgboost,
#     train_lightgbm
# )  # Temporarily disabled
from models.bayesian_optimizer import BayesianOptimizer, optimize_lgbm_params
# from models.dean_integration import get_dean_integrator  # Temporarily disabled
from models.sentiment_integration import get_sentiment_integrator
from typing import Optional, List

logger = ProjectLogger.get_logger(__name__)
MODEL_SAVE_PATH = "models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# -------------------------------
# Масшandбування data
# -------------------------------
def scale_data(X: np.ndarray, y: np.ndarray):
    if X.size == 0 or y.size == 0:
        raise ValueError("Порожнand данand for масшandбування")
    if X.shape[0] < 50:
        raise ValueError("Недосandтньо рядкandв for тренування моwhereлей")

    X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0), posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=np.nanmedian(y), posinf=0.0, neginf=0.0)

    variances = np.var(X, axis=0)
    non_zero_indices = np.where(variances > 1e-6)[0]
    if len(non_zero_indices) < X.shape[1]:
        logger.warning(f"[ERROR] Видалено {X.shape[1] - len(non_zero_indices)} оwithнак with нульовою дисперсandєю")
        X = X[:, non_zero_indices]

    if X.shape[1] == 0:
        raise ValueError("Всand оwithнаки мали нульову дисперсandю")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X.astype(np.float32))
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1).astype(np.float32))

    logger.info(f"[OK] Данand масшandбованand: {X.shape[0]} withраwithкandв, {X.shape[1]} оwithнак.")
    return X_scaled, y_scaled, scaler_X, scaler_y

# -------------------------------
# Вибandр оwithнак
# -------------------------------
def select_features(X: np.ndarray, y: np.ndarray, max_features: int = 20):
    if X.shape[1] == 0:
        logger.warning("Немає оwithнак for вибору пandсля scale_data!")
        return np.array([])

    k = min(max_features, X.shape[1])
    if k == X.shape[1]:
        return np.arange(X.shape[1])

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y.ravel())
    selected = selector.get_support(indices=True)
    logger.info(f"[OK] Вибрано {len(selected)} оwithнак andwith {X.shape[1]}")
    return selected

# -------------------------------
# Тренування LSTM
# -------------------------------
def train_lstm_model(X_scaled, y_scaled, seq_len: int = 10):
    if X_scaled.shape[0] <= seq_len:
        logger.warning(f"LSTM: notдосandтньо рядкandв ({X_scaled.shape[0]}) for seq_len={seq_len}")
        seq_len = max(1, X_scaled.shape[0] // 2)

    if X_scaled.shape[1] == 0:
        logger.warning("LSTM: Немає оwithнак for тренування.")
        return None

    lstm = LSTMModel(input_size=X_scaled.shape[1], hidden_size=64, output_size=1)
    try:
        lstm, val_loss = train_lstm(
            lstm, X_scaled, y_scaled,
            seq_len=seq_len, batch_size=32,
            validation_split=0.2, early_stopping=10, epochs=50
        )
        joblib.dump(lstm, os.path.join(MODEL_SAVE_PATH, "lstm_model.pkl"))
        logger.info(f"[OK] LSTM натренована. Best val loss: {val_loss:.6f}")
        return lstm
    except Exception as e:
        logger.exception(f"[ERROR] Error тренування LSTM: {e}")
        return None

# -------------------------------
# Тренування MLP
# -------------------------------
def train_mlp_model(X_scaled, y_scaled):
    if X_scaled.shape[1] == 0:
        logger.warning("MLP: Немає оwithнак for тренування.")
        return None

    try:
        model, val_loss = train_mlp(
            X_train=X_scaled, y_train=y_scaled,
            input_size=X_scaled.shape[1],
            epochs=50, lr=0.001, batch_size=32,
            validation_split=0.2, early_stopping=10
        )
        joblib.dump(model, os.path.join(MODEL_SAVE_PATH, "mlp_model.pkl"))
        logger.info(f"[OK] MLP успandшно натренований. Best val loss: {val_loss:.6f}")
        return model
    except Exception as e:
        logger.exception(f"[ERROR] Error тренування MLP: {e}")
        return None

# -------------------------------
# Інandцandалandforцandя allх моwhereлей
# -------------------------------
from config.feature_layers import get_features_by_layer

def initialize_models(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 10,
    max_features: int = 20,
    feature_layers: Optional[List[str]] = None
) -> Dict[str, Any]:
    results = {
        'models': {},
        'failed_models': [],
        'feature_scaler': None,
        'target_scaler': None,
        'features': [],
        'feature_layers': feature_layers or []
    }

    # [BRAIN] Логування шарandв
    if feature_layers:
        ProjectLogger.log_feature_layers(feature_layers)

    try:
        X_scaled, y_scaled, scaler_X, scaler_y = scale_data(X, y)
        results['feature_scaler'] = scaler_X
        results['target_scaler'] = scaler_y
    except Exception as e:
        logger.exception(f"[ERROR] Error масшandбування: {e}")
        raise

    # [BRAIN] Вибandр фandчей по шарах
    if feature_layers:
        selected_feature_names = [
            f for layer in feature_layers
            for f in get_features_by_layer(layer)
        ]
        logger.info(f"[BRAIN] Вибрано фandчand по шарах: {selected_feature_names}")
        df_features = pd.DataFrame(X_scaled, columns=selected_feature_names)
        X_scaled = df_features[selected_feature_names].values
        results['features'] = selected_feature_names
    else:
        selected = select_features(X_scaled, y_scaled, max_features=max_features)
        X_scaled = X_scaled[:, selected]
        results['features'] = selected.tolist()

    #  LSTM
    lstm_model = train_lstm_model(X_scaled, y_scaled, seq_len=seq_len)
    if lstm_model:
        results['models']['lstm_model'] = lstm_model
    else:
        results['failed_models'].append('lstm_model')

    #  MLP
    mlp_model = train_mlp_model(X_scaled, y_scaled)
    if mlp_model:
        results['models']['mlp_model'] = mlp_model
    else:
        results['failed_models'].append('mlp_model')

    #  Tree-based models
    tree_funcs = {
        'random_forest': train_random_forest,
        'catboost': train_catboost,
        'xgboost': train_xgboost,
        'lightgbm': train_lightgbm
    }

    if X_scaled.shape[1] == 0:
        logger.error("Немає оwithнак for whereревяних моwhereлей!")
        results['failed_models'].extend(tree_funcs.keys())
    else:
        for name, func in tree_funcs.items():
            try:
                model = func(X_scaled, y_scaled)
                joblib.dump(model, os.path.join(MODEL_SAVE_PATH, f"{name}_model.pkl"))
                results['models'][name] = model
                logger.info(f"[OK] {name} успandшно натренований.")
            except Exception as e:
                logger.exception(f"[ERROR] Error тренування {name}: {e}")
                results['failed_models'].append(name)

    joblib.dump(scaler_X, os.path.join(MODEL_SAVE_PATH, "feature_scaler.pkl"))
    joblib.dump(scaler_y, os.path.join(MODEL_SAVE_PATH, "target_scaler.pkl"))

    logger.info(f" Інandцandалandforцandя forвершена. Успandшнand: {list(results['models'].keys())}, "
                f"Проваленand: {results['failed_models']}")

    return results

# -------------------------------
# Bayesian Optimization
# -------------------------------
def train_with_bayesian_optimization(X, y, model_type='lgbm'):
    """Тренування з байєсівською оптимізацією гіперпараметрів"""
    try:
        if model_type == 'lgbm':
            best_params = optimize_lgbm_params(X, y)
            if best_params:
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(**best_params)
                model.fit(X, y)
                joblib.dump(model, os.path.join(MODEL_SAVE_PATH, "lgbm_bayesian_model.pkl"))
                logger.info(f"[OK] LGBM з байєсівською оптимізацією натренований")
                return model
        return None
    except Exception as e:
        logger.error(f"[ERROR] Помилка байєсівської оптимізації: {e}")
        return None

# -------------------------------
# Dean Models Training
# -------------------------------
def train_dean_models(training_data: Dict[str, Any]) -> Dict[str, float]:
    """Тренування Dean RL моделей"""
    try:
        dean_integrator = get_dean_integrator()
        improvements = dean_integrator.train_models(training_data)
        logger.info(f"[DEAN] Тренування Dean моделей завершено: {improvements}")
        return improvements
    except Exception as e:
        logger.error(f"[ERROR] Помилка тренування Dean моделей: {e}")
        return {'error': str(e)}

# -------------------------------
# Sentiment Models Training
# -------------------------------
def train_sentiment_models(news_data: pd.DataFrame) -> Dict[str, Any]:
    """Тренування sentiment моделей"""
    try:
        sentiment_integrator = get_sentiment_integrator()
        
        # Ініціалізація sentiment pipeline
        if not sentiment_integrator.initialize():
            return {'error': 'Failed to initialize sentiment models'}
        
        # Аналізуємо сентимент
        if 'title' in news_data.columns:
            texts = news_data['title'].fillna('').tolist()
        else:
            texts = news_data['text'].fillna('').tolist()
        
        texts = [text for text in texts if text.strip()]
        
        if texts:
            sentiment_df = sentiment_integrator.analyze_news_sentiment(texts)
            logger.info(f"[SENTIMENT] Проаналізовано {len(texts)} новин для тренування")
            
            return {
                'sentiment_trained': True,
                'texts_analyzed': len(texts),
                'sentiment_distribution': sentiment_df['label'].value_counts().to_dict(),
                'average_confidence': sentiment_df['score'].mean()
            }
        else:
            return {'error': 'No valid texts for sentiment analysis'}
            
    except Exception as e:
        logger.error(f"[ERROR] Помилка тренування sentiment моделей: {e}")
        return {'error': str(e)}

# -------------------------------
# Enhanced Training with All Models
# -------------------------------
def train_all_models_enhanced(X, y, seq_len=10, max_features=20, use_bayesian=True, use_dean=True, use_sentiment=False, news_data=None):
    """Розширене тренування всіх моделей"""
    results = initialize_models(X, y, seq_len, max_features)
    
    # Bayesian Optimization
    if use_bayesian and X.shape[1] > 0:
        try:
            bayesian_model = train_with_bayesian_optimization(X, y, 'lgbm')
            if bayesian_model:
                results['models']['lgbm_bayesian'] = bayesian_model
        except Exception as e:
            logger.warning(f"[WARN] Bayesian optimization failed: {e}")
    
    # Dean Models
    if use_dean:
        try:
            # Підготовка data для Dean моделей
            dean_training_data = {
                'trade_results': [],  # Тут мають бути реальні результати торгів
                'simulation_results': [],
                'market_data': {'X': X, 'y': y}
            }
            dean_improvements = train_dean_models(dean_training_data)
            results['dean_improvements'] = dean_improvements
        except Exception as e:
            logger.warning(f"[WARN] Dean models training failed: {e}")
    
    # Sentiment Models
    if use_sentiment and news_data is not None:
        try:
            sentiment_results = train_sentiment_models(news_data)
            results['sentiment_results'] = sentiment_results
        except Exception as e:
            logger.warning(f"[WARN] Sentiment models training failed: {e}")
    
    return results

# -------------------------------
# Унandверсальна обгортка for ModelManager
# -------------------------------
def train(model, X, y, **kwargs):
    """Унandверсальна функцandя for моwhereлей"""
    if hasattr(model, "fit"):
        return model.fit(X, y, **kwargs)
    elif isinstance(model, LSTMModel):
        return train_lstm(model, X, y, **kwargs)
    elif isinstance(model, MLPRegressor):
        model, _ = train_mlp(X, y, input_size=X.shape[1], **kwargs)
        return model
    else:
        raise ValueError(f"[ERROR] Невandдомий тип моwhereлand: {type(model).__name__}")