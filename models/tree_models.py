# models/tree_models.py

import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

logger = logging.getLogger(__name__)

# --------------------
# Пandдготовка data
# --------------------
def _prepare_Xy(X, y, selected_features=None):
    if X is None or y is None or len(X) == 0:
        raise ValueError("X or y порожнand")

    # Очистка дублandкатandв колонок
    if hasattr(X, "columns"):
        X = X.loc[:, ~X.columns.duplicated()]

    X_safe = np.nan_to_num(np.asarray(X), nan=0.0, posinf=0.0, neginf=0.0)
    y_safe = np.nan_to_num(np.asarray(y), nan=0.0, posinf=0.0, neginf=0.0).ravel()

    if selected_features is not None:
        X_safe = X_safe[:, selected_features]

    return X_safe, y_safe

# --------------------
# Random Forest
# --------------------
def train_rf_regressor(X, y, selected_features=None):
    X_safe, y_safe = _prepare_Xy(X, y, selected_features)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_safe, y_safe)
    logger.info("[OK] RandomForestRegressor натренований")
    return model

def train_rf_classifier(X, y, selected_features=None):
    X_safe, y_safe = _prepare_Xy(X, y, selected_features)
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_safe, y_safe)
    logger.info("[OK] RandomForestClassifier натренований")
    return model

# --------------------
# CatBoost
# --------------------
def train_catboost_regressor(X, y, selected_features=None):
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise ImportError("[ERROR] CatBoost not всandновлено. Всandнови череwith: pip install catboost")

    X_safe, y_safe = _prepare_Xy(X, y, selected_features)
    model = CatBoostRegressor(
        iterations=200, depth=6, learning_rate=0.05,
        loss_function='RMSE', verbose=0, random_seed=42
    )
    model.fit(X_safe, y_safe)
    logger.info("[OK] CatBoostRegressor натренований")
    return model

def train_catboost_classifier(X, y, selected_features=None):
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError("[ERROR] CatBoost not всandновлено. Всandнови череwith: pip install catboost")

    X_safe, y_safe = _prepare_Xy(X, y, selected_features)
    model = CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.05,
        loss_function='Logloss', verbose=0, random_seed=42
    )
    model.fit(X_safe, y_safe)
    logger.info("[OK] CatBoostClassifier натренований")
    return model

# --------------------
# XGBoost
# --------------------
def train_xgb_regressor(X, y, selected_features=None):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("[ERROR] XGBoost not всandновлено. Всandнови череwith: pip install xgboost")

    X_safe, y_safe = _prepare_Xy(X, y, selected_features)
    model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1, tree_method='hist'
    )
    model.fit(X_safe, y_safe)
    logger.info("[OK] XGBRegressor натренований")
    return model

def train_xgb_classifier(X, y, selected_features=None):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("[ERROR] XGBoost not всandновлено. Всandнови череwith: pip install xgboost")

    X_safe, y_safe = _prepare_Xy(X, y, selected_features)
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1, tree_method='hist',
        use_label_encoder=False, eval_metric="logloss"
    )
    model.fit(X_safe, y_safe)
    logger.info("[OK] XGBClassifier натренований")
    return model

# --------------------
# LightGBM
# --------------------
def train_lgbm_regressor(X, y, selected_features=None, **kwargs):
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError("[ERROR] LightGBM not всandновлено. Всandнови череwith: pip install lightgbm")

    X_safe, y_safe = _prepare_Xy(X, y, selected_features)
    model = LGBMRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=-1, random_state=42, **kwargs
    )
    model.fit(X_safe, y_safe)
    logger.info("[OK] LGBMRegressor натренований with параметрами: %s", kwargs)
    return model

def train_lgbm_classifier(X, y, selected_features=None, **kwargs):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("[ERROR] LightGBM not всandновлено. Всandнови череwith: pip install lightgbm")

    X_safe, y_safe = _prepare_Xy(X, y, selected_features)
    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=-1, class_weight="balanced",
        random_state=42, **kwargs
    )
    model.fit(X_safe, y_safe)
    logger.info("[OK] LGBMClassifier натренований with параметрами: %s", kwargs)
    return model

def train_multi_target_regressor(X, y, selected_features=None):
    from sklearn.multioutput import MultiOutputRegressor
    X_safe, y_safe = _prepare_Xy(X, y, selected_features)
    model = MultiOutputRegressor(LGBMRegressor(
        n_estimators=300, learning_rate=0.05, random_state=42
    ))
    model.fit(X_safe, y_safe)
    return model

def train_confidence_models(X, y, model_type="lgbm"):
    """Навчає confidence score for будь-якої моwhereлand"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    
    X_safe, y_safe = _prepare_Xy(X, y)
    
    if model_type == "rf":
        confidence_model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_type == "xgb":
        from xgboost import XGBRegressor
        confidence_model = XGBRegressor(n_estimators=200, random_state=42)
    elif model_type == "catboost":
        from catboost import CatBoostRegressor
        confidence_model = CatBoostRegressor(iterations=200, verbose=0, random_state=42)
    else:  # lgbm for forмовчуванням
        from lightgbm import LGBMRegressor
        confidence_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    
    # Навчаємо на абсолютних errorх
    y_errors = np.abs(y_safe)
    confidence_model.fit(X_safe, y_errors)
    
    return confidence_model

def predict_with_confidence(model, confidence_model, X):
    """Прогноwith with confidence score"""
    predictions = model.predict(X)
    confidence_scores = confidence_model.predict(X)
    
    # Чим нижча confidence_score, тим вища впевnotнandсть
    confidence_normalized = 1.0 / (1.0 + confidence_scores)
    
    return {
        "predictions": predictions,
        "confidence": confidence_normalized,
        "uncertainty": confidence_scores
    }

# --------------------
# Реєстр моwhereлей
# --------------------
MODEL_REGISTRY = {
    "rf_regressor": train_rf_regressor,
    "rf_classifier": train_rf_classifier,
    "catboost_regressor": train_catboost_regressor,
    "catboost_classifier": train_catboost_classifier,
    "xgb_regressor": train_xgb_regressor,
    "xgb_classifier": train_xgb_classifier,
    "lgbm_regressor": train_lgbm_regressor,
    "lgbm_classifier": train_lgbm_classifier,
    "multi_target_regressor": train_multi_target_regressor,
}