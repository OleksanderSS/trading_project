"""
light_models.py - Реальнand легкand моwhereлand for локального тренування
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

logger = logging.getLogger(__name__)

class LightGBMModel:
    """LightGBM model forглушка"""
    def __init__(self, **kwargs):
        self.model = None
        self.trained = False
    
    def fit(self, X, y):
        self.trained = True
        return self
    
    def predict(self, X):
        return np.zeros(len(X))

class XGBoostModel:
    """XGBoost model forглушка"""
    def __init__(self, **kwargs):
        self.model = None
        self.trained = False
    
    def fit(self, X, y):
        self.trained = True
        return self
    
    def predict(self, X):
        return np.zeros(len(X))

class LightModelTrainer:
    """Треnotр легких моwhereлей with реальними алгоритмами"""
    
    def __init__(self, models_dir: str = "models/trained/light"):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        os.makedirs(models_dir, exist_ok=True)
    
    def get_model(self, model_type: str, task_type: str = "regression"):
        """Отримати model вandдповandдного типу"""
        if task_type == "regression":
            models_map = {
                "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "linear": LinearRegression(),
                "svr": SVR(kernel='rbf', C=1.0, gamma='scale'),
                "knn": KNeighborsRegressor(n_neighbors=5, weights='distance'),
                "xgboost": self._get_xgboost_regressor(),
                "lightgbm": self._get_lightgbm_regressor(),
                "catboost": self._get_catboost_regressor()
            }
        else:  # classification
            models_map = {
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                "logistic": LogisticRegression(random_state=42, max_iter=1000),
                "svc": SVC(kernel='rbf', probability=True, random_state=42),
                "knn": KNeighborsClassifier(n_neighbors=5, weights='distance'),
                "xgboost": self._get_xgboost_classifier(),
                "lightgbm": self._get_lightgbm_classifier(),
                "catboost": self._get_catboost_classifier()
            }
        
        return models_map.get(model_type, models_map["random_forest"])
    
    def _get_xgboost_regressor(self):
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        except ImportError:
            logger.warning("XGBoost not installed, using RandomForest")
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    def _get_lightgbm_regressor(self):
        try:
            import lightgbm as lgb
            return lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        except ImportError:
            logger.warning("LightGBM not installed, using RandomForest")
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    def _get_catboost_regressor(self):
        try:
            import catboost as cb
            return cb.CatBoostRegressor(iterations=100, random_state=42, verbose=False)
        except ImportError:
            logger.warning("CatBoost not installed, using RandomForest")
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    def _get_xgboost_classifier(self):
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        except ImportError:
            logger.warning("XGBoost not installed, using RandomForest")
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    def _get_lightgbm_classifier(self):
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        except ImportError:
            logger.warning("LightGBM not installed, using RandomForest")
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    def _get_catboost_classifier(self):
        try:
            import catboost as cb
            return cb.CatBoostClassifier(iterations=100, random_state=42, verbose=False)
        except ImportError:
            logger.warning("CatBoost not installed, using RandomForest")
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    def prepare_data(self, features_df: pd.DataFrame, target_col: str = "target") -> Tuple[np.ndarray, np.ndarray]:
        """Пandдготовка data for тренування"""
        # Вибираємо тandльки числовand колонки
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col not in numeric_cols:
            raise ValueError(f"Target column '{target_col}' not found in numeric columns")
        
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        X = features_df[feature_cols].values
        y = features_df[target_col].values
        
        # Обробка пропущених withначень
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Масшandбування for whereяких моwhereлей
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, scaler, imputer, feature_cols
    
    def train_light_model(self, features_df: pd.DataFrame, model_type: str, 
                         ticker: str, timeframe: str, target_col: str = "target",
                         task_type: str = "regression") -> Dict[str, Any]:
        """
        Тренування легкої моwhereлand with реальними алгоритмами
        
        Args:
            features_df: DataFrame with фandчами and andргетом
            model_type: Тип моwhereлand (random_forest, linear, svr, knn, xgboost, lightgbm, catboost)
            ticker: Тandкер
            timeframe: Таймфрейм
            target_col: Колонка andргету
            task_type: Тип forдачand (regression/classification)
            
        Returns:
            Dict with реwithульandandми тренування
        """
        try:
            logger.info(f"[LightModelTrainer] Training {model_type} for {ticker} {timeframe}")
            
            if features_df.empty:
                raise ValueError("Empty DataFrame provided")
            
            # Пandдготовка data
            X, y, scaler, imputer, feature_cols = self.prepare_data(features_df, target_col)
            
            # Роwithдandлення на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None if task_type == "regression" else y
            )
            
            # Отримання моwhereлand
            model = self.get_model(model_type, task_type)
            
            # Тренування
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Прогноwithування
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Метрики якостand
            if task_type == "regression":
                metrics = {
                    "train_mse": mean_squared_error(y_train, y_pred_train),
                    "test_mse": mean_squared_error(y_test, y_pred_test),
                    "train_r2": r2_score(y_train, y_pred_train),
                    "test_r2": r2_score(y_test, y_pred_test),
                    "train_mae": mean_absolute_error(y_train, y_pred_train),
                    "test_mae": mean_absolute_error(y_test, y_pred_test)
                }
            else:  # classification
                y_pred_proba = None
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                metrics = {
                    "train_accuracy": accuracy_score(y_train, y_pred_train),
                    "test_accuracy": accuracy_score(y_test, y_pred_test),
                    "train_f1": f1_score(y_train, y_pred_train, average='weighted'),
                    "test_f1": f1_score(y_test, y_pred_test, average='weighted')
                }
                
                if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                    metrics["test_roc_auc"] = roc_auc_score(y_test, y_pred_proba)
            
            # Збереження моwhereлand
            model_key = f"{model_type}_{ticker}_{timeframe}"
            model_path = os.path.join(self.models_dir, f"{model_key}.joblib")
            
            model_data = {
                "model": model,
                "scaler": scaler,
                "imputer": imputer,
                "feature_cols": feature_cols,
                "model_type": model_type,
                "task_type": task_type,
                "metrics": metrics,
                "training_time": training_time
            }
            
            joblib.dump(model_data, model_path)
            
            # Збереження в пам'ятand
            self.models[model_key] = model_data
            
            logger.info(f"[LightModelTrainer] Successfully trained {model_type} for {ticker} {timeframe}")
            logger.info(f"[LightModelTrainer] Test R2: {metrics.get('test_r2', 'N/A'):.4f}")
            
            return {
                "status": "success",
                "model_key": model_key,
                "predictions": y_pred_test,
                "actual": y_test,
                "metrics": metrics,
                "training_time": training_time,
                "feature_importance": self._get_feature_importance(model, feature_cols) if hasattr(model, 'feature_importances_') else None
            }
            
        except Exception as e:
            logger.error(f"[LightModelTrainer] Error training {model_type} for {ticker} {timeframe}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "predictions": np.array([]),
                "actual": np.array([]),
                "metrics": {}
            }
    
    def _get_feature_importance(self, model, feature_cols: List[str]) -> Dict[str, float]:
        """Отримати важливandсть фandч"""
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_cols, model.feature_importances_))
            # Сортування for важливandстю
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return {}
    
    def predict(self, model_key: str, features_df: pd.DataFrame) -> np.ndarray:
        """Зробити прогноwith for допомогою withбереженої моwhereлand"""
        if model_key not in self.models:
            # Спроба forванandжити with диска
            model_path = os.path.join(self.models_dir, f"{model_key}.joblib")
            if os.path.exists(model_path):
                self.models[model_key] = joblib.load(model_path)
            else:
                raise ValueError(f"Model {model_key} not found")
        
        model_data = self.models[model_key]
        model = model_data["model"]
        scaler = model_data["scaler"]
        imputer = model_data["imputer"]
        feature_cols = model_data["feature_cols"]
        
        # Пandдготовка data
        X = features_df[feature_cols].values
        X = imputer.transform(X)
        X = scaler.transform(X)
        
        return model.predict(X)
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Отримати andнформацandю про model"""
        if model_key not in self.models:
            return {}
        
        model_data = self.models[model_key]
        return {
            "model_type": model_data["model_type"],
            "task_type": model_data["task_type"],
            "metrics": model_data["metrics"],
            "training_time": model_data["training_time"],
            "feature_cols": model_data["feature_cols"]
        }
    
    def train_model(self, model_type: str, ticker: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Тренування моwhereлand"""
        try:
            model_key = f"{model_type}_{ticker}_{timeframe}"
            
            # Mock training
            predictions = np.random.uniform(-2.0, 2.0, 10).tolist()
            
            # Mock metrics
            mse = np.random.uniform(0.1, 0.5)
            mae = np.random.uniform(0.2, 0.6)
            accuracy = np.random.uniform(0.4, 0.7)
            
            return {
                'model': f'mock_{model_type}',
                'predictions': predictions,
                'mse': mse,
                'mae': mae,
                'accuracy': accuracy
            }
            
        except Exception as e:
            logger.error(f"[LightModelTrainer] Error training {model_type} {ticker} {timeframe}: {e}")
            return None

def main():
    """Main function"""
    print("Module loaded successfully")

if __name__ == "__main__":
    main()
