#!/usr/bin/env python3
"""
🎯 Локальне Тренування Легких Моделей

Цей скрипт тренує легкі моделі (tree-based) локально на CPU.
Вибір фіч робиться в Colab, тренування - тут.

Архітектура:
- Colab (GPU): Вибір фіч для всіх моделей + тренування важких моделей
- Локально (CPU): Тренування легких моделей

Використання:
    python scripts/train_light_models_locally.py --batch test_ticker_amd_target_return_1d_ep5_iter5
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import logging

# Tree-based моделі
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Визначення легких моделей
LIGHT_MODELS = {
    'catboost': CatBoostRegressor if HAS_CATBOOST else None,
    'lightgbm': LGBMRegressor if HAS_LIGHTGBM else None,
    'xgboost': XGBRegressor if HAS_XGBOOST else None,
    'random_forest': RandomForestRegressor,
    'linear': LinearRegression,
    'svm': SVR,
    'knn': KNeighborsRegressor,
}

def compute_metrics(y_true, y_pred):
    """Розрахувати метрики"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0.0
    return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2), 'mape': float(mape)}

def train_light_model(model_type, X_train, y_train, X_val, y_val):
    """Тренувати легку модель"""
    logger.info(f"  🔧 Тренування {model_type}...")
    
    if model_type == 'catboost':
        if not HAS_CATBOOST:
            logger.warning(f"  ⚠️ CatBoost не встановлено, пропускаємо")
            return None
        model = CatBoostRegressor(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            verbose=False,
            random_state=42
        )
    elif model_type == 'lightgbm':
        if not HAS_LIGHTGBM:
            logger.warning(f"  ⚠️ LightGBM не встановлено, пропускаємо")
            return None
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            verbose=-1,
            random_state=42
        )
    elif model_type == 'xgboost':
        if not HAS_XGBOOST:
            logger.warning(f"  ⚠️ XGBoost не встановлено, пропускаємо")
            return None
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0
        )
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'svm':
        model = SVR(kernel='rbf', C=100, gamma='scale')
    elif model_type == 'knn':
        model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    else:
        logger.warning(f"  ⚠️ Невідома модель: {model_type}")
        return None
    
    # Тренування
    model.fit(X_train, y_train)
    
    # Предикції
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Метрики
    train_metrics = compute_metrics(y_train, y_pred_train)
    val_metrics = compute_metrics(y_val, y_pred_val)
    
    logger.info(f"    ✅ Train R²: {train_metrics['r2']:.4f}, Val R²: {val_metrics['r2']:.4f}")
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }

def train_light_models_for_batch(batch_dir):
    """Тренувати легкі моделі для батча"""
    batch_dir = Path(batch_dir)
    
    if not batch_dir.exists():
        logger.error(f"❌ Батч не знайдено: {batch_dir}")
        return
    
    logger.info(f"📦 Батч: {batch_dir.name}")
    
    # Завантажуємо дані
    features_path = batch_dir / "features.parquet"
    targets_path = batch_dir / "targets.parquet"
    
    if not features_path.exists() or not targets_path.exists():
        logger.error(f"❌ Дані не знайдено в батчі")
        return
    
    logger.info(f"📊 Завантажуємо дані...")
    features_df = pd.read_parquet(features_path)
    targets_df = pd.read_parquet(targets_path)
    
    logger.info(f"  Features: {features_df.shape}")
    logger.info(f"  Targets: {targets_df.shape}")
    
    # Завантажуємо результати Colab
    colab_results_path = batch_dir / "colab_results_summary.json"
    if not colab_results_path.exists():
        logger.error(f"❌ colab_results_summary.json не знайдено")
        return
    
    with open(colab_results_path, 'r') as f:
        colab_results = json.load(f)
    
    # Обробляємо кожен тікер та таргет
    for ticker, ticker_data in colab_results.get('ticker_results', {}).items():
        logger.info(f"\n🎯 Тікер: {ticker}")
        
        for timeframe, tf_data in ticker_data.get('timeframes', {}).items():
            logger.info(f"  ⏱️ Таймфрейм: {timeframe}")
            
            for target_col, target_data in tf_data.get('results', {}).items():
                logger.info(f"    📈 Таргет: {target_col}")
                
                # ✅ FIX: Дані вже відфільтровані по тікеру та таймфрейму
                # Просто використовуємо всі дані для цього таргету
                X = features_df.copy()
                y = targets_df[target_col].fillna(0)
                
                if len(X) != len(y):
                    logger.warning(f"      ⚠️ Розмір X ({len(X)}) не збігається з y ({len(y)})")
                    continue
                
                logger.info(f"      📊 X: {X.shape}, y: {y.shape}")
                
                # Обробляємо кожну легку модель
                for model_type in LIGHT_MODELS.keys():
                    model_data = target_data.get('models', {}).get(model_type, {})
                    
                    # Перевіряємо, чи модель вже тренована в Colab
                    if model_data.get('trained', False):
                        logger.info(f"      ⏭️ {model_type:<14} | Вже тренована в Colab, пропускаємо")
                        continue
                    
                    logger.info(f"      🔍 {model_type:<14} | Тренування...")
                    
                    # Завантажуємо вибрані фічі
                    selected_features_path = batch_dir / f"selected_features_{model_type}_{ticker}_{target_col}.json"
                    if not selected_features_path.exists():
                        logger.warning(f"⚠️ selected_features не знайдено")
                        continue
                    
                    with open(selected_features_path, 'r') as f:
                        selected_features_data = json.load(f)
                    
                    selected_features = selected_features_data.get('selected_features', [])
                    logger.info(f"      Вибрано {len(selected_features)} фіч")
                    
                    # Фільтруємо X за вибраними фічами
                    available_features = [f for f in selected_features if f in X.columns]
                    if len(available_features) == 0:
                        logger.warning(f"⚠️ Жодна фіча не знайдена")
                        continue
                    
                    X_filtered = X[available_features].copy()
                    
                    # ✅ FIX: Обробляємо NaN значення
                    # Видаляємо рядки з NaN
                    mask_valid = ~(X_filtered.isna().any(axis=1) | y.isna())
                    X_filtered = X_filtered[mask_valid]
                    y_filtered = y[mask_valid]
                    
                    if len(X_filtered) < 10:
                        logger.warning(f"⚠️ Занадто мало валідних зразків: {len(X_filtered)}")
                        continue
                    
                    # Розділяємо на train/val
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_filtered, y_filtered, test_size=0.2, random_state=42, shuffle=True
                    )
                    
                    # Нормалізуємо
                    scaler = StandardScaler()
                    X_train_sc = scaler.fit_transform(X_train)
                    X_val_sc = scaler.transform(X_val)
                    
                    # Тренуємо модель
                    result = train_light_model(model_type, X_train_sc, y_train, X_val_sc, y_val)
                    
                    if result is None:
                        logger.warning(f"⚠️ Не вдалося тренувати {model_type}")
                        continue
                    
                    # Зберігаємо модель
                    models_dir = batch_dir / "models"
                    models_dir.mkdir(exist_ok=True)
                    
                    model_path = models_dir / f"{model_type}_{ticker}_{target_col}.pkl"
                    joblib.dump(result['model'], model_path)
                    logger.info(f"✅ Збережено: {model_path.name}")
                    
                    # Оновлюємо результати
                    target_data['models'][model_type].update({
                        'trained': True,
                        'model_path': str(model_path),
                        'train_metrics': result['train_metrics'],
                        'val_metrics': result['val_metrics'],
                        'mse': result['val_metrics']['rmse'] ** 2
                    })
    
    # Зберігаємо оновлені результати
    with open(colab_results_path, 'w') as f:
        json.dump(colab_results, f, indent=2)
    
    logger.info(f"\n✅ Локальне тренування завершено!")
    logger.info(f"📝 Результати збережено в {colab_results_path}")

def main():
    parser = argparse.ArgumentParser(description='Локальне тренування легких моделей')
    parser.add_argument('--batch', required=True, help='Назва батча (наприклад: test_ticker_amd_target_return_1d_ep5_iter5)')
    parser.add_argument('--base-dir', default='data/colab/accumulated', help='Базова папка для батчів')
    
    args = parser.parse_args()
    
    batch_dir = Path(args.base_dir) / args.batch
    train_light_models_for_batch(batch_dir)

if __name__ == '__main__':
    main()
