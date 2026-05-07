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
import re

def sanitize_path_input(path_input: str) -> str:
    """
    Sanitize path input to prevent path traversal attacks.
    
    Args:
        path_input: Input string that will be used in file paths
        
    Returns:
        Sanitized string safe for path construction
    """
    if not path_input:
        return ""
    
    # Remove path traversal characters
    sanitized = re.sub(r'[./\\]', '_', path_input)
    
    # Remove null bytes and other dangerous characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    
    # Limit length to prevent path overflow
    sanitized = sanitized[:100]
    
    return sanitized

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

def train_light_model(model_type, X_train, y_train, x_val, y_val):
    """Тренувати легку модель"""
    logger.info("  🔧 Тренування {}...".format(model_type))
    
    if model_type == 'catboost':
        if not HAS_CATBOOST:
            logger.warning("  ⚠️ CatBoost не встановлено, пропускаємо")
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
            logger.warning("  ⚠️ LightGBM не встановлено, пропускаємо")
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
            logger.warning("  ⚠️ XGBoost не встановлено, пропускаємо")
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
            min_samples_leaf=1,
            max_features='sqrt',
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
        logger.warning("  ⚠️ Невідома модель: {}".format(model_type))
        return None
    
    # Тренування
    model.fit(X_train, y_train)
    
    # Предикції
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(x_val)
    
    # Метрики
    train_metrics = compute_metrics(y_train, y_pred_train)
    val_metrics = compute_metrics(y_val, y_pred_val)
    
    logger.info("    ✅ Train R²: {:.4f}, Val R²: {:.4f}".format(train_metrics['r2'], val_metrics['r2']))
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }

def train_light_models_for_batch(batch_dir):
    """Train light models for batch"""
    sanitized_batch_dir = sanitize_path_input(batch_dir)
    batch_dir = Path(sanitized_batch_dir)
    
    if not _validate_batch_directory(batch_dir):
        return
    
    batch_data = _load_batch_data(batch_dir)
    if batch_data is None:
        return
    
    colab_results = _load_colab_results(batch_dir)
    if colab_results is None:
        return
    
    _process_all_tickers(colab_results, batch_data, batch_dir)
    _save_updated_results(colab_results, batch_dir)

def _validate_batch_directory(batch_dir: Path) -> bool:
    """Validate batch directory exists."""
    if not batch_dir.exists():
        logger.error("❌ Батч не знайдено: {}".format(batch_dir))
        return False
    
    logger.info("📦 Батч: {}".format(batch_dir.name))
    return True

def _load_batch_data(batch_dir: Path) -> Optional[Dict]:
    """Load features and targets data."""
    features_path = batch_dir / "features.parquet"
    targets_path = batch_dir / "targets.parquet"
    
    if not features_path.exists() or not targets_path.exists():
        logger.error("❌ Дані не знайдено в батчі")
        return None
    
    logger.info("📊 Завантажуємо дані...")
    features_df = pd.read_parquet(features_path)
    targets_df = pd.read_parquet(targets_path)
    
    logger.info("  Features: {}".format(features_df.shape))
    logger.info("  Targets: {}".format(targets_df.shape))
    
    return {
        'features_df': features_df,
        'targets_df': targets_df
    }

def _load_colab_results(batch_dir: Path) -> Optional[Dict]:
    """Load Colab results summary."""
    colab_results_path = batch_dir / "colab_results_summary.json"
    if not colab_results_path.exists():
        logger.error("❌ colab_results_summary.json не знайдено")
        return None
    
    with open(colab_results_path, 'r') as f:
        return json.load(f)

def _process_all_tickers(colab_results: Dict, batch_data: Dict, batch_dir: Path):
    """Process all tickers in the batch."""
    for ticker, ticker_data in colab_results.get('ticker_results', {}).items():
        logger.info("\n🎯 Тікер: {}".format(ticker))
        _process_ticker_timeframes(ticker, ticker_data, batch_data, batch_dir)

def _process_ticker_timeframes(ticker: str, ticker_data: Dict, batch_data: Dict, batch_dir: Path):
    """Process all timeframes for a ticker."""
    for timeframe, tf_data in ticker_data.get('timeframes', {}).items():
        logger.info("  ⏱️ Таймфрейм: {}".format(timeframe))
        _process_timeframe_targets(ticker, tf_data, batch_data, batch_dir)

def _process_timeframe_targets(ticker: str, tf_data: Dict, batch_data: Dict, batch_dir: Path):
    """Process all targets for a timeframe."""
    for target_col, target_data in tf_data.get('results', {}).items():
        logger.info("    📈 Таргет: {}".format(target_col))
        _process_target_model(ticker, target_col, target_data, batch_data, batch_dir)

def _process_target_model(ticker: str, target_col: str, target_data: Dict, batch_data: Dict, batch_dir: Path):
    """Process all models for a target."""
    X = batch_data['features_df'].copy()
    y = batch_data['targets_df'][target_col].fillna(0)
    
    if not _validate_data_sizes(X, y):
        return
    
    logger.info("      📊 X: {}, y: {}".format(X.shape, y.shape))
    
    for model_type in LIGHT_MODELS.keys():
        _train_single_model(ticker, target_col, model_type, target_data, X, y, batch_dir)

def _validate_data_sizes(X: pd.DataFrame, y: pd.Series) -> bool:
    """Validate X and y have same size."""
    if len(X) != len(y):
        logger.warning("      ⚠️ Розмір X ({}) не збігається з y ({})".format(len(X), len(y)))
        return False
    return True

def _train_single_model(ticker: str, target_col: str, model_type: str, target_data: Dict, X: pd.DataFrame, y: pd.Series, batch_dir: Path):
    """Train a single model."""
    model_data = target_data.get('models', {}).get(model_type, {})
    
    if model_data.get('trained', False):
        logger.info("      ⏭️ {:<14} | Вже тренована в Colab, пропускаємо".format(model_type))
        return
    
    logger.info("      🔍 {:<14} | Тренування...".format(model_type))
    
    selected_features = _load_selected_features(batch_dir, model_type, ticker, target_col)
    if selected_features is None:
        return
    
    training_data = _prepare_training_data(X, y, selected_features)
    if training_data is None:
        return
    
    result = train_light_model(model_type, training_data['x_train_sc'], training_data['y_train'], training_data['x_val_sc'], training_data['y_val'])
    
    if result is None:
        logger.warning("⚠️ Не вдалося тренувати {}".format(model_type))
        return
    
    _save_model_and_update_results(ticker, target_col, model_type, result, target_data, batch_dir)

def _load_selected_features(batch_dir: Path, model_type: str, ticker: str, target_col: str) -> Optional[List]:
    """Load selected features for model."""
    selected_features_path = batch_dir / f"selected_features_{model_type}_{ticker}_{target_col}.json"
    if not selected_features_path.exists():
        logger.warning("⚠️ selected_features не знайдено")
        return None
    
    with open(selected_features_path, 'r') as f:
        selected_features_data = json.load(f)
    
    selected_features = selected_features_data.get('selected_features', [])
    logger.info("      Вибрано {} фіч".format(len(selected_features)))
    return selected_features

def _prepare_training_data(X: pd.DataFrame, y: pd.Series, selected_features: List) -> Optional[Dict]:
    """Prepare and split training data."""
    available_features = [f for f in selected_features if f in X.columns]
    if len(available_features) == 0:
        logger.warning("⚠️ Жодна фіча не знайдена")
        return None
    
    x_filtered = X[available_features].copy()
    
    # Обробляємо NaN значення
    mask_valid = ~(x_filtered.isna().any(axis=1) | y.isna())
    x_filtered = x_filtered[mask_valid]
    y_filtered = y[mask_valid]
    
    if len(x_filtered) < 10:
        logger.warning("⚠️ Занадто мало валідних зразків: {}".format(len(x_filtered)))
        return None
    
    # Розділяємо на train/val
    x_train, x_val, y_train, y_val = train_test_split(
        x_filtered, y_filtered, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Нормалізуємо
    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_val_sc = scaler.transform(x_val)
    
    return {
        'x_train_sc': x_train_sc,
        'x_val_sc': x_val_sc,
        'y_train': y_train,
        'y_val': y_val
    }

def _save_model_and_update_results(ticker: str, target_col: str, model_type: str, result: Dict, target_data: Dict, batch_dir: Path):
    """Save model and update results."""
    # Зберігаємо модель
    models_dir = batch_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / f"{model_type}_{ticker}_{target_col}.pkl"
    joblib.dump(result['model'], model_path)
    logger.info("✅ Збережено: {}".format(model_path.name))
    
    # Оновлюємо результати
    target_data['models'][model_type].update({
        'trained': True,
        'model_path': str(model_path),
        'train_metrics': result['train_metrics'],
        'val_metrics': result['val_metrics'],
        'mse': result['val_metrics']['rmse'] ** 2
    })

def _save_updated_results(colab_results: Dict, batch_dir: Path):
    """Save updated results to file."""
    colab_results_path = batch_dir / "colab_results_summary.json"
    with open(colab_results_path, 'w') as f:
        json.dump(colab_results, f, indent=2)
    
    logger.info("\n✅ Локальне тренування завершено!")
    logger.info("📝 Результати збережено в {}".format(colab_results_path))

def main():
    parser = argparse.ArgumentParser(description='Локальне тренування легких моделей')
    parser.add_argument('--batch', required=True, help='Назва батча (наприклад: test_ticker_amd_target_return_1d_ep5_iter5)')
    parser.add_argument('--base-dir', default='data/colab/accumulated', help='Базова папка для батчів')
    
    args = parser.parse_args()
    
    sanitized_batch = sanitize_path_input(args.batch)
    sanitized_base_dir = sanitize_path_input(args.base_dir)
    batch_dir = Path(sanitized_base_dir) / sanitized_batch
    train_light_models_for_batch(batch_dir)

if __name__ == '__main__':
    main()
