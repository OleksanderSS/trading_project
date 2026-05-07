# src/models/adapters/data_preparation.py - Уніфікована підготовка даних для ML моделей

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Union, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import skew, kurtosis
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DataPreparationAdapter")

def prepare_data_for_models(
    df: pd.DataFrame, 
    ticker: str, 
    timeframe: str,
    target_cols: List[str],
    seq_len: int = 10,
    val_size: float = 0.1,
    test_size: float = 0.2,
    scale_target: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Уніфікований ML адаптер для підготовки даних.
    Зосереджений виключно на кодуванні, масштабуванні та створенні послідовностей.
    Дані мають бути попередньо очищені (DataCleaner) та містити таргети (TargetGenerator).
    """
    try:
        if not target_cols:
            logger.error("target_cols є обов'язковим параметром.")
            return None

        # 1. Фільтрація
        filtered_df = filter_data_by_ticker_timeframe(df, ticker, timeframe)
        if filtered_df.empty:
            logger.warning(f"Немає даних для {ticker} {timeframe}")
            return None
        
        # 2. Перевірка наявності таргетів та очищеності
        for col in target_cols:
            if col not in filtered_df.columns:
                logger.error(f"Колонка таргета '{col}' не знайдена. Використовуйте TargetGenerator.")
                return None
        
        if filtered_df.isna().sum().sum() > 0:
            logger.warning("DataFrame містить NaN. Рекомендується обробка через DataCleaner перед підготовкою.")

        # 3. Обробка категоріальних фіч
        df_processed, categorical_info = handle_categorical_features(filtered_df, target_cols)
        
        # 4. Feature selection (numeric only)
        feature_cols = [c for c in df_processed.select_dtypes(include=[np.number]).columns 
                        if c not in target_cols and c not in ['datetime', 'date']]
        
        if len(feature_cols) < 1:
            logger.error("Відсутні числови ознаки для моделювання.")
            return None
            
        X = df_processed[feature_cols].replace([np.inf, -np.inf], np.nan)
        y = df_processed[target_cols]
        
        log_data_distribution(X)
        
        # 5. Split into вибірки
        total_len = len(X)
        test_idx = int(total_len * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size / (1 - test_size)))
        
        x_train, x_val, x_test = X.iloc[:val_idx], X.iloc[val_idx:test_idx], X.iloc[test_idx:]
        y_train, y_val, y_test = y.iloc[:val_idx], y.iloc[val_idx:test_idx], y.iloc[test_idx:]
        
        # 6. ML Трансформації (Імпутація та Скейлінг)
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        # ELITE FIX: Preserving column names by converting back to DataFrame after scaling
        x_train_imputed = imputer.fit_transform(x_train)
        x_val_imputed = imputer.transform(x_val)
        x_test_imputed = imputer.transform(x_test)
        
        x_train_scaled_arr = scaler.fit_transform(x_train_imputed)
        x_val_scaled_arr = scaler.transform(x_val_imputed)
        x_test_scaled_arr = scaler.transform(x_test_imputed)
        
        # Create DataFrames for Light Models (preserves feature names)
        x_train_scaled_df = pd.DataFrame(x_train_scaled_arr, columns=feature_cols, index=x_train.index)
        x_val_scaled_df = pd.DataFrame(x_val_scaled_arr, columns=feature_cols, index=x_val.index)
        x_test_scaled_df = pd.DataFrame(x_test_scaled_arr, columns=feature_cols, index=x_test.index)
        
        target_scaler = None
        if scale_target:
            target_scaler = StandardScaler()
            y_train_processed = target_scaler.fit_transform(y_train)
            y_val_processed = target_scaler.transform(y_val)
            y_test_processed = target_scaler.transform(y_test)
        else:
            y_train_processed, y_val_processed, y_test_processed = y_train.values, y_val.values, y_test.values
        
        light_data = {
            'X_train': x_train_scaled_df, 'X_val': x_val_scaled_df, 'X_test': x_test_scaled_df,
            'y_train': y_train_processed, 'y_val': y_val_processed, 'y_test': y_test_processed,
            'imputer': imputer, 'scaler': scaler, 'target_scaler': target_scaler,
            'feature_names': feature_cols, 'categorical_info': categorical_info
        }

        # Heavy models expect 3D sequences from numpy arrays
        heavy_data = prepare_sequence_data_optimized(
            x_train_scaled_arr, x_val_scaled_arr, x_test_scaled_arr,
            y_train_processed, y_val_processed, y_test_processed,
            seq_len
        )
        
        return {
            'ticker': ticker, 'timeframe': timeframe, 'target_cols': target_cols,
            'light_models': light_data, 'heavy_models': heavy_data,
            'metadata': {'feature_count': len(feature_cols), 'samples': total_len}
        }
    except Exception as e:
        logger.error(f"Критична помилка підготовки даних: {e}", exc_info=True)
        return None

def handle_categorical_features(df: pd.DataFrame, exclude_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """Кодує категоріальні колонки."""
    df_out = df.copy()
    cat_cols = [c for c in df_out.select_dtypes(include=['object', 'category']).columns 
                if c not in exclude_cols and 'ticker' not in c.lower() and 'timeframe' not in c.lower()]
    
    info = {}
    for col in cat_cols:
        nunique = df_out[col].nunique()
        if nunique < 2:
            df_out.drop(columns=[col], inplace=True)
            continue
        if nunique <= 5:
            dummies = pd.get_dummies(df_out[col], prefix=col, drop_first=True)
            df_out = pd.concat([df_out, dummies], axis=1).drop(columns=[col])
            info[col] = 'one-hot'
        else:
            le = LabelEncoder()
            df_out[col] = le.fit_transform(df_out[col].astype(str))
            info[col] = 'label'
    return df_out, info

def log_data_distribution(df: pd.DataFrame):
    """Логує статистичні показники розподілу ознак."""
    if df.empty: return
    stats = []
    for col in df.columns[:5]:
        vals = df[col].dropna()
        if len(vals) > 0:
            stats.append(f"{col}(S:{skew(vals):.2f},K:{kurtosis(vals):.2f})")
    logger.debug(f"Feature distribution: {', '.join(stats)}")

def prepare_sequence_data_optimized(x_tr, x_va, x_te, y_tr, y_va, y_te, seq_len) -> Dict[str, Any]:
    """Створення 3D вікон для Neural Networks за допомогою numpy strides."""
    def strided_window(x, y, window):
        if len(x) <= window: return np.array([]), np.array([])
        shape = (x.shape[0] - window, window, x.shape[1])
        strides = (x.strides[0], x.strides[0], x.strides[1])
        x_win = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        y_win = y[window:]
        return x_win, y_win

    x_train_s, y_train_s = strided_window(x_tr, y_tr, seq_len)
    x_val_s, y_val_s = strided_window(x_va, y_va, seq_len)
    x_test_s, y_test_s = strided_window(x_te, y_te, seq_len)
    
    return {
        'X_train': x_train_s, 'X_val': x_val_s, 'X_test': x_test_s,
        'y_train': y_train_s, 'y_val': y_val_s, 'y_test': y_test_s,
        'seq_len': seq_len, 'n_features': x_tr.shape[1]
    }

def filter_data_by_ticker_timeframe(df: pd.DataFrame, ticker: str, timeframe: str) -> pd.DataFrame:
    """Фільтрація вхідного набору даних."""
    t_cols = [c for c in df.columns if 'ticker' in c.lower() or 'symbol' in c.lower()]
    tf_cols = [c for c in df.columns if 'timeframe' in c.lower() or 'interval' in c.lower()]
    if t_cols and tf_cols:
        return df[(df[t_cols[0]] == ticker) & (df[tf_cols[0]] == timeframe)]
    return df

def validate_data_shapes(data: Dict[str, Any]) -> bool:
    """Перевірка розмірностей вихідних даних."""
    if not data: return False
    for m_type in ['light_models', 'heavy_models']:
        d = data.get(m_type, {})
        if not d: continue
        for subset in ['train', 'val', 'test']:
            x, y = d.get(f'X_{subset}'), d.get(f'y_{subset}')
            if x is not None and y is not None and len(x) != len(y):
                logger.warning(f"{m_type} {subset}: X/y length mismatch")
                return False
    return True