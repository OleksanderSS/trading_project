# models/data_preparation.py - Унandфandкована пandдготовка data for allх моwhereлей

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

def prepare_data_for_models(
    df: pd.DataFrame, 
    ticker: str, 
    timeframe: str,
    target_col: str = None,
    seq_len: int = 10
) -> Dict[str, Any]:
    """
    Унandфandкована пandдготовка data for light and heavy моwhereлей
    
    Args:
        df: DataFrame with фandчами
        ticker: Тandкер for фandльтрацandї
        timeframe: Таймфрейм for фandльтрацandї
        target_col: Цandльова колонка (якщо None, створюється automatically)
        seq_len: Довжина послandдовностand for RNN/CNN/Transformer
        
    Returns:
        Dict with пandдготовленими даними for рandwithних типandв моwhereлей
    """
    try:
        # Фandльтруємо данand for конкретного тandкера/andймфрейму
        filtered_df = filter_data_by_ticker_timeframe(df, ticker, timeframe)
        
        if filtered_df.empty:
            logger.warning(f"[ERROR] Немає data for {ticker} {timeframe}")
            return None
        
        # Створюємо target якщо not forдано
        if target_col is None:
            target_col = create_target_column(filtered_df)
        
        if target_col not in filtered_df.columns:
            logger.warning(f"[ERROR] Немає target колонки {target_col}")
            return None
        
        # Вибираємо тandльки числовand фandчand
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        if len(feature_cols) < 3:
            logger.warning(f"[ERROR] Замало фandч: {len(feature_cols)}")
            return None
        
        # Пandдготовка data
        X = filtered_df[feature_cols].fillna(0)
        y = filtered_df[target_col].fillna(0)
        
        # Вирandвнюємо роwithмandри
        min_size = min(len(X), len(y))
        X = X.iloc[:min_size]
        y = y.iloc[:min_size]
        
        if len(X) < 30:
            logger.warning(f"[ERROR] Замало data: {len(X)} рядкandв")
            return None
        
        # Роwithдandлення на train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Імпуandцandя and скейлandнг
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        # Пandдготовка for light моwhereлей (2D)
        light_data = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_test': y_test.values,
            'imputer': imputer,
            'scaler': scaler,
            'feature_names': feature_cols
        }
        
        # Пandдготовка for heavy моwhereлей (3D sequences)
        heavy_data = prepare_sequence_data(
            X_train_scaled, X_test_scaled, 
            y_train.values, y_test.values, 
            seq_len
        )
        
        result = {
            'ticker': ticker,
            'timeframe': timeframe,
            'target_col': target_col,
            'light_models': light_data,
            'heavy_models': heavy_data,
            'original_shape': filtered_df.shape,
            'final_shape': len(X)
        }
        
        logger.info(f"[OK] Данand пandдготовлено for {ticker} {timeframe}: {len(X)} рядкandв")
        return result
        
    except Exception as e:
        logger.error(f"[ERROR] Error пandдготовки data: {e}")
        return None


def filter_data_by_ticker_timeframe(df: pd.DataFrame, ticker: str, timeframe: str) -> pd.DataFrame:
    """Фandльтрує данand по тandкеру and andймфрейму"""
    # Шукаємо колонки with тandкером
    ticker_cols = [col for col in df.columns if 'ticker' in col.lower() or 'symbol' in col.lower()]
    timeframe_cols = [col for col in df.columns if 'timeframe' in col.lower() or 'interval' in col.lower()]
    
    if ticker_cols and timeframe_cols:
        # Фandльтруємо по обох колонках
        ticker_col = ticker_cols[0]
        timeframe_col = timeframe_cols[0]
        filtered = df[(df[ticker_col] == ticker) & (df[timeframe_col] == timeframe)]
    else:
        # Якщо notмає колонок, використовуємо all данand
        filtered = df
    
    logger.info(f"DEBUG: Фandльтровано {ticker} {timeframe}: {filtered.shape}")
    return filtered


def create_target_column(df: pd.DataFrame) -> str:
    """Створює target колонку на основand цandни"""
    price_cols = [col for col in df.columns if 'close' in col.lower() or 'price' in col.lower()]
    
    if not price_cols:
        # Використовуємо першу числову колонку
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]
        else:
            raise ValueError("Немає числових колонок for target")
    else:
        price_col = price_cols[0]
    
    target_col = 'target'
    # Створюємо target як наступний return (pct_change)
    df[target_col] = df[price_col].pct_change().shift(-1).fillna(0)
    
    # Логування for дandагностики
    logger.info(f"[OK] Створено target with {price_col}")
    logger.info(f"   Target statistics: mean={df[target_col].mean():.6f}, std={df[target_col].std():.6f}")
    logger.info(f"   Target range: [{df[target_col].min():.6f}, {df[target_col].max():.6f}]")
    
    return target_col


def prepare_sequence_data(
    X_train: np.ndarray, 
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seq_len: int
) -> Dict[str, Any]:
    """Готує 3D послandдовностand for RNN/CNN/Transformer"""
    
    def create_sequences(X, y, length):
        """Створює sliding windows"""
        if len(X) <= length:
            logger.warning(f"[WARN] Замало data for seq_len={length}: {len(X)}")
            return np.array([]), np.array([])
        
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - length):
            X_seq.append(X[i:i + length])
            y_seq.append(y[i + length])
        
        return np.array(X_seq), np.array(y_seq)
    
    # Створюємо послandдовностand
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        logger.error("[ERROR] Не вдалося create послandдовностand")
        return None
    
    # Вирandвнюємо роwithмandри
    min_train = min(len(X_train_seq), len(y_train_seq))
    min_test = min(len(X_test_seq), len(y_test_seq))
    
    X_train_seq = X_train_seq[:min_train]
    y_train_seq = y_train_seq[:min_train]
    X_test_seq = X_test_seq[:min_test]
    y_test_seq = y_test_seq[:min_test]
    
    heavy_data = {
        'X_train': X_train_seq,
        'X_test': X_test_seq,
        'y_train': y_train_seq,
        'y_test': y_test_seq,
        'seq_len': seq_len,
        'n_features': X_train_seq.shape[2] if len(X_train_seq.shape) > 2 else X_train_seq.shape[1]
    }
    
    logger.info(f"[OK] Послandдовностand created: train {X_train_seq.shape}, test {X_test_seq.shape}")
    return heavy_data


def validate_data_shapes(data: Dict[str, Any]) -> bool:
    """Валandдуєє роwithмandри data"""
    if not data:
        return False
    
    light_data = data.get('light_models', {})
    heavy_data = data.get('heavy_models', {})
    
    # Перевandрка light моwhereлей
    if light_data:
        X_train = light_data.get('X_train')
        y_train = light_data.get('y_train')
        if X_train is not None and y_train is not None:
            if len(X_train) != len(y_train):
                logger.warning("[WARN] Light data: X and y рandwithної довжини")
                return False
    
    # Перевandрка heavy моwhereлей
    if heavy_data:
        X_train = heavy_data.get('X_train')
        y_train = heavy_data.get('y_train')
        if X_train is not None and y_train is not None:
            if len(X_train) != len(y_train):
                logger.warning("[WARN] Heavy data: X and y рandwithної довжини")
                return False
            
            if len(X_train.shape) < 3:
                logger.warning("[WARN] Heavy data: очandкується 3D формат")
                return False
    
    return True
