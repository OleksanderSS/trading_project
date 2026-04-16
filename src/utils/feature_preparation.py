"""
Utility functions for feature preparation
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def prepare_features_for_training(
    features_df: pd.DataFrame,
    remove_metadata: bool = True,
    fill_na: bool = True,
    verbose: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Підготовка фіч для тренування моделей
    
    Args:
        features_df: DataFrame з фічами
        remove_metadata: Видаляти метадані (news_id, news_title, etc.)
        fill_na: Заповнювати NaN нулями
        verbose: Виводити детальну інформацію
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: (очищені фічі, список назв фіч)
    """
    
    # Колонки-метадані, які не є фічами для моделей
    metadata_cols = [
        'news_id',           # ID новини (метадані)
        'news_title',        # Текст заголовка (метадані)
        'ticker',            # Тікер (метадані)
        'datetime',          # Час (метадані)
        'published_at',      # Час публікації (метадані)
    ]
    
    df_clean = features_df.copy()
    
    # Видаляємо метадані
    if remove_metadata:
        cols_to_drop = [c for c in metadata_cols if c in df_clean.columns]
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            if verbose:
                logger.info(f"Dropped metadata columns: {cols_to_drop}")
    
    # Вибираємо тільки числові колонки
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df_clean[numeric_cols]
    
    if verbose:
        logger.info(f"Numeric features: {len(numeric_cols)}")
        logger.info(f"Shape before cleaning: {df_numeric.shape}")
    
    # Заповнюємо NaN
    if fill_na:
        df_numeric = df_numeric.fillna(0)
    
    # Замінюємо inf на 0
    df_numeric = df_numeric.replace([np.inf, -np.inf], 0)
    
    if verbose:
        logger.info(f"Shape after cleaning: {df_numeric.shape}")
        logger.info(f"NaN count: {df_numeric.isna().sum().sum()}")
        logger.info(f"Inf count: {np.isinf(df_numeric.values).sum()}")
        
        # Перевіряємо важливі фічі
        important_features = ['news_sentiment', 'AMD_15m_close', 'AMD_1h_close']
        for feat in important_features:
            if feat in numeric_cols:
                logger.info(f"  ✅ {feat} included")
    
    return df_numeric, numeric_cols


def align_features_with_model(
    X: pd.DataFrame,
    model_feature_names: List[str]
) -> pd.DataFrame:
    """
    Вирівнює фічі з тими, що очікує модель
    
    Args:
        X: DataFrame з фічами
        model_feature_names: Список фіч, які очікує модель
        
    Returns:
        pd.DataFrame: DataFrame з правильними фічами
    """
    
    # Знаходимо доступні фічі
    available_features = [f for f in model_feature_names if f in X.columns]
    
    if len(available_features) < len(model_feature_names):
        missing = set(model_feature_names) - set(available_features)
        logger.warning(f"Missing {len(missing)} features: {list(missing)[:5]}...")
    
    # Повертаємо тільки доступні фічі в правильному порядку
    return X[available_features]
