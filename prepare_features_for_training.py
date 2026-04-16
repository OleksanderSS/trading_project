"""
Правильна підготовка фіч для тренування
"""
import pandas as pd
import numpy as np
from pathlib import Path

def prepare_features(features_df: pd.DataFrame, verbose: bool = True):
    """
    Підготовка фіч для тренування
    
    Правила:
    1. Видаляємо метадані (news_id, news_title, ticker, datetime)
    2. Зберігаємо числові фічі (включаючи news_sentiment)
    3. Заповнюємо NaN нулями
    """
    
    # Колонки-метадані, які не є фічами
    metadata_cols = [
        'news_id',           # ID новини (метадані)
        'news_title',        # Текст заголовка (метадані)
        'ticker',            # Тікер (метадані)
        'datetime',          # Час (метадані)
        'published_at',      # Час публікації (метадані)
    ]
    
    # Видаляємо метадані
    cols_to_drop = [c for c in metadata_cols if c in features_df.columns]
    df_clean = features_df.drop(columns=cols_to_drop)
    
    if verbose:
        print(f"Dropped metadata columns: {cols_to_drop}")
    
    # Вибираємо тільки числові колонки
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df_clean[numeric_cols]
    
    if verbose:
        print(f"Numeric features: {len(numeric_cols)}")
        print(f"Shape before fillna: {df_numeric.shape}")
    
    # Заповнюємо NaN нулями
    df_filled = df_numeric.fillna(0)
    
    # Замінюємо inf на 0
    df_filled = df_filled.replace([np.inf, -np.inf], 0)
    
    if verbose:
        print(f"Shape after cleaning: {df_filled.shape}")
        print(f"NaN count: {df_filled.isna().sum().sum()}")
        print(f"Inf count: {np.isinf(df_filled.values).sum()}")
    
    return df_filled, numeric_cols


if __name__ == "__main__":
    # Тест
    batch_dir = Path("data/colab/accumulated/test_ticker_amd_target_return_1d_ep5_iter5")
    features_path = batch_dir / "features.parquet"
    
    print("=" * 80)
    print("FEATURE PREPARATION TEST")
    print("=" * 80)
    
    features_df = pd.read_parquet(features_path)
    print(f"\nOriginal shape: {features_df.shape}")
    print(f"Original columns (first 10): {features_df.columns.tolist()[:10]}")
    
    X_clean, feature_names = prepare_features(features_df, verbose=True)
    
    print(f"\nCleaned shape: {X_clean.shape}")
    print(f"\nFeature names (first 10): {feature_names[:10]}")
    print(f"\nImportant features:")
    print(f"  - news_sentiment: {'news_sentiment' in feature_names}")
    print(f"  - AMD_15m_close: {'AMD_15m_close' in feature_names}")
    print(f"  - macro features: {len([f for f in feature_names if 'macro_' in f])}")
    
    print("\n✅ Feature preparation successful!")
