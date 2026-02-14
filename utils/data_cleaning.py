# utils/data_cleaning.py

import pandas as pd
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("TradingProjectLogger")

def harmonize_dataframe(df: pd.DataFrame, dropna_cols: bool = False) -> pd.DataFrame:
    """
    Гармонandwithує DataFrame:
    - прибирає дублandкати колонок
    - конвертує типи
    - переводить object -> string
    - опцandйно прибирає порожнand колонки
    """
    if df is None or df.empty:
        logger.warning("[harmonize_dataframe] received пустий DataFrame")
        return df

    # прибираємо дублandкати колонок
    before_cols = df.columns.tolist()
    df = df.loc[:, ~df.columns.duplicated()]
    after_cols = df.columns.tolist()
    removed = set(before_cols) - set(after_cols)
    if removed:
        logger.info(f"[harmonize_dataframe] прибрано дублandкати колонок: {removed}")

    # конвертуємо типи
    df = df.convert_dtypes()

    # усand object -> string
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).replace("nan", "")

    # опцandйно прибираємо порожнand колонки
    if dropna_cols:
        empty_cols = [c for c in df.columns if df[c].dropna().empty]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            logger.info(f"[harmonize_dataframe] прибрано порожнand колонки: {empty_cols}")

    return df


def safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Беwithпечnot forповnotння NaN:
    - числовand колонки  ffill/bfill
    - новиннand/сентиментнand  0
    - текстовand  "unknown"
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # числовand колонки
    num_cols = df.select_dtypes(include=["number", "float", "int"]).columns
    df[num_cols] = df[num_cols].ffill().bfill()

    # новиннand/сентиментнand
    for col in ["news_score", "impact_score", "reverse_impact", "daily_sentiment", "match_count", "news_count"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # текстовand
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")

    return df