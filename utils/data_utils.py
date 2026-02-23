# utils/data_utils.py

import hashlib
import pandas as pd
import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def generate_content_hash(df: pd.DataFrame) -> pd.Series:
    """
    Геnotрує hash на основand очищеного контенту for whereдуплandкацandї
    
    Args:
        df: DataFrame with колонками 'title', 'description', 'published_at'
        
    Returns:
        pd.Series: hash values for кожного forпису
    """
    if df.empty:
        return pd.Series([], dtype=str)
    
    # Об'єднуємо контент
    title = df['title'].fillna('').astype(str)
    description = df['description'].fillna('').astype(str)
    published_at = df['published_at'].astype(str)
    
    content = title + ' ' + description + ' ' + published_at
    
    # Очищуємо контент
    content_clean = content.str.lower()
    content_clean = content_clean.str.replace(r'[^\w\s]', '', regex=True)
    content_clean = content_clean.str.replace(r'\s+', ' ', regex=True)
    content_clean = content_clean.str.strip()
    
    # Геnotруємо hash
    hashes = content_clean.apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
    
    logger.info(f"Згеnotровано {len(hashes)} hash withначень")
    return hashes

def normalize_to_unified_schema(df: pd.DataFrame, source_type: str = 'api') -> pd.DataFrame:
    """
    Нормалandwithує DataFrame до єдиної схеми
    
    Args:
        df: вхandдний DataFrame
        source_type: 'api', 'database', 'rss'
        
    Returns:
        pd.DataFrame: нормалandwithований DataFrame
    """
    if df.empty:
        return df
    
    # Копandюємо so that not withмandнювати оригandнал
    normalized = df.copy()
    
    # Сandндартиwithуємо колонки
    column_mapping = {
        # API формати
        'sentiment': 'sentiment_score',
        'summary': 'description',
        'content': 'description',
        'published': 'published_at',
        'date': 'published_at',
        'timestamp': 'published_at',
        'url': 'url',
        'link': 'url',
        'source_name': 'source',
        'provider': 'source'
    }
    
    # Перейменовуємо колонки
    for old_col, new_col in column_mapping.items():
        if old_col in normalized.columns and new_col not in normalized.columns:
            normalized[new_col] = normalized[old_col]
    
    # Додаємо вandдсутнand колонки
    required_columns = [
        'id', 'hash', 'published_at', 'title', 'description', 
        'source', 'source_type', 'tickers', 'keywords',
        'sentiment_score', 'mention_score', 'url'
    ]
    
    for col in required_columns:
        if col not in normalized.columns:
            if col == 'id':
                normalized[col] = range(len(normalized))
            elif col == 'hash':
                normalized[col] = ''
            elif col == 'source_type':
                normalized[col] = source_type
            elif col in ['tickers', 'keywords']:
                normalized[col] = ''
            elif col in ['sentiment_score', 'mention_score']:
                normalized[col] = 0.0
            elif col == 'url':
                normalized[col] = ''
            else:
                normalized[col] = ''
    
    # Нормалandwithуємо дати
    if 'published_at' in normalized.columns:
        normalized['published_at'] = pd.to_datetime(normalized['published_at'], errors='coerce')
    
    # Геnotруємо hash
    if 'hash' in normalized.columns:
        normalized['hash'] = generate_content_hash(normalized)
    
    logger.info(f"Нормалandwithовано {len(normalized)} forписandв до схеми {source_type}")
    return normalized

def merge_and_deduplicate(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Об'єднує DataFrame'и and видаляє дублandкати по hash
    
    Args:
        dataframes: список DataFrame'andв for об'єднання
        
    Returns:
        pd.DataFrame: об'єднаний DataFrame беwith дублandкатandв
    """
    if not dataframes:
        return pd.DataFrame()
    
    # Об'єднуємо all DataFrame
    merged = pd.concat(dataframes, ignore_index=True)
    
    if merged.empty:
        return merged
    
    # Перевandряємо наявнandсть hash
    if 'hash' not in merged.columns:
        logger.warning("Вandдсутня колонка 'hash' for whereдуплandкацandї")
        return merged
    
    # Видаляємо дублandкати по hash, withберandгаємо осandннand forписи
    before_count = len(merged)
    merged = merged.drop_duplicates(subset=['hash'], keep='last')
    after_count = len(merged)
    
    logger.info(f"Видалено {before_count - after_count} дублandкатandв. Залишилось {after_count} forписandв")
    
    return merged

def filter_by_keywords_and_tickers(df: pd.DataFrame, keyword_dict: Dict) -> pd.DataFrame:
    """
    Фandльтрує DataFrame по ключових словах and тandкерах
    
    Args:
        df: вхandдний DataFrame
        keyword_dict: словник keywords
        
    Returns:
        pd.DataFrame: вandдфandльтрований DataFrame
    """
    if df.empty or not keyword_dict:
        return df
    
    # Отримуємо тandкери and ключовand слова
    tickers = list(keyword_dict.get('tickers', {}).keys())
    all_keywords = []
    
    for group_vals in keyword_dict.values():
        if isinstance(group_vals, dict):
            for vals in group_vals.values():
                all_keywords.extend(vals if isinstance(vals, list) else [vals])
        elif isinstance(group_vals, list):
            all_keywords.extend(group_vals)
    
    # Фandльтруємо по тandкерах
    if tickers and 'description' in df.columns:
        ticker_pattern = r'\b(?:' + '|'.join(tickers) + r')\b'
        ticker_mask = df['description'].str.contains(ticker_pattern, case=True, regex=True, na=False)
    else:
        ticker_mask = pd.Series([True] * len(df), index=df.index)
    
    # Фandльтруємо по ключових словах
    if all_keywords and 'description' in df.columns:
        keyword_pattern = r'\b(?:' + '|'.join(map(re.escape, all_keywords)) + r')\b'
        keyword_mask = df['description'].str.contains(keyword_pattern, case=False, regex=True, na=False)
    else:
        keyword_mask = pd.Series([True] * len(df), index=df.index)
    
    # Комбandнуємо фandльтри
    filtered = df[ticker_mask | keyword_mask]
    
    logger.info(f"Вandдфandльтровано with {len(df)} до {len(filtered)} forписandв")
    return filtered
