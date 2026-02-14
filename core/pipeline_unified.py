#!/usr/bin/env python3
"""
Unified Trading Pipeline with Clean Architecture
Реалізує логіку описану користувачем з уніфікованою архітектурою
"""

import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import pandas as pd
import numpy as np
import logging

from utils.logger_fixed import ProjectLogger
from config.config import TICKERS, TIME_FRAMES

logger = ProjectLogger.get_logger("UnifiedPipeline")

class DataParser:
    """Етап 1: Парсинг сирих даних"""
    
    def __init__(self, cache_dir: str = "data/cache/raw"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_all_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Збирає всі сири дані (новини, ціни, макро)"""
        logger.info("[DataParser] Початок парсингу сирих даних...")
        
        cache_file = self.cache_dir / "raw_data.pkl"
        
        if not force_refresh and cache_file.exists():
            try:
                cached_data = pickle.load(open(cache_file, 'rb'))
                if self._is_data_fresh(cached_data):
                    logger.info("[DataParser] Використовуємо кешовані дані")
                    return cached_data
            except Exception as e:
                logger.warning(f"[DataParser] Помилка завантаження кешу: {e}")
        
        # Збираємо свіжі дані
        from core.stages.stage_1_collectors_layer import IdealStage1Collector
        
        collector = IdealStage1Collector(
            tickers=TICKERS,
            timeframes=TIME_FRAMES,
            use_free_data=True,
            enable_cache=True
        )
        
        raw_data = collector.run_stage_1(
            tickers=TICKERS,
            timeframes=TIME_FRAMES,
            use_free_data=True,
            enable_cache=True
        )
        
        # Додаємо метадані
        raw_data['_metadata'] = {
            'parsing_time': datetime.now(),
            'tickers': list(TICKERS.keys()),
            'timeframes': TIME_FRAMES
        }
        
        # Зберігаємо кеш
        pickle.dump(raw_data, open(cache_file, 'wb'))
        logger.info(f"[DataParser] Зібрано даних: {list(raw_data.keys())}")
        
        return raw_data
    
    def _is_data_fresh(self, data: Dict[str, Any], max_age_hours: int = 1) -> bool:
        """Перевіряє чи дані свіжі"""
        if '_metadata' not in data:
            return False
        
        parsing_time = data['_metadata'].get('parsing_time')
        if not parsing_time:
            return False
        
        age = datetime.now() - parsing_time
        return age.total_seconds() < max_age_hours * 3600

class DataEnricher:
    """Етап 2: Збагачення даних з макро, показниками, сентиментом"""
    
    def __init__(self, cache_dir: str = "data/cache/enriched"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def enrich_data(self, raw_data: Dict[str, Any], force_refresh: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Збагачує сири дані технічними індикаторами, макро, сентиментом"""
        logger.info("[DataEnricher] Початок збагачення даних...")
        
        cache_file = self.cache_dir / "enriched_data.pkl"
        
        if not force_refresh and cache_file.exists():
            try:
                cached_data = pickle.load(open(cache_file, 'rb'))
                if self._is_enrichment_fresh(cached_data):
                    logger.info("[DataEnricher] Використовуємо кешовані збагачені дані")
                    return cached_data['merged_df'], cached_data['metadata']
            except Exception as e:
                logger.warning(f"[DataEnricher] Помилка завантаження кешу: {e}")
        
        # Запускаємо збагачення
        from core.stages.stage_2_enrichment import run_stage_2_enrich_optimized
        
        raw_news, merged_df, pivots = run_stage_2_enrich_optimized(
            stage1_data=raw_data,
            tickers=TICKERS,
            time_frames=TIME_FRAMES
        )
        
        # Додаємо ковзні середні та інші розрахунки
        merged_df = self._add_calculations(merged_df)
        
        # Створюємо метадані
        metadata = {
            'enrichment_time': datetime.now(),
            'raw_news_count': len(raw_news) if raw_news else 0,
            'merged_shape': merged_df.shape if merged_df is not None else None,
            'pivots': list(pivots.keys()) if pivots else [],
            'features_added': self._detect_added_features(raw_data, merged_df),
            'calculations_added': ['moving_averages', 'price_changes', 'volatility']
        }
        
        # Зберігаємо кеш
        enriched_data = {
            'merged_df': merged_df,
            'metadata': metadata,
            'raw_news': raw_news,
            'pivots': pivots
        }
        pickle.dump(enriched_data, open(cache_file, 'wb'))
        
        logger.info(f"[DataEnricher] Збагачено даних: {merged_df.shape if merged_df is not None else None}")
        return merged_df, metadata
    
    def _add_calculations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Додає ковзні середні та інші розрахунки до DataFrame"""
        if df is None or df.empty:
            return df
        
        logger.info("[DataEnricher] Додавання ковзних середніх та розрахунків...")
        
        # Ковзні середні для різних періодів
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Волатильність
        df['volatility_20'] = df['close'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_20'] / df['volatility_20'].rolling(window=50).mean()
        
        # Зміни цін
        df['price_change_1d'] = df['close'].pct_change(1)
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_abs'] = abs(df['price_change_1d'])
        
        logger.info(f"[DataEnricher] Додано розрахунків: {df.shape}")
        return df
    
    def _is_enrichment_fresh(self, data: Dict[str, Any], max_age_hours: int = 6) -> bool:
        """Перевіряє чи збагачені дані свіжі"""
        enrichment_time = data.get('metadata', {}).get('enrichment_time')
        if not enrichment_time:
            return False
        
        age = datetime.now() - enrichment_time
        return age.total_seconds() < max_age_hours * 3600
    
    def _detect_added_features(self, raw_data: Dict[str, Any], enriched_df: pd.DataFrame) -> List[str]:
        """Визначає які фічі були додані під час збагачення"""
        if enriched_df is None:
            return []
        
        feature_categories = {
            'technical': ['rsi', 'sma', 'ema', 'macd', 'bollinger'],
            'macro': ['gdp', 'inflation', 'unemployment', 'interest'],
            'sentiment': ['sentiment', 'news_score', 'keywords'],
            'volume': ['volume', 'volatility'],
            'price_action': ['gap', 'pivot', 'support', 'resistance']
        }
        
        added_features = []
        columns_lower = [col.lower() for col in enriched_df.columns]
        
        for category, keywords in feature_categories.items():
            if any(keyword in ' '.join(columns_lower) for keyword in keywords):
                added_features.append(category)
        
        return added_features

class FeatureSelector:
    """Етап 3: Гнучка система вибору фіч під кожен тікер, таймфрейм, таргет"""
    
    def __init__(self, cache_dir: str = "data/cache/features"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_features(self, enriched_df: pd.DataFrame, 
                        target_config: Optional[Dict] = None,
                        force_refresh: bool = False) -> Dict[str, Any]:
        """Готує гнучкі фічі для різних таргетів"""
        logger.info("[FeatureSelector] Початок підготовки фіч...")
        
        if target_config is None:
            target_config = self._get_default_target_config()
        
        cache_file = self.cache_dir / f"features_{hashlib.md5(str(target_config).encode()).hexdigest()[:8]}.pkl"
        
        if not force_refresh and cache_file.exists():
            try:
                cached_data = pickle.load(open(cache_file, 'rb'))
                logger.info("[FeatureSelector] Використовуємо кешовані фічі")
                return cached_data
            except Exception as e:
                logger.warning(f"[FeatureSelector] Помилка завантаження кешу: {e}")
        
        # Готуємо фічі використовуючи етап 3
        from core.stages.stage_3_features import prepare_stage3_datasets
        
        stage1_data = {}
        stage2_data = {'merged_data': enriched_df}
        config = {'targets': target_config}
        
        feature_results = prepare_stage3_datasets(stage1_data, stage2_data, config)
        
        # Організовуємо результати
        features_dict = {
            'features_by_target': {},
            'context_data': {},
            'metadata': {
                'preparation_time': datetime.now(),
                'target_config': target_config,
                'feature_count': len(enriched_df.columns) if enriched_df is not None else 0
            }
        }
        
        if isinstance(feature_results, dict):
            features_dict['features_by_target'] = feature_results.get('features', {})
            features_dict['context_data'] = feature_results.get('context', pd.DataFrame())
        else:
            merged_stage3, context_df, features_df, trigger_data = feature_results
            features_dict['features_by_target'] = {'default': features_df}
            features_dict['context_data'] = context_df
        
        # Зберігаємо кеш
        pickle.dump(features_dict, open(cache_file, 'wb'))
        
        total_features = sum(len(df.columns) if hasattr(df, 'columns') else 0 
                           for df in features_dict['features_by_target'].values())
        logger.info(f"[FeatureSelector] Підготовлено {total_features} фіч для таргетів")
        
        return features_dict
    
    def _get_default_target_config(self) -> Dict[str, Any]:
        """Отримує конфігурацію таргетів за замовчуванням"""
        return {
            'price_direction': {
                'threshold': 0.02,
                'lookahead_periods': [1, 3, 5],
                'noise_filter': 0.005
            },
            'price_change_pct': {
                'threshold': 0.01,
                'lookahead_periods': [1, 3, 5],
                'noise_filter': 0.003
            },
            'volatility_target': {
                'threshold': 0.015,
                'lookahead_periods': [1, 3],
                'noise_filter': 0.002
            }
        }

class ModelTrainer:
    """Етап 4: Тренування моделей з розділенням локальних/Colab"""
    
    def __init__(self, cache_dir: str = "data/cache/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def train_models(self, features_dict: Dict[str, Any], 
                    model_config: Optional[Dict] = None,
                    force_refresh: bool = False) -> Dict[str, Any]:
        """Тренує моделі з розділенням локальних/Colab"""
        logger.info("[ModelTrainer] Початок тренування моделей...")
        
        if model_config is None:
            model_config = self._get_default_model_config()
        
        cache_file = self.cache_dir / f"training_results_{hashlib.md5(str(model_config).encode()).hexdigest()[:8]}.pkl"
        
        if not force_refresh and cache_file.exists():
            try:
                cached_results = pickle.load(open(cache_file, 'rb'))
                if self._are_models_fresh(cached_results):
                    logger.info("[ModelTrainer] Використовуємо кешовані моделі")
                    return cached_results
            except Exception as e:
                logger.warning(f"[ModelTrainer] Помилка завантаження кешу: {e}")
        
        training_results = {
            'light_models': {},
            'heavy_models': {},
            'metadata': {
                'training_time': datetime.now(),
                'model_config': model_config
            }
        }
        
        # Тренуємо легкі моделі локально
        for target_name, features_df in features_dict['features_by_target'].items():
            if features_df is not None and not features_df.empty:
                light_results = self._train_light_models(features_df, target_name, model_config)
                training_results['light_models'][target_name] = light_results
                
                # Готуємо важкі моделі для Colab
                heavy_data = self._prepare_heavy_models_data(features_df, target_name)
                training_results['heavy_models'][target_name] = heavy_data
        
        # Зберігаємо кеш
        pickle.dump(training_results
