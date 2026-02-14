#!/usr/bin/env python3
"""
Final Unified Trading Pipeline - Complete Implementation
Реалізує повну логіку описану користувачем з уніфікованою архітектурою
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

logger = ProjectLogger.get_logger("FinalPipeline")

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
        pickle.dump(training_results, open(cache_file, 'wb'))
        
        logger.info(f"[ModelTrainer] Треновано {len(training_results['light_models'])} легких моделей")
        return training_results
    
    def _train_light_models(self, features_df: pd.DataFrame, target_name: str, 
                           model_config: Dict) -> Dict[str, Any]:
        """Тренує легкі моделі локально"""
        from core.stages.stage_4_benchmark import benchmark_all_models
        
        light_models = [model for model in model_config.get('light_models', []) 
                       if model in ['linear', 'random_forest', 'xgboost', 'lightgbm']]
        
        try:
            results = benchmark_all_models(features_df, models=light_models)
            return {
                'results': results,
                'model_type': 'light',
                'target': target_name,
                'training_time': datetime.now()
            }
        except Exception as e:
            logger.error(f"[ModelTrainer] Помилка тренування легких моделей для {target_name}: {e}")
            return {'error': str(e), 'model_type': 'light', 'target': target_name}
    
    def _prepare_heavy_models_data(self, features_df: pd.DataFrame, target_name: str) -> Dict[str, Any]:
        """Готує дані для тренування важких моделей в Colab"""
        colab_dir = Path("data/colab/for_training")
        colab_dir.mkdir(parents=True, exist_ok=True)
        
        # Зберігаємо фічі для Colab
        features_file = colab_dir / f"features_{target_name}.parquet"
        features_df.to_parquet(features_file)
        
        # Створюємо конфігурацію тренування
        training_config = {
            'target_name': target_name,
            'features_shape': features_df.shape,
            'features_file': str(features_file),
            'heavy_models': ['lstm', 'transformer', 'bert', 'gpt'],
            'training_params': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'validation_split': 0.2
            }
        }
        
        # Зберігаємо конфігурацію
        config_file = colab_dir / f"config_{target_name}.json"
        with open(config_file, 'w') as f:
            json.dump(training_config, f, indent=2)
        
        return {
            'config': training_config,
            'features_file': str(features_file),
            'config_file': str(config_file),
            'model_type': 'heavy',
            'target': target_name,
            'preparation_time': datetime.now()
        }
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        """Отримує конфігурацію моделей за замовчуванням"""
        return {
            'light_models': ['linear', 'random_forest', 'xgboost', 'lightgbm'],
            'heavy_models': ['lstm', 'transformer', 'bert', 'gpt'],
            'evaluation_metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'random_state': 42
        }
    
    def _are_models_fresh(self, cached_results: Dict[str, Any], max_age_hours: int = 24) -> bool:
        """Перевіряє чи моделі свіжі"""
        training_time = cached_results.get('metadata', {}).get('training_time')
        if not training_time:
            return False
        
        age = datetime.now() - training_time
        return age.total_seconds() < max_age_hours * 3600

class ModelComparator:
    """Етап 5: Порівняння моделей та генерація фінальних сигналів"""
    
    def __init__(self, cache_dir: str = "data/cache/comparison"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def compare_models(self, training_results: Dict[str, Any], 
                       force_refresh: bool = False) -> Dict[str, Any]:
        """Порівнює моделі та вибирає найкращі"""
        logger.info("[ModelComparator] Початок порівняння моделей...")
        
        cache_file = self.cache_dir / "comparison_results.pkl"
        
        if not force_refresh and cache_file.exists():
            try:
                cached_results = pickle.load(open(cache_file, 'rb'))
                if self._is_comparison_fresh(cached_results):
                    logger.info("[ModelComparator] Використовуємо кешовані результати порівняння")
                    return cached_results
            except Exception as e:
                logger.warning(f"[ModelComparator] Помилка завантаження кешу: {e}")
        
        comparison_results = {
            'best_models': {},
            'model_rankings': {},
            'performance_metrics': {},
            'final_signals': {},
            'metadata': {
                'comparison_time': datetime.now(),
                'total_models_compared': 0
            }
        }
        
        # Порівнюємо легкі моделі
        for target_name, light_results in training_results['light_models'].items():
            if 'error' not in light_results:
                best_light = self._find_best_model(light_results['results'], 'light')
                comparison_results['best_models'][f'{target_name}_light'] = best_light
                comparison_results['model_rankings'][f'{target_name}_light'] = self._rank_models(light_results['results'])
        
        # Порівнюємо важкі моделі (якщо є результати)
        heavy_results_path = Path("data/colab/from_training")
        if heavy_results_path.exists():
            for target_name, heavy_results in training_results['heavy_models'].items():
                results_file = heavy_results_path / f"results_{target_name}.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        heavy_data = json.load(f)
                        best_heavy = self._find_best_model(heavy_data, 'heavy')
                        comparison_results['best_models'][f'{target_name}_heavy'] = best_heavy
                        comparison_results['model_rankings'][f'{target_name}_heavy'] = self._rank_models(heavy_data)
        
        # Генеруємо фінальні сигнали
        comparison_results['final_signals'] = self._generate_final_signals(comparison_results['best_models'])
        
        # Підраховуємо загальну кількість моделей
        total_models = sum(len(ranking) for ranking in comparison_results['model_rankings'].values())
        comparison_results['metadata']['total_models_compared'] = total_models
        
        # Зберігаємо кеш
        pickle.dump(comparison_results, open(cache_file, 'wb'))
        
        logger.info(f"[ModelComparator] Порівняно {total_models} моделей")
        return comparison_results
    
    def _find_best_model(self, model_results: Dict[str, Dict], model_type: str) -> Dict[str, Any]:
        """Знаходить найкращу модель серед результатів"""
        if not model_results:
            return {'model': 'none', 'score': 0, 'metrics': {}}
        
        # Використовуємо F1 score як основну метрику
        best_model = None
        best_score = -1
        
        for model_name, metrics in model_results.items():
            score = metrics.get('f1_score', 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if best_model:
            return {
                'model': best_model,
                'score': best_score,
                'metrics': model_results[best_model],
                'type': model_type
            }
        
        return {'model': 'none', 'score': 0, 'metrics': {}, 'type': model_type}
    
    def _rank_models(self, model_results: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Ранжує моделі за продуктивністю"""
        if not model_results:
            return []
        
        ranked_models = []
        for model_name, metrics in model_results.items():
            ranked_models.append({
                'model': model_name,
                'f1_score': metrics.get('f1_score', 0),
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0)
            })
        
        # Сортуємо за F1 score
        ranked_models.sort(key=lambda x: x['f1_score'], reverse=True)
        return ranked_models
    
    def _generate_final_signals(self, best_models: Dict[str, Dict]) -> Dict[str, Dict]:
        """Генерує фінальні торгові сигнали"""
        signals = {}
        
        for target_key, best_model_info in best_models.items():
            target_name = target_key.replace('_light', '').replace('_heavy', '')
            model_type = 'light' if '_light' in target_key else 'heavy'
            
            # Симулюємо сигнал на основі метрик моделі
            accuracy = best_model_info['metrics'].get('accuracy', 0.5)
            
            # Генеруємо сигнал на основі точності
            if accuracy > 0.7:
                signal = 1  # BUY
                confidence = accuracy
            elif accuracy < 0.4:
                signal = -1  # SELL
                confidence = 1 - accuracy
            else:
                signal = 0  # HOLD
                confidence = 0.5
            
            signals[target_name] = {
                'signal': signal,
                'confidence': confidence,
                'recommended_model': best_model_info['model'],
                'model_type': model_type,
                'accuracy': accuracy,
                'f1_score': best_model_info['metrics'].get('f1_score', 0)
            }
        
        return signals
    
    def _is_comparison_fresh(self, data: Dict[str, Any], max_age_hours: int = 6) -> bool:
        """Перевіряє чи результати порівняння свіжі"""
        comparison_time = data.get('metadata', {}).get('comparison_time')
        if not comparison_time:
            return False
        
        age = datetime.now() - comparison_time
        return age.total_seconds() < max_age_hours * 3600

class FinalPipeline:
    """Фінальний уніфікований пайплайн"""
    
    def __init__(self, cache_base_dir: str = "data/cache"):
        self.cache_base_dir = Path(cache_base_dir)
        
        # Ініціалізуємо всі етапи
        self.data_parser = DataParser(str(self.cache_base_dir / "raw"))
        self.data_enricher = DataEnricher(str(self.cache_base_dir / "enriched"))
        self.feature_selector = FeatureSelector(str(self.cache_base_dir / "features"))
        self.model_trainer = ModelTrainer(str(self.cache_base_dir / "models"))
        self.model_comparator = ModelComparator(str(self.cache_base_dir / "comparison"))
        
        logger.info("[FinalPipeline] Initialized all pipeline stages")
    
    def run_complete_pipeline(self, 
                            target_config: Optional[Dict] = None,
                            model_config: Optional[Dict] = None,
                            force_refresh: bool = False) -> Dict[str, Any]:
        """Запускає повний пайплайн від початку до кінця"""
        logger.info("[FinalPipeline] Запуск повного пайплайну...")
        start_time = datetime.now()
        
        results = {
            'metadata': {
                'start_time': start_time,
                'pipeline_version': 'final_unified',
                'force_refresh': force_refresh
            }
        }
        
        try:
            # Етап 1: Парсинг даних
            logger.info("🔄 Етап 1: Парсинг сирих даних...")
            raw_data = self.data_parser.parse_all_data(force_refresh)
            results['stage_1_collection'] = {
                'status': 'completed',
                'data_types': list(raw_data.keys()),
                'timestamp': datetime.now()
            }
            
            # Етап 2: Збагачення даних
            logger.info("🔄 Етап 2: Збагачення даних...")
            enriched_df, enrichment_metadata = self.data_enricher.enrich_data(raw_data, force_refresh)
            results['stage_2_enrichment'] = {
                'status': 'completed',
                'shape': enriched_df.shape if enriched_df is not None else None,
                'metadata': enrichment_metadata,
                'timestamp': datetime.now()
            }
            
            # Етап 3: Вибір фіч
            logger.info("🔄 Етап 3: Вибір фіч...")
            features_dict = self.feature_selector.prepare_features(enriched_df, target_config, force_refresh)
            results['stage_3_features'] = {
                'status': 'completed',
                'targets': list(features_dict['features_by_target'].keys()),
                'metadata': features_dict['metadata'],
                'timestamp': datetime.now()
            }
            
            # Етап 4: Тренування моделей
            logger.info("🔄 Етап 4: Тренування моделей...")
            training_results = self.model_trainer.train_models(features_dict, model_config, force_refresh)
            results['stage_4_training'] = {
                'status': 'completed',
                'light_models': list(training_results['light_models'].keys()),
                'heavy_models': list(training_results['heavy_models'].keys()),
                'metadata': training_results['metadata'],
                'timestamp': datetime.now()
            }
            
            # Етап 5: Порівняння моделей
            logger.info("🔄 Етап 5: Порівняння моделей...")
            comparison_results = self.model_comparator.compare_models(training_results, force_refresh)
            results['stage_5_comparison'] = {
                'status': 'completed',
                'best_models': list(comparison_results['best_models'].keys()),
                'final_signals': list(comparison_results['final_signals'].keys()),
                'metadata': comparison_results['metadata'],
                'timestamp': datetime.now()
            }
            
            # Фінальні сигнали
            results['final_signals'] = comparison_results['final_signals']
            
            # Розрахунок тривалості
            end_time = datetime.now()
            duration = end_time - start_time
            results['metadata']['end_time'] = end_time
            results['metadata']['duration'] = str(duration)
            results['metadata']['status'] = 'success'
            
            logger.info(f"[FinalPipeline] ✅ Пайплайн успішно завершено за {duration}")
            
        except Exception as e:
            logger.error(f"[FinalPipeline] ❌ Помилка в пайплайні: {e}")
            results['metadata']['status'] = 'failed'
            results['metadata']['error'] = str(e)
            results['metadata']['end_time'] = datetime.now()
            
            raise
        
        return results
    
    def run_stage_only(self, stage: int, **kwargs) -> Dict[str, Any]:
        """Запускає тільки вказаний етап"""
        logger.info(f"[FinalPipeline] Запуск тільки етапу {stage}...")
        
        if stage == 1:
            raw_data = self.data_parser.parse_all_data(kwargs.get('force_refresh', False))
            return {'status': 'completed', 'data': raw_data}
        
        elif stage == 2:
            # Потрібні дані з етапу 1
            raw_data = kwargs.get('raw_data')
            if not raw_data:
                raw_data = self.data_parser.parse_all_data(kwargs.get('force_refresh', False))
            
            enriched_df, metadata = self.data_enricher.enrich_data(raw_data, kwargs.get('force_refresh', False))
            return {'status': 'completed', 'data': enriched_df, 'metadata': metadata}
        
        elif stage == 3:
            # Потрібні дані з етапу 2
            enriched_df = kwargs.get('enriched_df')
            if not enriched_df:
                # Запускаємо етапи 1-2
                raw_data = self.data_parser.parse_all_data(kwargs.get('force_refresh', False))
                enriched_df, _ = self.data_enricher.enrich_data(raw_data, kwargs.get('force_refresh', False))
            
            features_dict = self.feature_selector.prepare_features(
                enriched_df, 
                kwargs.get('target_config'), 
                kwargs.get('force_refresh', False)
            )
            return {'status': 'completed', 'data': features_dict}
        
        elif stage == 4:
            # Потрібні дані з етапу 3
            features_dict = kwargs.get('features_dict')
            if not features_dict:
                # Запускаємо етапи 1-3
                raw_data = self.data_parser.parse_all_data(kwargs.get('force_refresh', False))
                enriched_df, _ = self.data_enricher.enrich_data(raw_data, kwargs.get('force_refresh', False))
                features_dict = self.feature_selector.prepare_features(
                    enriched_df, 
                    kwargs.get('target_config'), 
                    kwargs.get('force_refresh', False)
                )
            
            training_results = self.model_trainer.train_models(
                features_dict, 
                kwargs.get('model_config'), 
                kwargs.get('force_refresh', False)
            )
            return {'status': 'completed', 'data': training_results}
        
        elif stage == 5:
            # Потрібні дані з етапу 4
            training_results = kwargs.get('training_results')
            if not training_results:
                # Запускаємо етапи 1-4
                raw_data = self.data_parser.parse_all_data(kwargs.get('force_refresh', False))
                enriched_df, _ = self.data_enricher.enrich_data(raw_data, kwargs.get('force_refresh', False))
                features_dict = self.feature_selector.prepare_features(
                    enriched_df, 
                    kwargs.get('target_config'), 
                    kwargs.get('force_refresh', False)
                )
                training_results = self.model_trainer.train_models(
                    features_dict, 
                    kwargs.get('model_config'), 
                    kwargs.get('force_refresh', False)
                )
            
            comparison_results = self.model_comparator.compare_models(
                training_results, 
                kwargs.get('force_refresh', False)
            )
            return {'status': 'completed', 'data': comparison_results}
        
        else:
            raise ValueError(f"Невідомий етап: {stage}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Отримує статус пайплайну та кешів"""
        status = {
            'pipeline_version': 'final_unified',
            'cache_status': {},
            'last_runs': {}
        }
        
        # Перевіряємо статус кешів
        cache_dirs = ['raw', 'enriched', 'features', 'models', 'comparison']
        for cache_dir in cache_dirs:
            cache_path = self.cache_base_dir / cache_dir
            if cache_path.exists():
                files = list(cache_path.glob("*"))
                status['cache_status'][cache_dir] = {
                    'exists': True,
                    'files_count': len(files),
                    'latest_file': max(files, key=lambda f: f.stat().st_mtime).name if files else None
                }
            else:
                status['cache_status'][cache_dir] = {'exists': False, 'files_count': 0}
        
        return status

# Глобальний екземпляр для зручного використання
final_pipeline = FinalPipeline()
