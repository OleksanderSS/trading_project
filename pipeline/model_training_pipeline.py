#!/usr/bin/env python3
"""
Complete Model Training Pipeline - повний пайплайн для навчання моделі
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json

from utils.intelligent_data_filter import IntelligentDataFilter
from utils.model_feature_engineering import ModelFeatureEngineering
# ВИПРАВЛЕНО: використовуємо StageManager замість видаленої функції
from core.stages.stage_manager import StageManager
from core.stages.stage_2_enrichment import run_stage_2_enrich_ideal

logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    """
    [START] Повний пайплайн для навчання моделі - як я б зробив для себе
    """
    
    def __init__(self):
        self.data_filter = IntelligentDataFilter()
        self.feature_engineer = ModelFeatureEngineering()
        
        # [TARGET] НАЛАШТУВАННЯ ПАЙПЛАЙНУ
        self.config = {
            'min_candles_per_timeframe': 2,
            'min_quality_score': 0.6,
            'target_timeframes': ['15m', '1h', '1d'],
            'feature_selection': True,
            'validation_split': 0.2,
            'time_series_split': True
        }
    
    def run_complete_pipeline(self, tickers=None, timeframes=None, use_free_data=True) -> Dict:
        """
        [START] Запускаємо повний пайплайн
        """
        logger.info("[START] Starting Complete Model Training Pipeline")
        
        # [TARGET] ЕТАП 1: ЗБІР ДАНИХ
        logger.info("[DATA] Step 1: Data Collection")
        raw_data = self._collect_data(tickers, timeframes, use_free_data)
        
        # [TARGET] ЕТАП 2: ФІЛЬТРАЦІЯ ТА АНАЛІЗ ЯКОСТІ
        logger.info("[SEARCH] Step 2: Data Filtering & Quality Analysis")
        filtered_results = self._filter_and_analyze_data(raw_data)
        
        # [TARGET] ЕТАП 3: ВИВЧЕННЯ ПАТЕРНІВ
        logger.info("[TARGET] Step 3: Pattern Extraction")
        patterns = filtered_results['patterns']
        
        # [TARGET] ЕТАП 4: СТВОРЕННЯ ОСОБЛИВОСТЕЙ
        logger.info("⚙️ Step 4: Feature Engineering")
        features = self._create_features(filtered_results['filtered_data'], patterns)
        
        # [TARGET] ЕТАП 5: ПІДГОТОВКА ДАНИХ ДЛЯ МОДЕЛІ
        logger.info("[LIST] Step 5: Model Data Preparation")
        model_data = self._prepare_model_data(features, patterns)
        
        # [TARGET] ЕТАП 6: ВАЛІДАЦІЯ ТА РОЗДІЛЕННЯ
        logger.info("[OK] Step 6: Validation & Splitting")
        train_data, val_data, test_data = self._validate_and_split_data(model_data)
        
        # [TARGET] ЕТАП 7: ЗВІТ ПРО ПІДГОТОВКУ
        logger.info("[UP] Step 7: Preparation Report")
        report = self._create_preparation_report(filtered_results, features, model_data)
        
        logger.info("[OK] Pipeline completed successfully!")
        
        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'filtered_data': filtered_results['filtered_data'],
            'patterns': patterns,
            'features': features,
            'quality_report': filtered_results['quality_report'],
            'preparation_report': report
        }
    
    def _collect_data(self, tickers=None, timeframes=None, use_free_data=True):
        """
        [START] Збираємо дані з усіх джерел
        """
        logger.info(f"[DATA] Collecting data for tickers: {tickers}, timeframes: {timeframes}")
        
        # [TARGET] Етап 1: Збір сирих data
        stage1_data = run_stage_1_collect_intelligent(
            tickers=tickers,
            timeframes=timeframes,
            use_free_data=use_free_data,
            enable_cache=True,
            cache_ttl_hours=24
        )
        
        # [TARGET] Етап 2: Збагачення data
        stage2_data = run_stage_2_enrich_ideal(
            stage1_data=stage1_data,
            tickers=tickers,
            timeframes=timeframes,
            use_free_data=use_free_data
        )
        
        # [TARGET] Об'єднуємо дані
        combined_data = {
            **stage1_data,
            **stage2_data
        }
        
        logger.info(f"[DATA] Data collection completed: {len(combined_data)} data sources")
        return combined_data
    
    def _filter_and_analyze_data(self, raw_data: Dict) -> Dict:
        """
        [START] Фільтруємо та аналізуємо якість data
        """
        logger.info("[SEARCH] Filtering and analyzing data quality")
        
        # [TARGET] Інтелектуальна фільтрація
        filtered_results = self.data_filter.filter_quality_data(raw_data)
        
        # [TARGET] Логуємо результати фільтрації
        quality_report = filtered_results['quality_report']
        filtering_summary = filtered_results['filtering_summary']
        
        logger.info(f"[DATA] Quality Report:")
        logger.info(f"   Total sources: {filtering_summary['total_data_sources']}")
        logger.info(f"   Accepted: {filtering_summary['accepted_sources']}")
        logger.info(f"   Rejected: {filtering_summary['rejected_sources']}")
        logger.info(f"   Overall quality: {filtering_summary['overall_quality_score']:.2f}")
        
        return filtered_results
    
    def _create_features(self, filtered_data: Dict, patterns: Dict) -> Dict:
        """
        [START] Створюємо фічі для моделі
        """
        logger.info("⚙️ Creating features for model")
        
        # [TARGET] Створення фіч
        features = self.feature_engineer.create_model_features(filtered_data, patterns)
        
        # [TARGET] Логуємо статистику фіч
        feature_stats = self._calculate_feature_statistics(features)
        
        logger.info(f"[DATA] Feature Statistics:")
        for category, stats in feature_stats.items():
            logger.info(f"   {category}: {stats['total_features']} features")
        
        return features
    
    def _prepare_model_data(self, features: Dict, patterns: Dict) -> Dict:
        """
        [START] Готуємо дані для моделі
        """
        logger.info("[LIST] Preparing data for model training")
        
        model_data = {}
        
        # [TARGET] Обробляємо кожен timeframe
        for timeframe in self.config['target_timeframes']:
            if timeframe in features.get('price_features', {}):
                logger.info(f"[DATA] Processing {timeframe} timeframe")
                
                # [TARGET] Збираємо всі фічі для timeframe
                timeframe_data = self._collect_timeframe_data(features, timeframe)
                
                # [TARGET] Вирівнюємо дані
                aligned_data = self._align_timeframe_data(timeframe_data)
                
                # [TARGET] Очищуємо та фінально готуємо
                final_data = self._final_data_preparation(aligned_data)
                
                if final_data['X'].shape[0] > 0:  # Перевіряємо що є дані
                    model_data[timeframe] = final_data
                    logger.info(f"[OK] {timeframe}: {final_data['X'].shape[0]} samples, {final_data['X'].shape[1]} features")
                else:
                    logger.warning(f"[WARN] {timeframe}: No valid data after preparation")
        
        return model_data
    
    def _collect_timeframe_data(self, features: Dict, timeframe: str) -> Dict:
        """
        [START] Збираємо всі фічі для конкретного timeframe
        """
        timeframe_data = {}
        
        # [TARGET] Price features
        if timeframe in features.get('price_features', {}):
            timeframe_data.update(features['price_features'][timeframe])
        
        # [TARGET] Volume features
        if timeframe in features.get('volume_features', {}):
            timeframe_data.update(features['volume_features'][timeframe])
        
        # [TARGET] Pattern features
        if timeframe in features.get('pattern_features', {}):
            timeframe_data.update(features['pattern_features'][timeframe])
        
        # [TARGET] Target labels
        if timeframe in features.get('target_labels', {}):
            timeframe_data['targets'] = features['target_labels'][timeframe]
        
        # [TARGET] Global features (однакові для всіх timeframe)
        for category in ['sentiment_features', 'news_features', 'regime_features']:
            if category in features:
                for key, value in features[category].items():
                    if isinstance(value, (int, float)):
                        timeframe_data[f'global_{key}'] = value
        
        return timeframe_data
    
    def _align_timeframe_data(self, timeframe_data: Dict) -> Dict:
        """
        [START] Вирівнюємо дані за часом та довжиною
        """
        # [TARGET] Знаходимо мінімальну довжину
        min_length = float('inf')
        
        for key, value in timeframe_data.items():
            if key != 'targets' and isinstance(value, np.ndarray):
                if len(value) < min_length:
                    min_length = len(value)
        
        if min_length == float('inf'):
            return {'X': np.array([]), 'y': np.array([]), 'feature_names': []}
        
        # [TARGET] Вирівнюємо всі масиви
        aligned_data = {}
        feature_names = []
        
        for key, value in timeframe_data.items():
            if key != 'targets' and isinstance(value, np.ndarray):
                if len(value) >= min_length:
                    aligned_data[key] = value[:min_length]
                    feature_names.append(key)
                else:
                    # Якщо масив коротший - розширюємо NaN
                    padded = np.full(min_length, np.nan)
                    padded[:len(value)] = value
                    aligned_data[key] = padded
                    feature_names.append(key)
        
        # [TARGET] Створюємо матрицю фіч
        if aligned_data:
            X = np.column_stack([aligned_data[key] for key in feature_names])
        else:
            X = np.array([])
        
        # [TARGET] Обробляємо таргети
        y = np.array([])
        if 'targets' in timeframe_data:
            targets = timeframe_data['targets']
            
            # Вибираємо основний таргет (наприклад, future_return_1)
            target_keys = [k for k in targets.keys() if 'future_return_1' in k]
            if target_keys:
                y = targets[target_keys[0]][:min_length] if len(targets[target_keys[0]]) >= min_length else np.full(min_length, np.nan)
        
        return {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'original_data': aligned_data
        }
    
    def _final_data_preparation(self, aligned_data: Dict) -> Dict:
        """
        [START] Фінальна підготовка data
        """
        X = aligned_data['X']
        y = aligned_data['y']
        feature_names = aligned_data['feature_names']
        
        # [TARGET] Видаляємо рядки з NaN в таргетах
        valid_indices = ~np.isnan(y)
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        
        # [TARGET] Видаляємо фічі з занадто багато NaN
        feature_nan_ratios = np.isnan(X_clean).sum(axis=0) / len(X_clean)
        valid_features = feature_nan_ratios < 0.5  # Менше 50% NaN
        
        X_clean = X_clean[:, valid_features]
        feature_names = [feature_names[i] for i, valid in enumerate(valid_features) if valid]
        
        # [TARGET] Заповнюємо NaN в фічах
        for i in range(X_clean.shape[1]):
            nan_mask = np.isnan(X_clean[:, i])
            if nan_mask.any():
                # Заповнюємо медіаною
                median_val = np.nanmedian(X_clean[:, i])
                X_clean[nan_mask, i] = median_val
        
        # [TARGET] Видаляємо константні фічі
        feature_std = np.std(X_clean, axis=0)
        non_constant_features = feature_std > 1e-8
        
        X_clean = X_clean[:, non_constant_features]
        feature_names = [feature_names[i] for i, valid in enumerate(non_constant_features) if valid]
        
        # [TARGET] Нормалізація фіч (optional)
        # X_clean = (X_clean - X_clean.mean(axis=0)) / X_clean.std(axis=0)
        
        return {
            'X': X_clean,
            'y': y_clean,
            'feature_names': feature_names,
            'samples': len(X_clean),
            'features': len(feature_names)
        }
    
    def _validate_and_split_data(self, model_data: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        [START] Валідуємо та розподіляємо дані
        """
        train_data = {}
        val_data = {}
        test_data = {}
        
        for timeframe, data in model_data.items():
            X, y = data['X'], data['y']
            
            if len(X) < 10:
                logger.warning(f"[WARN] {timeframe}: Insufficient data for splitting")
                continue
            
            # [TARGET] Time series split (важливо для фінансових data!)
            n_samples = len(X)
            val_size = int(n_samples * self.config['validation_split'])
            test_size = int(n_samples * self.config['validation_split'])
            
            # [TARGET] Хронологічний спліт
            X_train = X[:n_samples - val_size - test_size]
            y_train = y[:n_samples - val_size - test_size]
            
            X_val = X[n_samples - val_size - test_size:n_samples - test_size]
            y_val = y[n_samples - val_size - test_size:n_samples - test_size]
            
            X_test = X[n_samples - test_size:]
            y_test = y[n_samples - test_size:]
            
            train_data[timeframe] = {
                'X': X_train,
                'y': y_train,
                'feature_names': data['feature_names']
            }
            
            val_data[timeframe] = {
                'X': X_val,
                'y': y_val,
                'feature_names': data['feature_names']
            }
            
            test_data[timeframe] = {
                'X': X_test,
                'y': y_test,
                'feature_names': data['feature_names']
            }
            
            logger.info(f"[DATA] {timeframe} split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return train_data, val_data, test_data
    
    def _calculate_feature_statistics(self, features: Dict) -> Dict:
        """
        [START] Розраховуємо статистику фіч
        """
        stats = {}
        
        for category, category_features in features.items():
            if isinstance(category_features, dict):
                total_features = 0
                for key, value in category_features.items():
                    if isinstance(value, dict):
                        total_features += len(value)
                    elif isinstance(value, (int, float)):
                        total_features += 1
                    elif isinstance(value, np.ndarray):
                        total_features += 1
                
                stats[category] = {
                    'total_features': total_features,
                    'subcategories': len(category_features) if isinstance(category_features, dict) else 1
                }
            else:
                stats[category] = {
                    'total_features': 1,
                    'subcategories': 1
                }
        
        return stats
    
    def _create_preparation_report(self, filtered_results: Dict, features: Dict, model_data: Dict) -> Dict:
        """
        [START] Створюємо звіт про підготовку
        """
        report = {
            'pipeline_summary': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'success': True
            },
            'data_quality_summary': filtered_results['filtering_summary'],
            'feature_summary': self._calculate_feature_statistics(features),
            'model_data_summary': {},
            'recommendations': []
        }
        
        # [TARGET] Статистика модельних data
        for timeframe, data in model_data.items():
            report['model_data_summary'][timeframe] = {
                'samples': data['samples'],
                'features': data['features'],
                'feature_names': data['feature_names'][:10]  # Перші 10 фіч
            }
        
        # [TARGET] Рекомендації
        if report['data_quality_summary']['overall_quality_score'] < 0.7:
            report['recommendations'].append("Consider improving data quality filters")
        
        total_samples = sum(data['samples'] for data in model_data.values())
        if total_samples < 1000:
            report['recommendations'].append("Consider collecting more data for better model performance")
        
        return report
    
    def save_pipeline_results(self, results: Dict, filepath: str = None):
        """
        [START] Зберігаємо результати пайплайну
        """
        if filepath is None:
            filepath = f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # [TARGET] Конвертуємо numpy arrays в lists для JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # [TARGET] Готуємо дані для збереження
        saveable_results = convert_numpy(results)
        
        # [TARGET] Зберігаємо тільки метадані (не самі дані)
        metadata_to_save = {
            'preparation_report': saveable_results['preparation_report'],
            'quality_report': saveable_results['quality_report'],
            'feature_summary': saveable_results['feature_summary'],
            'model_data_summary': saveable_results['model_data_summary']
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[SAVE] Pipeline results saved to {filepath}")
        return filepath


# [TARGET] ГОЛОВНА ФУНКЦІЯ
def run_complete_model_training_pipeline(tickers=None, timeframes=None, use_free_data=True):
    """
    [START] Запускаємо повний пайплайн для навчання моделі
    """
    pipeline = ModelTrainingPipeline()
    return pipeline.run_complete_pipeline(tickers, timeframes, use_free_data)


if __name__ == "__main__":
    print("Complete Model Training Pipeline - готовий до використання")
    print("[START] Повний пайплайн: збір data → фільтрація → патерни → фічі → модель")
    print("[DATA] Не видаляємо аномалії, а використовуємо їх як сигнали для моделі!")
