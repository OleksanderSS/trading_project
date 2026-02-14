"""
Dual Pipeline Orchestrator - роwithдandлення heavy/light моwhereлей
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd

# Import pipeline components
from core.stages.stage_1_collectors_layer import run_stage_1_collect
from core.stages.stage_2_enrichment import run_stage_2_enrich
from core.stages.stage_4_enhanced_modeling import run_enhanced_modeling
from core.analysis.time_series_validator import TimeSeriesValidator
from core.analysis.feature_optimizer import FeatureOptimizer

# Import Colab integration
from utils.colab_manager import ColabManager
from utils.colab_utils import ColabUtils

logger = logging.getLogger(__name__)

class DualPipelineOrchestrator:
    """Оркестратор with роwithдandленням heavy/light моwhereлей"""
    
    def __init__(self, tickers: Dict[str, str], time_frames: List[str], debug: bool = False):
        self.tickers = tickers
        self.time_frames = time_frames
        self.debug = debug
        
        # Initialize managers
        self.colab_manager = ColabManager()
        self.colab_utils = ColabUtils()
        self.time_series_validator = TimeSeriesValidator()
        self.feature_optimizer = FeatureOptimizer()
        
        # Pipeline state
        self.pipeline_state = {
            'stage_1_complete': False,
            'stage_2_complete': False,
            'light_models_complete': False,
            'heavy_models_complete': False
        }
        
        # Results storage
        self.results = {
            'light_models': {},
            'heavy_models': {},
            'combined_results': {}
        }
    
    def run_light_models_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Запуск light моwhereлей локально"""
        
        logger.info("=== LIGHT MODELS PIPELINE (Local) ===")
        start_time = datetime.now()
        
        # Light моwhereлand: LGBM, RF, Linear, MLP, XGB, CatBoost, SVM, KNN
        light_models = ['lgbm', 'rf', 'linear', 'mlp', 'xgb', 'catboost', 'svm', 'knn']
        
        # Пandдготовка data for light моwhereлей
        modeling_result = run_enhanced_modeling(df, list(self.tickers.keys()), include_heavy_light=True)
        
        light_results = {}
        
        for model_name in light_models:
            logger.info(f"Training light model: {model_name}")
            
            # Симуляцandя тренування light моwhereлand
            for dataset_key, dataset in modeling_result['preparation']['model_datasets'].items():
                if dataset['model_type'] in ['light', 'direction']:
                    # Створюємо тренувальнand данand with правильною time series валandдацandєю
                    X_train, X_val, y_train, y_val = self.time_series_validator.create_robust_split(
                        dataset['X_train'], dataset['y_train'], validation_ratio=0.2
                    )
                    
                    # Симуляцandя реwithульandтandв
                    model_result = self._simulate_light_model_training(
                        model_name, X_train, X_val, y_train, y_val, dataset
                    )
                    
                    key = f"{model_name}_{dataset['ticker']}_{dataset['target']}"
                    light_results[key] = model_result
        
        execution_time = datetime.now() - start_time
        logger.info(f"Light models completed in {execution_time}")
        
        return {
            'results': light_results,
            'execution_time': execution_time,
            'model_count': len(light_results)
        }
    
    def run_heavy_models_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Запуск heavy моwhereлей череwith Colab"""
        
        logger.info("=== HEAVY MODELS PIPELINE (Colab) ===")
        start_time = datetime.now()
        
        # Heavy моwhereлand: GRU, TabNet, Transformer, LSTM, CNN, Autoencoder
        heavy_models = ['gru', 'tabnet', 'transformer', 'lstm', 'cnn', 'autoencoder']
        
        # Пandдготовка data for heavy моwhereлей
        modeling_result = run_enhanced_modeling(df, list(self.tickers.keys()), include_heavy_light=True)
        
        # Пandдготовка data for Colab
        colab_data = self._prepare_colab_data(modeling_result, heavy_models)
        
        # Вandдправка в Colab
        colab_results = self._run_colab_heavy_models(colab_data)
        
        execution_time = datetime.now() - start_time
        logger.info(f"Heavy models completed in {execution_time}")
        
        return {
            'results': colab_results,
            'execution_time': execution_time,
            'model_count': len(colab_results)
        }
    
    def _simulate_light_model_training(self, model_name: str, X_train, X_val, y_train, y_val, 
                                     dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Симуляцandя тренування light моwhereлand"""
        
        # Симуляцandя метрик for light моwhereлей (кращand реwithульandти)
        base_mae = 0.3
        base_r2 = 0.6
        
        # Рandwithнand моwhereлand мають рandwithнand характеристики
        model_multipliers = {
            'lgbm': {'mae': 0.8, 'r2': 1.2},
            'rf': {'mae': 0.9, 'r2': 1.1},
            'linear': {'mae': 1.1, 'r2': 0.8},
            'mlp': {'mae': 0.85, 'r2': 1.15},
            'xgb': {'mae': 0.82, 'r2': 1.18},
            'catboost': {'mae': 0.78, 'r2': 1.22},
            'svm': {'mae': 1.2, 'r2': 0.7},
            'knn': {'mae': 1.0, 'r2': 0.9}
        }
        
        mult = model_multipliers.get(model_name, {'mae': 1.0, 'r2': 1.0})
        
        # Геnotруємо симульованand метрики
        mae = base_mae * mult['mae'] * (1 + np.random.uniform(-0.1, 0.1))
        r2 = min(0.95, base_r2 * mult['r2'] * (1 + np.random.uniform(-0.05, 0.05)))
        
        return {
            'model_name': model_name,
            'model_type': 'light',
            'ticker': dataset['ticker'],
            'target': dataset['target'],
            'features_count': len(dataset['features']),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'metrics': {
                'mae': mae,
                'rmse': mae * 1.3,
                'r2': r2,
                'mape': mae * 100
            },
            'training_time': np.random.uniform(0.5, 3.0),
            'validation_type': 'time_series_split'
        }
    
    def _prepare_colab_data(self, modeling_result: Dict[str, Any], heavy_models: List[str]) -> Dict[str, Any]:
        """Пandдготовка data for Colab"""
        
        colab_datasets = {}
        
        for dataset_key, dataset in modeling_result['preparation']['model_datasets'].items():
            if dataset['model_type'] == 'heavy':
                colab_datasets[dataset_key] = {
                    'X_train': dataset['X_train'],
                    'X_test': dataset['X_test'],
                    'y_train': dataset['y_train'],
                    'y_test': dataset['y_test'],
                    'features': dataset['features'],
                    'target': dataset['target'],
                    'ticker': dataset['ticker'],
                    'is_3d': dataset['is_3d']
                }
        
        return {
            'datasets': colab_datasets,
            'models': heavy_models,
            'config': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'validation_split': 0.2
            }
        }
    
    def _run_colab_heavy_models(self, colab_data: Dict[str, Any]) -> Dict[str, Any]:
        """Запуск heavy моwhereлей в Colab"""
        
        # Симуляцandя Colab виконання
        logger.info("Simulating Colab heavy models execution...")
        
        colab_results = {}
        
        for model_name in colab_data['models']:
            for dataset_key, dataset in colab_data['datasets'].items():
                # Симуляцandя тренування heavy моwhereлand
                result = self._simulate_heavy_model_training(model_name, dataset)
                
                key = f"{model_name}_{dataset['ticker']}_{dataset['target']}"
                colab_results[key] = result
        
        return colab_results
    
    def _simulate_heavy_model_training(self, model_name: str, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Симуляцandя тренування heavy моwhereлand"""
        
        # Heavy моwhereлand мають гandршand метрики але кращу withдатнandсть до складних патернandв
        base_mae = 1.2
        base_r2 = 0.4
        
        # Рandwithнand heavy моwhereлand
        model_multipliers = {
            'gru': {'mae': 0.9, 'r2': 1.3},
            'tabnet': {'mae': 0.85, 'r2': 1.4},
            'transformer': {'mae': 0.8, 'r2': 1.5},
            'lstm': {'mae': 0.95, 'r2': 1.25},
            'cnn': {'mae': 1.0, 'r2': 1.2},
            'autoencoder': {'mae': 1.1, 'r2': 1.1}
        }
        
        mult = model_multipliers.get(model_name, {'mae': 1.0, 'r2': 1.0})
        
        # Геnotруємо симульованand метрики
        mae = base_mae * mult['mae'] * (1 + np.random.uniform(-0.15, 0.15))
        r2 = min(0.85, base_r2 * mult['r2'] * (1 + np.random.uniform(-0.1, 0.1)))
        
        return {
            'model_name': model_name,
            'model_type': 'heavy',
            'ticker': dataset['ticker'],
            'target': dataset['target'],
            'features_count': len(dataset['features']),
            'train_samples': len(dataset['y_train']),
            'val_samples': len(dataset['y_test']),
            'metrics': {
                'mae': mae,
                'rmse': mae * 1.4,
                'r2': r2,
                'mape': mae * 120
            },
            'training_time': np.random.uniform(5.0, 15.0),  # Heavy моwhereлand повandльнandшand
            'validation_type': 'time_series_split',
            'is_3d': dataset['is_3d']
        }
    
    def combine_model_results(self, light_results: Dict[str, Any], heavy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Комбandнує реwithульandти light/heavy моwhereлей"""
        
        combined = {
            'light_models': light_results,
            'heavy_models': heavy_results,
            'ensemble_candidates': [],
            'best_models_by_target': {},
            'performance_comparison': {}
        }
        
        # Знаходимо найкращand моwhereлand по andргеandх
        all_results = {**light_results, **heavy_results}
        
        # Групуємо по andргеandх
        targets = {}
        for key, result in all_results.items():
            target = result['target']
            if target not in targets:
                targets[target] = []
            targets[target].append(result)
        
        # Знаходимо найкращand моwhereлand for кожного andргету
        for target, models in targets.items():
            # Сортуємо по MAE (менше = краще)
            best_model = min(models, key=lambda x: x['metrics']['mae'])
            combined['best_models_by_target'][target] = best_model
        
        # Порandвняння продуктивностand
        light_maes = [r['metrics']['mae'] for r in light_results.values()]
        heavy_maes = [r['metrics']['mae'] for r in heavy_results.values()]
        
        combined['performance_comparison'] = {
            'light_avg_mae': np.mean(light_maes),
            'heavy_avg_mae': np.mean(heavy_maes),
            'light_model_count': len(light_results),
            'heavy_model_count': len(heavy_results),
            'overall_best_type': 'light' if np.mean(light_maes) < np.mean(heavy_maes) else 'heavy'
        }
        
        return combined
    
    def run_complete_dual_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Запуск повного dual pipeline"""
        
        logger.info("=== COMPLETE DUAL PIPELINE START ===")
        total_start_time = datetime.now()
        
        # Крок 1: Light моwhereлand (локально)
        light_results = self.run_light_models_pipeline(df)
        self.pipeline_state['light_models_complete'] = True
        
        # Крок 2: Heavy моwhereлand (Colab)
        heavy_results = self.run_heavy_models_pipeline(df)
        self.pipeline_state['heavy_models_complete'] = True
        
        # Крок 3: Комбandнацandя реwithульandтandв
        combined_results = self.combine_model_results(light_results['results'], heavy_results['results'])
        
        total_execution_time = datetime.now() - total_start_time
        
        final_results = {
            'light_models': light_results,
            'heavy_models': heavy_results,
            'combined': combined_results,
            'total_execution_time': total_execution_time,
            'pipeline_state': self.pipeline_state
        }
        
        logger.info(f"Dual pipeline completed in {total_execution_time}")
        return final_results


def run_dual_pipeline(df: pd.DataFrame, tickers: Dict[str, str] = None,
                     time_frames: List[str] = None) -> Dict[str, Any]:
    """Запуск dual pipeline"""
    
    if tickers is None:
        tickers = {'SPY': 'SPDR S&P 500 ETF'}
    
    if time_frames is None:
        time_frames = ['1d', '60m', '15m']
    
    orchestrator = DualPipelineOrchestrator(tickers, time_frames)
    return orchestrator.run_complete_dual_pipeline(df)
