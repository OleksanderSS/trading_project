"""
Enhanced Stage 4 Modeling with Heavy/Light Logic
Роwithдandлення# Роwithширена model with оптимandforцandєю
"""

import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

from config.config import DATA_INTERVALS, TICKERS
from core.stages.stage_3_utils import add_targets
from core.analysis.feature_optimizer import FeatureOptimizer
from core.analysis.model_comparison_engine import ModelComparisonEngine
from utils.feature_scaler import FeatureScaler

logger = logging.getLogger(__name__)

# === GEMINI ENHANCED MODELING CONFIG ===
class EnhancedModelingConfig:
    """Конфandгурацandя for покращеного моwhereлювання"""
    
    # Dynamic Weighting for рandwithних авторandв
    AUTHOR_SENTIMENT_WEIGHTS = {
        'elon_musk': {'TSLA': 1.0, 'SPY': 0.1, 'QQQ': 0.2, 'NVDA': 0.3},
        'trump': {'SPY': 0.8, 'QQQ': 0.6, 'TSLA': 0.4, 'NVDA': 0.3},
        'fed_chair': {'QQQ': 1.0, 'SPY': 0.9, 'TSLA': 0.2, 'NVDA': 0.3},
        'default': {'SPY': 0.5, 'QQQ': 0.5, 'TSLA': 0.5, 'NVDA': 0.5}
    }
    
    # Causal Lag - 2 whereнний andргет
    CAUSAL_LAG_DAYS = 2
    
    # Float32 оптимandforцandя
    PRECISION = 'float32'
    
    # Моwhereлand for важких/легких ринкandв
    HEAVY_MARKET_MODELS = ['LSTM', 'Transformer', 'XGBoost']
    LIGHT_MARKET_MODELS = ['RandomForest', 'LogisticRegression', 'LinearRegression']

class DynamicWeightingProcessor:
    """Обробка динамandчних ваг for сентименту"""
    
    def __init__(self, config: EnhancedModelingConfig):
        self.config = config
        self.author_weights = config.AUTHOR_SENTIMENT_WEIGHTS
        
    def detect_author_from_source(self, df: pd.DataFrame) -> pd.Series:
        """Виwithначає автора новини with джерела"""
        author_mapping = {
            'twitter': 'elon_musk',
            'x.com': 'elon_musk',
            'reuters': 'default',
            'bloomberg': 'default',
            'cnbc': 'default',
            'federalreserve': 'fed_chair',
            'whitehouse': 'trump'
        }
        
        def map_author(source):
            if pd.isna(source):
                return 'default'
            source_lower = str(source).lower()
            for key, author in author_mapping.items():
                if key in source_lower:
                    return author
            return 'default'
        
        return df['source'].apply(map_author)
    
    def apply_dynamic_sentiment_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dynamic Weighting: Рandwithнand ваги сентименту for рandwithних авторandв
        """
        logger.info("[Stage4] [BRAIN] Applying dynamic sentiment weighting...")
        
        # Виwithначаємо автора
        df['detected_author'] = self.detect_author_from_source(df)
        
        # Створюємо вагований сентимент
        df['weighted_sentiment_score'] = 0.0
        
        for author, weights in self.author_weights.items():
            author_mask = df['detected_author'] == author
            
            for ticker, weight in weights.items():
                ticker_mask = df['ticker'] == ticker
                combined_mask = author_mask & ticker_mask
                
                if combined_mask.any():
                    df.loc[combined_mask, 'weighted_sentiment_score'] = (
                        df.loc[combined_mask, 'sentiment_score'] * weight
                    )
        
        # Логуємо сandтистику
        author_stats = df.groupby('detected_author')['weighted_sentiment_score'].agg(['mean', 'count'])
        logger.info(f"[Stage4] Author sentiment weighting applied:")
        for author, stats in author_stats.iterrows():
            logger.info(f"  - {author}: mean={stats['mean']:.3f}, count={stats['count']}")
        
        return df

class CausalLagProcessor:
    """Обробка Causal Lag for 2-whereнного andргету"""
    
    def __init__(self, config: EnhancedModelingConfig):
        self.lag_days = config.CAUSAL_LAG_DAYS
        
    def create_2day_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Causal Lag: 2-whereнний andргет for вandдсandчення емоцandйних реакцandй
        """
        logger.info(f"[Stage4]  Creating {self.lag_days}-day causal lag targets...")
        
        df = df.sort_values(['ticker', 'published_at']).copy()
        
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_data = df[ticker_mask].copy()
            
            # Створюємо 2-whereнний andргет for кожної цandни
            price_cols = [col for col in ticker_data.columns if 'close' in col.lower()]
            
            for price_col in price_cols:
                # Оригandнальна цandна
                original_price = ticker_data[price_col].values
                
                # Змandщуємо на 2 днand вперед
                future_price = np.roll(original_price, -self.lag_days)
                
                # Calculating withмandну
                price_change = (future_price - original_price) / original_price * 100
                
                # Створюємо andргет
                target_col = f"target_{price_col}_2d"
                df.loc[ticker_mask, target_col] = price_change
                
                # Створюємо напрямковий andргет
                direction_col = f"target_{price_col}_direction_2d"
                df.loc[ticker_mask, direction_col] = (price_change > 0).astype(int)
        
        # Логуємо сandтистику
        target_cols = [col for col in df.columns if '_2d' in col]
        logger.info(f"[Stage4] Created {len(target_cols)} 2-day targets")
        
        return df

class Float32Optimizer:
    """Оптимandforцandя пам'ятand череwith Float32"""
    
    def __init__(self, config: EnhancedModelingConfig):
        self.precision = config.PRECISION
        
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Float32 Precision: Переводить фandнансовand колонки в float32
        """
        logger.info("[Stage4] [START] Optimizing memory with float32 precision...")
        
        original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Конвертуємо числовand колонки в float32
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
        
        # Конвертуємо int64 в int32 where можливо
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.info(f"[Stage4] Memory optimization: {original_memory:.1f}MB -> {optimized_memory:.1f}MB ({memory_reduction:.1f}% reduction)")
        
        return df

# Глобальнand функцandї for викорисandння
def apply_enhanced_modeling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Застосовує all покращення for моwhereлювання
    """
    config = EnhancedModelingConfig()
    
    # 1. Dynamic Weighting
    dynamic_weighter = DynamicWeightingProcessor(config)
    df = dynamic_weighter.apply_dynamic_sentiment_weights(df)
    
    # 2. Causal Lag (2-whereнний andргет)
    causal_processor = CausalLagProcessor(config)
    df = causal_processor.create_2day_targets(df)
    
    # 3. Float32 оптимandforцandя
    memory_optimizer = Float32Optimizer(config)
    df = memory_optimizer.optimize_dataframe_memory(df)
    
    logger.info("[Stage4] [OK] Enhanced modeling features applied")
    return df

class EnhancedModelingStage:
    """Покращена сandдandя моwhereлювання with heavy/light логandкою"""
    
    def __init__(self, correlation_threshold: float = 0.8, max_features: int = 50):
        self.feature_optimizer = FeatureOptimizer(correlation_threshold)
        self.model_comparison = ModelComparisonEngine()
        self.max_features = max_features
        self.scaler = FeatureScaler()
        
        # Виvalues типandв моwhereлей
        self.heavy_models = ['gru', 'tabnet', 'transformer', 'lstm', 'cnn', 'autoencoder']
        self.light_models = ['lgbm', 'rf', 'linear', 'mlp', 'xgb', 'catboost', 'svm', 'knn']
        
    def prepare_data_for_modeling(self, df: pd.DataFrame, tickers: List[str],
                                include_heavy_light: bool = True) -> Dict[str, Any]:
        """Пandдготовка data for моwhereлювання"""
        
        logger.info("=== ENHANCED MODELING PREPARATION ===")
        
        # Крок 1: Додавання heavy/light andргетandв
        df_with_targets = add_targets(df.copy(), tickers, include_heavy_light=include_heavy_light)
        
        # Крок 2: Оптимandforцandя фandчей
        target_cols = [col for col in df_with_targets.columns if 'target_' in col]
        
        optimization_result = self.feature_optimizer.optimize_feature_set(
            df_with_targets, target_cols, self.max_features
        )
        
        # Крок 3: Пandдготовка data for моwhereлей
        model_data = self._prepare_model_datasets(optimization_result, tickers)
        
        return {
            'optimized_df': optimization_result['optimized_df'],
            'model_datasets': model_data,
            'optimization_result': optimization_result
        }
    
    def _prepare_model_datasets(self, optimization_result: Dict[str, Any], 
                               tickers: List[str]) -> Dict[str, Any]:
        """Пandдготовка нorрandв data for рandwithних типandв моwhereлей"""
        
        df = optimization_result['optimized_df']
        final_features = optimization_result['final_feature_set']
        
        # Роwithподandляємо andргети по типах
        heavy_targets = [col for col in df.columns if 'target_heavy' in col]
        light_targets = [col for col in df.columns if 'target_light' in col]
        direction_targets = [col for col in df.columns if 'target_direction' in col]
        
        model_datasets = {}
        
        for ticker in tickers:
            ticker_lower = ticker.lower()
            
            # Heavy моwhereлand
            for target in heavy_targets:
                if ticker_lower in target:
                    dataset = self._create_dataset(df, final_features, target, ticker, 'heavy')
                    if dataset:
                        model_datasets[f"heavy_{ticker}_{target}"] = dataset
            
            # Light моwhereлand  
            for target in light_targets:
                if ticker_lower in target:
                    dataset = self._create_dataset(df, final_features, target, ticker, 'light')
                    if dataset:
                        model_datasets[f"light_{ticker}_{target}"] = dataset
                        
            # Direction моwhereлand (for класифandкацandї)
            for target in direction_targets:
                if ticker_lower in target:
                    dataset = self._create_dataset(df, final_features, target, ticker, 'direction')
                    if dataset:
                        model_datasets[f"direction_{ticker}_{target}"] = dataset
        
        logger.info(f"Created {len(model_datasets)} model datasets")
        return model_datasets
    
    def _create_dataset(self, df: pd.DataFrame, features: List[str], 
                       target: str, ticker: str, model_type: str) -> Optional[Dict[str, Any]]:
        """Створення нorру data for конкретної моwhereлand"""
        
        # Фandльтруємо по тandкеру
        ticker_data = df[df['ticker'] == ticker].copy()
        
        if ticker_data.empty:
            logger.warning(f"No data for ticker {ticker}")
            return None
        
        # Вибираємо доступнand фandчand
        available_features = [f for f in features if f in ticker_data.columns]
        
        if not available_features:
            logger.warning(f"No available features for {ticker} {target}")
            return None
        
        # Очищення data
        X = ticker_data[available_features].fillna(0).astype(np.float32)
        y = ticker_data[target].fillna(0).astype(np.float32)
        
        # Видаляємо консandнтнand фandчand
        constant_features = [f for f in available_features if X[f].std() == 0]
        if constant_features:
            X = X.drop(columns=constant_features)
            available_features = [f for f in available_features if f not in constant_features]
        
        if X.empty or len(available_features) < 5:
            logger.warning(f"Insufficient features for {ticker} {target}")
            return None
        
        # Масшandбування
        scaled_data = self.scaler.scale(pd.concat([X, y], axis=1), key=target, fit_new=True)
        X_scaled = scaled_data[available_features]
        y_scaled = scaled_data[target]
        
        # Time series split
        if len(X_scaled) < 10:
            logger.warning(f"Insufficient data for {ticker} {target}: {len(X_scaled)} samples")
            return None
        
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X_scaled))
        train_idx, test_idx = splits[-1]  # Використовуємо осandннandй split
        
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y_scaled.iloc[train_idx], y_scaled.iloc[test_idx]
        
        # Для 3D моwhereлей (heavy)
        if model_type == 'heavy' and len(X_train) >= 20:
            X_train_3d, y_train_3d = self._create_time_windows(X_train, y_train, window_size=10)
            X_test_3d, y_test_3d = self._create_time_windows(X_test, y_test, window_size=10)
            
            return {
                'X_train': X_train_3d,
                'X_test': X_test_3d,
                'y_train': y_train_3d,
                'y_test': y_test_3d,
                'features': available_features,
                'target': target,
                'ticker': ticker,
                'model_type': model_type,
                'is_3d': True
            }
        else:
            # 2D моwhereлand (light, direction)
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'features': available_features,
                'target': target,
                'ticker': ticker,
                'model_type': model_type,
                'is_3d': False
            }
    
    def _create_time_windows(self,
        X: pd.DataFrame,
        y: pd.Series,
        window_size: int = 10) -> Tuple[np.ndarray,
        np.ndarray]:
        """Створення часових вandкон for 3D моwhereлей"""
        
        X_values = X.values
        y_values = y.values
        
        X_windows, y_windows = [], []
        
        for i in range(len(X_values) - window_size):
            X_windows.append(X_values[i:i+window_size])
            y_windows.append(y_values[i+window_size])
        
        return np.array(X_windows), np.array(y_windows)
    
    def get_model_config_for_target(self, target: str, model_type: str) -> Dict[str, Any]:
        """Отримати конфandгурацandю моwhereлand for andргету"""
        
        if model_type == 'heavy':
            available_models = self.heavy_models
            task_type = 'regression'
        elif model_type == 'light':
            available_models = self.light_models
            task_type = 'regression'
        else:  # direction
            available_models = self.light_models  # Light моwhereлand for класифandкацandї
            task_type = 'classification'
        
        return {
            'available_models': available_models,
            'task_type': task_type,
            'target_type': model_type,
            'target': target
        }
    
    def run_model_training_simulation(self, model_datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Симуляцandя тренування моwhereлей (беwith реального тренування)"""
        
        logger.info("=== MODEL TRAINING SIMULATION ===")
        
        training_results = {}
        
        for dataset_key, dataset in model_datasets.items():
            model_type = dataset['model_type']
            target = dataset['target']
            ticker = dataset['ticker']
            
            # Отримуємо конфandгурацandю
            config = self.get_model_config_for_target(target, model_type)
            
            # Симуляцandя реwithульandтandв
            X_train_shape = dataset['X_train'].shape
            y_train_shape = dataset['y_train'].shape
            
            # Геnotруємо фейковand реwithульandти
            if model_type == 'heavy':
                # Heavy моwhereлand forwithвичай мають гandршand метрики
                mae = np.random.uniform(0.5, 2.0)
                r2 = np.random.uniform(0.1, 0.6)
            else:
                # Light моwhereлand forwithвичай кращand
                mae = np.random.uniform(0.1, 0.8)
                r2 = np.random.uniform(0.3, 0.8)
            
            training_results[dataset_key] = {
                'config': config,
                'data_shape': {
                    'X_train': X_train_shape,
                    'X_test': dataset['X_test'].shape,
                    'y_train': y_train_shape,
                    'y_test': dataset['y_test'].shape
                },
                'simulated_metrics': {
                    'mae': mae,
                    'r2': r2,
                    'samples': len(dataset['y_train'])
                },
                'features_count': len(dataset['features']),
                'is_3d': dataset['is_3d']
            }
        
        # Створюємо withвandт
        report = self._generate_training_report(training_results)
        
        return {
            'training_results': training_results,
            'report': report
        }
    
    def _generate_training_report(self, training_results: Dict[str, Any]) -> str:
        """Геnotрацandя withвandту тренування"""
        
        report = []
        report.append("=== ENHANCED MODELING REPORT ===")
        
        # Сandтистика по типах моwhereлей
        heavy_count = sum(1 for k, v in training_results.items() if v['config']['target_type'] == 'heavy')
        light_count = sum(1 for k, v in training_results.items() if v['config']['target_type'] == 'light')
        direction_count = sum(1 for k, v in training_results.items() if v['config']['target_type'] == 'direction')
        
        report.append(f"Heavy model datasets: {heavy_count}")
        report.append(f"Light model datasets: {light_count}")
        report.append(f"Direction model datasets: {direction_count}")
        
        # Середнand метрики
        heavy_mae = [v['simulated_metrics']['mae'] for v in training_results.values() 
                    if v['config']['target_type'] == 'heavy']
        light_mae = [v['simulated_metrics']['mae'] for v in training_results.values() 
                    if v['config']['target_type'] == 'light']
        
        if heavy_mae:
            report.append(f"Heavy models avg MAE: {np.mean(heavy_mae):.3f}")
        if light_mae:
            report.append(f"Light models avg MAE: {np.mean(light_mae):.3f}")
        
        # 3D datasets
        datasets_3d = sum(1 for v in training_results.values() if v['is_3d'])
        report.append(f"3D datasets: {datasets_3d}")
        
        return "\n".join(report)


def run_enhanced_modeling(df: pd.DataFrame, tickers: List[str] = None,
                         include_heavy_light: bool = True) -> Dict[str, Any]:
    """Запуск покращеного моwhereлювання"""
    
    if tickers is None:
        tickers = ['SPY', 'QQQ']
    
    modeling_stage = EnhancedModelingStage()
    
    # Пandдготовка data
    preparation_result = modeling_stage.prepare_data_for_modeling(df, tickers, include_heavy_light)
    
    # Симуляцandя тренування
    training_result = modeling_stage.run_model_training_simulation(preparation_result['model_datasets'])
    
    return {
        'preparation': preparation_result,
        'training': training_result
    }
