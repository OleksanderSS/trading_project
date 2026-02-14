"""
Enhanced Pipeline with Feature Optimization and Model Comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

from core.stages.stage_3_utils import add_targets
from core.analysis.feature_optimizer import FeatureOptimizer
from core.analysis.model_comparison_engine import ModelComparisonEngine

logger = logging.getLogger(__name__)

class EnhancedPipeline:
    """Покращений пайплайн with оптимandforцandєю фandчей and порandвнянням моwhereлей"""
    
    def __init__(self, correlation_threshold: float = 0.8, max_features: int = 50):
        self.feature_optimizer = FeatureOptimizer(correlation_threshold)
        self.model_comparison = ModelComparisonEngine()
        self.max_features = max_features
        self.pipeline_results = {}
        
    def run_enhanced_pipeline(self, df: pd.DataFrame, tickers: List[str],
                            include_heavy_light: bool = True) -> Dict[str, Any]:
        """Запустити покращений пайплайн"""
        
        logger.info("=== ENHANCED PIPELINE START ===")
        start_time = datetime.now()
        
        # Крок 1: Додавання andргетandв
        logger.info("Step 1: Adding targets...")
        df_with_targets = add_targets(df.copy(), tickers, 
                                   include_heavy_light=include_heavy_light)
        
        # Крок 2: Оптимandforцandя фandчей
        logger.info("Step 2: Optimizing features...")
        target_cols = [col for col in df_with_targets.columns if 'target_' in col]
        
        optimization_result = self.feature_optimizer.optimize_feature_set(
            df_with_targets, target_cols, self.max_features
        )
        
        # Крок 3: Аналandwith якостand data
        logger.info("Step 3: Data quality analysis...")
        quality_analysis = self._analyze_data_quality(optimization_result['optimized_df'])
        
        # Крок 4: Пandдготовка for моwhereлей
        logger.info("Step 4: Model preparation...")
        model_preparation = self._prepare_for_models(optimization_result, tickers)
        
        # Крок 5: Геnotрацandя withвandтandв
        logger.info("Step 5: Generating reports...")
        reports = self._generate_reports(optimization_result, quality_analysis, 
                                       model_preparation)
        
        # Збереження реwithульandтandв
        self.pipeline_results = {
            'start_time': start_time,
            'end_time': datetime.now(),
            'original_shape': df.shape,
            'final_shape': optimization_result['optimized_df'].shape,
            'optimization_result': optimization_result,
            'quality_analysis': quality_analysis,
            'model_preparation': model_preparation,
            'reports': reports
        }
        
        logger.info(f"Pipeline completed in {datetime.now() - start_time}")
        return self.pipeline_results
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Аналandwith якостand data"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_data = df[numeric_cols].isnull().sum()
        
        quality_stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_cols),
            'missing_data_summary': {
                'total_missing': missing_data.sum(),
                'columns_with_missing': (missing_data > 0).sum(),
                'max_missing_pct': (missing_data / len(df) * 100).max()
            },
            'data_types': df.dtypes.value_counts().to_dict()
        }
        
        return quality_stats
    
    def _prepare_for_models(self, optimization_result: Dict[str, Any], 
                           tickers: List[str]) -> Dict[str, Any]:
        """Пandдготовка data for моwhereлей"""
        
        df = optimization_result['optimized_df']
        final_features = optimization_result['final_feature_set']
        
        # Роwithподandляємо andргети по типах моwhereлей
        heavy_targets = [col for col in df.columns if 'target_heavy' in col]
        light_targets = [col for col in df.columns if 'target_light' in col]
        direction_targets = [col for col in df.columns if 'target_direction' in col]
        
        model_preparation = {
            'feature_set': final_features,
            'heavy_targets': heavy_targets,
            'light_targets': light_targets,
            'direction_targets': direction_targets,
            'model_configurations': self._generate_model_configs(tickers)
        }
        
        return model_preparation
    
    def _generate_model_configs(self, tickers: List[str]) -> Dict[str, Any]:
        """Геnotрацandя конфandгурацandй моwhereлей"""
        
        configs = {
            'heavy_models': {
                'GRU': {'type': 'neural', 'target_type': 'heavy'},
                'TabNet': {'type': 'neural', 'target_type': 'heavy'},
                'Transformer': {'type': 'neural', 'target_type': 'heavy'}
            },
            'light_models': {
                'LGBM': {'type': 'tree', 'target_type': 'light'},
                'RF': {'type': 'tree', 'target_type': 'light'},
                'Linear': {'type': 'linear', 'target_type': 'light'},
                'MLP': {'type': 'neural', 'target_type': 'light'}
            }
        }
        
        return configs
    
    def _generate_reports(self, optimization_result: Dict[str, Any],
                         quality_analysis: Dict[str, Any],
                         model_preparation: Dict[str, Any]) -> Dict[str, str]:
        """Геnotрацandя withвandтandв"""
        
        reports = {}
        
        # Звandт оптимandforцandї фandчей
        reports['feature_optimization'] = self.feature_optimizer.generate_feature_report(
            optimization_result
        )
        
        # Звandт якостand data
        reports['data_quality'] = self._generate_quality_report(quality_analysis)
        
        # Звandт пandдготовки моwhereлей
        reports['model_preparation'] = self._generate_model_report(model_preparation)
        
        return reports
    
    def _generate_quality_report(self, quality_analysis: Dict[str, Any]) -> str:
        """Згеnotрувати withвandт якостand data"""
        
        report = []
        report.append("=== DATA QUALITY REPORT ===")
        report.append(f"Total rows: {quality_analysis['total_rows']}")
        report.append(f"Total columns: {quality_analysis['total_columns']}")
        report.append(f"Numeric columns: {quality_analysis['numeric_columns']}")
        
        missing = quality_analysis['missing_data_summary']
        report.append(f"Missing data: {missing['total_missing']} values")
        report.append(f"Columns with missing: {missing['columns_with_missing']}")
        report.append(f"Max missing %: {missing['max_missing_pct']:.1f}%")
        
        return "\n".join(report)
    
    def _generate_model_report(self, model_preparation: Dict[str, Any]) -> str:
        """Згеnotрувати withвandт пandдготовки моwhereлей"""
        
        report = []
        report.append("=== MODEL PREPARATION REPORT ===")
        report.append(f"Feature set size: {len(model_preparation['feature_set'])}")
        report.append(f"Heavy targets: {len(model_preparation['heavy_targets'])}")
        report.append(f"Light targets: {len(model_preparation['light_targets'])}")
        report.append(f"Direction targets: {len(model_preparation['direction_targets'])}")
        
        return "\n".join(report)
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Отримати пandдсумок пайплайну"""
        
        if not self.pipeline_results:
            return {}
            
        return {
            'execution_time': self.pipeline_results['end_time'] - self.pipeline_results['start_time'],
            'data_reduction': {
                'original_features': self.pipeline_results['original_shape'][1],
                'final_features': self.pipeline_results['final_shape'][1],
                'reduction_pct': (1 - self.pipeline_results['final_shape'][1] / 
                                self.pipeline_results['original_shape'][1]) * 100
            },
            'model_readiness': {
                'heavy_targets_ready': len(self.pipeline_results['model_preparation']['heavy_targets']),
                'light_targets_ready': len(self.pipeline_results['model_preparation']['light_targets']),
                'features_ready': len(self.pipeline_results['model_preparation']['feature_set'])
            }
        }
    
    def save_optimized_data(self, output_path: str = None):
        """Зберегти оптимandwithованand данand"""
        
        if not self.pipeline_results:
            logger.warning("No pipeline results to save")
            return
            
        if output_path is None:
            output_path = f"c:/trading_project/data/stages/enhanced_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        
        self.pipeline_results['optimization_result']['optimized_df'].to_parquet(output_path)
        logger.info(f"Optimized data saved to: {output_path}")
        
        return output_path
