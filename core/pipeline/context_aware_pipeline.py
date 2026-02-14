"""
Context-Aware Pipeline - andнтеграцandя andнтелектуального вибору моwhereлей
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime

from core.analysis.context_aware_model_selector import ContextAwareModelSelector
from core.stages.stage_3_utils import add_targets
from core.analysis.feature_optimizer import FeatureOptimizer

logger = logging.getLogger(__name__)

class ContextAwarePipeline:
    """Пайплайн with контекстно-forлежним вибором моwhereлей"""
    
    def __init__(self):
        self.context_selector = ContextAwareModelSelector()
        self.feature_optimizer = FeatureOptimizer()
        self.model_registry = {}
        self.performance_history = []
        
    def register_model(self, model_name: str, model_type: str, 
                      target_types: List[str], model_instance):
        """Реєстрацandя моwhereлand в системand"""
        
        self.model_registry[model_name] = {
            'type': model_type,  # 'heavy' or 'light'
            'target_types': target_types,  # ['heavy', 'light', 'direction']
            'instance': model_instance,
            'performance_history': []
        }
        
        logger.info(f"Registered model: {model_name} ({model_type}) for targets: {target_types}")
    
    def analyze_current_context(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Аналandwith поточного контексту ринку"""
        
        # Виvalues цandнових колонок
        price_cols = [col for col in df.columns if ticker.lower() in col.lower() and 
                     any(x in col.lower() for x in ['close', 'open', 'high', 'low'])]
        
        if not price_cols:
            logger.warning(f"No price columns found for {ticker}")
            return {}
        
        # Використовуємо close колонку for аналandwithу
        close_col = [col for col in price_cols if 'close' in col.lower()][0]
        
        # Створення DataFrame for аналandwithу
        analysis_df = df[['date', close_col]].copy() if 'date' in df.columns else df[[close_col]].copy()
        analysis_df.columns = ['date', 'close']
        
        # Додавання обсягandв якщо є
        volume_col = [col for col in df.columns if ticker.lower() in col.lower() and 'volume' in col.lower()]
        if volume_col:
            analysis_df['volume'] = df[volume_col[0]].values
        
        # Екстракцandя контексту
        context = self.context_selector.extract_context_from_data(analysis_df, ticker)
        
        return context
    
    def get_optimal_model_for_target(self, df: pd.DataFrame, ticker: str, 
                                    target_type: str) -> Dict[str, Any]:
        """Отримати оптимальну model for конкретного andргету"""
        
        # Аналandwith контексту
        context = self.analyze_current_context(df, ticker)
        
        # Фandльтрацandя доступних моwhereлей for andргету
        available_models = {
            name: info for name, info in self.model_registry.items()
            if target_type in info['target_types']
        }
        
        if not available_models:
            logger.warning(f"No models available for target type: {target_type}")
            return {}
        
        # Прогноwith найкращої моwhereлand
        selection_result = self.context_selector.predict_best_model_for_context(
            context, available_models, target_type
        )
        
        return {
            'context': context,
            'selection': selection_result,
            'available_models': list(available_models.keys()),
            'target_type': target_type
        }
    
    def prepare_target_specific_data(self, df: pd.DataFrame, ticker: str, 
                                   target_type: str) -> Dict[str, Any]:
        """Пandдготовка data for конкретного andргету"""
        
        # Додавання andргетandв
        df_with_targets = add_targets(df.copy(), [ticker], include_heavy_light=True)
        
        # Вибandр конкретного andргету
        target_cols = [col for col in df_with_targets.columns if f'target_{target_type}_' in col]
        
        if not target_cols:
            logger.warning(f"No {target_type} targets found")
            return {}
        
        target_col = target_cols[0]  # Використовуємо перший available
        
        # Оптимandforцandя фandчей
        optimization_result = self.feature_optimizer.optimize_feature_set(
            df_with_targets, [target_col], max_features=100
        )
        
        return {
            'optimized_df': optimization_result['optimized_df'],
            'target_col': target_col,
            'feature_set': optimization_result['final_feature_set'],
            'original_features': len(df.columns),
            'optimized_features': len(optimization_result['final_feature_set'])
        }
    
    def run_context_aware_analysis(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Запуск контекстно-forлежного аналandwithу"""
        
        logger.info(f"Starting context-aware analysis for {ticker}")
        
        results = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'context_analysis': {},
            'target_specific_analysis': {},
            'model_recommendations': {},
            'risk_assessment': {}
        }
        
        # Аналandwith контексту
        context = self.analyze_current_context(df, ticker)
        results['context_analysis'] = context
        
        # Рекомендацandї по контексту
        recommendations = self.context_selector.get_model_recommendations(context)
        results['model_recommendations'] = recommendations
        results['risk_assessment'] = recommendations['risk_assessment']
        
        # Аналandwith for кожного типу andргету
        target_types = ['heavy', 'light', 'direction']
        
        for target_type in target_types:
            logger.info(f"Analyzing {target_type} targets...")
            
            # Пandдготовка data
            target_data = self.prepare_target_specific_data(df, ticker, target_type)
            
            if target_data:
                # Вибandр оптимальної моwhereлand
                model_selection = self.get_optimal_model_for_target(df, ticker, target_type)
                
                results['target_specific_analysis'][target_type] = {
                    'data_preparation': target_data,
                    'model_selection': model_selection,
                    'optimal_model': model_selection.get('selection', {}).get('best_model'),
                    'selection_confidence': model_selection.get('selection', {}).get('selection_confidence'),
                    'predicted_performance': model_selection.get('selection', {}).get('predicted_performance')
                }
        
        return results
    
    def generate_live_recommendations(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Геnotрацandя рекомендацandй for live trading"""
        
        # Запуск повного аналandwithу
        analysis_results = self.run_context_aware_analysis(df, ticker)
        
        # Формування рекомендацandй
        recommendations = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'market_context': analysis_results['context_analysis'],
            'risk_level': analysis_results['risk_assessment']['risk_level'],
            'optimal_approaches': []
        }
        
        # Рекомендацandї по кожному andргету
        for target_type, analysis in analysis_results['target_specific_analysis'].items():
            if analysis['optimal_model'] and analysis['selection_confidence'] > 0.6:
                recommendations['optimal_approaches'].append({
                    'target_type': target_type,
                    'recommended_model': analysis['optimal_model'],
                    'confidence': analysis['selection_confidence'],
                    'expected_performance': analysis['predicted_performance'],
                    'data_quality': {
                        'original_features': analysis['data_preparation']['original_features'],
                        'optimized_features': analysis['data_preparation']['optimized_features']
                    }
                })
        
        # Сортування по впевnotностand
        recommendations['optimal_approaches'].sort(
            key=lambda x: x['confidence'], reverse=True
        )
        
        return recommendations
    
    def update_model_performance(self, model_name: str, target_type: str,
                                context: Dict[str, float], actual_metrics: Dict[str, float]):
        """Оновлення продуктивностand моwhereлand"""
        
        # Оновлення в контекстному селекторand
        self.context_selector.update_model_performance_db(
            model_name, target_type, context, actual_metrics
        )
        
        # Оновлення в реєстрand моwhereлей
        if model_name in self.model_registry:
            self.model_registry[model_name]['performance_history'].append({
                'timestamp': datetime.now(),
                'context': context,
                'metrics': actual_metrics
            })
        
        # Додавання в forгальну andсторandю
        self.performance_history.append({
            'model_name': model_name,
            'target_type': target_type,
            'context': context,
            'metrics': actual_metrics,
            'timestamp': datetime.now()
        })
        
        logger.info(f"Updated performance for {model_name} ({target_type})")


def create_context_aware_pipeline():
    """Створення контекстно-forлежного пайплайну"""
    
    pipeline = ContextAwarePipeline()
    
    # Реєстрацandя моwhereлей (приклад)
    # pipeline.register_model('lgbm', 'light', ['light', 'direction'], lgbm_model)
    # pipeline.register_model('gru', 'heavy', ['heavy'], gru_model)
    
    return pipeline


if __name__ == "__main__":
    # Тестування
    pipeline = create_context_aware_pipeline()
    
    # Симуляцandя реєстрацandї моwhereлей
    pipeline.register_model('lgbm', 'light', ['light', 'direction'], None)
    pipeline.register_model('rf', 'light', ['light', 'direction'], None)
    pipeline.register_model('gru', 'heavy', ['heavy'], None)
    pipeline.register_model('transformer', 'heavy', ['heavy'], None)
    
    # Load тестових data
    try:
        df = pd.read_parquet('c:/trading_project/data/stages/enhanced_with_temporal_features.parquet')
        logger.info(f"Loaded data: {df.shape}")
        
        # Геnotрацandя рекомендацandй
        recommendations = pipeline.generate_live_recommendations(df, 'SPY')
        
        logger.info("\n=== LIVE RECOMMENDATIONS ===")
        logger.info(f"Ticker: {recommendations['ticker']}")
        logger.info(f"Risk Level: {recommendations['risk_level']}")
        logger.info(f"Market Context: {recommendations['market_context']}")
        
        logger.info(f"\nOptimal Approaches ({len(recommendations['optimal_approaches'])}):")
        for i, approach in enumerate(recommendations['optimal_approaches'], 1):
            logger.info(f"{i}. {approach['target_type'].upper()} - {approach['recommended_model']}")
            logger.info(f"   Confidence: {approach['confidence']:.2f}")
            logger.info(f"   Expected Performance: {approach['expected_performance']:.3f}")
        
    except Exception as e:
        logger.info(f"Error loading data: {e}")
        logger.info("Pipeline created successfully, but no test data available")