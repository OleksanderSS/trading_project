"""
Analyzes and compares the performance of different ML models.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

from ..interfaces import IAnalyzer

logger = logging.getLogger(__name__)

class ModelComparisonAnalyzer(IAnalyzer):
    """
    Performs analysis on DataFrames of model results to compare architectures,
    find the best models, and assess performance stability.
    """

    # Constants for model types, can be configured during initialization
    HEAVY_MODELS = ["gru", "tabnet", "transformer", "cnn", "lstm", "autoencoder"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.heavy_models = self.config.get('heavy_models', self.HEAVY_MODELS)
        logger.info("ModelComparisonAnalyzer initialized.")

    def analyze(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        Main entry point for the analyzer.
        
        Args:
            data (Dict[str, pd.DataFrame]): A dictionary containing a 'results' DataFrame 
                                          with model performance data.
            **kwargs: Can specify which analyses to run, e.g., 
                      run_architectures=True, run_best_models=True.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the requested analyses.
        """
        results_df = data.get('results')
        if not isinstance(results_df, pd.DataFrame) or results_df.empty:
            logger.error("Input must be a dict with a non-empty 'results' DataFrame.")
            return {"error": "Invalid input format. Expected a 'results' DataFrame."}
        
        # Add model_type column if not present
        if 'model_type' not in results_df.columns:
            results_df['model_type'] = results_df['model'].apply(
                lambda x: 'heavy' if x in self.heavy_models else 'light'
            )

        analysis_results = {}

        # Conditionally run different analyses based on kwargs
        if kwargs.get("run_architecture_comparison", True):
            analysis_results['architecture_comparison'] = self._compare_architectures(results_df)
        
        if kwargs.get("run_best_model_finder", True):
            analysis_results['best_models_by_type'] = self._get_best_models_by_type(results_df)

        if kwargs.get("run_overall_summary", True):
            analysis_results['overall_summary'] = self._summarize_by_type(results_df)

        return analysis_results

    def _compare_architectures(self, results_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyzes performance stability and reliability across different model architectures.
        """
        if 'accuracy' not in results_df.columns:
            return []

        arch_stats = []
        for arch, group in results_df.groupby('model'):
            metrics = {
                'architecture': arch,
                'type': group['model_type'].iloc[0],
                'mean_accuracy': group['accuracy'].mean(),
                'stability_std': group['accuracy'].std(),
                'sample_size': len(group),
                'best_performance': group['accuracy'].max()
            }
            # Reliability Score: Performance penalized by instability (1 - std)
            metrics['reliability_score'] = metrics['mean_accuracy'] * (1 - metrics.get('stability_std', 0.0))
            arch_stats.append(metrics)
        
        return sorted(arch_stats, key=lambda x: x['reliability_score'], reverse=True)

    def _get_best_models_by_type(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Finds the best performing model for each type (light/heavy) within each ticker/timeframe group.
        """
        if 'accuracy' not in results_df.columns:
            return {}
        
        best_models = {'light': {}, 'heavy': {}}
        
        # Find best model for each group (ticker, timeframe)
        best_indices = results_df.loc[results_df.groupby(['ticker', 'timeframe', 'model_type'])['accuracy'].idxmax()]

        for _, row in best_indices.iterrows():
            model_type = row['model_type']
            group_key = f"{row['ticker']}_{row['timeframe']}"
            best_models[model_type][group_key] = {
                'model': row['model'],
                'accuracy': row['accuracy'],
            }
            
        return best_models

    def _summarize_by_type(self, results_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generates a high-level summary comparing light vs. heavy models.
        """
        if 'accuracy' not in results_df.columns:
            return []
        
        comparison = []
        for model_type, group in results_df.groupby('model_type'):
            comparison.append({
                'model_type': model_type,
                'avg_accuracy': group['accuracy'].mean(),
                'best_accuracy': group['accuracy'].max(),
                'worst_accuracy': group['accuracy'].min(),
                'model_count': group['model'].nunique(),
                'record_count': len(group),
                'ticker_coverage': group['ticker'].nunique(),
                'timeframe_coverage': group['timeframe'].nunique()
            })
        
        return comparison

    def compare_models(self, training_results: Dict[str, Any], market_context: str = 'neutral') -> Dict[str, Any]:
        """
        Порівнює моделі з training_results та вибирає чемпіона.
        
        Логіка:
        1. Важкі моделі порівнюються між собою
        2. Легкі моделі порівнюються між собою
        3. Найкраща важка vs найкраща легка → чемпіон
        
        Args:
            training_results: Результати тренування з UnifiedTrainingManager
            market_context: Контекст ринку (для майбутньої логіки)
            
        Returns:
            Dict з champion_model, selection_reason, heavy_best, light_best
        """
        logger.info("🏆 Порівняння моделей для вибору чемпіона...")
        
        # Витягуємо результати по тікерам
        tickers_results = training_results.get('tickers_results', {})
        
        if not tickers_results:
            logger.warning("⚠️ Немає результатів тренування для порівняння")
            return {
                'champion_model': 'unknown',
                'selection_reason': 'No training results available',
                'heavy_best': None,
                'light_best': None
            }
        
        # Збираємо всі моделі з метриками
        all_models = []
        
        for ticker, ticker_data in tickers_results.items():
            if ticker_data.get('status') != 'success':
                continue
            
            winner = ticker_data.get('winner', 'unknown')
            metrics = ticker_data.get('metrics', {})
            
            # Визначаємо тип моделі
            model_type = 'heavy' if winner.lower() in [m.lower() for m in self.heavy_models] else 'light'
            
            # Витягуємо accuracy (може бути в різних місцях)
            accuracy = metrics.get('accuracy', metrics.get('test_accuracy', metrics.get('r2', 0.0)))
            
            all_models.append({
                'model_name': winner,
                'model_type': model_type,
                'accuracy': accuracy,
                'metrics': metrics,
                'ticker': ticker
            })
        
        if not all_models:
            logger.warning("⚠️ Не знайдено жодної успішної моделі")
            return {
                'champion_model': 'unknown',
                'selection_reason': 'No successful models',
                'heavy_best': None,
                'light_best': None
            }
        
        # Розділяємо на важкі та легкі
        heavy_models = [m for m in all_models if m['model_type'] == 'heavy']
        light_models = [m for m in all_models if m['model_type'] == 'light']
        
        # Знаходимо найкращі в кожній категорії
        best_heavy = max(heavy_models, key=lambda x: x['accuracy']) if heavy_models else None
        best_light = max(light_models, key=lambda x: x['accuracy']) if light_models else None
        
        logger.info(f"📊 Важкі моделі: {len(heavy_models)}, Легкі моделі: {len(light_models)}")
        if best_heavy:
            logger.info(f"🏆 Найкраща важка: {best_heavy['model_name']} (accuracy: {best_heavy['accuracy']:.4f})")
        if best_light:
            logger.info(f"💡 Найкраща легка: {best_light['model_name']} (accuracy: {best_light['accuracy']:.4f})")
        
        # Вибираємо чемпіона
        if best_heavy and best_light:
            if best_heavy['accuracy'] >= best_light['accuracy']:
                champion = best_heavy['model_name']
                reason = f"Heavy model wins: {best_heavy['accuracy']:.4f} vs {best_light['accuracy']:.4f}"
            else:
                champion = best_light['model_name']
                reason = f"Light model wins: {best_light['accuracy']:.4f} vs {best_heavy['accuracy']:.4f}"
        elif best_heavy:
            champion = best_heavy['model_name']
            reason = "Only heavy model available"
        elif best_light:
            champion = best_light['model_name']
            reason = "Only light model available"
        else:
            champion = 'unknown'
            reason = "No models available"
        
        logger.info(f"🏆 ЧЕМПІОН: {champion} ({reason})")
        
        return {
            'champion_model': champion,
            'selection_reason': reason,
            'heavy_best': best_heavy,
            'light_best': best_light,
            'all_models': all_models
        }
