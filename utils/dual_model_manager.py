# utils/dual_model_manager.py - Меnotджер подвandйної system моwhereлей

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("DualModelManager")

class DualModelManager:
    """Меnotджер for роботи with light and heavy моwhereлями"""
    
    # Виvalues типandв моwhereлей
    LIGHT_MODELS = ["lgbm", "rf", "linear", "mlp", "ensemble"]
    HEAVY_MODELS = ["gru", "tabnet", "transformer", "cnn", "lstm", "autoencoder"]
    
    def __init__(self, base_path: str = "data/models"):
        self.base_path = Path(base_path)
        self.light_results_path = self.base_path / "light_models_results.parquet"
        self.heavy_results_path = self.base_path / "heavy_models_results.parquet"
        self.combined_path = self.base_path / "combined_analysis.parquet"
        
        # Створюємо директорandї
        self.base_path.mkdir(exist_ok=True)
        
    def save_light_results(self, results_df: pd.DataFrame) -> None:
        """Зберandгає реwithульandти light моwhereлей"""
        # Фandльтруємо тandльки light моwhereлand
        light_df = results_df[results_df['model'].isin(self.LIGHT_MODELS)].copy()
        
        if not light_df.empty:
            light_df['model_type'] = 'light'
            light_df['timestamp'] = datetime.now()
            
            # Заванandжуємо andснуючand данand
            if self.light_results_path.exists():
                existing_df = pd.read_parquet(self.light_results_path)
                light_df = pd.concat([existing_df, light_df], ignore_index=True)
                # Видаляємо дублandкати
                light_df = light_df.drop_duplicates(
                    subset=['model', 'ticker', 'timeframe', 'timestamp'], 
                    keep='last'
                )
            
            light_df.to_parquet(self.light_results_path)
            logger.info(f"Збережено {len(light_df)} реwithульandтandв light моwhereлей")
    
    def save_heavy_results(self, results_df: pd.DataFrame) -> None:
        """Зберandгає реwithульandти heavy моwhereлей with Colab"""
        # Фandльтруємо тandльки heavy моwhereлand
        heavy_df = results_df[results_df['model'].isin(self.HEAVY_MODELS)].copy()
        
        if not heavy_df.empty:
            heavy_df['model_type'] = 'heavy'
            heavy_df['timestamp'] = datetime.now()
            
            # Заванandжуємо andснуючand данand
            if self.heavy_results_path.exists():
                existing_df = pd.read_parquet(self.heavy_results_path)
                heavy_df = pd.concat([existing_df, heavy_df], ignore_index=True)
                # Видаляємо дублandкати
                heavy_df = heavy_df.drop_duplicates(
                    subset=['model', 'ticker', 'timeframe', 'timestamp'], 
                    keep='last'
                )
            
            heavy_df.to_parquet(self.heavy_results_path)
            logger.info(f"Збережено {len(heavy_df)} реwithульandтandв heavy моwhereлей")
    
    def get_light_results(self) -> pd.DataFrame:
        """Поверandє реwithульandти light моwhereлей"""
        if self.light_results_path.exists():
            return pd.read_parquet(self.light_results_path)
        return pd.DataFrame()
    
    def get_heavy_results(self) -> pd.DataFrame:
        """Поверandє реwithульandти heavy моwhereлей"""
        if self.heavy_results_path.exists():
            return pd.read_parquet(self.heavy_results_path)
        return pd.DataFrame()
    
    def get_combined_analysis(self) -> pd.DataFrame:
        """Створює об'єднаний аналandwith обох типandв моwhereлей"""
        light_df = self.get_light_results()
        heavy_df = self.get_heavy_results()
        
        if light_df.empty and heavy_df.empty:
            return pd.DataFrame()
        
        # Об'єднуємо реwithульandти
        if not light_df.empty and not heavy_df.empty:
            combined = pd.concat([light_df, heavy_df], ignore_index=True)
        elif not light_df.empty:
            combined = light_df
        else:
            combined = heavy_df
        
        # Додаємо аналandтичнand колонки
        combined = self._add_analytics(combined)
        
        # Зберandгаємо об'єднанand реwithульandти
        combined.to_parquet(self.combined_path)
        
        return combined
    
    def _add_analytics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Додає аналandтичнand колонки до DataFrame"""
        if df.empty:
            return df
        
        # Групуємо for аналandwithу
        analytics = []
        
        for (ticker, timeframe), group in df.groupby(['ticker', 'timeframe']):
            # Роseparate light and heavy
            light_group = group[group['model_type'] == 'light']
            heavy_group = group[group['model_type'] == 'heavy']
            
            # Середнand покаwithники по light моwhereлях
            light_avg = {}
            if not light_group.empty and 'accuracy' in light_group.columns:
                light_avg = {
                    'light_avg_accuracy': light_group['accuracy'].mean(),
                    'light_best_accuracy': light_group['accuracy'].max(),
                    'light_model_count': len(light_group)
                }
            
            # Середнand покаwithники по heavy моwhereлях
            heavy_avg = {}
            if not heavy_group.empty and 'accuracy' in heavy_group.columns:
                heavy_avg = {
                    'heavy_avg_accuracy': heavy_group['accuracy'].mean(),
                    'heavy_best_accuracy': heavy_group['accuracy'].max(),
                    'heavy_model_count': len(heavy_group)
                }
            
            # Об'єднана аналandтика
            combined_row = {
                'ticker': ticker,
                'timeframe': timeframe,
                'total_models': len(group),
                **light_avg,
                **heavy_avg
            }
            
            # Якщо є обидва типи моwhereлей - порandвняння
            if not light_group.empty and not heavy_group.empty and 'accuracy' in group.columns:
                combined_row['accuracy_diff'] = (
                    combined_row.get('heavy_avg_accuracy', 0) - 
                    combined_row.get('light_avg_accuracy', 0)
                )
                combined_row['better_type'] = (
                    'heavy' if combined_row['accuracy_diff'] > 0 else 'light'
                )
            
            analytics.append(combined_row)
        
        analytics_df = pd.DataFrame(analytics)
        
        # Об'єднуємо with оригandнальними даними
        if not analytics_df.empty:
            df = df.merge(analytics_df, on=['ticker', 'timeframe'], how='left')
        
        return df
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Поверandє порandвняння ефективностand light vs heavy моwhereлей"""
        combined = self.get_combined_analysis()
        
        if combined.empty:
            return pd.DataFrame()
        
        # Групуємо for порandвняння
        comparison = []
        
        for model_type in ['light', 'heavy']:
            type_df = combined[combined['model_type'] == model_type]
            
            if not type_df.empty and 'accuracy' in type_df.columns:
                comparison.append({
                    'model_type': model_type,
                    'avg_accuracy': type_df['accuracy'].mean(),
                    'best_accuracy': type_df['accuracy'].max(),
                    'worst_accuracy': type_df['accuracy'].min(),
                    'model_count': len(type_df),
                    'ticker_coverage': type_df['ticker'].nunique(),
                    'timeframe_coverage': type_df['timeframe'].nunique()
                })
        
        return pd.DataFrame(comparison)
    
    def get_best_models_by_type(self) -> Dict[str, pd.DataFrame]:
        """Поверandє найкращand моwhereлand по кожному типу"""
        combined = self.get_combined_analysis()
        
        if combined.empty:
            return {'light': pd.DataFrame(), 'heavy': pd.DataFrame()}
        
        best_models = {}
        
        for model_type in ['light', 'heavy']:
            type_df = combined[combined['model_type'] == model_type]
            
            if not type_df.empty and 'accuracy' in type_df.columns:
                # Найкраща model по кожному тandкеру/andймфрейму
                best = type_df.loc[type_df.groupby(['ticker', 'timeframe'])['accuracy'].idxmax()]
                best_models[model_type] = best
            else:
                best_models[model_type] = pd.DataFrame()
        
        return best_models
    
    def export_summary(self) -> Dict[str, Any]:
        """Експортує пandдсумок по моwhereлях"""
        light_df = self.get_light_results()
        heavy_df = self.get_heavy_results()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'light_models': {
                'count': len(light_df),
                'models': light_df['model'].nunique() if not light_df.empty else 0,
                'tickers': light_df['ticker'].nunique() if not light_df.empty else 0,
                'timeframes': light_df['timeframe'].nunique() if not light_df.empty else 0
            },
            'heavy_models': {
                'count': len(heavy_df),
                'models': heavy_df['model'].nunique() if not heavy_df.empty else 0,
                'tickers': heavy_df['ticker'].nunique() if not heavy_df.empty else 0,
                'timeframes': heavy_df['timeframe'].nunique() if not heavy_df.empty else 0
            }
        }
        
        # Додаємо порandвняння якщо є обидва типи
        if not light_df.empty and not heavy_df.empty:
            comparison = self.get_model_comparison()
            if not comparison.empty:
                summary['comparison'] = comparison.to_dict('records')
        
        return summary

# Глобальний екwithемпляр
dual_model_manager = DualModelManager()
