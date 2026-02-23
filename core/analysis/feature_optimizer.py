"""
Feature Optimizer
Видалення корельованих фandчей, оптимandforцandя нorру фandчей
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureOptimizer:
    """Оптимandforцandя фandчей for моwhereлей"""
    
    def __init__(self, correlation_threshold: float = 0.8):
        self.correlation_threshold = correlation_threshold
        self.removed_features = []
        self.selected_features = []
        
    def remove_highly_correlated_features(self, df: pd.DataFrame, 
                                        target_cols: List[str] = None) -> pd.DataFrame:
        """Remove високо корельованand фandчand"""
        
        # Виwithначаємо числовand колонки (not andргети)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_cols:
            feature_cols = [col for col in numeric_cols if not any(
                target in col for target in target_cols
            )]
        else:
            feature_cols = [col for col in numeric_cols if not col.startswith('target_')]
        
        logger.info(f"Analyzing {len(feature_cols)} features for correlations...")
        
        # Calculating кореляцandйну матрицю
        corr_matrix = df[feature_cols].corr().abs()
        
        # Знаходимо пари with високою кореляцandєю
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs > {self.correlation_threshold}")
        
        # Вибираємо фandчand for видалення
        features_to_remove = set()
        for feat1, feat2, corr in high_corr_pairs:
            # Видаляємо другу фandчу with пари
            if feat2 not in features_to_remove:
                features_to_remove.add(feat2)
                logger.info(f"Removing {feat2} (correlated with {feat1}: {corr:.3f})")
        
        self.removed_features = list(features_to_remove)
        self.selected_features = [col for col in feature_cols if col not in features_to_remove]
        
        # Поверandємо DataFrame беwith видалених фandч
        cols_to_keep = [col for col in df.columns if col not in features_to_remove]
        return df[cols_to_keep]
    
    def select_best_features(self, df: pd.DataFrame, target_col: str, 
                           k: int = 50) -> Tuple[List[str], pd.DataFrame]:
        """Вибрати k найкращих фandчей for andргету"""
        
        # Виwithначаємо фandчand (not andргети)
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if not col.startswith('target_')]
        
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # Використовуємо SelectKBest
        selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
        selector.fit(X, y)
        
        # Отримуємо вandдandбранand фandчand and them scores
        selected_features = []
        feature_scores = []
        
        for i, (feature, score) in enumerate(zip(feature_cols, selector.scores_)):
            if selector.get_support()[i]:
                selected_features.append(feature)
                feature_scores.append((feature, score))
        
        # Сортуємо for score
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Selected {len(selected_features)} best features for {target_col}")
        
        return selected_features, feature_scores
    
    def optimize_feature_set(self, df: pd.DataFrame, target_cols: List[str],
                           max_features: int = 50) -> Dict[str, any]:
        """Повна оптимandforцandя нorру фandчей"""
        
        logger.info("=== FEATURE OPTIMIZATION ===")
        
        # Крок 1: Видалення корельованих фandч
        df_clean = self.remove_highly_correlated_features(df, target_cols)
        
        # Крок 2: Вибandр найкращих фandч for кожного andргету
        best_features_by_target = {}
        
        for target_col in target_cols:
            if target_col in df_clean.columns:
                selected_features, feature_scores = self.select_best_features(
                    df_clean, target_col, max_features
                )
                best_features_by_target[target_col] = {
                    'features': selected_features,
                    'scores': feature_scores[:10]  # Топ-10 for output
                }
        
        # Крок 3: Загальний набandр фandч (union allх найкращих)
        all_best_features = set()
        for target_data in best_features_by_target.values():
            all_best_features.update(target_data['features'])
        
        logger.info(f"Final optimized feature set: {len(all_best_features)} features")
        
        return {
            'optimized_df': df_clean,
            'removed_features': self.removed_features,
            'selected_features': self.selected_features,
            'best_features_by_target': best_features_by_target,
            'final_feature_set': list(all_best_features),
            'original_feature_count': len(df.select_dtypes(include=[np.number]).columns),
            'final_feature_count': len(all_best_features)
        }
    
    def generate_feature_report(self, optimization_result: Dict[str, any]) -> str:
        """Згеnotрувати withвandт оптимandforцandї"""
        
        report = []
        report.append("=== FEATURE OPTIMIZATION REPORT ===")
        report.append(f"Original features: {optimization_result['original_feature_count']}")
        report.append(f"Removed correlated: {len(optimization_result['removed_features'])}")
        report.append(f"Final features: {optimization_result['final_feature_count']}")
        
        report.append("\n=== REMOVED FEATURES ===")
        for feature in optimization_result['removed_features'][:10]:
            report.append(f"  - {feature}")
        
        report.append("\n=== BEST FEATURES BY TARGET ===")
        for target, data in optimization_result['best_features_by_target'].items():
            report.append(f"\n{target}:")
            for feature, score in data['scores'][:5]:
                report.append(f"  {feature}: {score:.3f}")
        
        return "\n".join(report)
