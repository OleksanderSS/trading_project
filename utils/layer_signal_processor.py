# utils/layer_signal_processor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from config.feature_layers import get_layer_weight, get_features_by_layer, FEATURE_LAYERS
import logging

logger = logging.getLogger("LayerSignalProcessor")

class LayerSignalProcessor:
    """
    Обробляє сигнали with урахуванням багатошарової архandтектури.
    
    Поки що all шари мають вагу 1.0 (notйтрально), але логandка готова
    for майбутнього тюнandнгу пandсля тренування баwithових моwhereлей.
    """
    
    def __init__(self):
        self.layer_weights = {}
        self._initialize_layer_weights()
    
    def _initialize_layer_weights(self):
        """Інandцandалandwithує ваги шарandв. Поки all = 1.0"""
        for layer_name in FEATURE_LAYERS.keys():
            self.layer_weights[layer_name] = get_layer_weight(layer_name)
        
        logger.info(f"[LayerProcessor] Initialized {len(self.layer_weights)} шарandв")
        logger.info(f"[LayerProcessor] Поточнand ваги: {self.layer_weights}")
    
    def apply_layer_weights_to_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Застосовує ваги шарandв до фandчей в DataFrame.
        
        Args:
            df: DataFrame with фandчами
            
        Returns:
            DataFrame with модифandкованими фandчами withгandдно with вагами шарandв
        """
        df_weighted = df.copy()
        
        for layer_name, weight in self.layer_weights.items():
            if weight == 1.0:  # Пропускаємо notйтральнand шари for оптимandforцandї
                continue
                
            layer_features = get_features_by_layer(layer_name)
            
            # Знаходимо фandчand цього шару, якand є в DataFrame
            available_features = [f for f in layer_features if f in df.columns]
            
            if available_features:
                # Множимо фandчand на вагу шару
                df_weighted[available_features] = df_weighted[available_features] * weight
                logger.debug(f"[LayerProcessor] Шар '{layer_name}': {len(available_features)} фandчей * {weight}")
        
        return df_weighted
    
    def apply_layer_weights_to_predictions(self, predictions: Dict[str, float], 
                                         feature_importance: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Застосовує ваги шарandв до прогноwithandв моwhereлей.
        
        Args:
            predictions: Словник прогноwithandв {model_name: prediction_value}
            feature_importance: Важливandсть фandчей for виvalues впливу шарandв
            
        Returns:
            Словник with модифandкованими прогноforми
        """
        if not feature_importance:
            # Якщо notмає andнформацandї про важливandсть фandчей, поверandємо як є
            return predictions
        
        weighted_predictions = {}
        
        for model_name, prediction in predictions.items():
            # Calculating forгальний вплив шарandв на цю model
            total_layer_influence = 0.0
            total_importance = 0.0
            
            for layer_name, weight in self.layer_weights.items():
                if weight == 1.0:  # Пропускаємо notйтральнand шари
                    continue
                    
                layer_features = get_features_by_layer(layer_name)
                
                # Сума важливостand фandчей цього шару
                layer_importance = sum(feature_importance.get(f, 0.0) for f in layer_features)
                
                if layer_importance > 0:
                    # Вплив шару = (вага_шару - 1.0) * важливandсть_шару
                    layer_influence = (weight - 1.0) * layer_importance
                    total_layer_influence += layer_influence
                    total_importance += layer_importance
            
            # Застосовуємо корекцandю до прогноwithу
            if total_importance > 0:
                influence_factor = 1.0 + (total_layer_influence / total_importance)
                weighted_predictions[model_name] = prediction * influence_factor
                
                if influence_factor != 1.0:
                    logger.debug(f"[LayerProcessor] {model_name}: {prediction:.3f}  {weighted_predictions[model_name]:.3f} (factor: {influence_factor:.3f})")
            else:
                weighted_predictions[model_name] = prediction
        
        return weighted_predictions
    
    def get_layer_signal_breakdown(self, df: pd.DataFrame, 
                                 model_predictions: Dict[str, float]) -> Dict[str, Dict]:
        """
        Аналandwithує вклад кожного шару в фandнальний сигнал.
        
        Args:
            df: DataFrame with фandчами
            model_predictions: Прогноwithи моwhereлей
            
        Returns:
            Деandльний роwithбandр по шарах
        """
        breakdown = {}
        
        for layer_name in FEATURE_LAYERS.keys():
            layer_features = get_features_by_layer(layer_name)
            available_features = [f for f in layer_features if f in df.columns]
            
            if not available_features:
                continue
            
            # Сandтистика по шару
            layer_data = df[available_features].iloc[-1] if not df.empty else pd.Series()
            
            breakdown[layer_name] = {
                "weight": self.layer_weights.get(layer_name, 1.0),
                "features_count": len(available_features),
                "features_available": len([f for f in available_features if not pd.isna(layer_data.get(f, np.nan))]),
                "avg_value": layer_data.mean() if not layer_data.empty else 0.0,
                "std_value": layer_data.std() if not layer_data.empty else 0.0,
                "influence": "neutral" if self.layer_weights.get(layer_name, 1.0) == 1.0 else 
                           ("amplified" if self.layer_weights.get(layer_name, 1.0) > 1.0 else "dampened")
            }
        
        return breakdown
    
    def update_layer_weight(self, layer_name: str, new_weight: float):
        """
        Оновлює вагу конкретного шару.
        
        Args:
            layer_name: Наwithва шару
            new_weight: Нова вага (1.0 = notйтрально, >1.0 = пandдсилення, <1.0 = ослаблення)
        """
        if layer_name in FEATURE_LAYERS:
            old_weight = self.layer_weights.get(layer_name, 1.0)
            self.layer_weights[layer_name] = new_weight
            logger.info(f"[LayerProcessor] Шар '{layer_name}': {old_weight}  {new_weight}")
        else:
            logger.warning(f"[LayerProcessor] Невandдомий шар: {layer_name}")
    
    def reset_all_weights(self):
        """Скидає all ваги до notйтрального values 1.0"""
        for layer_name in self.layer_weights:
            self.layer_weights[layer_name] = 1.0
        logger.info("[LayerProcessor] Всand ваги скинуто до 1.0")
    
    def get_layer_summary(self) -> Dict[str, Dict]:
        """Поверandє withвеwhereння по allх шарах"""
        summary = {}
        
        for layer_name in FEATURE_LAYERS.keys():
            features = get_features_by_layer(layer_name)
            weight = self.layer_weights.get(layer_name, 1.0)
            
            summary[layer_name] = {
                "features_count": len(features),
                "weight": weight,
                "status": "neutral" if weight == 1.0 else ("amplified" if weight > 1.0 else "dampened"),
                "sample_features": features[:3] if features else []
            }
        
        return summary