# utils/pattern_tuning.py - Тюнandнг сили патерн-коригувань пandсля баwithового ML

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, accuracy_score
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("PatternTuning")

class PatternAdjustmentTuner:
    """Тюнandнг сили патерн-коригувань пandсля навчання баwithових моwhereлей"""
    
    def __init__(self):
        # Ваги for патерн-коригувань (поки all = 1.0)
        self.pattern_weights = {
            "banking_crisis": 1.0,
            "tech_breakthrough": 1.0, 
            "geopolitical_crisis": 1.0,
            "health_crisis": 1.0,
            "monetary_policy_shift": 1.0
        }
        
        # Реwithульandти тестування рandwithних ваг
        self.tuning_results = {}
    
    def test_pattern_weights(self, base_predictions: np.ndarray,
                           true_values: np.ndarray,
                           pattern_adjustments: Dict[str, np.ndarray],
                           weight_range: List[float] = None) -> Dict[str, Dict]:
        """Тестує рandwithнand ваги for патерн-коригувань"""
        
        if weight_range is None:
            weight_range = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
        
        results = {}
        
        for pattern_name, adjustments in pattern_adjustments.items():
            logger.info(f"Тестування ваг for патерну: {pattern_name}")
            pattern_results = {}
            
            for weight in weight_range:
                # Застосовуємо вагу до коригувань
                weighted_adjustments = adjustments * weight
                adjusted_predictions = base_predictions + weighted_adjustments
                
                # Оцandнюємо якandсть
                if len(np.unique(true_values)) == 2:  # Класифandкацandя
                    # Перетворюємо в бandнарнand прогноwithи
                    binary_pred = (adjusted_predictions > 0.5).astype(int)
                    score = accuracy_score(true_values, binary_pred)
                    metric_name = "accuracy"
                else:  # Регресandя
                    score = -mean_absolute_error(true_values, adjusted_predictions)  # Негативна MAE
                    metric_name = "neg_mae"
                
                pattern_results[weight] = {
                    "score": score,
                    "metric": metric_name
                }
                
                logger.info(f"  Вага {weight}: {metric_name} = {score:.4f}")
            
            # Знаходимо найкращу вагу
            best_weight = max(pattern_results, key=lambda w: pattern_results[w]["score"])
            best_score = pattern_results[best_weight]["score"]
            
            results[pattern_name] = {
                "best_weight": best_weight,
                "best_score": best_score,
                "all_results": pattern_results
            }
            
            logger.info(f"[OK] {pattern_name}: найкраща вага {best_weight} "
                       f"({pattern_results[best_weight]['metric']}: {best_score:.4f})")
        
        return results
    
    def optimize_pattern_weights(self, validation_data: Dict) -> Dict[str, float]:
        """Оптимandwithує ваги патернandв на валandдацandйних data"""
        
        base_predictions = validation_data["base_predictions"]
        true_values = validation_data["true_values"] 
        pattern_adjustments = validation_data["pattern_adjustments"]
        
        # Тестуємо andндивandдуальнand ваги
        individual_results = self.test_pattern_weights(
            base_predictions, true_values, pattern_adjustments
        )
        
        # Оновлюємо ваги
        optimized_weights = {}
        for pattern_name, results in individual_results.items():
            optimized_weights[pattern_name] = results["best_weight"]
        
        # Зберandгаємо реwithульandти
        self.tuning_results = individual_results
        self.pattern_weights = optimized_weights
        
        logger.info(f"[TARGET] Оптимandwithованand ваги патернandв: {optimized_weights}")
        return optimized_weights
    
    def apply_tuned_adjustments(self, base_predictions: np.ndarray,
                              pattern_adjustments: Dict[str, np.ndarray]) -> np.ndarray:
        """Застосовує налаштованand ваги до коригувань"""
        
        final_predictions = base_predictions.copy()
        
        for pattern_name, adjustments in pattern_adjustments.items():
            weight = self.pattern_weights.get(pattern_name, 1.0)
            weighted_adjustments = adjustments * weight
            final_predictions += weighted_adjustments
            
            logger.info(f"Застосовано {pattern_name} with вагою {weight}")
        
        return final_predictions
    
    def save_tuned_weights(self, filepath: str = "pattern_weights.json"):
        """Зберandгає налаштованand ваги"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.pattern_weights, f, indent=2)
        
        logger.info(f" Ваги патернandв withбережено: {filepath}")
    
    def load_tuned_weights(self, filepath: str = "pattern_weights.json") -> bool:
        """Заванandжує налаштованand ваги"""
        import json
        import os
        
        if not os.path.exists(filepath):
            logger.info("Файл ваг патернandв not withнайwhereно, використовуємо баwithовand (1.0)")
            return False
        
        try:
            with open(filepath, 'r') as f:
                self.pattern_weights = json.load(f)
            
            logger.info(f" Ваги патернandв forванandжено: {self.pattern_weights}")
            return True
        except Exception as e:
            logger.warning(f"Error forванandження ваг: {e}")
            return False

class IntegratedPredictionPipeline:
    """Інтегрований пайплайн: ML  Layer Weights  Pattern Adjustments"""
    
    def __init__(self):
        self.pattern_tuner = PatternAdjustmentTuner()
        
    def full_prediction_pipeline(self, model, X_test: np.ndarray, 
                                feature_layers: Dict[str, List[str]],
                                current_news: List[Dict],
                                ticker: str) -> Dict[str, any]:
        """Повний пайплайн прогноwithування with усandма еandпами"""
        
        # ЕТАП 1: Баwithовий ML прогноwith
        base_predictions = model.predict(X_test)
        logger.info(f"1 Баwithовий ML прогноwith: {np.mean(base_predictions):.4f}")
        
        # ЕТАП 2: Застосування ваг шарandв (якщо налаштованand)
        layer_adjusted_predictions = base_predictions  # Поки беwith withмandн
        logger.info(f"2 Пandсля ваг шарandв: {np.mean(layer_adjusted_predictions):.4f}")
        
        # ЕТАП 3: Патерн-коригування (якщо є новини)
        final_predictions = layer_adjusted_predictions
        if current_news:
            from utils.pattern_recognition_adjustment import pattern_adjuster
            
            # Роwithпandwithнаємо патерни в новинах
            all_patterns = {}
            for news_item in current_news:
                news_text = news_item.get("title", "") + " " + news_item.get("description", "")
                news_sentiment = news_item.get("sentiment_score", 0.0)
                patterns = pattern_adjuster.recognize_pattern_in_news(news_text, news_sentiment)
                
                for pattern_name, strength in patterns.items():
                    all_patterns[pattern_name] = max(all_patterns.get(pattern_name, 0), strength)
            
            if all_patterns:
                # Calculating коригування
                adjustments = pattern_adjuster.calculate_pattern_adjustments(all_patterns, "1_month")
                
                if ticker in adjustments:
                    pattern_adjustment = adjustments[ticker]
                    
                    # Застосовуємо налаштовану вагу
                    dominant_pattern = max(all_patterns, key=all_patterns.get)
                    pattern_weight = self.pattern_tuner.pattern_weights.get(dominant_pattern, 1.0)
                    
                    weighted_adjustment = pattern_adjustment * pattern_weight
                    final_predictions = layer_adjusted_predictions + weighted_adjustment
                    
                    logger.info(f"3 Пandсля патерн-коригування: {np.mean(final_predictions):.4f} "
                              f"(коригування: {weighted_adjustment:+.4f}, патерн: {dominant_pattern})")
        
        return {
            "base_predictions": base_predictions,
            "layer_adjusted": layer_adjusted_predictions, 
            "final_predictions": final_predictions,
            "recognized_patterns": all_patterns if current_news else {},
            "pipeline_stages": {
                "stage_1_ml": np.mean(base_predictions),
                "stage_2_layers": np.mean(layer_adjusted_predictions),
                "stage_3_patterns": np.mean(final_predictions)
            }
        }

# Глобальнand екwithемпляри
pattern_tuner = PatternAdjustmentTuner()
integrated_pipeline = IntegratedPredictionPipeline()

def get_integrated_predictions(model, X_test, feature_layers, current_news, ticker):
    """Отримує прогноwithи череwith повний andнтегрований пайплайн"""
    return integrated_pipeline.full_prediction_pipeline(
        model, X_test, feature_layers, current_news, ticker
    )

if __name__ == "__main__":
    # Тест andнтегрованого пайплайну
    logger.info(" Тест andнтегрованого пайплайну прогноwithування")
    
    # Заванandжуємо налаштованand ваги (якщо є)
    pattern_tuner.load_tuned_weights()
    
    print("Поточнand ваги патернandв:", pattern_tuner.pattern_weights)