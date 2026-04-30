# src/patterns/pattern_tuning.py
"""
Pattern Adjustment Tuning - Optimizes the strength of pattern-based adjustments post-ML inference.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_absolute_error, accuracy_score
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("PatternTuning")

class PatternAdjustmentTuner:
    """Computes optimal weights for pattern-based prediction adjustments."""
    
    def __init__(self):
        """Initializes tuner with default weights for known pattern categories."""
        # Sensitivity weights for adjustments (default = 1.0)
        self.pattern_weights = {
            "banking_crisis": 1.0,
            "tech_breakthrough": 1.0, 
            "geopolitical_crisis": 1.0,
            "health_crisis": 1.0,
            "monetary_policy_shift": 1.0
        }
        
        # Performance history of weight candidates
        self.tuning_results = {}
    
    def test_pattern_weights(self, base_predictions: np.ndarray,
                           true_values: np.ndarray,
                           pattern_adjustments: Dict[str, np.ndarray],
                           weight_range: List[float] = None) -> Dict[str, Dict]:
        """
        Grid searches for the optimal weight for each pattern type based on historical accuracy.
        
        Args:
            base_predictions: Raw ML model outputs.
            true_values: Ground truth market movements.
            pattern_adjustments: Raw adjustments suggested by PatternRecognitionAdjuster.
            weight_range: List of candidate multipliers to test.
        """
        
        if weight_range is None:
            weight_range = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
        
        results = {}
        
        for pattern_name, adjustments in pattern_adjustments.items():
            logger.info(f"Grid searching weights for pattern: {pattern_name}")
            pattern_results = {}
            
            for weight in weight_range:
                # Apply candidate weight to suggested adjustment
                weighted_adjustments = adjustments * weight
                adjusted_predictions = base_predictions + weighted_adjustments
                
                # Evaluate performance metric
                if len(np.unique(true_values)) == 2:  # Classification task
                    # Convert to binary outcomes
                    binary_predictions = (adjusted_predictions > 0.5).astype(int)
                    score = accuracy_score(true_values, binary_predictions)
                    metric_name = "accuracy"
                else:  # Regression task
                    score = -mean_absolute_error(true_values, adjusted_predictions)  # Using negative MAE for maximization
                    metric_name = "neg_mae"
                
                pattern_results[weight] = {
                    "score": score,
                    "metric": metric_name
                }
                
                logger.debug(f"  Candidate weight {weight}: {metric_name} = {score:.4f}")
            
            # Find the weight associated with the best score
            best_weight = max(pattern_results, key=lambda w, pr=pattern_results: pr[w]["score"])
            best_score = pattern_results[best_weight]["score"]
            
            results[pattern_name] = {
                "best_weight": best_weight,
                "best_score": best_score,
                "all_results": pattern_results
            }
            
            logger.info(f"Optimization complete for '{pattern_name}': best weight = {best_weight} "
                       f"({pattern_results[best_weight]['metric']}: {best_score:.4f})")
        
        return results
    
    def optimize_pattern_weights(self, validation_data: Dict) -> Dict[str, float]:
        """Runs the optimization protocol on provided validation samples."""
        
        base_predictions = validation_data["base_predictions"]
        true_values = validation_data["true_values"] 
        pattern_adjustments = validation_data["pattern_adjustments"]
        
        # Evaluate individual weights
        individual_results = self.test_pattern_weights(
            base_predictions, true_values, pattern_adjustments
        )
        
        # Synthesis into dictionary
        optimized_weights = {}
        for pattern_name, results in individual_results.items():
            optimized_weights[pattern_name] = results["best_weight"]
        
        # Update local state
        self.tuning_results = individual_results
        self.pattern_weights = optimized_weights
        
        logger.info(f"Pattern sensitivity re-calibrated: {optimized_weights}")
        return optimized_weights
    
    def apply_tuned_adjustments(self, base_predictions: np.ndarray,
                               pattern_adjustments: Dict[str, np.ndarray]) -> np.ndarray:
        """Applies calibrated weights to suggested adjustments during live inference."""
        
        final_predictions = base_predictions.copy()
        
        for pattern_name, adjustments in pattern_adjustments.items():
            weight = self.pattern_weights.get(pattern_name, 1.0)
            weighted_adjustments = adjustments * weight
            final_predictions += weighted_adjustments
            
            logger.debug(f"Applied pattern '{pattern_name}' with sensitivity {weight}")
        
        return final_predictions
    
    def save_tuned_weights(self, filepath: str = "pattern_weights.json"):
        """Persists calibrated sensitivity settings to disk."""
        with open(filepath, 'w') as f:
            json.dump(self.pattern_weights, f, indent=2)
        
        logger.info(f"Pattern weight configurations saved to: {filepath}")
    
    def load_tuned_weights(self, filepath: str = "pattern_weights.json") -> bool:
        """Loads sensitivity configurations from disk."""
        if not os.path.exists(filepath):
            logger.debug("Weight configuration file not found. Using conservative defaults (1.0)")
            return False
        
        try:
            with open(filepath, 'r') as f:
                self.pattern_weights = json.load(f)
            
            logger.info(f"Pattern weights synchronized: {self.pattern_weights}")
            return True
        except Exception as e:
            logger.warning(f"Failed to synchronize pattern weights: {e}")
            return False

class IntegratedPredictionPipeline:
    """Coordination logic for multi-layer inference: ML -> Layer Balancing -> Pattern Overlay."""
    
    def __init__(self):
        """Initializes the integration pipeline component."""
        self.pattern_tuner = PatternAdjustmentTuner()
        
    def full_prediction_pipeline(self, model, X_test: np.ndarray, 
                                current_news: List[Dict],
                                ticker: str) -> Dict[str, Any]:
        """
        Executes the complete predictive stack:
        1. Model inference
        2. Feature layer weighting
        3. Fundamental/News pattern adjustment
        """
        
        # STAGE 1: Raw Machine Learning Inference
        base_predictions = model.predict(X_test)
        logger.info(f"Inference Stage 1 (Base ML): {np.mean(base_predictions):.4f}")
        
        # STAGE 2: Feature Layer Weight Balancing (Post-processing)
        layer_adjusted_predictions = base_predictions  # Placeholder for future balance logic
        logger.info(f"Inference Stage 2 (Layer Balance): {np.mean(layer_adjusted_predictions):.4f}")
        
        # STAGE 3: Fundamental/Behavioral Pattern Overlay
        final_predictions = layer_adjusted_predictions
        all_patterns = {}
        if current_news:
            from src.patterns.pattern_recognition_adjustment import pattern_adjuster
            
            # Extract pattern signals from news text
            for news_item in current_news:
                news_text = news_item.get("title", "") + " " + news_item.get("description", "")
                news_sentiment = news_item.get("sentiment_score", 0.0)
                patterns = pattern_adjuster.recognize_pattern_in_news(news_text, news_sentiment)
                
                for pattern_name, strength in patterns.items():
                    all_patterns[pattern_name] = max(all_patterns.get(pattern_name, 0), strength)
            
            if all_patterns:
                # Calculate theoretical adjustment
                adjustments = pattern_adjuster.calculate_pattern_adjustments(all_patterns, "1_month")
                
                if ticker in adjustments:
                    pattern_adjustment = adjustments[ticker]
                    
                    # Apply calibrated sensitivity multiplier
                    dominant_pattern = max(all_patterns, key=all_patterns.get)
                    pattern_weight = self.pattern_tuner.pattern_weights.get(dominant_pattern, 1.0)
                    
                    weighted_adjustment = pattern_adjustment * pattern_weight
                    final_predictions = layer_adjusted_predictions + weighted_adjustment
                    
                    logger.info(f"Inference Stage 3 (Pattern Correction): {np.mean(final_predictions):.4f} "
                               f"(delta: {weighted_adjustment:+.4f}, pattern: {dominant_pattern})")
        
        return {
            "base_predictions": base_predictions,
            "layer_adjusted": layer_adjusted_predictions, 
            "final_predictions": final_predictions,
            "recognized_patterns": all_patterns if current_news else {},
            "pipeline_summary": {
                "base_ml_score": np.mean(base_predictions),
                "layer_balance_score": np.mean(layer_adjusted_predictions),
                "pattern_corrected_score": np.mean(final_predictions)
            }
        }

# Singleton instances for global access
pattern_tuner = PatternAdjustmentTuner()
integrated_pipeline = IntegratedPredictionPipeline()

def get_integrated_predictions(model, X_test, current_news, ticker):
    """Utility entry point for running the integrated predictive stack."""
    return integrated_pipeline.full_prediction_pipeline(
        model, X_test, current_news, ticker
    )

if __name__ == "__main__":
    ProjectLogger.setup_logging()
    logger.info("🎬 Initializing integrated predictive pipeline test sequence")
    
    # Synchronize calibrated sensitivities
    pattern_tuner.load_tuned_weights()
    
    logger.info(f"Operational pattern sensitivity coefficients: {pattern_tuner.pattern_weights}")