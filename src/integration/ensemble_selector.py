"""
Ensemble Selector
Intelligent selection of best ensemble method based on context and data
Chooses between LiveAdaptiveEnsemble, EnhancedEnsembleModel, and other ensembles
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.core.logging.logger import ProjectLogger

@dataclass
class EnsembleContext:
    """Context information for ensemble selection"""
    data_size: int
    has_real_time_data: bool
    model_count: int
    market_regime: str
    volatility_level: float
    prediction_frequency: str
    computational_resources: str  # 'low', 'medium', 'high'
    latency_requirement: str  # 'low', 'medium', 'high'
    
class EnsembleSelector:
    """
    Intelligent ensemble selector that chooses the best ensemble method
    based on data characteristics, market conditions, and requirements
    """
    
    def __init__(self, logger=None):
        """
        Initialize ensemble selector
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or ProjectLogger.get_logger(__name__)
        
        # Ensemble method configurations
        self.ensemble_methods = {
            'live_adaptive': {
                'class_path': 'src.trading.live_adaptive_ensemble.LiveAdaptiveEnsemble',
                'strengths': ['real_time_adaptation', 'performance_tracking', 'regime_aware'],
                'weaknesses': ['requires_historical_data', 'higher_latency'],
                'requirements': {'min_models': 3, 'real_time_data': True, 'history_days': 30},
                'best_for': ['live_trading', 'adaptive_strategies', 'multi_model_systems']
            },
            'stacked_ensemble': {
                'class_path': 'src.ensembling.stacked_ensemble.StackedEnsemble',
                'strengths': ['meta_learning', 'ridge_regression', 'live_efficiency_weighting'],
                'weaknesses': ['requires_training', 'meta_model_complexity'],
                'requirements': {'min_models': 2, 'training_data': True, 'meta_model': True},
                'best_for': ['meta_learning', 'complex_combinations', 'weighted_combinations']
            },
            'consensus_engine': {
                'class_path': 'src.trading.consensus_engine.ConsensusEngine',
                'strengths': ['decision_core', 'regime_aware', 'critic_filters', 'knn_patterns'],
                'weaknesses': ['complex_dependencies', 'requires_diary_engine'],
                'requirements': {'min_models': 3, 'experience_diary': True, 'threshold_analyzer': True},
                'best_for': ['final_decisions', 'risk_aware', 'quality_signals']
            },
            'simple_average': {
                'class_path': 'built_in',
                'strengths': ['fast', 'simple', 'reliable'],
                'weaknesses': ['no_adaptation', 'equal_weights'],
                'requirements': {'min_models': 1, 'real_time_data': False},
                'best_for': ['quick_predictions', 'baseline', 'low_resources']
            },
            'weighted_average': {
                'class_path': 'built_in',
                'strengths': ['performance_based', 'simple'],
                'weaknesses': ['static_weights', 'requires_performance_data'],
                'requirements': {'min_models': 1, 'performance_data': True},
                'best_for': ['performance_weighted', 'moderate_complexity']
            }
        }
        
        # Selection rules
        self.selection_rules = self._initialize_selection_rules()
        
        self.logger = ProjectLogger.get_logger(__name__)
        self.logger.info("✅ EnsembleSelector initialized with intelligent selection rules")
    
    def select_best_ensemble(self, context: EnsembleContext, 
                            available_models: List[str],
                            performance_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Select the best ensemble method based on context and requirements
        
        Args:
            context: Ensemble context information
            available_models: List of available model names
            performance_data: Optional performance data for models
            
        Returns:
            Dict with selected ensemble and reasoning
        """
        try:
            # Score each ensemble method
            ensemble_scores = {}
            for method_name, method_config in self.ensemble_methods.items():
                score = self._score_ensemble_method(method_name, method_config, context, available_models, performance_data)
                ensemble_scores[method_name] = score
            
            # Select best method
            best_method = max(ensemble_scores.keys(), key=lambda k: ensemble_scores[k])
            best_score = ensemble_scores[best_method]
            
            # Generate reasoning
            reasoning = self._generate_selection_reasoning(best_method, best_score, context, ensemble_scores)
            
            selection_result = {
                'selected_ensemble': best_method,
                'score': best_score,
                'reasoning': reasoning,
                'all_scores': ensemble_scores,
                'context': context,
                'available_models': available_models,
                'selection_time': datetime.now(),
                'confidence': self._calculate_selection_confidence(best_score, ensemble_scores)
            }
            
            self.logger.info(f"✅ Selected {best_method} ensemble (score: {best_score:.2f})")
            return selection_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to select ensemble: {e}")
            return self._get_fallback_selection(context, available_models)
    
    def _score_ensemble_method(self, method_name: str, method_config: Dict, 
                              context: EnsembleContext, available_models: List[str],
                              performance_data: Optional[Dict]) -> float:
        """Score an ensemble method for the given context"""
        try:
            score = 0.0
            
            # Check basic requirements
            if not self._check_requirements(method_config['requirements'], context, available_models, performance_data):
                return 0.0
            
            # Context-based scoring
            score += self._score_context_fit(method_config, context)
            
            # Performance-based scoring
            if performance_data:
                score += self._score_performance_fit(method_config, performance_data)
            
            # Resource-based scoring
            score += self._score_resource_fit(method_config, context)
            
            # Latency scoring
            score += self._score_latency_fit(method_config, context)
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"❌ Failed to score {method_name}: {e}")
            return 0.0
    
    def _check_requirements(self, requirements: Dict, context: EnsembleContext, 
                           available_models: List[str], performance_data: Optional[Dict]) -> bool:
        """Check if ensemble method requirements are met"""
        try:
            # Check minimum models
            if 'min_models' in requirements:
                if len(available_models) < requirements['min_models']:
                    return False
            
            # Check real-time data requirement
            if 'real_time_data' in requirements:
                if requirements['real_time_data'] and not context.has_real_time_data:
                    return False
            
            # Check performance data requirement
            if 'performance_data' in requirements:
                if requirements['performance_data'] and not performance_data:
                    return False
            
            # Check colab results requirement
            if 'colab_results' in requirements:
                if requirements['colab_results']:
                    # This would check if colab results are available
                    # For now, assume not available
                    return False
            
            # Check batch mode requirement
            if 'batch_mode' in requirements:
                if requirements['batch_mode'] and context.prediction_frequency == 'real_time':
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to check requirements: {e}")
            return False
    
    def _score_context_fit(self, method_config: Dict, context: EnsembleContext) -> float:
        """Score how well method fits the context"""
        try:
            score = 0.0
            
            # Market regime scoring
            if context.market_regime == 'volatile':
                if 'regime_aware' in method_config['strengths']:
                    score += 0.2
            
            # Data size scoring
            if context.data_size > 10000:
                if 'batch_processing' in method_config['strengths']:
                    score += 0.1
            elif context.data_size < 1000:
                if 'fast' in method_config['strengths']:
                    score += 0.1
            
            # Model count scoring
            if context.model_count > 5:
                if 'multi_model_systems' in method_config['best_for']:
                    score += 0.2
            
            return score
            
        except Exception as e:
            self.logger.error(f"❌ Failed to score context fit: {e}")
            return 0.0
    
    def _score_performance_fit(self, method_config: Dict, performance_data: Dict) -> float:
        """Score based on available performance data"""
        try:
            score = 0.0
            
            # If performance data is available, prefer adaptive methods
            if performance_data and len(performance_data) > 0:
                if 'performance_based' in method_config['strengths']:
                    score += 0.15
                if 'real_time_adaptation' in method_config['strengths']:
                    score += 0.15
            
            return score
            
        except Exception as e:
            self.logger.error(f"❌ Failed to score performance fit: {e}")
            return 0.0
    
    def _score_resource_fit(self, method_config: Dict, context: EnsembleContext) -> float:
        """Score based on computational resources"""
        try:
            score = 0.0
            
            # Low resources - prefer simple methods
            if context.computational_resources == 'low':
                if 'fast' in method_config['strengths'] or 'simple' in method_config['strengths']:
                    score += 0.2
                elif 'heavy_models' in method_config['strengths']:
                    score -= 0.1
            
            # High resources - can handle complex methods
            elif context.computational_resources == 'high':
                if 'heavy_models' in method_config['strengths'] or 'colab_integration' in method_config['strengths']:
                    score += 0.2
            
            return score
            
        except Exception as e:
            self.logger.error(f"❌ Failed to score resource fit: {e}")
            return 0.0
    
    def _score_latency_fit(self, method_config: Dict, context: EnsembleContext) -> float:
        """Score based on latency requirements"""
        try:
            score = 0.0
            
            # High latency requirement - prefer simple methods
            if context.latency_requirement == 'high':
                if 'fast' in method_config['strengths']:
                    score += 0.2
                elif 'real_time_adaptation' in method_config['weaknesses']:
                    score -= 0.1
            
            # Low latency requirement - can handle complex methods
            elif context.latency_requirement == 'low':
                if 'real_time_adaptation' in method_config['strengths']:
                    score += 0.2
            
            return score
            
        except Exception as e:
            self.logger.error(f"❌ Failed to score latency fit: {e}")
            return 0.0
    
    def _generate_selection_reasoning(self, best_method: str, best_score: float,
                                    context: EnsembleContext, all_scores: Dict) -> str:
        """Generate human-readable reasoning for selection"""
        try:
            method_config = self.ensemble_methods[best_method]
            
            reasoning_parts = []
            
            # Main reason
            if best_score > 0.8:
                reasoning_parts.append(f"Excellent fit for {best_method}")
            elif best_score > 0.6:
                reasoning_parts.append(f"Good fit for {best_method}")
            else:
                reasoning_parts.append(f"Selected {best_method} as best available option")
            
            # Context-based reasons
            if context.has_real_time_data and 'real_time_adaptation' in method_config['strengths']:
                reasoning_parts.append("real-time adaptation capability")
            
            if context.computational_resources == 'low' and 'fast' in method_config['strengths']:
                reasoning_parts.append("fast execution for low-resource environment")
            
            if context.market_regime == 'volatile' and 'regime_aware' in method_config['strengths']:
                reasoning_parts.append("regime-aware weighting for volatile markets")
            
            if context.model_count > 5 and 'multi_model_systems' in method_config['best_for']:
                reasoning_parts.append("optimized for multi-model systems")
            
            # Join reasoning
            if len(reasoning_parts) > 1:
                reasoning = f"{reasoning_parts[0]} due to {', '.join(reasoning_parts[1:])}"
            else:
                reasoning = reasoning_parts[0] if reasoning_parts else "Default selection"
            
            return reasoning
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate reasoning: {e}")
            return f"Selected {best_method} based on scoring algorithm"
    
    def _calculate_selection_confidence(self, best_score: float, all_scores: Dict) -> float:
        """Calculate confidence in selection"""
        try:
            if best_score > 0.8:
                return 0.9
            elif best_score > 0.6:
                return 0.7
            elif best_score > 0.4:
                return 0.5
            else:
                return 0.3
                
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate confidence: {e}")
            return 0.5
    
    def _get_fallback_selection(self, context: EnsembleContext, available_models: List[str]) -> Dict[str, Any]:
        """Get fallback selection if main selection fails"""
        try:
            # Always available fallback
            fallback_method = 'simple_average'
            
            return {
                'selected_ensemble': fallback_method,
                'score': 0.3,
                'reasoning': f"Fallback to {fallback_method} due to selection error",
                'all_scores': {fallback_method: 0.3},
                'context': context,
                'available_models': available_models,
                'selection_time': datetime.now(),
                'confidence': 0.3,
                'is_fallback': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get fallback selection: {e}")
            return {
                'selected_ensemble': 'simple_average',
                'score': 0.0,
                'reasoning': 'Emergency fallback',
                'confidence': 0.0
            }
    
    def _initialize_selection_rules(self) -> Dict[str, Any]:
        """Initialize selection rules for different scenarios"""
        return {
            'live_trading': {
                'preferred': ['live_adaptive'],
                'avoid': ['enhanced_batch'],
                'reasoning': 'Live trading requires real-time adaptation'
            },
            'batch_prediction': {
                'preferred': ['enhanced_batch', 'weighted_average'],
                'avoid': ['live_adaptive'],
                'reasoning': 'Batch processing can handle heavier models'
            },
            'low_resources': {
                'preferred': ['simple_average'],
                'avoid': ['enhanced_batch', 'live_adaptive'],
                'reasoning': 'Limited resources require simple methods'
            },
            'high_volatility': {
                'preferred': ['live_adaptive'],
                'avoid': ['simple_average'],
                'reasoning': 'Volatile markets need adaptive weighting'
            }
        }
    
    def create_ensemble_instance(self, method_name: str, **kwargs) -> Optional[Any]:
        """
        Create an instance of the selected ensemble method
        
        Args:
            method_name: Name of the ensemble method
            **kwargs: Additional parameters for ensemble initialization
            
        Returns:
            Ensemble instance or None if creation fails
        """
        try:
            if method_name == 'live_adaptive':
                from src.trading.live_adaptive_ensemble import LiveAdaptiveEnsemble
                return LiveAdaptiveEnsemble(**kwargs)
            
            elif method_name == 'enhanced_batch':
                from src.models.ensemble.enhanced_ensemble import EnhancedEnsembleModel
                return EnhancedEnsembleModel(**kwargs)
            
            elif method_name == 'stacked_ensemble':
                from src.ensembling.stacked_ensemble import StackedEnsemble
                return StackedEnsemble(**kwargs)
            
            elif method_name == 'consensus_engine':
                from src.trading.consensus_engine import ConsensusEngine
                return ConsensusEngine(**kwargs)
            
            elif method_name == 'simple_average':
                # Return a simple average ensemble function
                return self._create_simple_average_ensemble()
            
            elif method_name == 'weighted_average':
                # Return a weighted average ensemble function
                return self._create_weighted_average_ensemble()
            
            else:
                self.logger.error(f"❌ Unknown ensemble method: {method_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create ensemble instance: {e}")
            return None
    
    def _create_simple_average_ensemble(self):
        """Create simple average ensemble function"""
        def simple_average_ensemble(predictions: Dict[str, np.ndarray]) -> np.ndarray:
            if not predictions:
                return np.array([])
            
            # Stack all predictions and take mean
            stacked = np.stack(list(predictions.values()))
            return np.mean(stacked, axis=0)
        
        return simple_average_ensemble
    
    def _create_weighted_average_ensemble(self):
        """Create weighted average ensemble function"""
        def weighted_average_ensemble(predictions: Dict[str, np.ndarray], 
                                    weights: Optional[Dict[str, float]] = None) -> np.ndarray:
            if not predictions:
                return np.array([])
            
            if weights is None:
                # Equal weights if none provided
                weights = {k: 1.0/len(predictions) for k in predictions.keys()}
            
            # Calculate weighted average
            weighted_sum = np.zeros_like(list(predictions.values())[0])
            total_weight = 0.0
            
            for model_name, pred in predictions.items():
                weight = weights.get(model_name, 0.0)
                weighted_sum += weight * pred
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return np.mean(np.stack(list(predictions.values())), axis=0)
        
        return weighted_average_ensemble
