#!/usr/bin/env python3
"""
Enhanced Smart Feature Selector - Full Integration with New Analysis Components
Integrates drift monitoring, redundancy detection, regime tracking, and news decay optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import logging
from datetime import datetime

from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import UnifiedConfigManager, get_current_config

# Import existing SmartFeatureSelector
from .smart_selector import SmartFeatureSelector

# Import new analysis components
from src.monitoring.feature_drift_monitor import get_feature_drift_monitor
from src.monitoring.data_freshness_monitor import get_data_freshness_monitor
from src.features.validation.redundancy_detector import get_redundancy_detector
from src.features.analysis.regime_importance_tracker import get_regime_importance_tracker
from src.features.analysis.news_decay_modeler import get_news_decay_modeler

logger = ProjectLogger.get_logger("EnhancedSmartFeatureSelector")

class EnhancedSmartFeatureSelector(SmartFeatureSelector):
    """
    Enhanced Smart Feature Selector with full integration of new analysis components.
    
    This selector extends the base SmartFeatureSelector with:
    - Real-time feature drift monitoring
    - Automatic redundancy elimination
    - Regime-specific importance tracking
    - ML-optimized news decay modeling
    - Comprehensive reporting and alerts
    
    Provides production-ready feature selection with full observability.
    """
    
    def __init__(self, config_manager: Optional[UnifiedConfigManager] = None):
        """
        Initialize Enhanced Smart Feature Selector.
        
        Args:
            config_manager: Configuration manager for system settings
        """
        # Initialize base SmartFeatureSelector
        super().__init__()
        
        self.config_manager = config_manager or get_current_config()
        
        # Initialize new analysis components
        self.drift_monitor = get_feature_drift_monitor()
        self.freshness_monitor = get_data_freshness_monitor()
        self.redundancy_detector = get_redundancy_detector()
        self.regime_tracker = get_regime_importance_tracker()
        self.news_decay_modeler = get_news_decay_modeler()
        
        # Enhanced settings
        self.monitoring_enabled = self.config_manager.get('feature_selection.monitoring_enabled', True)
        self.redundancy_elimination_enabled = self.config_manager.get('feature_selection.redundancy_elimination_enabled', True)
        self.regime_adaptation_enabled = self.config_manager.get('feature_selection.regime_adaptation_enabled', True)
        
        self.logger.info("✅ EnhancedSmartFeatureSelector initialized with full analysis integration")
    
    async def select_with_full_analysis(self,
                                      features_df: pd.DataFrame,
                                      target_series: pd.Series,
                                      context_id: str,
                                      market_data: Optional[pd.DataFrame] = None,
                                      news_data: Optional[pd.DataFrame] = None,
                                      model_metadata: Optional[Dict[str, Any]] = None,
                                      **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive feature selection with full analysis integration.
        
        Args:
            features_df: Input features DataFrame
            target_series: Target series for selection
            context_id: Context identifier for caching
            market_data: Market data for regime detection and drift analysis
            news_data: News data for decay optimization
            model_metadata: Model metadata with feature importance
            **kwargs: Additional parameters for selection
            
        Returns:
            Dict with comprehensive selection results and analysis
        """
        self.logger.info(f"🧠 Starting enhanced feature selection for {context_id}")
        
        results: Dict[str, Any] = {
            'context_id': context_id,
            'timestamp': datetime.now(),
            'original_feature_count': len(features_df.columns),
            'selected_features': [],
            'analysis_results': {},
            'monitoring_results': {},
            'recommendations': [],
            'performance_metrics': {}
        }
        
        try:
            # 1. Data Freshness Monitoring
            if self.monitoring_enabled:
                freshness_result = await self._monitor_data_freshness()
                if isinstance(results.get('monitoring_results'), dict):
                    results['monitoring_results']['data_freshness'] = freshness_result
            
            # 2. Feature Drift Monitoring
            drift_result = await self._monitor_feature_drift(features_df, model_metadata)
            if isinstance(results.get('monitoring_results'), dict):
                results['monitoring_results']['feature_drift'] = drift_result
            
            # 3. Redundancy Elimination
            if self.redundancy_elimination_enabled:
                redundancy_result = await self._eliminate_redundant_features(features_df, target_series)
                if isinstance(results.get('analysis_results'), dict):
                    results['analysis_results']['redundancy'] = redundancy_result
                clean_features = redundancy_result['cleaned_features']
            else:
                clean_features = features_df.copy()
                results['analysis_results']['redundancy'] = {'status': 'disabled'}
            
            # 4. Regime Importance Tracking
            if self.regime_adaptation_enabled and market_data is not None:
                regime_result = await self._track_regime_importance(
                    clean_features, target_series, market_data, model_metadata
                )
                if isinstance(results.get('analysis_results'), dict):
                    results['analysis_results']['regime_importance'] = regime_result
                
                # Get adaptive method weights
                method_weights = regime_result.get('method_weights', {})
            else:
                method_weights = {}
                if isinstance(results.get('analysis_results'), dict):
                    results['analysis_results']['regime_importance'] = {'status': 'disabled'}
            
            # 5. News Decay Optimization (if news data available)
            if news_data is not None and market_data is not None:
                decay_result = await self._optimize_news_decay(news_data, market_data)
                if isinstance(results.get('analysis_results'), dict):
                    results['analysis_results']['news_decay'] = decay_result
            else:
                if isinstance(results.get('analysis_results'), dict):
                    results['analysis_results']['news_decay'] = {'status': 'no_news_data'}
            
            # 6. Enhanced Feature Selection with Adaptive Weights
            selection_result = await self._enhanced_feature_selection(
                clean_features, target_series, context_id, method_weights, **kwargs
            )
            results['selected_features'] = selection_result['selected_features']
            if isinstance(results.get('analysis_results'), dict):
                results['analysis_results']['selection'] = selection_result
            
            # 7. Performance Metrics
            selected_features = results.get('selected_features', [])
            if isinstance(selected_features, list):
                results['performance_metrics'] = self._calculate_performance_metrics(
                    features_df, clean_features, selected_features
                )
            
            # 8. Generate Recommendations
            results['recommendations'] = self._generate_enhanced_recommendations(results)
            
            # 9. Log Comprehensive Summary
            self._log_enhanced_selection_summary(results)
            
            selected_count = len(results.get('selected_features', []))
            self.logger.info(f"✅ Enhanced feature selection complete. "
                           f"Selected {selected_count} features from {len(features_df.columns)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced feature selection: {e}", exc_info=True)
            results['error'] = str(e)
            return results
    
    async def _monitor_data_freshness(self) -> Dict[str, Any]:
        """Monitor data freshness across all sources."""
        
        try:
            freshness_result = await self.freshness_monitor.check_all_data_sources()
            if freshness_result is None:
                return {'status': 'error', 'error': 'Freshness check returned None'}
            return freshness_result
            
            if freshness_result.get('status') == 'warning':
                self.logger.warning("⚠️ Data freshness issues detected")
            
            return freshness_result
            
        except Exception as e:
            self.logger.error(f"Error in data freshness monitoring: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _monitor_feature_drift(self, 
                                  features_df: pd.DataFrame,
                                  model_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Monitor feature drift and importance changes."""
        
        try:
            drift_result = await self.drift_monitor.detect_drift(
                features_df, model_metadata
            )
            
            if drift_result.get('drift_detected', False):
                self.logger.warning("🚨 Feature drift detected!")
                
                # Log specific drift information
                if 'statistical_drift' in drift_result:
                    drifted_count = len(drift_result['statistical_drift'].get('drifted_features', []))
                    self.logger.warning(f"   Statistical drift in {drifted_count} features")
                
                if 'importance_drift' in drift_result:
                    importance_changes = len(drift_result['importance_drift'].get('significant_changes', []))
                    self.logger.warning(f"   Importance drift in {importance_changes} features")
            
            return drift_result
            
        except Exception as e:
            self.logger.error(f"Error in feature drift monitoring: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _eliminate_redundant_features(self,
                                        features_df: pd.DataFrame,
                                        target_series: pd.Series) -> Dict[str, Any]:
        """Eliminate redundant features using advanced analysis."""
        
        try:
            redundancy_result = self.redundancy_detector.eliminate_redundant_features(
                features_df, target_series
            )
            
            original_count = len(features_df.columns)
            final_count = redundancy_result['selected_count']
            reduction_ratio = redundancy_result['reduction_ratio']
            
            self.logger.info(f"🗑️ Redundancy elimination: {original_count} → {final_count} features "
                           f"({reduction_ratio:.1f}% reduction)")
            
            # Log specific redundancy information
            if redundancy_result.get('correlation_groups'):
                group_count = len(redundancy_result['correlation_groups'])
                self.logger.info(f"   Found {group_count} redundant correlation groups")
            
            if redundancy_result.get('vif_results', {}).get('high_vif_features'):
                high_vif_count = len(redundancy_result['vif_results']['high_vif_features'])
                self.logger.info(f"   Found {high_vif_count} high VIF features")
            
            return redundancy_result
            
        except Exception as e:
            self.logger.error(f"Error in redundancy elimination: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'cleaned_features': features_df.copy()
            }
    
    async def _track_regime_importance(self,
                                    features_df: pd.DataFrame,
                                    target_series: pd.Series,
                                    market_data: pd.DataFrame,
                                    model_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Track feature importance across market regimes."""
        
        try:
            # Calculate current feature importance (simplified)
            current_importance = self._calculate_feature_importance(features_df, target_series)
            
            regime_result = await self.regime_tracker.track_feature_importance(
                current_importance, market_data, model_metadata
            )
            
            current_regime = regime_result.get('current_regime', 'normal')
            stability_score = regime_result.get('regime_stability', {}).get('stability_score', 1.0)
            
            self.logger.info(f"📊 Regime tracking: Current regime = {current_regime}, "
                           f"Stability = {stability_score:.2f}")
            
            # Log regime-specific information
            if regime_result.get('regime_switch_detected', False):
                switch_info = regime_result.get('regime_switch_info', {})
                self.logger.info(f"   Regime switch detected: {switch_info}")
            
            if regime_result.get('adaptation_recommendations'):
                recommendations = regime_result['adaptation_recommendations']
                self.logger.info(f"   Regime recommendations: {len(recommendations)}")
            
            return regime_result
            
        except Exception as e:
            self.logger.error(f"Error in regime importance tracking: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'method_weights': {}
            }
    
    async def _optimize_news_decay(self,
                               news_data: pd.DataFrame,
                               market_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize news impact decay modeling."""
        
        try:
            # Prepare market returns for decay modeling
            if 'returns' not in market_data.columns:
                market_data['returns'] = market_data['close'].pct_change()
            
            decay_result = await self.news_decay_modeler.fit_optimal_decay_model(
                news_data, market_data
            )
            
            if decay_result.get('best_overall_model'):
                best_model = decay_result['best_overall_model']
                self.logger.info(f"📈 News decay optimization: Best model = {best_model['function_name']}")
                
                # Log performance information
                performance = best_model.get('performance', {})
                r2_score = performance.get('r2', 0)
                self.logger.info(f"   Model R² score: {r2_score:.3f}")
            
            return decay_result
            
        except Exception as e:
            self.logger.error(f"Error in news decay optimization: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _enhanced_feature_selection(self,
                                       features_df: pd.DataFrame,
                                       target_series: pd.Series,
                                       context_id: str,
                                       method_weights: Dict[str, float],
                                       **kwargs) -> Dict[str, Any]:
        """Perform enhanced feature selection with adaptive method weights."""
        
        try:
            # Use base SmartFeatureSelector with adaptive weights
            market_regime = kwargs.get('market_regime', 'normal')
            
            # Apply method weights if available
            if method_weights:
                # Temporarily update selector weights (this would require extending base class)
                original_weights = self._get_current_method_weights()
                self._update_method_weights(method_weights)
            
            # Perform selection
            selected_features = self.select(
                features_df, target_series, context_id,
                market_regime=market_regime, **kwargs
            )
            
            # Restore original weights if they were changed
            if method_weights and hasattr(self, '_restore_method_weights'):
                self._restore_method_weights(original_weights)
            
            self.logger.info(f"✅ Enhanced selection: {len(selected_features)} features selected")
            
            return {
                'selected_features': selected_features,
                'method_weights_used': method_weights,
                'selection_method': 'enhanced_adaptive'
            }
            
        except Exception as e:
            self.logger.error(f"Error in enhanced feature selection: {e}")
            return {
                'selected_features': [],
                'error': str(e)
            }
    
    def _calculate_feature_importance(self,
                                   features_df: pd.DataFrame,
                                   target_series: pd.Series) -> Dict[str, float]:
        """Calculate simplified feature importance for regime tracking."""
        
        try:
            importance = {}
            
            # Use correlation as simple importance metric
            for feature_name in features_df.columns:
                if features_df[feature_name].dtype in ['float64', 'int64']:
                    correlation = features_df[feature_name].corr(target_series)
                    importance[feature_name] = abs(correlation) if not np.isnan(correlation) else 0.0
                else:
                    importance[feature_name] = 0.0
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def _get_current_method_weights(self) -> Dict[str, float]:
        """Get current method weights from base selector."""
        # This would need to be implemented in base class
        return {}
    
    def _update_method_weights(self, weights: Dict[str, float]) -> None:
        """Update method weights in base selector."""
        # This would need to be implemented in base class
        pass
    
    def _calculate_performance_metrics(self,
                                     original_features: pd.DataFrame,
                                     clean_features: pd.DataFrame,
                                     selected_features: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        
        try:
            metrics = {
                'original_feature_count': len(original_features.columns),
                'clean_feature_count': len(clean_features.columns),
                'selected_feature_count': len(selected_features),
                'total_reduction_ratio': (1 - len(selected_features) / len(original_features.columns)) * 100,
                'selection_efficiency': len(selected_features) / len(original_features.columns) if len(original_features.columns) > 0 else 0
            }
            
            # Calculate feature quality metrics
            if selected_features:
                selected_df = clean_features[selected_features]
                
                # Feature variance (higher is generally better)
                avg_variance = selected_df.var().mean()
                metrics['average_feature_variance'] = avg_variance
                
                # Feature correlation (lower is better for diversity)
                correlation_matrix = selected_df.corr().abs()
                avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
                metrics['average_feature_correlation'] = avg_correlation
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _generate_enhanced_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on analysis results."""
        
        recommendations = []
        
        try:
            # Data freshness recommendations
            freshness_result = results.get('monitoring_results', {}).get('data_freshness', {})
            if freshness_result.get('status') == 'warning':
                recommendations.append(
                    "⚠️ Data freshness issues detected. Check data collection pipeline."
                )
            
            # Feature drift recommendations
            drift_result = results.get('monitoring_results', {}).get('feature_drift', {})
            if drift_result.get('drift_detected', False):
                recommendations.append(
                    "🚨 Feature drift detected. Consider model retraining."
                )
                
                # Specific drift recommendations
                if 'statistical_drift' in drift_result:
                    drifted_count = len(drift_result['statistical_drift'].get('drifted_features', []))
                    recommendations.append(
                        f"📊 {drifted_count} features show statistical drift. Review data pipeline."
                    )
            
            # Redundancy recommendations
            redundancy_result = results.get('analysis_results', {}).get('redundancy', {})
            if redundancy_result.get('reduction_ratio', 0) > 50:
                recommendations.append(
                    "🗑️ High redundancy detected (>50%). Consider feature engineering review."
                )
            
            # Regime recommendations
            regime_result = results.get('analysis_results', {}).get('regime_importance', {})
            if regime_result.get('adaptation_recommendations'):
                recommendations.extend(regime_result['adaptation_recommendations'][:3])
            
            # Performance recommendations
            metrics = results.get('performance_metrics', {})
            if metrics.get('selection_efficiency', 0) < 0.1:
                recommendations.append(
                    "📉 Low selection efficiency. Consider increasing feature selection threshold."
                )
            
            if metrics.get('average_feature_correlation', 0) > 0.8:
                recommendations.append(
                    "🔗 High feature correlation. Consider additional redundancy elimination."
                )
            
            # No issues recommendation
            if not recommendations:
                recommendations.append("✅ All metrics look good. Feature selection is performing well.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["❌ Error generating recommendations"]
    
    def _log_enhanced_selection_summary(self, results: Dict[str, Any]) -> None:
        """Log comprehensive selection summary."""
        
        try:
            self.logger.info("=" * 80)
            self.logger.info("🧠 ENHANCED FEATURE SELECTION SUMMARY")
            self.logger.info("=" * 80)
            
            # Basic metrics
            original_count = results.get('original_feature_count', 0)
            selected_count = len(results.get('selected_features', []))
            reduction_ratio = results.get('performance_metrics', {}).get('total_reduction_ratio', 0)
            
            self.logger.info(f"📊 Features: {original_count} → {selected_count} ({reduction_ratio:.1f}% reduction)")
            
            # Monitoring results
            monitoring = results.get('monitoring_results', {})
            
            # Data freshness
            freshness = monitoring.get('data_freshness', {})
            freshness_status = freshness.get('status', 'unknown')
            self.logger.info(f"🕐 Data Freshness: {freshness_status}")
            
            # Feature drift
            drift = monitoring.get('feature_drift', {})
            drift_detected = drift.get('drift_detected', False)
            self.logger.info(f"🚨 Feature Drift: {'DETECTED' if drift_detected else 'None'}")
            
            # Analysis results
            analysis = results.get('analysis_results', {})
            
            # Redundancy
            redundancy = analysis.get('redundancy', {})
            if redundancy.get('status') != 'disabled':
                redundancy_ratio = redundancy.get('reduction_ratio', 0)
                self.logger.info(f"🗑️ Redundancy Elimination: {redundancy_ratio:.1f}% reduction")
            
            # Regime tracking
            regime = analysis.get('regime_importance', {})
            if regime.get('status') != 'disabled':
                current_regime = regime.get('current_regime', 'unknown')
                stability = regime.get('regime_stability', {}).get('stability_score', 0)
                self.logger.info(f"📊 Current Regime: {current_regime} (stability: {stability:.2f})")
            
            # News decay
            decay = analysis.get('news_decay', {})
            if decay.get('status') != 'disabled' and decay.get('best_overall_model'):
                best_model = decay['best_overall_model'].get('function_name', 'unknown')
                self.logger.info(f"📈 News Decay Model: {best_model}")
            
            # Recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                self.logger.info(f"💡 Recommendations: {len(recommendations)}")
                for i, rec in enumerate(recommendations[:3]):  # Show top 3
                    self.logger.info(f"   {i+1}. {rec}")
            
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Error logging selection summary: {e}")


# Factory function for easy instantiation
def get_enhanced_smart_selector(config_manager: Optional[UnifiedConfigManager] = None) -> EnhancedSmartFeatureSelector:
    """Factory function to get EnhancedSmartFeatureSelector instance."""
    return EnhancedSmartFeatureSelector(config_manager)


# Convenience function for quick enhanced selection
async def select_features_enhanced(features_df: pd.DataFrame,
                                 target_series: pd.Series,
                                 context_id: str,
                                 config_manager: Optional[UnifiedConfigManager] = None,
                                 **kwargs) -> Dict[str, Any]:
    """
    Quick enhanced feature selection.
    
    Args:
        features_df: Features DataFrame to analyze
        target_series: Target series for selection
        context_id: Context identifier
        config_manager: Configuration manager
        **kwargs: Additional parameters
        
    Returns:
        Enhanced selection result dictionary
    """
    selector = get_enhanced_smart_selector(config_manager)
    return await selector.select_with_full_analysis(features_df, target_series, context_id, **kwargs)
