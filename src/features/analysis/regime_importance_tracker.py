#!/usr/bin/env python3
"""
Regime Importance Tracker - Dynamic Feature Importance Tracking Across Market Regimes
Tracks and adapts feature importance changes across different market regimes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
import asyncio

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("RegimeImportanceTracker")

class RegimeImportanceTracker:
    """
    Tracks feature importance changes across market regimes and adapts selection strategies.
    
    This tracker monitors:
    - Feature importance stability across different market regimes
    - Regime-specific importance patterns
    - Automatic regime switching detection
    - Dynamic method weight adaptation for feature selection
    
    Critical for maintaining model performance across changing market conditions.
    """
    
    # Market regime definitions
    REGIME_TYPES = {
        'normal': {
            'description': 'Normal market conditions',
            'volatility_range': (0.01, 0.02),
            'trend_strength': (-0.001, 0.001)
        },
        'volatile': {
            'description': 'High volatility market',
            'volatility_range': (0.02, 0.05),
            'trend_strength': (-0.003, 0.003)
        },
        'trending_up': {
            'description': 'Strong uptrend market',
            'volatility_range': (0.015, 0.025),
            'trend_strength': (0.002, 0.005)
        },
        'trending_down': {
            'description': 'Strong downtrend market',
            'volatility_range': (0.015, 0.025),
            'trend_strength': (-0.005, -0.002)
        },
        'crisis': {
            'description': 'Market crisis conditions',
            'volatility_range': (0.04, 0.10),
            'trend_strength': (-0.01, 0.01)
        }
    }
    
    # Stability thresholds
    STABILITY_THRESHOLDS = {
        'importance_stability': 0.3,        # 30% change indicates instability
        'regime_detection_threshold': 0.7,   # 70% confidence for regime switch
        'min_samples_per_regime': 50,        # Minimum samples for reliable analysis
        'window_size_days': 30,               # Rolling window for analysis
        'significance_threshold': 0.05         # Statistical significance
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RegimeImportanceTracker.
        
        Args:
            config: Configuration dictionary for regime tracking
        """
        self.logger = logger
        self.config = config or {}
        
        # Override thresholds with config
        self.thresholds = self.STABILITY_THRESHOLDS.copy()
        self.thresholds.update(self.config.get('thresholds', {}))
        
        # Data storage
        self.importance_history: List[Dict[str, Any]] = []
        self.regime_history: List[Dict[str, Any]] = []
        self.regime_importance_cache: Dict[str, Any] = {}
        self.regime_switch_points: List[Dict[str, Any]] = []
        
        # Analysis settings
        self.window_size_days = self.config.get('window_size_days', 30)
        self.adaptation_enabled = self.config.get('adaptation_enabled', True)
        self.auto_regime_detection = self.config.get('auto_regime_detection', True)
        
        # Storage paths
        self.storage_path = Path(self.config.get('storage_path', 'data/analysis/regime_importance'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ RegimeImportanceTracker initialized")
    
    async def track_feature_importance(self, 
                                   current_importance: Dict[str, float],
                                   market_data: pd.DataFrame,
                                   model_metadata: Optional[Dict[str, Any]] = None,
                                   current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Track feature importance and analyze regime-specific patterns.
        
        Args:
            current_importance: Current feature importance dictionary
            market_data: Market data for regime detection
            model_metadata: Model metadata
            current_time: Current timestamp (uses now if None)
            
        Returns:
            Dict with regime analysis and recommendations
        """
        if current_time is None:
            current_time = datetime.now()
        
        self.logger.info(f"📊 Tracking feature importance at {current_time}")
        
        results = {
            'timestamp': current_time,
            'current_importance': current_importance,
            'current_regime': None,
            'regime_stability': {},
            'importance_changes': {},
            'regime_switch_detected': False,
            'adaptation_recommendations': [],
            'method_weights': {}
        }
        
        try:
            # 1. Detect current market regime
            current_regime = await self._detect_market_regime(market_data, current_time)
            results['current_regime'] = current_regime
            
            # 2. Store importance with regime
            importance_record = {
                'timestamp': current_time,
                'importance': current_importance.copy(),
                'regime': current_regime,
                'market_conditions': self._calculate_market_conditions(market_data)
            }
            
            self.importance_history.append(importance_record)
            
            # 3. Analyze importance stability
            stability_analysis = await self._analyze_importance_stability(current_regime, current_time)
            results['regime_stability'] = stability_analysis
            
            # 4. Detect regime switching
            regime_switch = await self._detect_regime_switch(current_regime, current_time)
            results['regime_switch_detected'] = regime_switch['switch_detected']
            
            if regime_switch['switch_detected']:
                results['regime_switch_info'] = regime_switch
                self.regime_switch_points.append(regime_switch)
            
            # 5. Calculate importance changes
            importance_changes = await self._calculate_importance_changes(current_importance, current_regime)
            results['importance_changes'] = importance_changes
            
            # 6. Generate adaptation recommendations
            if self.adaptation_enabled:
                recommendations = await self._generate_adaptation_recommendations(
                    current_regime, stability_analysis, importance_changes
                )
                results['adaptation_recommendations'] = recommendations
                
                # 7. Update method weights
                method_weights = await self._update_method_weights(current_regime, recommendations)
                results['method_weights'] = method_weights
            
            # 8. Clean old data
            self._clean_old_data()
            
            # 9. Store results
            self._store_tracking_results(results)
            
            self.logger.info(f"✅ Regime importance tracking complete. Regime: {current_regime}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in regime importance tracking: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': current_time
            }
    
    async def _detect_market_regime(self, 
                                 market_data: pd.DataFrame,
                                 current_time: datetime) -> str:
        """Detect current market regime based on market conditions."""
        
        try:
            # Calculate market conditions
            volatility = self._calculate_volatility(market_data)
            trend = self._calculate_trend(market_data)
            
            # Determine regime based on conditions
            for regime_name, regime_config in self.REGIME_TYPES.items():
                vol_range = regime_config['volatility_range']
                trend_range = regime_config['trend_strength']
                
                if (vol_range[0] <= float(volatility) <= vol_range[1] and
                    trend_range[0] <= float(trend) <= trend_range[1]):
                    
                    self.logger.debug(f"Detected regime: {regime_name} (vol={volatility:.4f}, trend={trend:.4f})")
                    return regime_name
            
            # Default to normal if no specific regime matches
            return 'normal'
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return 'normal'
    
    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate market volatility."""
        try:
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                return float(returns.std() * np.sqrt(252))  # Annualized volatility
            else:
                # Use any price column
                price_cols = [col for col in market_data.columns if 'price' in col.lower() or col in ['open', 'high', 'low', 'close']]
                if price_cols:
                    returns = market_data[price_cols[0]].pct_change().dropna()
                    return float(returns.std() * np.sqrt(252))
                return 0.02  # Default volatility
        except Exception:
            return 0.02
    
    def _calculate_trend(self, market_data: pd.DataFrame) -> float:
        """Calculate market trend."""
        try:
            if 'close' in market_data.columns:
                # Calculate linear trend over last 20 periods
                recent_prices = market_data['close'].tail(20)
                if len(recent_prices) >= 2:
                    x = np.arange(len(recent_prices))
                    slope = np.polyfit(x, recent_prices, 1)[0]
                    # Normalize by price level
                    normalized_trend = slope / recent_prices.mean()
                    return float(normalized_trend)
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive market conditions."""
        
        conditions = {}
        
        try:
            # Volatility
            conditions['volatility'] = self._calculate_volatility(market_data)
            
            # Trend
            conditions['trend'] = self._calculate_trend(market_data)
            
            # Volume (if available)
            if 'volume' in market_data.columns:
                recent_volume = market_data['volume'].tail(10).mean()
                historical_volume = market_data['volume'].mean()
                conditions['volume_ratio'] = recent_volume / historical_volume if historical_volume > 0 else 1.0
            else:
                conditions['volume_ratio'] = 1.0
            
            # Price momentum
            if 'close' in market_data.columns:
                momentum_5d = (market_data['close'].iloc[-1] / market_data['close'].iloc[-6] - 1) if len(market_data) >= 6 else 0
                momentum_20d = (market_data['close'].iloc[-1] / market_data['close'].iloc[-21] - 1) if len(market_data) >= 21 else 0
                conditions['momentum_5d'] = momentum_5d
                conditions['momentum_20d'] = momentum_20d
            
        except Exception as e:
            self.logger.error(f"Error calculating market conditions: {e}")
        
        return conditions
    
    async def _analyze_importance_stability(self, 
                                        current_regime: str,
                                        current_time: datetime) -> Dict[str, Any]:
        """Analyze feature importance stability for current regime."""
        
        stability_analysis = {
            'regime': current_regime,
            'stability_score': 1.0,
            'stable_features': [],
            'unstable_features': [],
            'importance_variance': {},
            'sample_count': 0
        }
        
        try:
            # Get historical importance for this regime
            regime_importance = [
                record['importance'] for record in self.importance_history
                if record['regime'] == current_regime
            ]
            
            stability_analysis['sample_count'] = len(regime_importance)
            
            if len(regime_importance) < self.thresholds['min_samples_per_regime']:
                stability_analysis['stability_score'] = 0.0  # Insufficient data
                return stability_analysis
            
            # Calculate importance variance for each feature
            feature_importances: Dict[str, List[float]] = {}
            for record in regime_importance:
                for feature_name, importance in record['importance'].items():
                    if feature_name not in feature_importances:
                        feature_importances[feature_name] = []
                    feature_importances[feature_name].append(importance)
            
            # Calculate stability metrics
            stability_threshold = self.thresholds['importance_stability']
            
            for feature_name, importance_values in feature_importances.items():
                if len(importance_values) >= 2:
                    importance_variance = np.var(importance_values)
                    importance_mean = np.mean(importance_values)
                    
                    # Calculate coefficient of variation
                    cv = np.std(importance_values) / importance_mean if importance_mean > 0 else float('inf')
                    
                    if isinstance(stability_analysis.get('importance_variance'), dict):
                        stability_analysis['importance_variance'][feature_name] = {
                        'variance': importance_variance,
                        'mean': importance_mean,
                        'cv': cv,
                        'values': importance_values
                    }
                    
                    # Determine stability
                    if cv <= stability_threshold:
                        if isinstance(stability_analysis.get('stable_features'), list):
                            stability_analysis['stable_features'].append(feature_name)
                    else:
                        if isinstance(stability_analysis.get('unstable_features'), list):
                            stability_analysis['unstable_features'].append(feature_name)
            
            # Calculate overall stability score
            total_features = len(stability_analysis.get('stable_features', [])) + len(stability_analysis.get('unstable_features', []))
            if total_features > 0:
                stability_analysis['stability_score'] = len(stability_analysis.get('stable_features', [])) / total_features
            
            self.logger.info(f"📊 Regime {current_regime} stability: {stability_analysis['stability_score']:.2f}")
            
            return stability_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing importance stability: {e}")
            return stability_analysis
    
    async def _detect_regime_switch(self, 
                                  current_regime: str,
                                  current_time: datetime) -> Dict[str, Any]:
        """Detect if market regime has switched."""
        
        switch_detection = {
            'switch_detected': False,
            'previous_regime': None,
            'switch_confidence': 0.0,
            'switch_reason': '',
            'time_since_last_switch': None
        }
        
        try:
            if not self.importance_history:
                return switch_detection
            
            # Get recent regime history
            recent_records = self.importance_history[-10:]  # Last 10 records
            
            if len(recent_records) < 2:
                return switch_detection
            
            # Check if regime has changed
            previous_regime = recent_records[-2]['regime']
            
            if previous_regime != current_regime:
                # Calculate switch confidence
                switch_detection['switch_detected'] = True
                switch_detection['previous_regime'] = previous_regime
                switch_detection['switch_confidence'] = 0.8  # High confidence for explicit switch
                switch_detection['switch_reason'] = f"Regime changed from {previous_regime} to {current_regime}"
                
                # Calculate time since last switch
                last_switch_time = None
                for switch_point in reversed(self.regime_switch_points):
                    if switch_point['timestamp'] < current_time:
                        last_switch_time = switch_point['timestamp']
                        break
                
                if last_switch_time:
                    time_since_last = current_time - last_switch_time
                    switch_detection['time_since_last_switch'] = time_since_last.total_seconds() / 3600  # hours
                
                self.logger.info(f"🔄 Regime switch detected: {previous_regime} -> {current_regime}")
            
            return switch_detection
            
        except Exception as e:
            self.logger.error(f"Error detecting regime switch: {e}")
            return switch_detection
    
    async def _calculate_importance_changes(self, 
                                       current_importance: Dict[str, float],
                                       current_regime: str) -> Dict[str, Any]:
        """Calculate changes in feature importance."""
        
        changes_analysis: Dict[str, Any] = {
            'significant_changes': [],
            'change_summary': {},
            'regime_comparison': {}
        }
        
        try:
            # Get last importance for this regime
            regime_importance = [
                record['importance'] for record in self.importance_history
                if record['regime'] == current_regime
            ]
            
            if not regime_importance:
                return changes_analysis
            
            # Get most recent importance for this regime
            last_importance = regime_importance[-1]
            
            # Calculate changes
            for feature_name, current_imp in current_importance.items():
                if feature_name in last_importance:
                    last_imp = last_importance[feature_name]
                    
                    # Calculate relative change
                    if last_imp != 0:
                        relative_change = abs(current_imp - last_imp) / last_imp
                    else:
                        relative_change = 1.0 if current_imp != 0 else 0.0
                    
                    # Determine significance
                    significance_threshold = self.thresholds['importance_stability']
                    
                    if isinstance(changes_analysis.get('change_summary'), dict):
                        changes_analysis['change_summary'][feature_name] = {
                        'last_importance': last_imp,
                        'current_importance': current_imp,
                        'absolute_change': abs(current_imp - last_imp),
                        'relative_change': relative_change,
                        'significant_change': relative_change > significance_threshold
                    }
                    
                    if relative_change > significance_threshold:
                        if isinstance(changes_analysis.get('significant_changes'), list):
                            changes_analysis['significant_changes'].append({
                            'feature': feature_name,
                            'change_type': 'importance_drift',
                            'relative_change': relative_change,
                            'last_value': last_imp,
                            'current_value': current_imp
                        })
            
            # Compare with other regimes
            other_regimes = set(record['regime'] for record in self.importance_history) - {current_regime}
            
            for other_regime in other_regimes:
                other_regime_importance = [
                    record['importance'] for record in self.importance_history
                    if record['regime'] == other_regime
                ]
                
                if other_regime_importance:
                    other_avg_importance: Dict[str, Any] = {}
                    
                    # Calculate average importance for other regime
                    for record in other_regime_importance:
                        for feature_name, importance in record['importance'].items():
                            if feature_name not in other_avg_importance:
                                other_avg_importance[feature_name] = []
                            other_avg_importance[feature_name].append(importance)
                    
                    # Average the importance values
                    for feature_name in other_avg_importance:
                        other_avg_importance[feature_name] = np.mean(other_avg_importance[feature_name])
                    
                    if isinstance(changes_analysis.get('regime_comparison'), dict):
                        changes_analysis['regime_comparison'][other_regime] = other_avg_importance
            
            return changes_analysis
            
        except Exception as e:
            self.logger.error(f"Error calculating importance changes: {e}")
            return changes_analysis
    
    async def _generate_adaptation_recommendations(self, 
                                               current_regime: str,
                                               stability_analysis: Dict[str, Any],
                                               importance_changes: Dict[str, Any]) -> List[str]:
        """Generate adaptation recommendations based on analysis."""
        
        recommendations = []
        
        try:
            # Stability-based recommendations
            stability_score = stability_analysis.get('stability_score', 1.0)
            
            if stability_score < 0.5:
                recommendations.append(
                    f"⚠️ Low importance stability ({stability_score:.2f}) in {current_regime} regime. "
                    "Consider increasing feature selection frequency."
                )
            
            # Unstable features recommendations
            unstable_features = stability_analysis.get('unstable_features', [])
            if unstable_features:
                recommendations.append(
                    f"🔄 {len(unstable_features)} features show unstable importance in {current_regime} regime. "
                    "Consider regime-specific feature selection."
                )
            
            # Significant changes recommendations
            significant_changes = importance_changes.get('significant_changes', [])
            if significant_changes:
                recommendations.append(
                    f"📊 {len(significant_changes)} features show significant importance changes. "
                    "Review feature engineering pipeline."
                )
            
            # Regime-specific recommendations
            regime_recommendations = self._get_regime_specific_recommendations(current_regime)
            recommendations.extend(regime_recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return recommendations
    
    def _get_regime_specific_recommendations(self, regime: str) -> List[str]:
        """Get regime-specific feature selection recommendations."""
        
        recommendations = []
        
        try:
            regime_config = self.REGIME_TYPES.get(regime, {})
            
            if regime == 'volatile':
                recommendations.extend([
                    "🌊 Volatile regime: Increase emphasis on volatility-based features",
                    "🌊 Use shorter lookback periods for technical indicators",
                    "🌊 Consider risk management features more heavily"
                ])
            elif regime == 'trending_up':
                recommendations.extend([
                    "📈 Uptrend regime: Emphasize momentum features",
                    "📈 Increase weight for trend-following indicators",
                    "📈 Consider breakout detection features"
                ])
            elif regime == 'trending_down':
                recommendations.extend([
                    "📉 Downtrend regime: Emphasize mean-reversion features",
                    "📉 Increase weight for contrarian indicators",
                    "📉 Consider short-selling signals"
                ])
            elif regime == 'crisis':
                recommendations.extend([
                    "🚨 Crisis regime: Emphasize safety features",
                    "🚨 Increase weight for defensive indicators",
                    "🚨 Consider market stress indicators"
                ])
            else:  # normal
                recommendations.extend([
                    "✅ Normal regime: Use balanced feature selection",
                    "✅ Maintain standard feature weights",
                    "✅ Regular model retraining schedule"
                ])
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting regime-specific recommendations: {e}")
            return recommendations
    
    async def _update_method_weights(self, 
                                  current_regime: str,
                                  recommendations: List[str]) -> Dict[str, float]:
        """Update feature selection method weights based on regime and recommendations."""
        
        # Base weights for different regimes
        base_weights = {
            'normal': {'correlation': 0.4, 'mutual_info': 0.3, 'lgbm': 0.2, 'rf': 0.1},
            'volatile': {'correlation': 0.1, 'mutual_info': 0.4, 'lgbm': 0.4, 'rf': 0.1},
            'trending_up': {'correlation': 0.3, 'mutual_info': 0.2, 'lgbm': 0.4, 'rf': 0.1},
            'trending_down': {'correlation': 0.3, 'mutual_info': 0.2, 'lgbm': 0.4, 'rf': 0.1},
            'crisis': {'correlation': 0.1, 'mutual_info': 0.5, 'lgbm': 0.3, 'rf': 0.1}
        }
        
        # Get base weights for current regime
        method_weights = base_weights.get(current_regime, base_weights['normal']).copy()
        
        # Adjust weights based on recommendations
        for recommendation in recommendations:
            if 'unstable importance' in recommendation:
                # Reduce correlation weight, increase model-based weights
                method_weights['correlation'] *= 0.8
                method_weights['lgbm'] *= 1.2
                method_weights['rf'] *= 1.1
            elif 'significant importance changes' in recommendation:
                # Increase mutual information weight for stability
                method_weights['mutual_info'] *= 1.3
            elif 'volatile regime' in recommendation:
                # Already handled in base weights
                pass
        
        # Normalize weights to sum to 1.0
        total_weight = sum(method_weights.values())
        if total_weight > 0:
            method_weights = {k: v / total_weight for k, v in method_weights.items()}
        
        return method_weights
    
    def _clean_old_data(self) -> None:
        """Clean old data to prevent memory issues."""
        
        try:
            # Keep only last 1000 importance records
            if len(self.importance_history) > 1000:
                self.importance_history = self.importance_history[-1000:]
            
            # Keep only last 100 regime switch points
            if len(self.regime_switch_points) > 100:
                self.regime_switch_points = self.regime_switch_points[-100:]
            
        except Exception as e:
            self.logger.error(f"Error cleaning old data: {e}")
    
    def _store_tracking_results(self, results: Dict[str, Any]) -> None:
        """Store tracking results for historical analysis."""
        
        try:
            # Store in JSON file
            timestamp = results['timestamp'].strftime('%Y%m%d_%H%M%S')
            filename = f"regime_tracking_{timestamp}.json"
            filepath = self.storage_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Keep only last 100 files
            files = list(self.storage_path.glob("regime_tracking_*.json"))
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for file_to_delete in files[100:]:
                file_to_delete.unlink()
                
        except Exception as e:
            self.logger.error(f"Failed to store tracking results: {e}")
    
    def get_regime_importance_summary(self, 
                                   days: int = 30) -> Dict[str, Any]:
        """Get summary of feature importance across regimes."""
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Filter recent records
        recent_records = [
            record for record in self.importance_history
            if record['timestamp'] >= cutoff_time
        ]
        
        if not recent_records:
            return {'error': 'No recent regime importance data available'}
        
        # Analyze by regime
        regime_analysis = {}
        for regime in self.REGIME_TYPES.keys():
            regime_records = [
                record for record in recent_records
                if record['regime'] == regime
            ]
            
            if regime_records:
                # Calculate average importance for this regime
                all_features = set()
                feature_importances: Dict[str, List[float]] = {}
                
                for record in regime_records:
                    for feature_name, importance in record['importance'].items():
                        all_features.add(feature_name)
                        if feature_name not in feature_importances:
                            feature_importances[feature_name] = []
                        feature_importances[feature_name].append(importance)
                
                # Calculate statistics
                regime_stats = {}
                for feature_name in all_features:
                    values = feature_importances[feature_name]
                    regime_stats[feature_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
                
                regime_analysis[regime] = {
                    'record_count': len(regime_records),
                    'feature_count': len(all_features),
                    'feature_importance': regime_stats
                }
        
        # Calculate regime transitions
        regime_transitions = []
        for i in range(1, len(recent_records)):
            if recent_records[i]['regime'] != recent_records[i-1]['regime']:
                regime_transitions.append({
                    'from_regime': recent_records[i-1]['regime'],
                    'to_regime': recent_records[i]['regime'],
                    'timestamp': recent_records[i]['timestamp']
                })
        
        summary = {
            'period_days': days,
            'total_records': len(recent_records),
            'regime_analysis': regime_analysis,
            'regime_transitions': regime_transitions,
            'most_common_regime': self._get_most_common_regime(recent_records),
            'regime_stability_scores': self._calculate_regime_stability_scores(recent_records)
        }
        
        return summary
    
    def _get_most_common_regime(self, records: List[Dict[str, Any]]) -> str:
        """Get the most common regime in the period."""
        
        regime_counts: Dict[str, int] = {}
        for record in records:
            regime = record['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        if regime_counts:
            return str(max(regime_counts.items(), key=lambda x: x[1])[0])
        return 'normal'
    
    def _calculate_regime_stability_scores(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate stability scores for each regime."""
        
        regime_stability = {}
        
        for regime in self.REGIME_TYPES.keys():
            regime_records = [
                record for record in records
                if record['regime'] == regime
            ]
            
            if len(regime_records) >= 2:
                # Calculate how long regime persists
                persistence_times = []
                current_persistence = 1
                
                for i in range(1, len(regime_records)):
                    if regime_records[i]['regime'] == regime:
                        current_persistence += 1
                    else:
                        persistence_times.append(current_persistence)
                        current_persistence = 1
                
                persistence_times.append(current_persistence)
                
                # Average persistence time
                avg_persistence = np.mean(persistence_times)
                regime_stability[regime] = avg_persistence
        
        return regime_stability


# Factory function for easy instantiation
def get_regime_importance_tracker(config: Optional[Dict[str, Any]] = None) -> RegimeImportanceTracker:
    """Factory function to get RegimeImportanceTracker instance."""
    return RegimeImportanceTracker(config)


# Convenience function for quick tracking
async def track_regime_importance_quick(current_importance: Dict[str, float],
                                       market_data: pd.DataFrame,
                                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Quick regime importance tracking.
    
    Args:
        current_importance: Current feature importance dictionary
        market_data: Market data for regime detection
        config: Configuration dictionary
        
    Returns:
        Regime tracking result dictionary
    """
    tracker = get_regime_importance_tracker(config)
    return await tracker.track_feature_importance(current_importance, market_data)
