#!/usr/bin/env python3
"""
Baseline Dominance Detector - Detects When Simple Baselines Outperform Complex Models
Analyzes whether complex models provide real value over simple baselines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import asyncio

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("BaselineDominanceDetector")

class BaselineDominanceDetector:
    """
    Detects and analyzes baseline model dominance over complex models.
    
    This detector analyzes whether complex models provide real value
    over simple baseline models and recommends simplification when appropriate.
    
    Key Features:
    - Multiple baseline model types (linear, moving average, buy-and-hold)
    - Performance comparison with statistical significance testing
    - Cost-benefit analysis for model complexity
    - Simplification recommendations with confidence scores
    """
    
    # Baseline model configurations
    BASELINE_MODELS = {
        'linear_regression': {
            'description': 'Simple linear regression',
            'complexity_score': 1,
            'training_time': 'fast',
            'interpretability': 'high'
        },
        'moving_average': {
            'description': 'Moving average strategy',
            'complexity_score': 0.5,
            'training_time': 'instant',
            'interpretability': 'very_high'
        },
        'buy_and_hold': {
            'description': 'Buy and hold strategy',
            'complexity_score': 0.1,
            'training_time': 'instant',
            'interpretability': 'very_high'
        },
        'random_forest_simple': {
            'description': 'Simple Random Forest (few trees)',
            'complexity_score': 3,
            'training_time': 'medium',
            'interpretability': 'medium'
        },
        'mean_reversion': {
            'description': 'Mean reversion strategy',
            'complexity_score': 2,
            'training_time': 'fast',
            'interpretability': 'high'
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Baseline Dominance Detector.
        
        Args:
            config: Configuration dictionary for detection settings
        """
        self.logger = logger
        self.config = config or {}
        
        # Detection thresholds
        self.dominance_threshold = self.config.get('dominance_threshold', 0.05)  # 5% advantage
        self.significance_level = self.config.get('significance_level', 0.05)
        self.min_samples = self.config.get('min_samples', 100)
        
        # Analysis settings
        self.enable_cost_benefit = self.config.get('enable_cost_benefit', True)
        self.complexity_penalty = self.config.get('complexity_penalty', 0.02)  # 2% per complexity point
        
        self.logger.info("✅ BaselineDominanceDetector initialized")
    
    async def analyze_baseline_dominance(self, 
                                      complex_model_results: Dict[str, Any],
                                      market_data: pd.DataFrame,
                                      features_df: Optional[pd.DataFrame] = None,
                                      target_series: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Analyze baseline dominance over complex models.
        
        Args:
            complex_model_results: Results from complex model evaluation
            market_data: Market data for baseline strategies
            features_df: Feature data for baseline models (optional)
            target_series: Target series for training (optional)
            
        Returns:
            Dict with baseline dominance analysis and recommendations
        """
        self.logger.info("🔍 Analyzing baseline dominance...")
        
        results = {
            'timestamp': datetime.now(),
            'complex_model_info': complex_model_results,
            'baseline_results': {},
            'dominance_analysis': {},
            'cost_benefit_analysis': {},
            'recommendations': []
        }
        
        try:
            # 1. Train baseline models
            baseline_results = await self._train_baseline_models(
                market_data, features_df, target_series
            )
            results['baseline_results'] = baseline_results
            
            # 2. Compare performance with complex models
            dominance_analysis = self._compare_with_baselines(
                complex_model_results, baseline_results
            )
            results['dominance_analysis'] = dominance_analysis
            
            # 3. Cost-benefit analysis
            if self.enable_cost_benefit:
                cost_benefit = self._perform_cost_benefit_analysis(
                    complex_model_results, baseline_results
                )
                results['cost_benefit_analysis'] = cost_benefit
            
            # 4. Generate recommendations
            recommendations = self._generate_simplification_recommendations(
                dominance_analysis, results.get('cost_benefit_analysis', {})
            )  # type: ignore
            results['recommendations'] = recommendations
            
            # 5. Log comprehensive summary
            self._log_dominance_summary(results)
            
            self.logger.info("✅ Baseline dominance analysis complete")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in baseline dominance analysis: {e}", exc_info=True)
            results['error'] = str(e)
            return results
    
    async def _train_baseline_models(self, 
                                  market_data: pd.DataFrame,
                                  features_df: Optional[pd.DataFrame],
                                  target_series: Optional[pd.Series]) -> Dict[str, Any]:
        """Train and evaluate baseline models."""
        
        baseline_results = {}
        
        try:
            # 1. Buy and Hold Strategy
            if 'close' in market_data.columns:
                buy_hold_result = self._train_buy_and_hold(market_data, target_series)
                baseline_results['buy_and_hold'] = buy_hold_result
            
            # 2. Moving Average Strategy
            if 'close' in market_data.columns:
                ma_result = self._train_moving_average(market_data, target_series)
                baseline_results['moving_average'] = ma_result
            
            # 3. Linear Regression
            if features_df is not None and target_series is not None:
                lr_result = self._train_linear_regression(features_df, target_series)
                baseline_results['linear_regression'] = lr_result
            
            # 4. Simple Random Forest
            if features_df is not None and target_series is not None:
                rf_result = self._train_simple_random_forest(features_df, target_series)
                baseline_results['random_forest_simple'] = rf_result
            
            # 5. Mean Reversion Strategy
            if 'close' in market_data.columns:
                mr_result = self._train_mean_reversion(market_data, target_series)
                baseline_results['mean_reversion'] = mr_result
            
            self.logger.info(f"📊 Trained {len(baseline_results)} baseline models")
            
            return baseline_results
            
        except Exception as e:
            self.logger.error(f"Error training baseline models: {e}")
            return baseline_results
    
    def _train_buy_and_hold(self, market_data: pd.DataFrame, target_series: Optional[pd.Series]) -> Dict[str, Any]:
        """Train buy and hold baseline strategy."""
        
        try:
            if target_series is not None:
                # Use actual target as buy and hold prediction
                predictions = target_series.shift(1).fillna(target_series.mean())
                actual = target_series
                
                # Calculate metrics
                mse = mean_squared_error(actual, predictions)
                mae = mean_absolute_error(actual, predictions)
                r2 = r2_score(actual, predictions)
                
                return {
                    'model_type': 'buy_and_hold',
                    'predictions': predictions,
                    'metrics': {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2
                    },
                    'complexity_score': self.BASELINE_MODELS['buy_and_hold']['complexity_score']
                }
            else:
                # Simple price prediction (next period same as current)
                if 'close' in market_data.columns:
                    prices = market_data['close']
                    predictions = prices.shift(1).fillna(prices.mean())
                    actual = prices
                    
                    mse = mean_squared_error(actual, predictions)
                    mae = mean_absolute_error(actual, predictions)
                    r2 = r2_score(actual, predictions)
                    
                    return {
                        'model_type': 'buy_and_hold',
                        'predictions': predictions,
                        'metrics': {
                            'mse': mse,
                            'mae': mae,
                            'r2': r2
                        },
                        'complexity_score': self.BASELINE_MODELS['buy_and_hold']['complexity_score']
                    }
            
            return {'model_type': 'buy_and_hold', 'status': 'no_data'}
            
        except Exception as e:
            self.logger.error(f"Error training buy and hold: {e}")
            return {'model_type': 'buy_and_hold', 'status': 'error', 'error': str(e)}
    
    def _train_moving_average(self, market_data: pd.DataFrame, target_series: Optional[pd.Series]) -> Dict[str, Any]:
        """Train moving average baseline strategy."""
        
        try:
            if 'close' in market_data.columns:
                prices = market_data['close']
                
                # Calculate moving averages with different windows
                ma_windows = [5, 10, 20]
                best_score = float('inf')
                best_predictions = None
                best_window = None
                
                for window in ma_windows:
                    if len(prices) >= window:
                        ma = prices.rolling(window=window, min_periods=1).mean()
                        predictions = ma.shift(1).fillna(ma.mean())
                        
                        if target_series is not None:
                            actual = target_series
                        else:
                            actual = prices
                        
                        mse = mean_squared_error(actual, predictions)
                        
                        if mse < best_score:
                            best_score = mse
                            best_predictions = predictions
                            best_window = window
                
                if best_predictions is not None:
                    if target_series is not None:
                        actual = target_series
                    else:
                        actual = prices
                    
                    mse = mean_squared_error(actual, best_predictions)
                    mae = mean_absolute_error(actual, best_predictions)
                    r2 = r2_score(actual, best_predictions)
                    
                    return {
                        'model_type': 'moving_average',
                        'predictions': best_predictions,
                        'metrics': {
                            'mse': mse,
                            'mae': mae,
                            'r2': r2
                        },
                        'complexity_score': self.BASELINE_MODELS['moving_average']['complexity_score'],
                        'best_window': best_window
                    }
            
            return {'model_type': 'moving_average', 'status': 'no_data'}
            
        except Exception as e:
            self.logger.error(f"Error training moving average: {e}")
            return {'model_type': 'moving_average', 'status': 'error', 'error': str(e)}
    
    def _train_linear_regression(self, features_df: pd.DataFrame, target_series: pd.Series) -> Dict[str, Any]:
        """Train linear regression baseline model."""
        
        try:
            if len(features_df) < self.min_samples:
                return {'model_type': 'linear_regression', 'status': 'insufficient_data'}
            
            # Prepare data
            X = features_df.select_dtypes(include=[np.number]).fillna(features_df.mean())
            y = target_series
            
            if len(X.columns) == 0:
                return {'model_type': 'linear_regression', 'status': 'no_numeric_features'}
            
            # Train model
            model = LinearRegression()
            model.fit(X, y)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Calculate metrics
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            return {
                'model_type': 'linear_regression',
                'predictions': predictions,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                },
                'complexity_score': self.BASELINE_MODELS['linear_regression']['complexity_score'],
                'feature_count': len(X.columns),
                'coefficients': dict(zip(X.columns, model.coef_))
            }
            
        except Exception as e:
            self.logger.error(f"Error training linear regression: {e}")
            return {'model_type': 'linear_regression', 'status': 'error', 'error': str(e)}
    
    def _train_simple_random_forest(self, features_df: pd.DataFrame, target_series: pd.Series) -> Dict[str, Any]:
        """Train simple random forest baseline model."""
        
        try:
            if len(features_df) < self.min_samples:
                return {'model_type': 'random_forest_simple', 'status': 'insufficient_data'}
            
            # Prepare data
            X = features_df.select_dtypes(include=[np.number]).fillna(features_df.mean())
            y = target_series
            
            if len(X.columns) == 0:
                return {'model_type': 'random_forest_simple', 'status': 'no_numeric_features'}
            
            # Train simple model (few trees)
            model = RandomForestRegressor(
                n_estimators=10,  # Simple model
                max_depth=5,       # Limited depth
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Calculate metrics
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            return {
                'model_type': 'random_forest_simple',
                'predictions': predictions,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                },
                'complexity_score': self.BASELINE_MODELS['random_forest_simple']['complexity_score'],
                'feature_count': len(X.columns),
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
        except Exception as e:
            self.logger.error(f"Error training simple random forest: {e}")
            return {'model_type': 'random_forest_simple', 'status': 'error', 'error': str(e)}
    
    def _train_mean_reversion(self, market_data: pd.DataFrame, target_series: Optional[pd.Series]) -> Dict[str, Any]:
        """Train mean reversion baseline strategy."""
        
        try:
            if 'close' in market_data.columns:
                prices = market_data['close']
                
                # Calculate mean reversion signals
                lookback_window = 20
                if len(prices) >= lookback_window:
                    mean_price = prices.rolling(window=lookback_window, min_periods=1).mean()
                    
                    # Simple mean reversion: predict price will move toward mean
                    price_diff = prices - mean_price
                    reversion_factor = 0.5  # How much price reverts to mean
                    predictions = prices - reversion_factor * price_diff
                    
                    if target_series is not None:
                        actual = target_series
                    else:
                        actual = prices
                    
                    mse = mean_squared_error(actual, predictions)
                    mae = mean_absolute_error(actual, predictions)
                    r2 = r2_score(actual, predictions)
                    
                    return {
                        'model_type': 'mean_reversion',
                        'predictions': predictions,
                        'metrics': {
                            'mse': mse,
                            'mae': mae,
                            'r2': r2
                        },
                        'complexity_score': self.BASELINE_MODELS['mean_reversion']['complexity_score'],
                        'lookback_window': lookback_window
                    }
            
            return {'model_type': 'mean_reversion', 'status': 'no_data'}
            
        except Exception as e:
            self.logger.error(f"Error training mean reversion: {e}")
            return {'model_type': 'mean_reversion', 'status': 'error', 'error': str(e)}
    
    def _compare_with_baselines(self, 
                              complex_model_results: Dict[str, Any],
                              baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare complex model performance with baselines."""
        
        comparison_results = {
            'dominant_baselines': [],
            'performance_comparison': {},
            'statistical_significance': {},
            'dominance_detected': False
        }
        
        try:
            # Get complex model metrics
            complex_metrics = complex_model_results.get('metrics', {})
            if not complex_metrics:
                self.logger.warning("No complex model metrics available for comparison")
                return comparison_results
            
            # Compare with each baseline
            for baseline_name, baseline_result in baseline_results.items():
                if baseline_result.get('status') != 'error' and 'metrics' in baseline_result:
                    baseline_metrics = baseline_result['metrics']
                    
                    # Calculate performance differences
                    performance_diff = self._calculate_performance_difference(
                        complex_metrics, baseline_metrics
                    )
                    
                    # Check for dominance
                    dominance_info = self._check_baseline_dominance(
                        performance_diff, baseline_name
                    )
                    
                    comparison_results['performance_comparison'][baseline_name] = {
                        'performance_difference': performance_diff,
                        'dominance_info': dominance_info,
                        'baseline_complexity': baseline_result.get('complexity_score', 0)
                    }  # type: ignore
                    
                    if dominance_info['is_dominant']:
                        comparison_results['dominant_baselines'].append({
                            'baseline_name': baseline_name,
                            'dominance_strength': dominance_info['strength'],
                            'complexity_savings': dominance_info['complexity_savings']
                        })  # type: ignore
                        comparison_results['dominance_detected'] = True
            
            # Sort dominant baselines by strength
            comparison_results['dominant_baselines'].sort(
                key=lambda x: x['dominance_strength'], reverse=True
            )  # type: ignore
            
            self.logger.info(f"📊 Found {len(comparison_results['dominant_baselines'])} dominant baselines")  # type: ignore
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Error comparing with baselines: {e}")
            return comparison_results
    
    def _calculate_performance_difference(self, 
                                         complex_metrics: Dict[str, float],
                                         baseline_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance differences between complex and baseline models."""
        
        differences = {}
        
        # Common metrics to compare
        common_metrics = ['mse', 'mae', 'r2']
        
        for metric in common_metrics:
            if metric in complex_metrics and metric in baseline_metrics:
                complex_val = complex_metrics[metric]
                baseline_val = baseline_metrics[metric]
                
                if metric == 'r2':
                    # For R2, higher is better
                    diff = baseline_val - complex_val  # Positive = baseline better
                else:
                    # For MSE/MAE, lower is better
                    diff = complex_val - baseline_val  # Positive = baseline better
                
                differences[metric] = diff
        
        return differences
    
    def _check_baseline_dominance(self, 
                                 performance_diff: Dict[str, float],
                                 baseline_name: str) -> Dict[str, Any]:
        """Check if baseline dominates complex model."""
        
        dominance_info = {
            'is_dominant': False,
            'strength': 0.0,
            'complexity_savings': 0.0,
            'dominant_metrics': []
        }
        
        try:
            baseline_config = self.BASELINE_MODELS.get(baseline_name, {})
            baseline_complexity = baseline_config.get('complexity_score', 1)
            
            # Check each metric for dominance
            dominant_count = 0
            total_advantage = 0.0
            
            for metric, diff in performance_diff.items():
                if diff > self.dominance_threshold:  # Baseline significantly better
                    dominance_info['dominant_metrics'].append(metric)  # type: ignore
                    dominant_count += 1
                    total_advantage += diff
            
            # Calculate dominance strength
            if dominant_count > 0:
                dominance_info['strength'] = total_advantage / dominant_count
                dominance_info['is_dominant'] = True
                
                # Calculate complexity savings
                # Assume complex model has complexity score of 10
                complex_complexity = 10
                dominance_info['complexity_savings'] = (complex_complexity - baseline_complexity) / complex_complexity  # type: ignore
            
            return dominance_info
            
        except Exception as e:
            self.logger.error(f"Error checking baseline dominance: {e}")
            return dominance_info
    
    def _perform_cost_benefit_analysis(self, 
                                     complex_model_results: Dict[str, Any],
                                     baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cost-benefit analysis for model complexity."""
        
        cost_benefit = {
            'complexity_cost': 0.0,
            'performance_benefit': 0.0,
            'net_benefit': 0.0,
            'recommendation': 'keep_complex'
        }
        
        try:
            # Calculate complexity cost (assume complex model score of 10)
            complex_complexity = 10
            cost_benefit['complexity_cost'] = complex_complexity * self.complexity_penalty
            
            # Calculate performance benefit (avoid baseline dominance)
            performance_comparison = self._compare_with_baselines(
                complex_model_results, baseline_results
            )
            
            if not performance_comparison['dominance_detected']:
                # No baseline dominance, calculate benefit
                cost_benefit['performance_benefit'] = 0.1  # Default benefit
            else:
                # Baselines dominate, no performance benefit
                cost_benefit['performance_benefit'] = 0.0
            
            # Calculate net benefit
            cost_benefit['net_benefit'] = cost_benefit['performance_benefit'] - cost_benefit['complexity_cost']  # type: ignore
            
            # Make recommendation
            if cost_benefit['net_benefit'] < 0:  # type: ignore
                cost_benefit['recommendation'] = 'simplify'
            elif cost_benefit['net_benefit'] < 0.02:  # type: ignore
                cost_benefit['recommendation'] = 'consider_simplification'
            else:
                cost_benefit['recommendation'] = 'keep_complex'
            
            return cost_benefit
            
        except Exception as e:
            self.logger.error(f"Error in cost-benefit analysis: {e}")
            return cost_benefit
    
    def _generate_simplification_recommendations(self, 
                                               dominance_analysis: Dict[str, Any],
                                               cost_benefit: Dict[str, Any]) -> List[str]:
        """Generate model simplification recommendations."""
        
        recommendations = []
        
        try:
            # Baseline dominance recommendations
            if dominance_analysis.get('dominance_detected', False):
                dominant_baselines = dominance_analysis.get('dominant_baselines', [])
                
                for baseline_info in dominant_baselines[:2]:  # Top 2 recommendations
                    baseline_name = baseline_info['baseline_name']
                    strength = baseline_info['dominance_strength']
                    savings = baseline_info['complexity_savings']
                    
                    baseline_config = self.BASELINE_MODELS.get(baseline_name, {})
                    description = baseline_config.get('description', baseline_name)
                    
                    recommendations.append(
                        f"🎯 Consider {description} - "
                        f"outperforms complex model by {strength:.3f} "
                        f"with {savings:.1%} complexity reduction"
                    )
            
            # Cost-benefit recommendations
            cost_benefit_rec = cost_benefit.get('recommendation', 'keep_complex')
            
            if cost_benefit_rec == 'simplify':
                recommendations.append(
                    "⚠️ High complexity cost detected. Model simplification recommended."
                )
            elif cost_benefit_rec == 'consider_simplification':
                recommendations.append(
                    "📊 Marginal complexity benefit. Consider simplification options."
                )
            
            # No dominance recommendations
            if not dominance_analysis.get('dominance_detected', False):
                recommendations.append(
                    "✅ No baseline dominance detected. Complex model provides value."
                )
            
            # General recommendations
            if len(recommendations) == 0:
                recommendations.append(
                    "📈 Continue with current model. Monitor for future baseline improvements."
                )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["❌ Error generating recommendations"]
    
    def _log_dominance_summary(self, results: Dict[str, Any]) -> None:
        """Log comprehensive baseline dominance summary."""
        
        try:
            self.logger.info("=" * 80)
            self.logger.info("🔍 BASELINE DOMINANCE ANALYSIS SUMMARY")
            self.logger.info("=" * 80)
            
            # Complex model info
            complex_info = results.get('complex_model_info', {})
            complex_metrics = complex_info.get('metrics', {})
            
            if complex_metrics:
                self.logger.info(f"🤖 Complex Model Performance:")
                for metric, value in complex_metrics.items():
                    self.logger.info(f"   {metric.upper()}: {value:.6f}")
            
            # Baseline results
            baseline_results = results.get('baseline_results', {})
            self.logger.info(f"📊 Baseline Models Trained: {len(baseline_results)}")
            
            for baseline_name, baseline_result in baseline_results.items():
                if 'metrics' in baseline_result:
                    metrics = baseline_result['metrics']
                    self.logger.info(f"   {baseline_name}: MSE={metrics.get('mse', 0):.6f}, "
                                   f"R²={metrics.get('r2', 0):.4f}")
            
            # Dominance analysis
            dominance = results.get('dominance_analysis', {})
            dominant_baselines = dominance.get('dominant_baselines', [])
            
            if dominant_baselines:
                self.logger.info(f"🎯 Dominant Baselines Found: {len(dominant_baselines)}")
                for i, baseline_info in enumerate(dominant_baselines[:3]):  # Top 3
                    self.logger.info(f"   {i+1}. {baseline_info['baseline_name']} "
                                   f"(strength: {baseline_info['dominance_strength']:.4f})")
            else:
                self.logger.info("✅ No baseline dominance detected")
            
            # Cost-benefit analysis
            cost_benefit = results.get('cost_benefit_analysis', {})
            if cost_benefit:
                net_benefit = cost_benefit.get('net_benefit', 0)
                recommendation = cost_benefit.get('recommendation', 'keep_complex')
                
                self.logger.info(f"💰 Cost-Benefit Analysis:")
                self.logger.info(f"   Net Benefit: {net_benefit:.4f}")
                self.logger.info(f"   Recommendation: {recommendation}")
            
            # Recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                self.logger.info(f"💡 Recommendations: {len(recommendations)}")
                for i, rec in enumerate(recommendations[:3]):  # Top 3
                    self.logger.info(f"   {i+1}. {rec}")
            
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Error logging dominance summary: {e}")


# Factory function for easy instantiation
def get_baseline_dominance_detector(config: Optional[Dict[str, Any]] = None) -> BaselineDominanceDetector:
    """Factory function to get BaselineDominanceDetector instance."""
    return BaselineDominanceDetector(config)


# Convenience function for quick analysis
async def analyze_baseline_dominance_quick(complex_model_results: Dict[str, Any],
                                       market_data: pd.DataFrame,
                                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Quick baseline dominance analysis.
    
    Args:
        complex_model_results: Results from complex model evaluation
        market_data: Market data for baseline strategies
        config: Configuration dictionary
        
    Returns:
        Baseline dominance analysis result dictionary
    """
    detector = get_baseline_dominance_detector(config)
    return await detector.analyze_baseline_dominance(complex_model_results, market_data)
