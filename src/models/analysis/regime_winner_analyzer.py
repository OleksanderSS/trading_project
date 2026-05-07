#!/usr/bin/env python3
"""
Regime Winner Analyzer - Analyzes Model Winner Consistency Across Market Regimes
Tracks and analyzes model performance patterns across different market regimes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json
from pathlib import Path

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("RegimeWinnerAnalyzer")

class RegimeWinnerAnalyzer:
    """
    Analyzes model winner consistency across market regimes and adapts selection strategies.
    
    This analyzer tracks:
    - Model winner consistency across different market regimes
    - Regime-specific performance patterns
    - Automatic regime switching detection
    - Dynamic model selection recommendations
    
    Critical for maintaining model performance across changing market conditions.
    """
    
    # Market regime definitions
    REGIME_TYPES = {
        'normal': {
            'description': 'Normal market conditions',
            'volatility_range': (0.01, 0.02),
            'trend_strength': (-0.001, 0.001),
            'typical_winners': ['ensemble', 'lgbm', 'rf']
        },
        'volatile': {
            'description': 'High volatility market',
            'volatility_range': (0.02, 0.05),
            'trend_strength': (-0.003, 0.003),
            'typical_winners': ['rf', 'ensemble', 'svm']
        },
        'trending_up': {
            'description': 'Strong uptrend market',
            'volatility_range': (0.015, 0.025),
            'trend_strength': (0.002, 0.005),
            'typical_winners': ['lgbm', 'xgboost', 'linear']
        },
        'trending_down': {
            'description': 'Strong downtrend market',
            'volatility_range': (0.015, 0.025),
            'trend_strength': (-0.005, -0.002),
            'typical_winners': ['rf', 'svm', 'ensemble']
        },
        'crisis': {
            'description': 'Market crisis conditions',
            'volatility_range': (0.04, 0.10),
            'trend_strength': (-0.01, 0.01),
            'typical_winners': ['rf', 'svm', 'conservative']
        }
    }
    
    # Consistency thresholds
    CONSISTENCY_THRESHOLDS = {
        'min_samples_per_regime': 30,        # Minimum samples for reliable analysis
        'consistency_threshold': 0.6,        # 60% consistency required
        'switch_detection_window': 5,         # Window for regime switching detection
        'performance_gap_threshold': 0.05,    # 5% performance gap for winner detection
        'stability_window_days': 30           # Window for stability analysis
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Regime Winner Analyzer.
        
        Args:
            config: Configuration dictionary for regime analysis
        """
        self.logger = logger
        self.config = config or {}
        
        # Override thresholds with config
        self.thresholds = self.CONSISTENCY_THRESHOLDS.copy()
        self.thresholds.update(self.config.get('thresholds', {}))
        
        # Data storage
        self.regime_performance_history = []
        self.winner_consistency_cache = {}
        self.regime_switch_points = []
        
        # Analysis settings
        self.min_samples_per_regime = self.thresholds['min_samples_per_regime']
        self.consistency_threshold = self.thresholds['consistency_threshold']
        
        # Storage paths
        self.storage_path = Path(self.config.get('storage_path', 'data/analysis/regime_winners'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ RegimeWinnerAnalyzer initialized")
    
    async def analyze_regime_consistency(self, 
                                     model_results: Dict[str, Any],
                                     market_data: pd.DataFrame,
                                     current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Analyze model winner consistency across market regimes.
        
        Args:
            model_results: Dictionary with model performance results
            market_data: Market data for regime detection
            current_time: Current timestamp (uses now if None)
            
        Returns:
            Dict with regime consistency analysis and recommendations
        """
        if current_time is None:
            current_time = datetime.now()
        
        self.logger.info(f"📊 Analyzing regime consistency at {current_time}")
        
        results = {
            'timestamp': current_time,
            'current_regime': None,
            'regime_performance': {},
            'consistency_analysis': {},
            'winner_patterns': {},
            'switching_analysis': {},
            'recommendations': []
        }
        
        try:
            # 1. Detect current market regime
            current_regime = await self._detect_market_regime(market_data, current_time)
            results['current_regime'] = current_regime
            
            # 2. Analyze winners by regime
            regime_winners = await self._analyze_winners_by_regime(
                model_results, market_data, current_regime
            )
            results['regime_performance'] = regime_winners
            
            # 3. Calculate consistency metrics
            consistency_metrics = await self._calculate_consistency_metrics(
                regime_winners, current_regime
            )
            results['consistency_analysis'] = consistency_metrics
            
            # 4. Analyze winner patterns
            winner_patterns = await self._analyze_winner_patterns(
                regime_winners, current_regime
            )
            results['winner_patterns'] = winner_patterns
            
            # 5. Detect regime switching
            switching_analysis = await self._detect_regime_switching(
                current_regime, current_time
            )
            results['switching_analysis'] = switching_analysis
            
            # 6. Generate recommendations
            recommendations = await self._generate_regime_recommendations(
                current_regime, consistency_metrics, winner_patterns
            )
            results['recommendations'] = recommendations
            
            # 7. Store results
            self._store_analysis_results(results)
            
            self.logger.info(f"✅ Regime consistency analysis complete. Regime: {current_regime}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in regime consistency analysis: {e}", exc_info=True)
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
                
                if (vol_range[0] <= volatility <= vol_range[1] and
                    trend_range[0] <= trend <= trend_range[1]):
                    
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
                return returns.std() * np.sqrt(252)  # Annualized volatility
            else:
                # Use any price column
                price_cols = [col for col in market_data.columns if 'price' in col.lower() or col in ['open', 'high', 'low', 'close']]
                if price_cols:
                    returns = market_data[price_cols[0]].pct_change().dropna()
                    return returns.std() * np.sqrt(252)
                return 0.02  # Default volatility
        except:
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
                    return normalized_trend
            return 0.0
        except:
            return 0.0
    
    async def _analyze_winners_by_regime(self, 
                                     model_results: Dict[str, Any],
                                     market_data: pd.DataFrame,
                                     current_regime: str) -> Dict[str, Any]:
        """Analyze model winners by regime."""
        
        regime_analysis = {
            'current_regime': current_regime,
            'model_performance': {},
            'winner_ranking': [],
            'regime_specific_metrics': {}
        }
        
        try:
            # Extract model performance data
            model_performance = {}
            
            # Handle different result formats
            if isinstance(model_results, dict):
                for model_name, result in model_results.items():
                    if isinstance(result, dict) and 'metrics' in result:
                        metrics = result['metrics']
                        
                        # Extract common metrics
                        performance_score = self._calculate_performance_score(metrics)
                        model_performance[model_name] = {
                            'score': performance_score,
                            'metrics': metrics,
                            'model_type': result.get('model_type', 'unknown')
                        }
            
            # Rank models by performance
            ranked_models = sorted(
                model_performance.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )
            
            regime_analysis['model_performance'] = model_performance
            regime_analysis['winner_ranking'] = [
                {
                    'model_name': model_name,
                    'score': perf_data['score'],
                    'model_type': perf_data['model_type']
                }
                for model_name, perf_data in ranked_models
            ]
            
            # Calculate regime-specific metrics
            if ranked_models:
                winner = ranked_models[0][0]
                winner_score = ranked_models[0][1]['score']
                
                regime_analysis['regime_specific_metrics'] = {
                    'best_model': winner,
                    'best_score': winner_score,
                    'total_models': len(ranked_models),
                    'score_gap': self._calculate_score_gap(ranked_models),
                    'expected_winner': self.REGIME_TYPES[current_regime]['typical_winners'][0]
                }
            
            # Store in history
            self.regime_performance_history.append({
                'timestamp': datetime.now(),
                'regime': current_regime,
                'performance': model_performance,
                'winner_ranking': regime_analysis['winner_ranking']
            })
            
            # Clean old data
            if len(self.regime_performance_history) > 1000:
                self.regime_performance_history = self.regime_performance_history[-1000:]
            
            return regime_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing winners by regime: {e}")
            return regime_analysis
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate unified performance score from metrics."""
        
        try:
            # Common metrics and their weights
            metric_weights = {
                'accuracy': 0.3,
                'precision': 0.2,
                'recall': 0.2,
                'f1': 0.2,
                'r2': 0.1,
                'mse': -0.1,  # Negative for error metrics
                'mae': -0.1,   # Negative for error metrics
                'rmse': -0.1   # Negative for error metrics
            }
            
            score = 0.0
            total_weight = 0.0
            
            for metric_name, value in metrics.items():
                if metric_name in metric_weights:
                    weight = metric_weights[metric_name]
                    
                    # For error metrics, lower is better, so invert
                    if metric_name in ['mse', 'mae', 'rmse']:
                        if value > 0:
                            normalized_value = 1.0 / (1.0 + value)
                        else:
                            normalized_value = 1.0
                    else:
                        # For performance metrics, higher is better
                        normalized_value = max(0.0, min(1.0, value))
                    
                    score += weight * normalized_value
                    total_weight += abs(weight)
            
            # Normalize score
            if total_weight > 0:
                score = score / total_weight
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating performance score: {e}")
            return 0.0
    
    def _calculate_score_gap(self, ranked_models: List[Tuple[str, Dict[str, Any]]]) -> float:
        """Calculate performance gap between winner and runner-up."""
        
        try:
            if len(ranked_models) >= 2:
                winner_score = ranked_models[0][1]['score']
                runner_up_score = ranked_models[1][1]['score']
                return winner_score - runner_up_score
            return 0.0
        except:
            return 0.0
    
    async def _calculate_consistency_metrics(self, 
                                         regime_winners: Dict[str, Any],
                                         current_regime: str) -> Dict[str, Any]:
        """Calculate consistency metrics across regimes."""
        
        consistency_metrics = {
            'current_regime': current_regime,
            'regime_consistency': {},
            'overall_consistency': 0.0,
            'stable_models': [],
            'inconsistent_models': []
        }
        
        try:
            # Group performance history by regime
            regime_history = defaultdict(list)
            
            for record in self.regime_performance_history:
                regime = record['regime']
                performance = record['performance']
                
                for model_name, perf_data in performance.items():
                    regime_history[regime].append({
                        'model_name': model_name,
                        'score': perf_data['score'],
                        'timestamp': record['timestamp']
                    })
            
            # Calculate consistency for each regime
            for regime, history in regime_history.items():
                if len(history) >= self.min_samples_per_regime:
                    # Calculate model consistency in this regime
                    model_consistency = self._calculate_model_consistency(history)
                    consistency_metrics['regime_consistency'][regime] = model_consistency
                    
                    # Identify stable and inconsistent models
                    for model_name, consistency_score in model_consistency.items():
                        if consistency_score >= self.consistency_threshold:
                            consistency_metrics['stable_models'].append({
                                'model_name': model_name,
                                'regime': regime,
                                'consistency': consistency_score
                            })
                        else:
                            consistency_metrics['inconsistent_models'].append({
                                'model_name': model_name,
                                'regime': regime,
                                'consistency': consistency_score
                            })
            
            # Calculate overall consistency
            if consistency_metrics['regime_consistency']:
                regime_consistencies = []
                
                for regime_data in consistency_metrics['regime_consistency'].values():
                    if regime_data:
                        avg_consistency = np.mean(list(regime_data.values()))
                        regime_consistencies.append(avg_consistency)
                
                if regime_consistencies:
                    consistency_metrics['overall_consistency'] = np.mean(regime_consistencies)
            
            return consistency_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating consistency metrics: {e}")
            return consistency_metrics
    
    def _calculate_model_consistency(self, regime_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate consistency scores for models in a regime."""
        
        model_consistency = {}
        
        try:
            # Group by model
            model_scores = defaultdict(list)
            
            for record in regime_history:
                model_name = record['model_name']
                score = record['score']
                model_scores[model_name].append(score)
            
            # Calculate consistency for each model
            for model_name, scores in model_scores.items():
                if len(scores) >= 3:  # Need minimum samples
                    # Calculate coefficient of variation (lower is more consistent)
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    
                    if mean_score > 0:
                        cv = std_score / mean_score
                        # Convert to consistency score (higher is more consistent)
                        consistency_score = 1.0 / (1.0 + cv)
                    else:
                        consistency_score = 0.0
                    
                    model_consistency[model_name] = consistency_score
            
            return model_consistency
            
        except Exception as e:
            self.logger.error(f"Error calculating model consistency: {e}")
            return model_consistency
    
    async def _analyze_winner_patterns(self, 
                                     regime_winners: Dict[str, Any],
                                     current_regime: str) -> Dict[str, Any]:
        """Analyze winner patterns across regimes."""
        
        winner_patterns = {
            'current_regime': current_regime,
            'expected_winners': {},
            'actual_winners': {},
            'pattern_deviations': [],
            'regime_specific_insights': {}
        }
        
        try:
            # Get expected winners for current regime
            expected_winners = self.REGIME_TYPES[current_regime]['typical_winners']
            winner_patterns['expected_winners'] = expected_winners
            
            # Get actual winners from ranking
            actual_ranking = regime_winners.get('winner_ranking', [])
            actual_winners = [item['model_name'] for item in actual_ranking[:3]]  # Top 3
            winner_patterns['actual_winners'] = actual_winners
            
            # Detect pattern deviations
            for i, expected_winner in enumerate(expected_winners):
                if i < len(actual_winners):
                    actual_winner = actual_winners[i]
                    
                    if actual_winner != expected_winner:
                        winner_patterns['pattern_deviations'].append({
                            'position': i + 1,
                            'expected': expected_winner,
                            'actual': actual_winner,
                            'severity': self._calculate_deviation_severity(expected_winner, actual_winner, i)
                        })
            
            # Generate regime-specific insights
            insights = self._generate_regime_insights(
                current_regime, expected_winners, actual_winners
            )
            winner_patterns['regime_specific_insights'] = insights
            
            return winner_patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing winner patterns: {e}")
            return winner_patterns
    
    def _calculate_deviation_severity(self, expected: str, actual: str, position: int) -> str:
        """Calculate severity of pattern deviation."""
        
        try:
            # Higher position deviations are more severe
            position_weight = position / 3.0  # Normalize by max position
            
            # Check if actual is in expected list (but wrong position)
            expected_winners = self.REGIME_TYPES.get('normal', {}).get('typical_winners', [])
            
            if actual in expected_winners:
                return 'low' if position_weight < 0.5 else 'medium'
            else:
                return 'high' if position_weight < 0.5 else 'critical'
                
        except:
            return 'medium'
    
    def _generate_regime_insights(self, 
                                 regime: str,
                                 expected_winners: List[str],
                                 actual_winners: List[str]) -> Dict[str, Any]:
        """Generate regime-specific insights."""
        
        insights = {
            'regime_characteristics': self.REGIME_TYPES[regime]['description'],
            'alignment_score': 0.0,
            'recommendations': []
        }
        
        try:
            # Calculate alignment score
            alignment_count = 0
            for i, expected in enumerate(expected_winners):
                if i < len(actual_winners) and actual_winners[i] == expected:
                    alignment_count += 1
            
            insights['alignment_score'] = alignment_count / len(expected_winners)
            
            # Generate recommendations based on alignment
            if insights['alignment_score'] >= 0.8:
                insights['recommendations'].append(
                    f"✅ Excellent alignment with {regime} regime expectations"
                )
            elif insights['alignment_score'] >= 0.5:
                insights['recommendations'].append(
                    f"⚠️ Moderate alignment with {regime} regime. Monitor for consistency"
                )
            else:
                insights['recommendations'].append(
                    f"🚨 Poor alignment with {regime} regime. Consider model selection review"
                )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating regime insights: {e}")
            return insights
    
    async def _detect_regime_switching(self, 
                                    current_regime: str,
                                    current_time: datetime) -> Dict[str, Any]:
        """Detect regime switching patterns."""
        
        switching_analysis = {
            'current_regime': current_regime,
            'switch_detected': False,
            'switch_frequency': 0.0,
            'recent_switches': [],
            'stability_analysis': {}
        }
        
        try:
            # Get recent regime history
            recent_history = [
                record for record in self.regime_performance_history
                if current_time - record['timestamp'] <= timedelta(days=self.thresholds['stability_window_days'])
            ]
            
            if len(recent_history) < 2:
                return switching_analysis
            
            # Detect regime switches
            switches = []
            for i in range(1, len(recent_history)):
                if recent_history[i]['regime'] != recent_history[i-1]['regime']:
                    switches.append({
                        'from_regime': recent_history[i-1]['regime'],
                        'to_regime': recent_history[i]['regime'],
                        'timestamp': recent_history[i]['timestamp']
                    })
            
            switching_analysis['recent_switches'] = switches
            
            # Calculate switch frequency
            if len(recent_history) > 0:
                switching_analysis['switch_frequency'] = len(switches) / len(recent_history)
                switching_analysis['switch_detected'] = len(switches) > 0
            
            # Stability analysis
            if switches:
                switching_analysis['stability_analysis'] = {
                    'most_frequent_switch': self._get_most_frequent_switch(switches),
                    'average_stable_period': self._calculate_average_stable_period(recent_history, switches),
                    'volatility_correlation': self._calculate_volatility_correlation(recent_history)
                }
            
            return switching_analysis
            
        except Exception as e:
            self.logger.error(f"Error detecting regime switching: {e}")
            return switching_analysis
    
    def _get_most_frequent_switch(self, switches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the most frequent regime switch."""
        
        try:
            switch_counts = {}
            
            for switch in switches:
                switch_key = f"{switch['from_regime']}->{switch['to_regime']}"
                switch_counts[switch_key] = switch_counts.get(switch_key, 0) + 1
            
            if switch_counts:
                most_frequent = max(switch_counts.items(), key=lambda x: x[1])
                
                from_regime, to_regime = most_frequent[0].split('->')
                
                return {
                    'from_regime': from_regime,
                    'to_regime': to_regime,
                    'count': most_frequent[1]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting most frequent switch: {e}")
            return None
    
    def _calculate_average_stable_period(self, 
                                        history: List[Dict[str, Any]], 
                                        switches: List[Dict[str, Any]]) -> float:
        """Calculate average stable period between switches."""
        
        try:
            if len(switches) < 2:
                return float('inf')
            
            # Calculate periods between switches
            stable_periods = []
            
            for i in range(1, len(switches)):
                prev_switch_time = switches[i-1]['timestamp']
                curr_switch_time = switches[i]['timestamp']
                
                stable_period = (curr_switch_time - prev_switch_time).total_seconds() / 3600  # hours
                stable_periods.append(stable_period)
            
            if stable_periods:
                return np.mean(stable_periods)
            
            return float('inf')
            
        except Exception as e:
            self.logger.error(f"Error calculating average stable period: {e}")
            return float('inf')
    
    def _calculate_volatility_correlation(self, history: List[Dict[str, Any]]) -> float:
        """Calculate correlation between regime changes and volatility."""
        
        try:
            # This would require market data correlation analysis
            # For now, return a placeholder
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility correlation: {e}")
            return 0.0
    
    async def _generate_regime_recommendations(self, 
                                           current_regime: str,
                                           consistency_metrics: Dict[str, Any],
                                           winner_patterns: Dict[str, Any]) -> List[str]:
        """Generate regime-specific recommendations."""
        
        recommendations = []
        
        try:
            # Consistency-based recommendations
            overall_consistency = consistency_metrics.get('overall_consistency', 0.0)
            
            if overall_consistency < 0.5:
                recommendations.append(
                    f"⚠️ Low model consistency ({overall_consistency:.2f}). "
                    "Consider ensemble methods for stability."
                )
            elif overall_consistency < 0.7:
                recommendations.append(
                    f"📊 Moderate model consistency ({overall_consistency:.2f}). "
                    "Monitor for performance degradation."
                )
            else:
                recommendations.append(
                    f"✅ High model consistency ({overall_consistency:.2f}). "
                    "Current model selection strategy is effective."
                )
            
            # Pattern-based recommendations
            pattern_deviations = winner_patterns.get('pattern_deviations', [])
            
            if len(pattern_deviations) > 2:
                recommendations.append(
                    f"🚨 High pattern deviations ({len(pattern_deviations)}). "
                    "Review model selection for current regime."
                )
            
            # Regime-specific recommendations
            regime_insights = winner_patterns.get('regime_specific_insights', {})
            regime_recommendations = regime_insights.get('recommendations', [])
            recommendations.extend(regime_recommendations)
            
            # Switching-based recommendations
            switching_analysis = self.regime_performance_history[-5:] if self.regime_performance_history else []
            recent_switches = sum(1 for i in range(1, len(switching_analysis)) 
                                if switching_analysis[i]['regime'] != switching_analysis[i-1]['regime'])
            
            if recent_switches > 3:
                recommendations.append(
                    f"🔄 High regime switching detected ({recent_switches} switches). "
                    "Consider adaptive model selection."
                )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return recommendations
    
    def _store_analysis_results(self, results: Dict[str, Any]) -> None:
        """Store analysis results for historical tracking."""
        
        try:
            # Store in JSON file
            timestamp = results['timestamp'].strftime('%Y%m%d_%H%M%S')
            filename = f"regime_analysis_{timestamp}.json"
            filepath = self.storage_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Keep only last 100 files
            files = list(self.storage_path.glob("regime_analysis_*.json"))
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for file_to_delete in files[100:]:
                file_to_delete.unlink()
                
        except Exception as e:
            self.logger.error(f"Failed to store analysis results: {e}")
    
    def get_regime_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of regime analysis over time period."""
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Filter recent records
        recent_records = [
            record for record in self.regime_performance_history
            if record['timestamp'] >= cutoff_time
        ]
        
        if not recent_records:
            return {'error': 'No recent regime analysis data available'}
        
        # Analyze regime distribution
        regime_counts = {}
        for record in recent_records:
            regime = record['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Calculate summary statistics
        summary = {
            'period_days': days,
            'total_analyses': len(recent_records),
            'regime_distribution': regime_counts,
            'most_common_regime': max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else 'unknown',
            'regime_stability': self._calculate_regime_stability(recent_records),
            'winner_consistency': self._calculate_winner_consistency(recent_records)
        }
        
        return summary
    
    def _calculate_regime_stability(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate stability metrics for each regime."""
        
        regime_stability = {}
        
        try:
            # Group by regime
            regime_sequences = defaultdict(list)
            
            for record in records:
                regime = record['regime']
                regime_sequences[regime].append(record)
            
            # Calculate stability for each regime
            for regime, sequence in regime_sequences.items():
                if len(sequence) >= 2:
                    # Calculate how long regime persists
                    persistence_times = []
                    current_persistence = 1
                    
                    for i in range(1, len(sequence)):
                        if sequence[i]['regime'] == regime:
                            current_persistence += 1
                        else:
                            persistence_times.append(current_persistence)
                            current_persistence = 1
                    
                    persistence_times.append(current_persistence)
                    
                    # Average persistence time
                    avg_persistence = np.mean(persistence_times)
                    regime_stability[regime] = avg_persistence
        
            return regime_stability
            
        except Exception as e:
            self.logger.error(f"Error calculating regime stability: {e}")
            return regime_stability
    
    def _calculate_winner_consistency(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate winner consistency across regimes."""
        
        try:
            # Group by regime and get winners
            regime_winners = defaultdict(list)
            
            for record in records:
                regime = record['regime']
                winner_ranking = record.get('winner_ranking', [])
                
                if winner_ranking:
                    winner = winner_ranking[0]['model_name']
                    regime_winners[regime].append(winner)
            
            # Calculate consistency scores
            winner_consistency = {}
            
            for regime, winners in regime_winners.items():
                if len(winners) >= 3:
                    # Calculate frequency of most common winner
                    winner_counts = {}
                    for winner in winners:
                        winner_counts[winner] = winner_counts.get(winner, 0) + 1
                    
                    most_common_winner = max(winner_counts.items(), key=lambda x: x[1])
                    consistency_score = most_common_winner[1] / len(winners)
                    
                    winner_consistency[regime] = consistency_score
            
            return winner_consistency
            
        except Exception as e:
            self.logger.error(f"Error calculating winner consistency: {e}")
            return {}


# Factory function for easy instantiation
def get_regime_winner_analyzer(config: Optional[Dict[str, Any]] = None) -> RegimeWinnerAnalyzer:
    """Factory function to get RegimeWinnerAnalyzer instance."""
    return RegimeWinnerAnalyzer(config)


# Convenience function for quick analysis
async def analyze_regime_consistency_quick(model_results: Dict[str, Any],
                                        market_data: pd.DataFrame,
                                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Quick regime consistency analysis.
    
    Args:
        model_results: Model performance results
        market_data: Market data for regime detection
        config: Configuration dictionary
        
    Returns:
        Regime consistency analysis result dictionary
    """
    analyzer = get_regime_winner_analyzer(config)
    return await analyzer.analyze_regime_consistency(model_results, market_data)
