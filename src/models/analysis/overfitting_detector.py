#!/usr/bin/env python3
"""
Overfitting Detector - Advanced Overfitting Detection and Prevention
Detects and prevents overfitting in machine learning models with comprehensive analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("OverfittingDetector")

class OverfittingDetector:
    """
    Advanced overfitting detection and prevention system.
    
    This detector analyzes and prevents overfitting through:
    - Learning curve analysis with visual inspection
    - Cross-validation consistency checking
    - Train-validation gap detection
    - Regularization recommendations
    - Model complexity assessment
    
    Critical for maintaining robust and generalizable models.
    """
    
    # Overfitting signal types
    OVERFITTING_SIGNALS = {
        'train_val_gap': {
            'description': 'Large train-validation performance gap',
            'threshold': 0.1,  # 10% gap threshold
            'severity': 'high'
        },
        'learning_curve': {
            'description': 'Unstable learning curve patterns',
            'threshold': 0.15,  # 15% variance threshold
            'severity': 'medium'
        },
        'cv_variance': {
            'description': 'High cross-validation variance',
            'threshold': 0.05,  # 5% variance threshold
            'severity': 'medium'
        },
        'complexity_penalty': {
            'description': 'Model complexity vs performance trade-off',
            'threshold': 0.02,  # 2% per complexity point
            'severity': 'low'
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Overfitting Detector.
        
        Args:
            config: Configuration dictionary for detection settings
        """
        self.logger = logger
        self.config = config or {}
        
        # Detection thresholds
        self.thresholds = self.OVERFITTING_SIGNALS.copy()
        self.thresholds.update(self.config.get('thresholds', {}))
        
        # Analysis settings
        self.cv_folds = self.config.get('cv_folds', 5)
        self.scoring_metric = self.config.get('scoring_metric', 'neg_mean_squared_error')
        self.train_sizes = self.config.get('train_sizes', np.linspace(0.1, 1.0, 10))
        
        # Visualization settings
        self.enable_visualization = self.config.get('enable_visualization', True)
        self.save_plots = self.config.get('save_plots', True)
        
        # Storage paths
        self.storage_path = Path(self.config.get('storage_path', 'data/analysis/overfitting'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ OverfittingDetector initialized")
    
    async def detect_overfitting(self, 
                               model: Any,
                               X_train: pd.DataFrame,
                               y_train: pd.Series,
                               X_val: Optional[pd.DataFrame] = None,
                               y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Detect overfitting signals in a trained model.
        
        Args:
            model: Trained machine learning model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dict with overfitting analysis and recommendations
        """
        self.logger.info("🔍 Detecting overfitting signals...")
        
        results = {
            'timestamp': datetime.now(),
            'model_type': type(model).__name__,
            'data_info': self._analyze_data_characteristics(X_train, X_val),
            'learning_curve': {},
            'cv_results': {},
            'overfitting_signals': {},
            'recommendations': []
        }
        
        try:
            # 1. Generate learning curve
            learning_curve_result = await self._generate_learning_curve(
                model, X_train, y_train
            )
            results['learning_curve'] = learning_curve_result
            
            # 2. Cross-validation analysis
            cv_result = await self._perform_cv_analysis(
                model, X_train, y_train
            )
            results['cv_results'] = cv_result
            
            # 3. Train-validation gap analysis (if validation data available)
            if X_val is not None and y_val is not None:
                gap_analysis = self._analyze_train_val_gap(
                    model, X_train, y_train, X_val, y_val
                )
                results['train_val_gap'] = gap_analysis
            else:
                results['train_val_gap'] = {'status': 'no_validation_data'}
            
            # 4. Overfitting signal detection
            overfitting_signals = self._detect_overfitting_signals(
                results['learning_curve'],
                results['cv_results'],
                results.get('train_val_gap', {})
            )
            results['overfitting_signals'] = overfitting_signals
            
            # 5. Generate prevention recommendations
            recommendations = self._generate_prevention_recommendations(
                overfitting_signals,
                results['data_info']
            )
            results['recommendations'] = recommendations
            
            # 6. Create visualizations
            if self.enable_visualization:
                await self._create_overfitting_visualizations(results)
            
            # 7. Store results
            self._store_detection_results(results)
            
            self.logger.info(f"✅ Overfitting detection complete. Signals: {len(overfitting_signals)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in overfitting detection: {e}", exc_info=True)
            results['error'] = str(e)
            return results
    
    async def _generate_learning_curve(self, 
                                     model: Any,
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series) -> Dict[str, Any]:
        """Generate and analyze learning curve."""
        
        learning_curve_result = {
            'status': 'completed',
            'train_scores': [],
            'val_scores': [],
            'train_sizes': [],
            'analysis': {}
        }
        
        try:
            # Generate learning curve
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train,
                cv=self.cv_folds,
                scoring=self.scoring_metric,
                train_sizes=self.train_sizes,
                n_jobs=-1,
                random_state=42
            )
            
            learning_curve_result['train_sizes'] = train_sizes.tolist()
            learning_curve_result['train_scores'] = train_scores.tolist()
            learning_curve_result['val_scores'] = val_scores.tolist()
            
            # Analyze learning curve
            analysis = self._analyze_learning_curve_patterns(
                train_scores, val_scores, train_sizes
            )
            learning_curve_result['analysis'] = analysis
            
            self.logger.info("📈 Learning curve generated successfully")
            
            return learning_curve_result
            
        except Exception as e:
            self.logger.error(f"Error generating learning curve: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_learning_curve_patterns(self, 
                                       train_scores: np.ndarray,
                                       val_scores: np.ndarray,
                                       train_sizes: np.ndarray) -> Dict[str, Any]:
        """Analyze learning curve patterns for overfitting indicators."""
        
        analysis = {
            'train_score_mean': np.mean(train_scores),
            'val_score_mean': np.mean(val_scores),
            'train_score_std': np.std(train_scores),
            'val_score_std': np.std(val_scores),
            'final_gap': train_scores[-1] - val_scores[-1],
            'max_gap': np.max(train_scores - val_scores),
            'overfitting_indicators': [],
            'underfitting_indicators': [],
            'ideal_learning': False
        }
        
        try:
            # Check for overfitting indicators
            final_gap = analysis['final_gap']
            max_gap = analysis['max_gap']
            
            # Large final gap indicates overfitting
            if final_gap > self.thresholds['train_val_gap']['threshold']:
                analysis['overfitting_indicators'].append({
                    'type': 'large_final_gap',
                    'value': final_gap,
                    'threshold': self.thresholds['train_val_gap']['threshold']
                })
            
            # Large max gap also indicates overfitting
            if max_gap > self.thresholds['train_val_gap']['threshold'] * 1.5:
                analysis['overfitting_indicators'].append({
                    'type': 'large_max_gap',
                    'value': max_gap,
                    'threshold': self.thresholds['train_val_gap']['threshold'] * 1.5
                })
            
            # Check for underfitting indicators
            if analysis['train_score_mean'] < 0.5:  # Assuming scores are 0-1 or negative for MSE
                analysis['underfitting_indicators'].append({
                    'type': 'low_train_performance',
                    'value': analysis['train_score_mean']
                })
            
            if analysis['val_score_mean'] < 0.5:
                analysis['underfitting_indicators'].append({
                    'type': 'low_val_performance',
                    'value': analysis['val_score_mean']
                })
            
            # Check for ideal learning (converging curves)
            train_std = analysis['train_score_std']
            val_std = analysis['val_score_std']
            
            # Low variance and convergence indicates ideal learning
            if (train_std < 0.05 and val_std < 0.05 and
                final_gap < 0.02):  # Small gap indicates convergence
                analysis['ideal_learning'] = True
            
            # Calculate learning curve stability
            train_variance = np.var(train_scores)
            val_variance = np.var(val_scores)
            
            analysis['train_variance'] = train_variance
            analysis['val_variance'] = val_variance
            analysis['stability_score'] = 1.0 / (1.0 + train_variance + val_variance)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing learning curve patterns: {e}")
            return analysis
    
    async def _perform_cv_analysis(self, 
                               model: Any,
                               X_train: pd.DataFrame,
                               y_train: pd.Series) -> Dict[str, Any]:
        """Perform cross-validation analysis."""
        
        cv_result = {
            'status': 'completed',
            'cv_scores': [],
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'cv_variance': 0.0,
            'stability_score': 0.0
        }
        
        try:
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.cv_folds,
                scoring=self.scoring_metric,
                n_jobs=-1
            )
            
            cv_result['cv_scores'] = cv_scores.tolist()
            cv_result['cv_mean'] = np.mean(cv_scores)
            cv_result['cv_std'] = np.std(cv_scores)
            cv_result['cv_variance'] = np.var(cv_scores)
            
            # Calculate stability score (inverse of variance)
            cv_result['stability_score'] = 1.0 / (1.0 + cv_result['cv_variance'])
            
            # Check for high variance (unstable model)
            if cv_result['cv_variance'] > self.thresholds['cv_variance']['threshold']:
                cv_result['high_variance'] = True
            else:
                cv_result['high_variance'] = False
            
            self.logger.info(f"📊 Cross-validation completed. Stability: {cv_result['stability_score']:.3f}")
            
            return cv_result
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_train_val_gap(self, 
                                model: Any,
                                X_train: pd.DataFrame,
                                y_train: pd.Series,
                                X_val: pd.DataFrame,
                                y_val: pd.Series) -> Dict[str, Any]:
        """Analyze train-validation performance gap."""
        
        gap_analysis = {
            'status': 'completed',
            'train_metrics': {},
            'val_metrics': {},
            'gap_analysis': {}
        }
        
        try:
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred)
            val_metrics = self._calculate_metrics(y_val, y_val_pred)
            
            gap_analysis['train_metrics'] = train_metrics
            gap_analysis['val_metrics'] = val_metrics
            
            # Calculate gaps
            gap_analysis['gap_analysis'] = {
                'mse_gap': train_metrics['mse'] - val_metrics['mse'],
                'mae_gap': train_metrics['mae'] - val_metrics['mae'],
                'r2_gap': train_metrics['r2'] - val_metrics['r2'],
                'relative_mse_gap': (train_metrics['mse'] - val_metrics['mse']) / val_metrics['mse'] if val_metrics['mse'] > 0 else 0,
                'relative_mae_gap': (train_metrics['mae'] - val_metrics['mae']) / val_metrics['mae'] if val_metrics['mae'] > 0 else 0
            }
            
            # Check for significant gaps
            mse_gap = gap_analysis['gap_analysis']['mse_gap']
            relative_mse_gap = gap_analysis['gap_analysis']['relative_mse_gap']
            
            if mse_gap > self.thresholds['train_val_gap']['threshold']:
                gap_analysis['significant_gap'] = True
                gap_analysis['gap_type'] = 'overfitting'
            elif mse_gap < -self.thresholds['train_val_gap']['threshold']:
                gap_analysis['significant_gap'] = True
                gap_analysis['gap_type'] = 'underfitting'
            else:
                gap_analysis['significant_gap'] = False
                gap_analysis['gap_type'] = 'well_balanced'
            
            self.logger.info(f"📈 Train-Val gap: {mse_gap:.6f} ({gap_analysis['gap_type']})")
            
            return gap_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing train-val gap: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _detect_overfitting_signals(self, 
                                    learning_curve: Dict[str, Any],
                                    cv_results: Dict[str, Any],
                                    gap_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect overfitting signals from analysis results."""
        
        signals = {
            'total_signals': 0,
            'signal_details': [],
            'severity_breakdown': {
                'critical': [],
                'high': [],
                'medium': [],
                'low': []
            }
        }
        
        try:
            # Check learning curve signals
            if learning_curve.get('status') == 'completed':
                lc_analysis = learning_curve.get('analysis', {})
                
                # Overfitting indicators from learning curve
                for indicator in lc_analysis.get('overfitting_indicators', []):
                    signals['signal_details'].append({
                        'source': 'learning_curve',
                        'type': indicator['type'],
                        'value': indicator['value'],
                        'threshold': indicator['threshold'],
                        'severity': self.OVERFITTING_SIGNALS['learning_curve']['severity']
                    })
                    signals['severity_breakdown'][self.OVERFITTING_SIGNALS['learning_curve']['severity']].append(indicator)
                    signals['total_signals'] += 1
                
                # Underfitting indicators
                for indicator in lc_analysis.get('underfitting_indicators', []):
                    signals['signal_details'].append({
                        'source': 'learning_curve',
                        'type': indicator['type'],
                        'value': indicator['value'],
                        'severity': 'medium'  # Underfitting is less severe
                    })
                    signals['severity_breakdown']['medium'].append(indicator)
                    signals['total_signals'] += 1
            
            # Check cross-validation signals
            if cv_results.get('status') == 'completed':
                if cv_results.get('high_variance', False):
                    signals['signal_details'].append({
                        'source': 'cross_validation',
                        'type': 'high_variance',
                        'value': cv_results['cv_variance'],
                        'threshold': self.thresholds['cv_variance']['threshold'],
                        'severity': self.OVERFITTING_SIGNALS['cv_variance']['severity']
                    })
                    signals['severity_breakdown'][self.OVERFITTING_SIGNALS['cv_variance']['severity']].append({
                        'type': 'high_variance',
                        'value': cv_results['cv_variance']
                    })
                    signals['total_signals'] += 1
                
                # Low stability score
                stability_score = cv_results.get('stability_score', 1.0)
                if stability_score < 0.5:
                    signals['signal_details'].append({
                        'source': 'cross_validation',
                        'type': 'low_stability',
                        'value': stability_score,
                        'threshold': 0.5,
                        'severity': 'medium'
                    })
                    signals['severity_breakdown']['medium'].append({
                        'type': 'low_stability',
                        'value': stability_score
                    })
                    signals['total_signals'] += 1
            
            # Check train-validation gap signals
            if gap_analysis.get('status') == 'completed':
                if gap_analysis.get('significant_gap', False):
                    gap_type = gap_analysis.get('gap_type', 'unknown')
                    
                    signals['signal_details'].append({
                        'source': 'train_val_gap',
                        'type': gap_type,
                        'value': gap_analysis['gap_analysis'].get('relative_mse_gap', 0),
                        'threshold': self.thresholds['train_val_gap']['threshold'],
                        'severity': 'high' if gap_type == 'overfitting' else 'medium'
                    })
                    
                    severity = 'high' if gap_type == 'overfitting' else 'medium'
                    signals['severity_breakdown'][severity].append({
                        'type': gap_type,
                        'value': gap_analysis['gap_analysis'].get('relative_mse_gap', 0)
                    })
                    signals['total_signals'] += 1
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error detecting overfitting signals: {e}")
            return signals
    
    def _generate_prevention_recommendations(self, 
                                           overfitting_signals: Dict[str, Any],
                                           data_info: Dict[str, Any]) -> List[str]:
        """Generate overfitting prevention recommendations."""
        
        recommendations = []
        
        try:
            signal_count = overfitting_signals['total_signals']
            
            # High severity recommendations
            critical_signals = overfitting_signals['severity_breakdown']['critical']
            if critical_signals:
                recommendations.append(
                    f"🚨 CRITICAL: {len(critical_signals)} critical overfitting signals detected."
                )
                
                for signal in critical_signals:
                    if signal['source'] == 'learning_curve':
                        recommendations.append(
                            f"   • Learning curve {signal['type']}: {signal['value']:.3f} (threshold: {signal['threshold']:.3f})"
                        )
                    elif signal['source'] == 'train_val_gap':
                        recommendations.append(
                            f"   • Train-Val gap {signal['type']}: {signal['value']:.3f} (threshold: {signal['threshold']:.3f})"
                        )
                
                recommendations.append(
                    "   → STOP: Model requires significant regularization or simplification."
                )
            
            # High severity recommendations
            high_signals = overfitting_signals['severity_breakdown']['high']
            if high_signals:
                recommendations.append(
                    f"⚠️ HIGH: {len(high_signals)} high overfitting signals detected."
                )
                
                # Specific recommendations for high severity
                if any(s['source'] == 'learning_curve' for s in high_signals):
                    recommendations.append(
                        "   • Reduce model complexity (fewer layers/parameters)"
                    )
                    recommendations.append(
                        "   • Increase regularization strength"
                    )
                    recommendations.append(
                        "   • Add dropout layers (for neural networks)"
                    )
                
                if any(s['source'] == 'cross_validation' for s in high_signals):
                    recommendations.append(
                        "   • Increase cross-validation folds"
                    )
                    recommendations.append(
                        "   • Use ensemble methods to reduce variance"
                    )
                
                if any(s['source'] == 'train_val_gap' for s in high_signals):
                    recommendations.append(
                        "   • Use early stopping to prevent overfitting"
                    )
                    recommendations.append(
                        "   • Increase training data or use data augmentation"
                    )
                
                recommendations.append(
                    "   → Model likely overfits. Apply regularization immediately."
                )
            
            # Medium severity recommendations
            medium_signals = overfitting_signals['severity_breakdown']['medium']
            if medium_signals:
                recommendations.append(
                    f"⚠️ MEDIUM: {len(medium_signals)} medium overfitting signals detected."
                )
                
                # Check specific medium signals
                if any(s['type'] == 'low_stability' for s in medium_signals):
                    recommendations.append(
                        "   • Consider ensemble methods for stability"
                    )
                    recommendations.append(
                        "   • Use more consistent cross-validation"
                    )
                
                if any(s['type'] in ['low_train_performance', 'low_val_performance'] for s in medium_signals):
                    recommendations.append(
                        "   • Model may be underfitting. Consider increasing complexity"
                    )
                    recommendations.append(
                        "   • Add more features or use more powerful model"
                    )
                
                recommendations.append(
                    "   → Monitor model performance and consider adjustments."
                )
            
            # Low severity recommendations
            low_signals = overfitting_signals['severity_breakdown']['low']
            if low_signals:
                recommendations.append(
                    f"📊 LOW: {len(low_signals)} low overfitting signals detected."
                )
                
                recommendations.append(
                    "   → Model appears well-balanced. Continue monitoring."
                )
            
            # No signals
            if signal_count == 0:
                recommendations.append(
                    "✅ NO OVERFITTING: Model appears well-balanced."
                )
                recommendations.append(
                    "   → Continue with current model configuration."
                )
            
            # Data-specific recommendations
            data_size = data_info.get('sample_count', 0)
            feature_count = data_info.get('feature_count', 0)
            
            if data_size < 1000:
                recommendations.append(
                    f"📊 DATA CONCERN: Small dataset ({data_size} samples). "
                    "Consider data augmentation or collecting more data."
                )
            
            if feature_count > data_size:
                recommendations.append(
                    f"📊 CURSE OF DIMENSIONALITY: {feature_count} features vs {data_size} samples. "
                    "Consider feature selection or dimensionality reduction."
                )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return recommendations
    
    def _analyze_data_characteristics(self, 
                                     X_train: pd.DataFrame,
                                     X_val: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze data characteristics for context."""
        
        data_info = {
            'train_shape': X_train.shape,
            'val_shape': X_val.shape if X_val is not None else None,
            'sample_count': len(X_train),
            'feature_count': len(X_train.columns),
            'numeric_features': len(X_train.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(X_train.select_dtypes(exclude=[np.number]).columns),
            'missing_values': X_train.isnull().sum().sum(),
            'feature_types': X_train.dtypes.value_counts().to_dict()
        }
        
        return data_info
    
    async def _create_overfitting_visualizations(self, results: Dict[str, Any]) -> None:
        """Create visualizations for overfitting analysis."""
        
        try:
            # Create plots directory
            plots_dir = self.storage_path / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Learning curve plot
            if results['learning_curve'].get('status') == 'completed':
                self._plot_learning_curve(
                    results['learning_curve'],
                    plots_dir / f'learning_curve_{timestamp}.png'
                )
            
            # Cross-validation plot
            if results['cv_results'].get('status') == 'completed':
                self._plot_cv_distribution(
                    results['cv_results'],
                    plots_dir / f'cv_distribution_{timestamp}.png'
                )
            
            self.logger.info("📊 Overfitting visualizations created")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
    
    def _plot_learning_curve(self, learning_curve: Dict[str, Any], save_path: Path) -> None:
        """Create learning curve visualization."""
        
        try:
            plt.figure(figsize=(10, 6))
            
            train_sizes = learning_curve['train_sizes']
            train_scores = learning_curve['train_scores']
            val_scores = learning_curve['val_scores']
            
            plt.plot(train_sizes, train_scores, 'o-', color='blue', label='Training Score')
            plt.plot(train_sizes, val_scores, 'o-', color='red', label='Validation Score')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('Score')
            plt.title('Learning Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add annotations for key points
            final_gap = learning_curve['analysis'].get('final_gap', 0)
            max_gap = learning_curve['analysis'].get('max_gap', 0)
            
            if final_gap > 0.1:
                plt.annotate(
                    f'Final Gap: {final_gap:.3f}',
                    xy=(train_sizes[-1], train_scores[-1]),
                    xytext=(10, -10),
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7),
                    fontsize=9,
                    color='red'
                )
            
            if self.save_plots:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting learning curve: {e}")
    
    def _plot_cv_distribution(self, cv_results: Dict[str, Any], save_path: Path) -> None:
        """Create cross-validation distribution visualization."""
        
        try:
            plt.figure(figsize=(8, 5))
            
            cv_scores = cv_results['cv_scores']
            
            plt.hist(cv_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            
            plt.xlabel('Cross-Validation Score')
            plt.ylabel('Frequency')
            plt.title('Cross-Validation Score Distribution')
            
            # Add statistics
            mean_score = cv_results['cv_mean']
            std_score = cv_results['cv_std']
            
            plt.axvline(mean_score, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_score:.3f}')
            plt.axvline(mean_score + std_score, color='orange', linestyle='--', alpha=0.8, label=f'±1 STD: {mean_score + std_score:.3f}')
            plt.axvline(mean_score - std_score, color='orange', linestyle='--', alpha=0.8, label=f'±1 STD: {mean_score - std_score:.3f}')
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if self.save_plots:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting CV distribution: {e}")
    
    def _store_detection_results(self, results: Dict[str, Any]) -> None:
        """Store overfitting detection results for historical tracking."""
        
        try:
            # Store in JSON file
            timestamp = results['timestamp'].strftime('%Y%m%d_%H%M%S')
            filename = f"overfitting_detection_{timestamp}.json"
            filepath = self.storage_path / filename
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Keep only last 100 files
            files = list(self.storage_path.glob("overfitting_detection_*.json"))
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for file_to_delete in files[100:]:
                file_to_delete.unlink()
                
        except Exception as e:
            self.logger.error(f"Failed to store detection results: {e}")
    
    def get_overfitting_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of overfitting detection over time period."""
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Load recent detection results
        recent_files = list(self.storage_path.glob("overfitting_detection_*.json"))
        recent_files = [
            f for f in recent_files
            if datetime.fromtimestamp(f.stat().st_mtime) >= cutoff_time
        ]
        
        if not recent_files:
            return {'error': 'No recent overfitting detection data available'}
        
        # Analyze overfitting trends
        overfitting_history = []
        
        for file_path in recent_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    overfitting_history.append(data)
            except Exception as e:
                self.logger.error(f"Error loading overfitting history from {file_path}: {e}")
        
        # Calculate summary statistics
        summary = {
            'period_days': days,
            'total_detections': len(overfitting_history),
            'overfitting_trends': self._analyze_overfitting_trends(overfitting_history),
            'common_signals': self._get_common_overfitting_signals(overfitting_history),
            'model_types_analyzed': self._get_model_types_analyzed(overfitting_history)
        }
        
        return summary
    
    def _analyze_overfitting_trends(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overfitting trends over time."""
        
        try:
            trends = {
                'signal_count_trend': [],
                'severity_trend': [],
                'most_common_signal': None
            }
            
            # Extract signal counts over time
            for record in history:
                signal_count = record.get('overfitting_signals', {}).get('total_signals', 0)
                severity_breakdown = record.get('overfitting_signals', {}).get('severity_breakdown', {})
                
                trends['signal_count_trend'].append({
                    'timestamp': record['timestamp'],
                    'signal_count': signal_count,
                    'critical_count': len(severity_breakdown.get('critical', [])),
                    'high_count': len(severity_breakdown.get('high', [])),
                    'medium_count': len(severity_breakdown.get('medium', [])),
                    'low_count': len(severity_breakdown.get('low', []))
                })
            
            # Calculate trends
            if len(trends['signal_count_trend']) >= 2:
                signal_counts = [t['signal_count'] for t in trends['signal_count_trend']]
                
                # Simple trend analysis
                if len(signal_counts) > 1:
                    slope = np.polyfit(range(len(signal_counts)), signal_counts, 1)[0]
                    trends['signal_count_trend'] = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
                else:
                    trends['signal_count_trend'] = 'stable'
            
            # Get most common signal
            all_signals = []
            for record in history:
                signals = record.get('overfitting_signals', {}).get('signal_details', [])
                all_signals.extend([s['type'] for s in signals])
            
            if all_signals:
                signal_counts = {}
                for signal in all_signals:
                    signal_counts[signal] = signal_counts.get(signal, 0) + 1
                
                trends['most_common_signal'] = max(signal_counts.items(), key=lambda x: x[1])[0]
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing overfitting trends: {e}")
            return trends
    
    def _get_common_overfitting_signals(self, history: List[Dict[str, Any]]) -> List[str]:
        """Get most common overfitting signals."""
        
        try:
            signal_counts = {}
            
            for record in history:
                signals = record.get('overfitting_signals', {}).get('signal_details', [])
                for signal in signals:
                    signal_key = f"{signal['source']}_{signal['type']}"
                    signal_counts[signal_key] = signal_counts.get(signal_key, 0) + 1
            
            # Return top 5 most common signals
            sorted_signals = sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)
            return [signal[0] for signal in sorted_signals[:5]]
            
        except Exception as e:
            self.logger.error(f"Error getting common overfitting signals: {e}")
            return []
    
    def _get_model_types_analyzed(self, history: List[Dict[str, Any]]) -> List[str]:
        """Get model types that have been analyzed."""
        
        try:
            model_types = set()
            
            for record in history:
                model_type = record.get('model_type', 'unknown')
                model_types.add(model_type)
            
            return list(model_types)
            
        except Exception as e:
            self.logger.error(f"Error getting model types analyzed: {e}")
            return []


# Factory function for easy instantiation
def get_overfitting_detector(config: Optional[Dict[str, Any]] = None) -> OverfittingDetector:
    """Factory function to get OverfittingDetector instance."""
    return OverfittingDetector(config)


# Convenience function for quick detection
async def detect_overfitting_quick(model: Any,
                                  X_train: pd.DataFrame,
                                  y_train: pd.Series,
                                  config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Quick overfitting detection.
    
    Args:
        model: Trained model to analyze
        X_train: Training features
        y_train: Training targets
        config: Configuration dictionary
        
    Returns:
        Overfitting detection result dictionary
    """
    detector = get_overfitting_detector(config)
    return await detector.detect_overfitting(model, X_train, y_train)
