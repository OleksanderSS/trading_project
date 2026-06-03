# src/validation/temporal_feature_separator.py
"""
TemporalFeatureSeparator Implementation.
Validates temporal ordering and prevents lookahead bias in time-series features.
Identifies and isolates features that may contain future information.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("TemporalFeatureSeparator")

class TemporalFeatureSeparator:
    """
    Separates features based on their temporal alignment to identify potential data leakage.
    
    Financial time-series are susceptible to lookahead bias where features at time T
    accidentally contain information about targets at time T+n.
    
    Capabilities:
    - Temporal correlation analysis: Checks feature alignment with future targets.
    - Pattern-based leakage detection: Identifies direct and delayed target leaks.
    - Dataset sanitization: Prunes suspicious features to ensure causal integrity.
    """

    def __init__(self, project_path: Optional[str] = None):
        """
        Initializes the separator environment.
        
        Args:
            project_path: Optional path for persisting analysis reports.
        """
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self.temporal_analysis: Dict[str, Any] = {}
        self.suspicious_features: List[str] = []
        self.safe_features: List[str] = []

    def _validate_feature_target_alignment(self, df_features: pd.DataFrame, target_series: pd.Series) -> bool:
        """Validates that features and target have matching lengths."""
        if len(df_features) != len(target_series):
            logger.warning(f"Feature/Target alignment error: Features({len(df_features)}) vs Target({len(target_series)})")
            return False
        return True

    def _calculate_lookahead_correlations(self, feature_values: np.ndarray, target_series: pd.Series, max_lookahead: int) -> List[Dict[str, Any]]:
        """Calculates correlations for different lookahead periods."""
        lookahead_correlations = []
        
        for lag in range(1, max_lookahead + 1):
            if lag >= len(target_series):
                break

            shifted_target = target_series.iloc[lag:].reset_index(drop=True)
            trimmed_features = feature_values[:-lag]

            if len(trimmed_features) > 2:
                valid_idx = ~(np.isnan(trimmed_features) | np.isnan(shifted_target))
                if valid_idx.sum() > 2:
                    corr = np.corrcoef(
                        trimmed_features[valid_idx],
                        shifted_target[valid_idx]
                    )[0, 1]
                    lookahead_correlations.append({
                        'lookahead_periods': lag,
                        'correlation': float(corr) if not np.isnan(corr) else 0.0
                    })
        
        return lookahead_correlations

    def _evaluate_feature_suspicion(self, max_corr: float) -> Tuple[bool, str]:
        """Evaluates if a feature is suspicious based on correlation threshold."""
        if max_corr > 0.7:
            return True, f"Critical lookahead correlation detected ({max_corr:.3f})"
        elif max_corr > 0.5:
            return True, f"Elevated lookahead correlation detected ({max_corr:.3f})"
        return False, ""

    def _create_feature_analysis(self, col: str, max_corr: float, lookahead_correlations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Creates analysis result for a single feature."""
        suspicious, reason = self._evaluate_feature_suspicion(max_corr)
        
        return {
            'feature_name': col,
            'suspicious': suspicious,
            'reason': reason,
            'max_lookahead_correlation': max_corr,
            'lookahead_analysis': lookahead_correlations,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _update_feature_lists(self, col: str, suspicious: bool) -> None:
        """Updates suspicious or safe feature lists."""
        if suspicious:
            self.suspicious_features.append(col)
        else:
            self.safe_features.append(col)

    def analyze_temporal_correlation(
        self, 
        df_features: pd.DataFrame, 
        target_series: pd.Series, 
        max_lookahead: int = 5
    ) -> Dict[str, Any]:
        """
        Analyzes the correlation of features at time T with target values at time T+n.
        High future correlation indicates potential leakage (causal violation).
        
        Args:
            df_features: Feature matrix.
            target_series: Target vector.
            max_lookahead: Number of future periods to audit.
        
        Returns:
            Dictionary containing granular analysis results for each audited feature.
        """
        if not self._validate_feature_target_alignment(df_features, target_series):
            return {}

        analysis_results = {}

        for col in df_features.columns:
            feature_values = df_features[col].values
            lookahead_correlations = self._calculate_lookahead_correlations(
                feature_values, target_series, max_lookahead
            )
            
            max_corr = max([abs(c['correlation']) for c in lookahead_correlations], default=0.0)
            
            analysis_results[col] = self._create_feature_analysis(
                col, max_corr, lookahead_correlations
            )
            
            self._update_feature_lists(col, analysis_results[col]['suspicious'])

        self.temporal_analysis = analysis_results
        return analysis_results

    def _calculate_correlation(self, feature: np.ndarray, target: np.ndarray) -> Optional[float]:
        """Safely calculates correlation between feature and target."""
        if len(feature) <= 1 or len(target) <= 1:
            return None
            
        valid_idx = ~(np.isnan(feature) | np.isnan(target))
        if valid_idx.sum() <= 2:
            return None
            
        corr = np.corrcoef(feature[valid_idx], target[valid_idx])[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def _detect_direct_target_proxy(self, col: str, feature: np.ndarray, target: np.ndarray, threshold: float) -> Optional[Dict[str, Any]]:
        """Detects direct correlation between feature and target."""
        direct_corr = self._calculate_correlation(feature, target)
        
        if direct_corr is not None and abs(direct_corr) > threshold:
            return {
                'feature': col,
                'pattern': 'direct_target_proxy',
                'correlation': direct_corr,
                'severity': 'CRITICAL' if abs(direct_corr) > 0.9 else 'HIGH'
            }
        return None

    def _detect_delayed_target_redundancy(self, col: str, feature: np.ndarray, target: np.ndarray, threshold: float) -> Optional[Dict[str, Any]]:
        """Detects correlation between feature and delayed target."""
        if len(feature) <= 1 or len(target) <= 1:
            return None
            
        delayed_target = np.roll(target, 1)[1:]
        trimmed_feature = feature[1:]

        delayed_corr = self._calculate_correlation(trimmed_feature, delayed_target)
        
        if delayed_corr is not None and abs(delayed_corr) > threshold:
            return {
                'feature': col,
                'pattern': 'delayed_target_redundancy',
                'correlation': delayed_corr,
                'severity': 'HIGH'
            }
        return None

    def detect_leakage_pattern(
        self, 
        df_features: pd.DataFrame, 
        target_series: pd.Series, 
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Detects specific archetypes of data leakage commonly found in trading signal pipelines.
        
        Archetypes:
        - Direct Correlation: Feature is nearly identical to the ground truth target.
        - Delayed Target Leak: Feature at T is a simple transformation of target at T-1.
        - Lookahead Bias: Feature at T contains information from T+n.
        """
        leakage_patterns = []
        target = target_series.values

        for col in df_features.columns:
            feature = df_features[col].values

            # Pattern 1: Direct Target Proximity (Feature ~ Target)
            direct_pattern = self._detect_direct_target_proxy(col, feature, target, threshold)
            if direct_pattern:
                leakage_patterns.append(direct_pattern)

            # Pattern 2: Causal Leakage (Feature[T] = Target[T-1] or similar)
            delayed_pattern = self._detect_delayed_target_redundancy(col, feature, target, threshold)
            if delayed_pattern:
                leakage_patterns.append(delayed_pattern)

        return leakage_patterns

    def get_safe_features(self, threshold: float = 0.5) -> List[str]:
        """Filters analysis results to return assets below the sensitivity threshold."""
        safe = []
        for col, analysis in self.temporal_analysis.items():
            if abs(analysis['max_lookahead_correlation']) < threshold:
                safe.append(col)
        return safe

    def get_suspicious_features(self, threshold: float = 0.5) -> List[str]:
        """Identifies features exceeding the temporal sensitivity threshold."""
        suspicious = []
        for col, analysis in self.temporal_analysis.items():
            if abs(analysis['max_lookahead_correlation']) >= threshold:
                suspicious.append(col)
        return suspicious

    def create_clean_dataset(
        self, 
        df_features: pd.DataFrame, 
        remove_suspicious: bool = True, 
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Generates a sanitized dataset by excluding features identified as non-causal or leaky.
        
        Args:
            df_features: Raw input features.
            remove_suspicious: If True, prunes suspicious features.
            threshold: Correlation ceiling for pruning decisions.
        
        Returns:
            Sanitized DataFrame ready for training.
        """
        if remove_suspicious:
            safe_cols = self.get_safe_features(threshold=threshold)
            clean_df = df_features[safe_cols].copy()
        else:
            clean_df = df_features.copy()

        logger.info(
            f"Dataset Sanitized: Initial={len(df_features.columns)}, "
            f"Safe={len(clean_df.columns)}, Pruned={len(df_features.columns) - len(clean_df.columns)}"
        )

        return clean_df

    def save_temporal_analysis(self, filepath: Optional[Path] = None) -> Path:
        """Persists the temporal audit results to a JSON report."""
        if filepath is None:
            filepath = self.project_path / "temporal_analysis_report.json"

        # Prepare serializable state
        serializable_analysis = {}
        for key, value in self.temporal_analysis.items():
            serializable_analysis[key] = {
                'feature_name': value['feature_name'],
                'suspicious': value['suspicious'],
                'reason': value['reason'],
                'max_lookahead_correlation': float(value['max_lookahead_correlation']),
                'analysis_timestamp': value['analysis_timestamp']
            }

        report = {
            'temporal_analysis_matrix': serializable_analysis,
            'suspicious_features': self.suspicious_features,
            'safe_features': self.safe_features,
            'total_suspicious': len(self.suspicious_features),
            'total_safe': len(self.safe_features),
            'generated_at': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Temporal intelligence persisted to {filepath}")
        return filepath


# Operational Demonstration
if __name__ == "__main__":
    logger.info("Executing TemporalFeatureSeparator functional demonstration\n")

    # Time-series data simulation
    rng = np.random.default_rng(42)
    n_samples = 100
    dates = pd.date_range('2026-01-01', periods=n_samples, freq='D')

    # Synthetic Ground Truth Target
    target = rng.standard_normal(n_samples).cumsum() * 0.01

    # Feature simulation with synthetic leakage points
    X = pd.DataFrame({
        'safe_factor_alpha': rng.standard_normal(n_samples) * 0.1,
        'safe_factor_beta': rng.standard_normal(n_samples) * 0.1,
        'leaked_delayed_target': np.roll(target, 1),  # Potential Causal Leak (T-1 info)
        'leaked_future_signal': np.roll(target, -2) + rng.standard_normal(n_samples) * 0.01  # Lookahead Leak (T+2 info)
    })

    # Execution of the audit pipeline
    separator = TemporalFeatureSeparator()
    
    logger.info("Simulating temporal correlation matrix analysis...")
    analysis = separator.analyze_temporal_correlation(X, pd.Series(target), max_lookahead=5)

    logger.info("\n--- Audit Results ---")
    for col, result in analysis.items():
        status = "🚨 SUSPICIOUS" if result['suspicious'] else "✅ SAFE"
        logger.info(f"\n{status}: Asset '{col}'")
        logger.info(f"  Max Lookahead Correlation: {result['max_lookahead_correlation']:.4f}")
        if result['reason']:
            logger.info(f"  Alert Rationale: {result['reason']}")

    # Pattern recognition audit
    leakage = separator.detect_leakage_pattern(X, pd.Series(target), threshold=0.6)
    
    if leakage:
        logger.info("\n--- Identified Leakage Archetypes ---")
        for pattern in leakage:
            logger.warning(f"  🚨 Detected: {pattern['feature']} -> {pattern['pattern']} (Correlation: {pattern['correlation']:.4f})")
    
    # Dataset cleansing
    clean_X = separator.create_clean_dataset(X, remove_suspicious=True)
    
    # Synchronization to disk
    separator.save_temporal_analysis()
    logger.info("\n✅ Demonstration sequence completed successfully.")
