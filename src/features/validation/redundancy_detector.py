#!/usr/bin/env python3
"""
Redundancy Detector - Advanced Feature Redundancy Detection and Elimination
Detects and eliminates redundant features using correlation clustering and VIF analysis.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("RedundancyDetector")

class RedundancyDetector:
    """
    Advanced redundancy detection and automatic feature elimination.

    This detector identifies redundant features through multiple methods:
    - High correlation clustering (threshold 0.95)
    - Variance Inflation Factor (VIF) analysis (threshold 10)
    - Feature grouping and representative selection
    - Automatic redundant feature elimination

    Reduces dimensionality while preserving maximum information.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize RedundancyDetector.

        Args:
            config: Configuration dictionary for redundancy detection
        """
        self.logger = logger
        self.config = config or {}

        # Redundancy detection thresholds
        self.REDUNDANCY_THRESHOLDS = {
            'correlation_threshold': 0.95,      # Features with correlation >0.95 are redundant
            'vif_threshold': 10.0,              # VIF >10 indicates multicollinearity
            'min_group_size': 2,                 # Minimum features in a redundant group
            'max_features_per_group': 1,          # Keep only 1 feature per redundant group
            'variance_threshold': 0.01           # Features with variance <0.01 are useless
        }

        # Override thresholds with config
        self.thresholds = self.REDUNDANCY_THRESHOLDS.copy()
        self.thresholds.update(self.config.get('thresholds', {}))

        # Analysis settings
        self.scaler = StandardScaler()
        self.use_clustering = self.config.get('use_clustering', True)
        self.use_vif = self.config.get('use_vif', True)

        self.logger.info("✅ RedundancyDetector initialized")

    def _filter_numeric_features(self, features_df: pd.DataFrame) -> tuple:
        """Filter numeric and non-numeric features."""
        numeric_features = features_df.select_dtypes(include=[np.number])
        non_numeric_features = features_df.select_dtypes(exclude=[np.number])
        return numeric_features, non_numeric_features

    def _combine_redundant_features(self, low_variance: list, correlation_redundant: list, high_vif: list) -> list:
        """Combine all redundant feature lists."""
        return list(set(low_variance + correlation_redundant + high_vif))

    def _create_final_feature_set(self, selected_features: pd.DataFrame, non_numeric_features: pd.DataFrame) -> pd.DataFrame:
        """Combine selected features with non-numeric features."""
        final_features = selected_features.copy()
        if not non_numeric_features.empty:
            final_features = pd.concat([final_features, non_numeric_features], axis=1)
        return final_features

    def _update_results(self, results: dict, redundant_features: list, final_features: pd.DataFrame, original_count: int) -> dict:
        """Update results with final statistics."""
        results.update({
            'redundant_features': redundant_features,
            'selected_features': list(final_features.columns),
            'selected_count': len(final_features.columns),
            'reduction_ratio': (len(redundant_features) / original_count) * 100,
            'cleaned_features': final_features
        })
        return results

    def eliminate_redundant_features(self,
                                   features_df: pd.DataFrame,
                                   target_series: pd.Series | None = None) -> dict[str, Any]:
        """
        Eliminate redundant features using multiple methods.

        Args:
            features_df: DataFrame with features to analyze
            target_series: Target series (optional, for VIF calculation)

        Returns:
            Dict with analysis results and cleaned features
        """
        self.logger.info(f"🔍 Analyzing {len(features_df.columns)} features for redundancy")

        results = {
            'original_features': list(features_df.columns),
            'original_count': len(features_df.columns),
            'redundant_features': [],
            'correlation_groups': {},
            'vif_results': {},
            'low_variance_features': [],
            'selected_features': [],
            'selected_count': 0,
            'reduction_ratio': 0.0,
            'cleaned_features': pd.DataFrame()
        }

        try:
            # 1. Filter out non-numeric features
            numeric_features, non_numeric_features = self._filter_numeric_features(features_df)

            if numeric_features.empty:
                self.logger.warning("No numeric features found for redundancy analysis")
                return self._create_empty_result(features_df, results)

            # 2. Remove low variance features
            variance_results = self._remove_low_variance_features(numeric_features)
            numeric_features = variance_results['remaining_features']
            results['low_variance_features'] = variance_results['removed_features']

            if numeric_features.empty:
                self.logger.warning("All features removed due to low variance")
                return self._create_empty_result(features_df, results)

            # 3. Correlation-based redundancy detection
            correlation_results = self._detect_correlation_redundancy(numeric_features)
            results['correlation_groups'] = correlation_results['redundant_groups']

            # 4. VIF analysis (if target provided)
            vif_results = {}
            if self.use_vif and target_series is not None:
                vif_results = self._calculate_vif_analysis(numeric_features, target_series)
                results['vif_results'] = vif_results

            # 5. Select representative features
            selection_results = self._select_representative_features(
                numeric_features, correlation_results, vif_results
            )

            # 6. Combine redundant features
            redundant_features = self._combine_redundant_features(
                results['low_variance_features'],
                correlation_results['redundant_features'],
                vif_results.get('high_vif_features', [])
            )

            # 7. Create final feature set
            selected_features = selection_results['selected_features']
            final_features = self._create_final_feature_set(selected_features, non_numeric_features)

            # 8. Update results
            self._update_results(results, redundant_features, final_features, len(features_df.columns))

            self._log_redundancy_summary(results)

            return results

        except Exception as e:
            self.logger.error(f"Error in redundancy detection: {e}", exc_info=True)
            return self._create_empty_result(features_df, results)

    def _remove_low_variance_features(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """Remove features with very low variance."""

        results = {
            'removed_features': [],
            'remaining_features': features_df.copy()
        }

        variance_threshold = self.thresholds['variance_threshold']

        for feature_name in features_df.columns:
            feature_variance = features_df[feature_name].var()

            if feature_variance < variance_threshold:
                results['removed_features'].append(feature_name)
                results['remaining_features'] = results['remaining_features'].drop(columns=[feature_name])

        if results['removed_features']:
            self.logger.info(f"🗑️ Removed {len(results['removed_features'])} low-variance features")

        return results

    def _detect_correlation_redundancy(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """Detect redundant features using correlation clustering."""

        results = {
            'redundant_groups': {},
            'redundant_features': [],
            'correlation_matrix': pd.DataFrame()
        }

        if not self.use_clustering:
            return results

        try:
            # Calculate correlation matrix
            correlation_matrix = features_df.corr().abs()
            results['correlation_matrix'] = correlation_matrix

            # Create correlation-based distance matrix
            distance_matrix = 1 - correlation_matrix
            np.fill_diagonal(distance_matrix.values, 0)

            # Perform hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - self.thresholds['correlation_threshold'],
                linkage='average',
                affinity='precomputed'
            )

            cluster_labels = clustering.fit_predict(distance_matrix)

            # Group features by cluster
            feature_clusters: dict[int, list[str]] = {}
            for feature_name, cluster_id in zip(features_df.columns, cluster_labels, strict=False):
                if cluster_id not in feature_clusters:
                    feature_clusters[cluster_id] = []
                feature_clusters[cluster_id].append(feature_name)

            # Identify redundant groups
            correlation_threshold = self.thresholds['correlation_threshold']
            min_group_size = self.thresholds['min_group_size']

            for cluster_id, cluster_features in feature_clusters.items():
                if len(cluster_features) >= min_group_size:
                    # Check if this cluster is actually redundant
                    cluster_corr = correlation_matrix.loc[cluster_features, cluster_features]

                    # Check if average correlation exceeds threshold
                    avg_correlation = cluster_corr.values[np.triu_indices_from(cluster_corr.shape, k=1)].mean()

                    if avg_correlation >= correlation_threshold:
                        results['redundant_groups'][f'cluster_{cluster_id}'] = {
                            'features': cluster_features,
                            'average_correlation': avg_correlation,
                            'size': len(cluster_features)
                        }

                        # All but one feature in this group are redundant
                        redundant_in_group = cluster_features[1:]  # Keep first feature
                        results['redundant_features'].extend(redundant_in_group)

            self.logger.info(f"🔗 Found {len(results['redundant_groups'])} redundant correlation groups")

            return results

        except Exception as e:
            self.logger.error(f"Error in correlation redundancy detection: {e}")
            return results

    def _calculate_vif_analysis(self,
                              features_df: pd.DataFrame,
                              target_series: pd.Series) -> dict[str, Any]:
        """Calculate Variance Inflation Factor (VIF) for multicollinearity detection."""

        results = {
            'vif_scores': {},
            'high_vif_features': [],
            'vif_threshold': self.thresholds['vif_threshold']
        }

        try:
            # Prepare data for VIF calculation
            X = features_df.copy()
            y = target_series.copy()

            # Remove any NaN or infinite values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())

            # Calculate VIF for each feature
            for feature_name in X.columns:
                # Regress feature against all other features
                other_features = X.drop(columns=[feature_name])

                if other_features.empty:
                    continue

                # Fit linear regression
                model = LinearRegression()
                model.fit(other_features, X[feature_name])

                # Calculate R-squared
                r_squared = model.score(other_features, X[feature_name])

                # Calculate VIF
                if r_squared >= 0.999:  # Perfect multicollinearity
                    vif = float('inf')
                else:
                    vif = 1 / (1 - r_squared)

                if isinstance(results.get('vif_scores'), dict):
                    results['vif_scores'][feature_name] = vif

                # Check if VIF exceeds threshold
                if vif > self.thresholds['vif_threshold']:
                    if isinstance(results.get('high_vif_features'), list):
                        results['high_vif_features'].append(feature_name)

            high_vif_count = len(results.get('high_vif_features', []))
            self.logger.info(f"📊 VIF analysis: {high_vif_count} features with high VIF")

            return results

        except Exception as e:
            self.logger.error(f"Error in VIF analysis: {e}")
            return {'vif_scores': {}, 'high_vif_features': [], 'vif_threshold': self.thresholds['vif_threshold'], 'error': str(e)}

    def _select_representative_features(self,
                                    features_df: pd.DataFrame,
                                    correlation_results: dict[str, Any],
                                    vif_results: dict[str, Any]) -> dict[str, Any]:
        """Select representative features from redundant groups."""

        results = {
            'selected_features': features_df.copy(),
            'selection_method': {},
            'removed_features': []
        }

        try:
            features_to_keep = set(features_df.columns)

            # 1. Handle correlation-based redundancy
            if correlation_results.get('redundant_groups'):
                for group_name, group_info in correlation_results['redundant_groups'].items():
                    group_features = group_info['features']

                    # Select best representative from group
                    best_feature = self._select_best_feature_from_group(
                        features_df[group_features], group_features
                    )

                    # Remove all other features from this group
                    redundant_in_group = [f for f in group_features if f != best_feature]
                    features_to_keep -= set(redundant_in_group)

                    results['selection_method'][best_feature] = {
                        'method': 'correlation_group_representative',
                        'group': group_name,
                        'group_size': len(group_features),
                        'replaced_features': redundant_in_group
                    }

                    results['removed_features'].extend(redundant_in_group)

            # 2. Handle VIF-based redundancy
            if vif_results.get('high_vif_features'):
                high_vif_features = vif_results['high_vif_features']

                # Sort by VIF score and remove highest VIF features
                vif_scores = vif_results.get('vif_scores', {})
                sorted_by_vif = sorted(
                    high_vif_features,
                    key=lambda f: vif_scores.get(f, float('inf')),
                    reverse=True
                )

                # Keep features with lower VIF
                # Remove top 25% of high VIF features, or at least 1
                num_to_remove = max(1, len(sorted_by_vif) // 4)
                features_to_remove = sorted_by_vif[:num_to_remove]

                features_to_keep -= set(features_to_remove)

                for feature in features_to_remove:
                    results['selection_method'][feature] = {
                        'method': 'high_vif_elimination',
                        'vif_score': vif_scores.get(feature, float('inf'))
                    }

                    results['removed_features'].append(feature)

            # 3. Create final selected features DataFrame
            final_features = features_df[list(features_to_keep)].copy()
            results['selected_features'] = final_features

            self.logger.info(f"✅ Selected {len(final_features.columns)} representative features")

            return results

        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return {'selected_features': features_df.copy(), 'selection_method': {}, 'removed_features': [], 'error': str(e)}

    def _select_best_feature_from_group(self,
                                    group_features: pd.DataFrame,
                                    feature_names: list[str]) -> str:
        """Select the best representative feature from a redundant group."""

        try:
            # Method 1: Highest variance (most informative)
            variances = group_features.var()
            best_by_variance = variances.idxmax()

            # Method 2: Lowest average correlation with others (most independent)
            correlation_matrix = group_features.corr().abs()
            avg_correlations = correlation_matrix.mean(axis=1)
            best_by_correlation = avg_correlations.idxmin()

            # Method 3: Simple heuristic - prefer features with 'close' or 'volume'
            priority_keywords = ['close', 'volume', 'high', 'low', 'open']
            best_by_keyword = None

            for keyword in priority_keywords:
                for feature_name in feature_names:
                    if keyword.lower() in feature_name.lower():
                        best_by_keyword = feature_name
                        break
                if best_by_keyword:
                    break

            # Decision: Prefer keyword > variance > correlation
            if best_by_keyword:
                return best_by_keyword
            elif best_by_variance in feature_names:
                return str(best_by_variance)
            else:
                return str(best_by_correlation)

        except Exception as e:
            self.logger.error(f"Error selecting best feature from group: {e}")
            return feature_names[0]  # Return first feature as fallback

    def _create_empty_result(self,
                           original_features: pd.DataFrame,
                           results: dict[str, Any]) -> dict[str, Any]:
        """Create result when all features are removed."""

        results.update({
            'selected_features': original_features.copy(),
            'selected_count': len(original_features.columns),
            'reduction_ratio': 0.0,
            'cleaned_features': original_features.copy()
        })

        return results

    def _log_redundancy_summary(self, results: dict[str, Any]) -> None:
        """Log comprehensive redundancy analysis summary."""

        original_count = results['original_count']
        selected_count = results['selected_count']
        redundant_count = len(results['redundant_features'])
        reduction_ratio = results['reduction_ratio']

        self.logger.info("=" * 60)
        self.logger.info("🔍 REDUNDANCY DETECTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Original Features: {original_count}")
        self.logger.info(f"Selected Features: {selected_count}")
        self.logger.info(f"Redundant Features: {redundant_count}")
        self.logger.info(f"Reduction Ratio: {reduction_ratio:.1f}%")

        # Low variance features
        if results['low_variance_features']:
            self.logger.info(f"🗑️ Low Variance: {len(results['low_variance_features'])}")
            self.logger.info(f"   {results['low_variance_features']}")

        # Correlation groups
        if results['correlation_groups']:
            self.logger.info(f"🔗 Correlation Groups: {len(results['correlation_groups'])}")
            for group_name, group_info in results['correlation_groups'].items():
                self.logger.info(f"   {group_name}: {group_info['size']} features "
                               f"(avg corr: {group_info['average_correlation']:.3f})")

        # VIF results
        if results['vif_results']:
            high_vif_count = len(results['vif_results'].get('high_vif_features', []))
            self.logger.info(f"📊 High VIF Features: {high_vif_count}")

            if high_vif_count > 0:
                vif_scores = results['vif_results'].get('vif_scores', {})
                high_vif_features = results['vif_results']['high_vif_features']

                for feature in high_vif_features[:5]:  # Show top 5
                    vif_score = vif_scores.get(feature, 0)
                    self.logger.info(f"   {feature}: VIF = {vif_score:.2f}")

        # Selection methods
        if results.get('selection_method'):
            self.logger.info("✅ Selection Methods Applied:")
            for feature, method_info in results['selection_method'].items():
                method = method_info['method']
                self.logger.info(f"   {feature}: {method}")

        self.logger.info("=" * 60)

    def get_redundancy_report(self,
                             hours: int = 24) -> dict[str, Any]:
        """Generate redundancy analysis report."""

        # This would load historical redundancy analysis results
        # For now, return current configuration
        return {
            'thresholds': self.thresholds,
            'methods_used': {
                'correlation_clustering': self.use_clustering,
                'vif_analysis': self.use_vif,
                'variance_filtering': True
            },
            'recommendations': [
                "Monitor redundancy ratios over time",
                "Adjust thresholds based on domain knowledge",
                "Consider feature importance when selecting representatives",
                "Validate redundancy elimination doesn't hurt model performance"
            ]
        }


# Factory function for easy instantiation
def get_redundancy_detector(config: dict[str, Any] | None = None) -> RedundancyDetector:
    """Factory function to get RedundancyDetector instance."""
    return RedundancyDetector(config)


# Convenience function for quick analysis
def eliminate_redundancy_quick(features_df: pd.DataFrame,
                               target_series: pd.Series | None = None,
                               config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Quick redundancy elimination.

    Args:
        features_df: Features DataFrame to analyze
        target_series: Target series (optional, for VIF calculation)
        config: Configuration dictionary

    Returns:
        Redundancy elimination result dictionary
    """
    detector = get_redundancy_detector(config)
    return detector.eliminate_redundant_features(features_df, target_series)
