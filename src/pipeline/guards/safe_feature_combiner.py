#!/usr/bin/env python3
"""
Safe Feature Combiner - Temporal-Safe Multi-Timeframe Feature Combination
Prevents temporal leakage when combining features from different timeframes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from src.core.logging.logger import ProjectLogger
from src.pipeline.guards.timeframe_alignment_guard import TimeframeAlignmentGuard

logger = ProjectLogger.get_logger("SafeFeatureCombiner")

class SafeFeatureCombiner:
    """
    Temporal-safe feature combination for multi-timeframe systems.
    
    This combiner ensures that features from different timeframes are combined
    without introducing temporal leakage - the most critical issue in multi-timeframe
    trading systems.
    
    Key features:
    - Validates temporal compatibility before combination
    - Adds timeframe prefixes to feature names
    - Preserves temporal structure
    - Handles missing data safely
    - Provides detailed combination reports
    """
    
    def __init__(self, alignment_guard: Optional[TimeframeAlignmentGuard] = None):
        """
        Initialize the SafeFeatureCombiner.
        
        Args:
            alignment_guard: TimeframeAlignmentGuard instance (creates if None)
        """
        self.alignment_guard = alignment_guard or TimeframeAlignmentGuard()
        self.logger = logger
    
    def combine_features_safe(self, 
                            features_by_tf: Dict[str, pd.DataFrame], 
                            current_time: pd.Timestamp) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Safely combine features from multiple timeframes with temporal validation.
        
        This is the main method that ensures safe feature combination.
        
        Args:
            features_by_tf: Dictionary of timeframes to feature DataFrames
            current_time: Current timestamp for validation
            
        Returns:
            Tuple of (combined_features_df, combination_report)
        """
        self.logger.info(f"🔗 Starting safe feature combination for {current_time}")
        
        # Step 1: Validate combination safety
        safety_result = self.alignment_guard.validate_feature_combination_safety(
            features_by_tf, current_time
        )
        
        if safety_result['status'] == 'unsafe':
            self.logger.error(f"❌ Feature combination unsafe: {safety_result['issues']}")
            return pd.DataFrame(), {
                'status': 'failed',
                'reason': 'unsafe_combination',
                'issues': safety_result['issues']
            }
        
        # Step 2: Filter to valid timeframes
        valid_timeframes = safety_result['valid_timeframes']
        valid_features = {tf: df for tf, df in features_by_tf.items() if tf in valid_timeframes}
        
        if len(valid_features) < 2:
            self.logger.warning(f"⚠️ Only {len(valid_features)} valid timeframe(s), skipping combination")
            # Return the single timeframe as-is
            if valid_features:
                tf, df = next(iter(valid_features.items()))
                return self._add_single_timeframe_prefixes(df, tf), {
                    'status': 'single_timeframe',
                    'timeframe': tf,
                    'shape': df.shape
                }
            else:
                return pd.DataFrame(), {
                    'status': 'failed',
                    'reason': 'no_valid_timeframes'
                }
        
        # Step 3: Prepare DataFrames for combination
        prepared_dfs = []
        combination_metadata = []
        
        for tf, df in valid_features.items():
            prepared_df, metadata = self._prepare_dataframe_for_combination(df, tf)
            prepared_dfs.append(prepared_df)
            combination_metadata.append(metadata)
        
        # Step 4: Combine DataFrames safely
        combined_df = self._combine_prepared_dataframes(prepared_dfs, valid_timeframes)
        
        # Step 5: Validate combined result
        validation_result = self._validate_combined_dataframe(combined_df, valid_timeframes)
        
        # Step 6: Generate combination report
        combination_report = self._generate_combination_report(
            valid_timeframes, combination_metadata, validation_result, current_time
        )
        
        self.logger.info(f"✅ Safe feature combination completed: {combined_df.shape}")
        
        return combined_df, combination_report
    
    def _prepare_dataframe_for_combination(self, 
                                        df: pd.DataFrame, 
                                        timeframe: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare a single timeframe DataFrame for combination.
        
        Args:
            df: DataFrame for single timeframe
            timeframe: Timeframe string (e.g., '15m', '1h', '1d')
            
        Returns:
            Tuple of (prepared_df, metadata)
        """
        df_prepared = df.copy()
        metadata = {
            'timeframe': timeframe,
            'original_shape': df.shape,
            'original_columns': list(df.columns)
        }
        
        # Ensure datetime column exists
        if 'datetime' not in df_prepared.columns:
            if isinstance(df_prepared.index, pd.DatetimeIndex):
                df_prepared = df_prepared.reset_index()
                df_prepared = df_prepared.rename(columns={'index': 'datetime'})
                metadata['datetime_source'] = 'index'
            else:
                raise ValueError(f"No datetime column or index found for timeframe {timeframe}")
        else:
            metadata['datetime_source'] = 'column'
        
        # Ensure datetime is properly typed
        df_prepared['datetime'] = pd.to_datetime(df_prepared['datetime'])
        
        # Sort by datetime to ensure proper temporal ordering
        df_prepared = df_prepared.sort_values('datetime')
        
        # Add timeframe prefixes to feature columns
        prefix = self._get_timeframe_prefix(timeframe)
        feature_cols = [col for col in df_prepared.columns 
                       if col not in ['datetime', 'ticker', 'interval'] and not col.startswith('target_')]
        
        # Rename feature columns with timeframe prefix
        rename_dict = {}
        for col in feature_cols:
            rename_dict[col] = f"{prefix}_{col}"
        
        df_prepared = df_prepared.rename(columns=rename_dict)
        
        # Update metadata
        metadata['prepared_shape'] = df_prepared.shape
        metadata['prepared_columns'] = list(df_prepared.columns)
        metadata['prefix'] = prefix
        metadata['renamed_columns'] = len(rename_dict)
        
        self.logger.debug(f"📝 Prepared {timeframe} DataFrame: {df_prepared.shape} with prefix '{prefix}'")
        
        return df_prepared, metadata
    
    def _get_timeframe_prefix(self, timeframe: str) -> str:
        """Get standardized prefix for timeframe."""
        prefix_map = {
            '15m': 'm15',
            '60m': 'h1', 
            '1d': 'd1'
        }
        return prefix_map.get(timeframe, timeframe.replace('m', 'm').replace('h', 'h').replace('d', 'd'))
    
    def _combine_prepared_dataframes(self, 
                                  prepared_dfs: List[pd.DataFrame], 
                                  timeframes: List[str]) -> pd.DataFrame:
        """
        Combine prepared DataFrames using temporal-safe merge.
        
        Args:
            prepared_dfs: List of prepared DataFrames
            timeframes: List of corresponding timeframes
            
        Returns:
            Combined DataFrame
        """
        self.logger.info(f"🔗 Combining {len(prepared_dfs)} DataFrames")
        
        # Start with the first DataFrame
        combined = prepared_dfs[0].copy()
        
        # Merge with remaining DataFrames
        for i, df in enumerate(prepared_dfs[1:], 1):
            timeframe = timeframes[i]
            
            # Use outer merge to preserve all timestamps
            merge_cols = ['datetime']
            if 'ticker' in combined.columns and 'ticker' in df.columns:
                merge_cols.append('ticker')
            
            combined = pd.merge(
                combined, 
                df, 
                on=merge_cols, 
                how='outer',
                suffixes=('', f'_{timeframe}_dup')
            )
            
            # Remove duplicate columns if any
            dup_cols = [col for col in combined.columns if col.endswith('_dup')]
            if dup_cols:
                combined.drop(columns=dup_cols, inplace=True)
                self.logger.debug(f"🗑️ Removed {len(dup_cols)} duplicate columns from {timeframe} merge")
        
        # Sort by datetime (and ticker if exists)
        sort_cols = ['datetime']
        if 'ticker' in combined.columns:
            sort_cols.append('ticker')
        
        combined = combined.sort_values(sort_cols)
        
        # Reset index for clean DataFrame
        combined = combined.reset_index(drop=True)
        
        self.logger.info(f"✅ Combined DataFrame shape: {combined.shape}")
        
        return combined
    
    def _validate_combined_dataframe(self, 
                                  combined_df: pd.DataFrame, 
                                  timeframes: List[str]) -> Dict[str, Any]:
        """
        Validate the combined DataFrame for integrity.
        
        Args:
            combined_df: Combined DataFrame
            timeframes: List of timeframes that were combined
            
        Returns:
            Validation result
        """
        validation_result = {
            'status': 'valid',
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check if DataFrame is empty
        if combined_df.empty:
            validation_result['status'] = 'invalid'
            validation_result['issues'].append("Combined DataFrame is empty")
            return validation_result
        
        # Check datetime column
        if 'datetime' not in combined_df.columns:
            validation_result['status'] = 'invalid'
            validation_result['issues'].append("Missing datetime column in combined DataFrame")
            return validation_result
        
        # Check for NaN values
        nan_counts = combined_df.isnull().sum()
        high_nan_cols = nan_counts[nan_counts > len(combined_df) * 0.5].index.tolist()
        
        if high_nan_cols:
            validation_result['warnings'].append(
                f"High NaN columns (>50%): {high_nan_cols}"
            )
        
        # Check timeframe prefixes
        expected_prefixes = [self._get_timeframe_prefix(tf) for tf in timeframes]
        prefix_counts = {}
        
        for prefix in expected_prefixes:
            prefix_cols = [col for col in combined_df.columns if col.startswith(f"{prefix}_")]
            prefix_counts[prefix] = len(prefix_cols)
        
        validation_result['statistics'] = {
            'total_rows': len(combined_df),
            'total_columns': len(combined_df.columns),
            'timeframe_prefix_counts': prefix_counts,
            'nan_percentage': (combined_df.isnull().sum().sum() / (len(combined_df) * len(combined_df.columns))) * 100,
            'date_range': {
                'start': combined_df['datetime'].min(),
                'end': combined_df['datetime'].max()
            }
        }
        
        # Check temporal ordering
        if not combined_df['datetime'].is_monotonic_increasing:
            validation_result['warnings'].append("DataFrame is not temporally sorted")
        
        return validation_result
    
    def _add_single_timeframe_prefixes(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Add prefixes to single timeframe DataFrame (when no combination needed).
        
        Args:
            df: Single timeframe DataFrame
            timeframe: Timeframe string
            
        Returns:
            DataFrame with prefixes added
        """
        df_prefixed = df.copy()
        prefix = self._get_timeframe_prefix(timeframe)
        
        # Add prefixes to feature columns
        feature_cols = [col for col in df_prefixed.columns 
                       if col not in ['datetime', 'ticker', 'interval'] and not col.startswith('target_')]
        
        rename_dict = {}
        for col in feature_cols:
            rename_dict[col] = f"{prefix}_{col}"
        
        df_prefixed = df_prefixed.rename(columns=rename_dict)
        
        return df_prefixed
    
    def _generate_combination_report(self, 
                                  timeframes: List[str],
                                  metadata_list: List[Dict[str, Any]],
                                  validation_result: Dict[str, Any],
                                  current_time: pd.Timestamp) -> Dict[str, Any]:
        """
        Generate comprehensive combination report.
        
        Args:
            timeframes: List of combined timeframes
            metadata_list: List of metadata for each timeframe
            validation_result: Validation result
            current_time: Current timestamp
            
        Returns:
            Combination report
        """
        report = {
            'status': 'success',
            'current_time': current_time,
            'combined_timeframes': timeframes,
            'timeframe_count': len(timeframes),
            'combination_metadata': metadata_list,
            'validation': validation_result,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Add summary statistics
        total_original_rows = sum(meta['original_shape'][0] for meta in metadata_list)
        total_original_cols = sum(meta['original_shape'][1] for meta in metadata_list)
        total_renamed_cols = sum(meta['renamed_columns'] for meta in metadata_list)
        
        report['summary'] = {
            'total_original_rows': total_original_rows,
            'total_original_columns': total_original_cols,
            'total_renamed_columns': total_renamed_cols,
            'final_shape': validation_result['statistics'].get('total_rows', 0),
            'final_columns': validation_result['statistics'].get('total_columns', 0)
        }
        
        return report
    
    def get_combination_preview(self, 
                               features_by_tf: Dict[str, pd.DataFrame],
                               current_time: pd.Timestamp) -> Dict[str, Any]:
        """
        Get a preview of what the combination would look like without actually combining.
        
        Useful for debugging and planning.
        
        Args:
            features_by_tf: Dictionary of timeframes to DataFrames
            current_time: Current timestamp
            
        Returns:
            Preview information
        """
        # Validate safety first
        safety_result = self.alignment_guard.validate_feature_combination_safety(
            features_by_tf, current_time
        )
        
        preview = {
            'safety_status': safety_result['status'],
            'valid_timeframes': safety_result['valid_timeframes'],
            'issues': safety_result.get('issues', []),
            'timeframe_info': {}
        }
        
        # Add information about each timeframe
        for tf, df in features_by_tf.items():
            if tf in safety_result['valid_timeframes']:
                prefix = self._get_timeframe_prefix(tf)
                feature_cols = [col for col in df.columns 
                               if col not in ['datetime', 'ticker', 'interval'] and not col.startswith('target_')]
                
                preview['timeframe_info'][tf] = {
                    'shape': df.shape,
                    'prefix': prefix,
                    'feature_count': len(feature_cols),
                    'sample_features': feature_cols[:5]  # First 5 features
                }
        
        return preview


# Factory function for easy instantiation
def get_safe_feature_combiner(alignment_guard: Optional[TimeframeAlignmentGuard] = None) -> SafeFeatureCombiner:
    """Factory function to get SafeFeatureCombiner instance."""
    return SafeFeatureCombiner(alignment_guard)


# Convenience function for quick combination
def combine_timeframes_quick(features_by_tf: Dict[str, pd.DataFrame], 
                           current_time: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Quick combination function with temporal validation.
    
    Args:
        features_by_tf: Dictionary of timeframes to DataFrames
        current_time: Current timestamp (uses now if None)
        
    Returns:
        Tuple of (combined_features_df, combination_report)
    """
    combiner = get_safe_feature_combiner()
    if current_time is None:
        current_time = pd.Timestamp.now()
    
    return combiner.combine_features_safe(features_by_tf, current_time)
