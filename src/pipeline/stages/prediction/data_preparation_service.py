"""
Data Preparation Service for Stage 5 Prediction.

Handles data preparation, validation, and ticker-specific data processing.
Extracted from stage_5_prediction.py to reduce coupling and improve testability.
"""
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
from src.core.logging.logger import ProjectLogger


class DataPreparationService:
    """
    Service for preparing and validating data for prediction.
    
    Responsibilities:
    - Input validation
    - Ticker-specific data extraction and preparation
    - Feature filtering and preservation of context columns
    - Data type conversion and cleaning
    """

    def __init__(self):
        self.logger = ProjectLogger.get_logger('DataPreparationService')

    def prepare_inputs(
        self,
        kwargs: Dict[str, Any],
        model_resolver
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], str]:
        """
        Prepare and validate inputs for prediction.
        
        Args:
            kwargs: Pipeline data dict with features_data and models_metadata
            model_resolver: ModelResolver instance for loading models from disk
            
        Returns:
            Tuple of (features_df, models_meta, market_regime)
        """
        features_df = self._extract_features_df(kwargs)
        models_meta = self._extract_models_meta(kwargs)
        market_regime = kwargs.get('market_regime', 'neutral')
        
        # Load models from disk if not provided
        if not models_meta:
            models_meta = model_resolver.load_models_metadata_from_disk(kwargs)
            if not models_meta:
                self.logger.warning('Failed to load models_metadata from disk')
                return None, {}, market_regime
            self.logger.info(f'Loaded {len(models_meta)} models from disk')
        
        is_valid = self._validate_inputs(features_df, models_meta)
        if not is_valid:
            return None, {}, market_regime
        
        if isinstance(features_df, pd.DataFrame):
            from src.features.utils.datetime_utils import normalize_metadata_columns
            features_df = normalize_metadata_columns(features_df)
            self.logger.info('Normalized features_df at stage entry')
        
        return features_df, models_meta, market_regime

    def _extract_features_df(self, kwargs: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Extract features DataFrame from kwargs with fallback keys."""
        return next(
            (kwargs[k] for k in ('features_data', 'features_df', 'enriched_data') 
             if k in kwargs and kwargs[k] is not None),
            None
        )

    def _extract_models_meta(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract models metadata from kwargs with fallback keys."""
        return kwargs.get('models_metadata') or kwargs.get('models_meta', {})

    def _validate_inputs(self, features_df: Optional[pd.DataFrame], models_meta: Dict[str, Any]) -> bool:
        """Validate that required inputs are present and non-empty."""
        if features_df is None or features_df.empty or not models_meta:
            self.logger.warning('Required features or model metadata not found. Skipping Stage 5.')
            self.logger.warning(f'  - features_df is None: {features_df is None}')
            self.logger.warning(
                f"  - features_df empty: {features_df.empty if features_df is not None else 'N/A'}"
            )
            self.logger.warning(f'  - models_meta empty: {not models_meta}')
            return False
        return True

    def prepare_ticker_data(
        self,
        features_df: pd.DataFrame,
        ticker: str
    ) -> Optional[pd.DataFrame]:
        """
        Prepare ticker-specific data for prediction.
        
        Args:
            features_df: Full features DataFrame
            ticker: Ticker symbol to extract
            
        Returns:
            Prepared DataFrame for the ticker, or None if no data
        """
        ticker_df = features_df[features_df['ticker'] == ticker].tail(50)
        if ticker_df.empty:
            self.logger.warning(f'⚠️ No data for ticker {ticker}')
            return None
        
        ticker_df_clean = ticker_df.copy()
        
        # Preserve context columns before numeric conversion
        preserved_cols = [
            'context_fingerprint',
            'context_pattern_id',
            'context_pattern_seq',
            'state_champion',
            'context_velocity',
        ]
        preserved_data = ticker_df_clean[
            [c for c in preserved_cols if c in ticker_df_clean.columns]
        ].copy()
        
        # Remove metadata columns
        metadata_cols = ['ticker', 'datetime', 'date', 'interval', 'timeframe', 'hash', 'symbol']
        ticker_df_clean = ticker_df_clean.drop(
            columns=[c for c in metadata_cols if c in ticker_df_clean.columns],
            errors='ignore'
        )
        
        # Convert to numeric, skip preserved columns
        for col in ticker_df_clean.columns:
            if col in preserved_cols:
                continue
            try:
                ticker_df_clean[col] = pd.to_numeric(ticker_df_clean[col], errors='coerce')
            except (ValueError, TypeError):
                ticker_df_clean = ticker_df_clean.drop(columns=[col], errors='ignore')
        
        # Fill NaN and inf
        ticker_df_clean = ticker_df_clean.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Restore preserved columns
        for c in preserved_data.columns:
            ticker_df_clean[c] = preserved_data[c]
        
        return ticker_df_clean

    def prepare_context_data(
        self,
        context_id: str,
        meta: Dict[str, Any],
        features_df: pd.DataFrame
    ) -> Optional[Tuple[pd.DataFrame, list]]:
        """
        Prepare context-specific data for prediction.
        
        Args:
            context_id: Context identifier
            meta: Model metadata
            features_df: Full features DataFrame
            
        Returns:
            Tuple of (ticker_df_clean, selected_features) or None if preparation fails
        """
        ticker = meta.get('ticker')
        if not ticker:
            self.logger.error(f'No ticker found in metadata for context {context_id}')
            return None
        
        self.logger.info(f'🔍 Processing context: {context_id}')
        
        ticker_df_clean = self.prepare_ticker_data(features_df, ticker)
        if ticker_df_clean is None:
            return None
        
        # Preserve critical context columns before filtering
        context_cols = [
            'context_fingerprint',
            'context_pattern_id',
            'context_pattern_seq',
            'state_champion',
            'context_velocity',
        ]
        context_data = ticker_df_clean[
            [c for c in context_cols if c in ticker_df_clean.columns]
        ].copy()
        
        selected_features = meta.get('selected_features', [])
        
        # Check for missing features
        missing_features = [f for f in selected_features if f not in ticker_df_clean.columns]
        if missing_features:
            self.logger.error(
                f'Context {context_id} missing {len(missing_features)} selected features; '
                f'skipping prediction instead of filling zeros.'
            )
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'Missing features for {context_id}: {missing_features}')
            return None
        
        if selected_features:
            ticker_df_clean_features = ticker_df_clean[selected_features].copy()
        else:
            ticker_df_clean_features = ticker_df_clean.copy()
        
        # Restore context columns
        for c in context_data.columns:
            ticker_df_clean_features[c] = context_data[c]
        
        return ticker_df_clean_features, selected_features

    def create_context_fingerprint(
        self,
        ticker_df: pd.DataFrame,
        market_regime: str
    ) -> str:
        """
        Create a context fingerprint using context_pattern_id.
        
        Args:
            ticker_df: Ticker DataFrame
            market_regime: Market regime string
            
        Returns:
            Context fingerprint string
        """
        if 'context_pattern_id' in ticker_df.columns and len(ticker_df) > 0:
            return str(ticker_df['context_pattern_id'].iloc[-1])
        
        # Fallback to legacy logic
        try:
            regime_map = {'bull': 1, 'bear': -1, 'sideways': 0, 'volatile': 2}
            regime_val = regime_map.get(market_regime.lower(), 0)
            return f"legacy_{regime_val}"
        except Exception as e:
            self.logger.error(f"Error creating context fingerprint: {e}", exc_info=True)
            return 'unknown_context'
