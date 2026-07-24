"""
Data Preparation Service for Stage 5 Prediction.

Handles data preparation, validation, and ticker-specific data processing.
Extracted from stage_5_prediction.py to reduce coupling and improve testability.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.pipeline.stages.prediction.lineage import (
    apply_lineage_attrs,
    source_lineage_attrs,
)


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
        kwargs: dict[str, Any],
        model_resolver
    ) -> tuple[pd.DataFrame | None, dict[str, Any], str]:
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

    def _extract_features_df(self, kwargs: dict[str, Any]) -> pd.DataFrame | None:
        """Extract features DataFrame from kwargs with fallback keys."""
        return next(
            (kwargs[k] for k in ('features_data', 'features_df', 'enriched_data')
             if k in kwargs and kwargs[k] is not None),
            None
        )

    def _extract_models_meta(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract models metadata from kwargs with fallback keys."""
        return kwargs.get('models_metadata') or kwargs.get('models_meta', {})

    def _validate_inputs(self, features_df: pd.DataFrame | None, models_meta: dict[str, Any]) -> bool:
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
    ) -> pd.DataFrame | None:
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
        lineage_attrs = source_lineage_attrs(ticker_df_clean)

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

        # Keep missing numeric values visible. Context-specific filtering below
        # decides whether a row is safe to send to a model.
        numeric_cols = [c for c in ticker_df_clean.columns if c not in preserved_cols]
        if numeric_cols:
            ticker_df_clean[numeric_cols] = ticker_df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Restore preserved columns
        for c in preserved_data.columns:
            ticker_df_clean[c] = preserved_data[c]

        return apply_lineage_attrs(ticker_df_clean, lineage_attrs)

    def prepare_context_data(
        self,
        context_id: str,
        meta: dict[str, Any],
        features_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list] | None:
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
        lineage_attrs = dict(ticker_df_clean.attrs)

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
        
        # Robustly exclude any remaining metadata columns like 'hash' from being expected
        metadata_cols = {'ticker', 'datetime', 'date', 'interval', 'timeframe', 'hash', 'symbol'}
        selected_features = [f for f in selected_features if f not in metadata_cols]

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

        model_feature_cols = [c for c in ticker_df_clean_features.columns if c not in context_cols]
        ticker_df_clean_features = self._drop_incomplete_model_rows(
            ticker_df_clean_features,
            model_feature_cols,
            context_id,
        )
        if ticker_df_clean_features is None:
            return None

        # Restore context columns
        for c in context_data.columns:
            ticker_df_clean_features[c] = context_data.reindex(ticker_df_clean_features.index)[c]

        return (
            apply_lineage_attrs(
                ticker_df_clean_features,
                lineage_attrs,
            ),
            selected_features,
        )

    def _drop_incomplete_model_rows(
        self,
        ticker_df: pd.DataFrame,
        model_feature_cols: list[str],
        context_id: str
    ) -> pd.DataFrame | None:
        """Drop rows with unavailable model inputs instead of fabricating zeros.

        Zero-filling a missing technical-indicator value (e.g. RSI, SMA)
        feeds the model a real, in-range number that looks like legitimate
        data -- the model has no way to know it's fabricated, so it can
        produce a confident, silently wrong prediction. Dropping the row is
        the honest choice: no prediction is safer than a wrong one that
        looks fine. (This function's own name and docstring already said
        this was the intent; the implementation was doing the opposite --
        always filling zeros and never dropping a row.)
        """
        if not model_feature_cols:
            return ticker_df

        complete_rows = ticker_df[model_feature_cols].notna().all(axis=1)
        if complete_rows.all():
            return ticker_df

        dropped = int((~complete_rows).sum())
        self.logger.warning(
            f'Context {context_id} has {dropped} incomplete feature row(s); '
            'dropping them rather than fabricating zeros.'
        )
        ticker_df = ticker_df[complete_rows].copy()

        if ticker_df.empty:
            self.logger.error(
                f'Context {context_id} has no data after dropping incomplete rows; skipping prediction.'
            )
            return None

        return ticker_df

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
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error creating context fingerprint: {e}", exc_info=True)
            return 'unknown_context'
