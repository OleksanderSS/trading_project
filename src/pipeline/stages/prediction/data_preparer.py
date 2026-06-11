import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger
from src.utils.artifact_security import resolve_trusted_artifact_path


class PredictionDataPreparer:
    """Prepares stage 5 inputs, context data, and target scaler lookup."""
    ACCUMULATION_OUTPUT_DIR_CONFIG = 'system.accumulation.output_dir'
    DEFAULT_ACCUMULATION_DIR = 'data/colab/accumulated'

    def __init__(self, config_manager: Any):
        self.logger = ProjectLogger.get_logger('PredictionDataPreparer')
        self.config_manager = config_manager

    def prepare_inputs(self, kwargs: dict[str, Any], model_resolver: Any
        ) ->tuple[pd.DataFrame | None, dict[str, Any], str]:
        features_df = next((kwargs[k] for k in ('features_data',
            'features_df', 'enriched_data') if k in kwargs and kwargs[k] is not
            None), None)
        models_meta = kwargs.get('models_metadata') or kwargs.get('models_meta'
            , {})
        market_regime = kwargs.get('market_regime', 'neutral')
        if not models_meta:
            models_meta = model_resolver.load_models_metadata_from_disk(kwargs)
            if not models_meta:
                self.logger.warning('Failed to load models_metadata from disk')
                return None, {}, market_regime
            self.logger.info(f'Loaded {len(models_meta)} models from disk')
        if isinstance(features_df, pd.DataFrame):
            features_df = self._normalize_metadata_columns(features_df)
            self.logger.info('Normalized features_df at stage entry')
        is_valid, _ = self.validate_inputs(features_df, models_meta)
        if not is_valid:
            return None, {}, market_regime
        return features_df, models_meta, market_regime

    def validate_inputs(self, features_df: Any, models_meta: dict[str, Any]
        ) ->tuple[bool, str]:
        if features_df is None or getattr(features_df, 'empty', False
            ) or not models_meta:
            self.logger.warning(
                'Required features or model metadata not found. Skipping Stage 5.'
                )
            self.logger.warning(
                f'  - features_df is None: {features_df is None}')
            self.logger.warning(
                f"  - features_df empty: {features_df.empty if features_df is not None else 'N/A'}"
                )
            self.logger.warning(f'  - models_meta empty: {not models_meta}')
            return False, 'Invalid inputs'
        return True, 'Valid inputs'

    def get_available_model_types(self) ->set[str]:
        try:
            base_dir = Path(self.config_manager.get(self.
                ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR))
            batch_dir = base_dir / 'main_database'
            if not batch_dir.exists():
                self.logger.warning(f'Model directory not found: {batch_dir}')
                return {'mlp', 'tabnet'}
            model_types = set()
            if list(batch_dir.glob('*.pkl')):
                model_types.add('mlp')
            if list(batch_dir.glob('*.zip')):
                model_types.add('tabnet')
            if list(batch_dir.glob('*.keras')):
                model_types.update(['cnn', 'lstm', 'gru', 'transformer',
                    'autoencoder'])
            self.logger.info(f'Available model types: {sorted(model_types)}')
            return model_types if model_types else {'mlp', 'tabnet'}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error scanning model types: {e}', exc_info=True)
            raise DataProcessingError(f"Failed to scan model types: {e}") from e

    def _handle_missing_features(self, context_id: str, ticker_df_clean: pd.DataFrame,
                                   selected_features: list[str]) -> pd.DataFrame | None:
        """Handle missing features through adaptive re-enrichment."""
        missing_features = [f for f in selected_features if f not in ticker_df_clean.columns]
        if missing_features:
            self.logger.info(
                f'⚠️ Context {context_id} missing {len(missing_features)} features. Attempting adaptive re-enrichment...'
            )
            ticker_df_clean = self.adaptive_re_enrichment(ticker_df_clean, missing_features)
            still_missing = [f for f in selected_features if f not in ticker_df_clean.columns]
            if still_missing:
                self.logger.error(
                    f'Context {context_id} still missing {len(still_missing)} selected features after re-enrichment; skipping prediction.'
                )
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'Remaining missing features for {context_id}: {still_missing}')
                return None
        return ticker_df_clean

    def _filter_features_for_model(self, ticker_df_clean: pd.DataFrame, selected_features: list[str],
                                    model_type: str, context_id: str) -> tuple[pd.DataFrame, list[str]] | None:
        """Filter features for the model and drop incomplete rows."""
        filtered_features_list = selected_features
        if selected_features:
            self.logger.info(f'✅ Using {len(filtered_features_list)} selected features for prediction')
        else:
            self.logger.warning('⚠️ No selected features specified in metadata')

        existing_cols = [c for c in filtered_features_list if c in ticker_df_clean.columns]
        if existing_cols:
            ticker_df_clean = ticker_df_clean[existing_cols]
            ticker_df_clean = self._drop_incomplete_model_rows(ticker_df_clean, existing_cols, context_id)
            if ticker_df_clean is None:
                return None
            self.logger.info(f' Using {len(existing_cols)} features for {model_type}')
        else:
            self.logger.warning(
                f' No selected features for {model_type}, using all {ticker_df_clean.shape[1]} columns'
            )
            filtered_features_list = ticker_df_clean.columns.tolist()

        if ticker_df_clean.empty:
            return None
        return ticker_df_clean, filtered_features_list

    def prepare_context_data(  # noqa: C901
        self, context_id: str, meta: dict[str, Any],
        features_df: pd.DataFrame) ->(tuple[pd.DataFrame, list[str]] | None):
        ticker = meta.get('ticker')
        if not ticker:
            self.logger.error(
                f'No ticker found in metadata for context {context_id}')
            return None
        target_col = meta.get('target', '')
        model_type = meta.get('model_type', '')
        self.logger.info(f'🔍 Processing context: {context_id}')
        self.logger.info(
            f'   ticker={ticker}, target={target_col}, model_type={model_type}'
            )
        ticker_df_clean = self.prepare_ticker_data(features_df, ticker)
        if ticker_df_clean is None:
            return None
        selected_features = meta.get('selected_features', [])
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f'🔍 Stage 5: selected_features from metadata: {len(selected_features)} features'
                )
        missing_features = [f for f in selected_features if f not in
            ticker_df_clean.columns]
        if missing_features:
            self.logger.info(
                f'⚠️ Context {context_id} missing {len(missing_features)} features. Attempting adaptive re-enrichment...'
                )
            ticker_df_clean = self.adaptive_re_enrichment(ticker_df_clean,
                missing_features)
            still_missing = [f for f in selected_features if f not in
                ticker_df_clean.columns]
            if still_missing:
                self.logger.error(
                    f'Context {context_id} still missing {len(still_missing)} selected features after re-enrichment; skipping prediction.'
                )
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f'Remaining missing features for {context_id}: {still_missing}'
                    )
                return None
        filtered_features_list = selected_features
        if selected_features:
            self.logger.info(
                f'✅ Using {len(filtered_features_list)} selected features for prediction'
                )
        else:
            self.logger.warning('⚠️ No selected features specified in metadata'
                )
        existing_cols = [c for c in filtered_features_list if c in
            ticker_df_clean.columns]
        if existing_cols:
            ticker_df_clean = ticker_df_clean[existing_cols]
            ticker_df_clean = self._drop_incomplete_model_rows(
                ticker_df_clean, existing_cols, context_id)
            if ticker_df_clean is None:
                return None
            self.logger.info(
                f' Using {len(existing_cols)} features for {model_type}')
        else:
            self.logger.warning(
                f' No selected features for {model_type}, using all {ticker_df_clean.shape[1]} columns'
                )
            filtered_features_list = ticker_df_clean.columns.tolist()
        if ticker_df_clean.empty:
            return None
        return ticker_df_clean, filtered_features_list

    def _drop_incomplete_model_rows(self, ticker_df: pd.DataFrame,
        model_feature_cols: list[str], context_id: str) ->pd.DataFrame | None:
        if not model_feature_cols:
            return ticker_df
        complete_rows = ticker_df[model_feature_cols].notna().all(axis=1)
        if complete_rows.all():
            return ticker_df
        dropped = int((~complete_rows).sum())
        self.logger.warning(
            f'Context {context_id} has {dropped} incomplete feature row(s); dropping instead of filling zeros.'
            )
        filtered = ticker_df.loc[complete_rows].copy()
        if filtered.empty:
            self.logger.error(
                f'Context {context_id} has no complete feature rows; skipping prediction.'
                )
            return None
        return filtered

    def adaptive_re_enrichment(self, df: pd.DataFrame, missing_features:
        list[str]) ->pd.DataFrame:
        try:
            from src.features.feature_orchestrator import FeatureOrchestrator
            required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_ohlcv):
                self.logger.warning(
                    f'Cannot re-enrich: missing OHLCV columns. Available: {df.columns.tolist()[:10]}'
                    )
                return df
            orchestrator = FeatureOrchestrator.create_from_config(self.
                config_manager)
            enriched_df = orchestrator.run(df)
            new_features_found = [f for f in missing_features if f in
                enriched_df.columns]
            if new_features_found:
                self.logger.info(
                    f'✅ Successfully recovered {len(new_features_found)} missing features via adaptive enrichment'
                    )
                return enriched_df
            self.logger.warning(
                '⚠️ Adaptive enrichment completed but missing features were not recovered.'
                )
            return enriched_df
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'❌ Adaptive re-enrichment failed: {e}', exc_info=True)
            return df

    def _set_datetime_index(self, ticker_df_clean: pd.DataFrame) -> None:
        """Set datetime column as index if available."""
        for dt_col in ['datetime', 'date']:
            if dt_col in ticker_df_clean.columns:
                try:
                    ticker_df_clean.index = pd.to_datetime(ticker_df_clean[dt_col])
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f'Moved {dt_col} to index for timestamp preservation')
                    break
                except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                    self.logger.error(f'Failed to move {dt_col} to index: {e}', exc_info=True)

    def _preserve_metadata_columns(self, ticker_df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Preserve metadata columns and return cleaned df and preserved data."""
        metadata_cols = ['ticker', 'datetime', 'date', 'interval', 'timeframe', 'hash', 'symbol', '_cache_ticker', '_cache_date', '_cache_config_hash', 'is_significant']
        preserved_cols = ['context_fingerprint', 'context_pattern_id', 'context_pattern_seq', 'state_champion', 'context_velocity']
        preserved_data = ticker_df_clean[[c for c in preserved_cols if c in ticker_df_clean.columns]].copy()
        ticker_df_clean = ticker_df_clean.drop(columns=[c for c in metadata_cols if c in ticker_df_clean.columns], errors='ignore')
        return ticker_df_clean, preserved_data

    def _convert_to_numeric(self, ticker_df_clean: pd.DataFrame, preserved_cols: list[str]) -> pd.DataFrame:
        """Convert columns to numeric, dropping non-convertible ones."""
        obj_cols = ticker_df_clean.select_dtypes(exclude=['number']).columns.tolist()
        if obj_cols:
            ticker_df_clean = ticker_df_clean.drop(columns=[c for c in obj_cols if c not in preserved_cols], errors='ignore')
        for col in ticker_df_clean.columns:
            if col in preserved_cols:
                continue
            try:
                ticker_df_clean[col] = pd.to_numeric(ticker_df_clean[col], errors='coerce')
            except (ValueError, TypeError) as e:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'Failed to convert column {col} to numeric: {e}')
                ticker_df_clean = ticker_df_clean.drop(columns=[col], errors='ignore')
        return ticker_df_clean

    def _handle_infinite_values(self, ticker_df_clean: pd.DataFrame, preserved_cols: list[str]) -> None:
        """Replace infinite values with NaN."""
        numeric_cols = [c for c in ticker_df_clean.columns if c not in preserved_cols]
        if numeric_cols:
            ticker_df_clean[numeric_cols] = ticker_df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)

    def _validate_numeric_data(self, ticker_df_clean: pd.DataFrame, ticker: str, preserved_cols: list[str]) -> bool:
        """Validate that data is numeric and return True if valid."""
        numeric_check_cols = [c for c in ticker_df_clean.columns if c not in preserved_cols]
        has_non_numeric = ticker_df_clean[numeric_check_cols].dtypes.apply(lambda x: x.kind not in 'biufc').any() if numeric_check_cols else False
        if ticker_df_clean.empty or has_non_numeric:
            self.logger.warning(f'⚠️ Data for {ticker} contains non-numeric columns, skipping')
            return False
        return True

    def prepare_ticker_data(self, features_df: pd.DataFrame, ticker: str) ->(pd.DataFrame | None):
        ticker_df = features_df[features_df['ticker'] == ticker].tail(250)
        if ticker_df.empty:
            self.logger.warning(f'⚠️ No data for ticker {ticker}')
            return None
        ticker_df_clean = ticker_df.copy()

        self._set_datetime_index(ticker_df_clean)

        preserved_cols = ['context_fingerprint', 'context_pattern_id', 'context_pattern_seq', 'state_champion', 'context_velocity']
        ticker_df_clean, preserved_data = self._preserve_metadata_columns(ticker_df_clean)

        ticker_df_clean = self._convert_to_numeric(ticker_df_clean, preserved_cols)
        self._handle_infinite_values(ticker_df_clean, preserved_cols)

        for c in preserved_data.columns:
            ticker_df_clean[c] = preserved_data[c]

        if not self._validate_numeric_data(ticker_df_clean, ticker, preserved_cols):
            return None
        return ticker_df_clean

    def _get_scaler_paths(self, meta: dict[str, Any]) -> tuple[Path, Path, str]:
        """Get base directory, scalers directory, and scaler filename from metadata."""
        ticker = meta.get('ticker', '')
        target_col = meta.get('target', '')
        base_dir = Path(self.config_manager.get(self.ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR))
        scalers_dir = Path(self.config_manager.get('paths.scalers', 'data/scalers'))
        scaler_filename = f'scaler_{ticker}_{target_col}.pkl'
        return base_dir, scalers_dir, scaler_filename

    def _try_scaler_in_scalers_dir(self, scalers_dir: Path, scaler_filename: str) -> Any | None:
        """Try loading scaler from scalers directory."""
        scaler_path = scalers_dir / scaler_filename
        if scaler_path.exists():
            return self._try_load_scaler(scaler_path)
        return None

    def _try_scaler_in_model_path(self, model_path_str: str, scaler_filename: str) -> Any | None:
        """Try loading scaler from model path directory."""
        if not model_path_str:
            return None
        model_path = Path(model_path_str.replace('/', os.sep))
        scaler_path = model_path.parent / scaler_filename
        if scaler_path.exists():
            return self._try_load_scaler(scaler_path)
        return None

    def _try_scaler_in_batch_dirs(self, base_dir: Path, scaler_filename: str) -> Any | None:
        """Try loading scaler from known batch directories."""
        for batch_name in ['main_database', 'test_ticker_AAPL_target_return_1d']:
            scaler_path = base_dir / batch_name / scaler_filename
            if scaler_path.exists():
                return self._try_load_scaler(scaler_path)
        return None

    def _try_scaler_from_model_parts(self, model_path_str: str, base_dir: Path, scaler_filename: str) -> Any | None:
        """Try loading scaler by extracting batch name from model path."""
        if not model_path_str:
            return None
        parts = model_path_str.replace('/', os.sep).split(os.sep)
        if 'models' in parts:
            models_idx = parts.index('models')
            if models_idx > 0:
                batch_name = parts[models_idx - 1]
                scaler_path = base_dir / batch_name / scaler_filename
                if scaler_path.exists():
                    return self._try_load_scaler(scaler_path)
        return None

    def _try_scaler_recursive_search(self, base_dir: Path, scaler_filename: str) -> Any | None:
        """Try loading scaler by recursively searching base directory."""
        try:
            for candidate in base_dir.rglob(scaler_filename):
                result = self._try_load_scaler(candidate)
                if result is not None:
                    return result
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'rglob for {scaler_filename} failed: {e}', exc_info=True)
            raise DataProcessingError(f"Failed to scan for scaler {scaler_filename}") from e
        return None

    def load_target_scaler(self, meta: dict[str, Any]) ->(Any | None):
        ticker = meta.get('ticker', '')
        target_col = meta.get('target', '')
        model_path_str = meta.get('model_path', '')

        base_dir, scalers_dir, scaler_filename = self._get_scaler_paths(meta)

        # Try scalers directory first
        result = self._try_scaler_in_scalers_dir(scalers_dir, scaler_filename)
        if result is not None:
            return result

        # Try model path directory
        result = self._try_scaler_in_model_path(model_path_str, scaler_filename)
        if result is not None:
            return result

        # Try known batch directories
        result = self._try_scaler_in_batch_dirs(base_dir, scaler_filename)
        if result is not None:
            return result

        # Try extracting batch name from model path
        result = self._try_scaler_from_model_parts(model_path_str, base_dir, scaler_filename)
        if result is not None:
            return result

        # Try recursive search
        result = self._try_scaler_recursive_search(base_dir, scaler_filename)
        if result is not None:
            return result

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'No target scaler found for {ticker}_{target_col}')
        return None

    def _try_load_scaler(self, scaler_path: Path) ->(Any | None):
        try:
            trusted_scaler_path = resolve_trusted_artifact_path(
                scaler_path,
                allowed_suffixes={'.pkl', '.joblib'},
                must_exist=True,
            )
            target_scaler = joblib.load(trusted_scaler_path)  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
            if hasattr(target_scaler, 'scale_'):
                if target_scaler.scale_.shape[0] == 1:
                    self.logger.info(
                        f'Loaded target scaler from {trusted_scaler_path}')
                    return target_scaler
                self.logger.error(
                    f'INVALID scaler at {scaler_path}: {target_scaler.scale_.shape[0]} features instead of 1'
                    )
            else:
                self.logger.warning(
                    f'Scaler at {scaler_path} has no scale_ attribute')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(
                f'Failed to load scaler from {scaler_path}: {e}', exc_info=True)
        return None

    def _create_fallback_scaler(self, meta: dict[str, Any]) ->(Any | None):
        try:
            ticker = meta.get('ticker', '')
            target_col = meta.get('target', '')
            fallback_scaler = StandardScaler()
            if 'return' in target_col.lower():
                dummy_data = np.array([[-0.05], [0.0], [0.05]])
            elif 'up' in target_col.lower() or 'down' in target_col.lower():
                dummy_data = np.array([[0.0], [0.5], [1.0]])
            elif 'multi' in target_col.lower():
                dummy_data = np.array([[0.0], [1.0], [2.0]])
            else:
                dummy_data = np.array([[0.0], [0.5], [1.0]])
            fallback_scaler.fit(dummy_data)
            self.logger.info(
                f'✅ Created fallback scaler for {ticker}_{target_col}')
            return fallback_scaler
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'❌ Failed to create fallback scaler: {e}', exc_info=True)
            raise DataProcessingError("Failed to create fallback target scaler") from e

    def _normalize_metadata_columns(self, features_df: pd.DataFrame
        ) ->pd.DataFrame:
        if 'datetime' in features_df.columns and features_df['datetime'
            ].dtype == object:
            try:
                features_df['datetime'] = pd.to_datetime(features_df[
                    'datetime'])
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f'Error normalizing datetime column: {e}', exc_info=True)
        return features_df
