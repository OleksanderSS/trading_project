"""
Data Preparation Service for Stage 5 Prediction.

Handles data preparation, validation, and ticker-specific data processing.
Extracted from stage_5_prediction.py to reduce coupling and improve testability.
"""
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.pipeline.timeframe_lineage import is_timeframe_token, normalize_timeframe
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
        ticker: str,
        timeframe: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Prepare ticker-specific data for prediction.

        Args:
            features_df: Full features DataFrame
            ticker: Ticker symbol to extract
            timeframe: The model's timeframe. Required in practice, optional
                in the signature only so existing callers keep working.

        Returns:
            Prepared DataFrame for the ticker, or None if no data

        The timeframe filter is the whole point. features.parquet stacks
        every timeframe in one frame -- AAPL is 325 rows of 1d and 372 of
        60m -- and each timeframe's features carry a suffix, so SMA_5_1d is
        NaN on every 60m row by construction. Taking `.tail(50)` blind
        therefore handed a 1d model fifty 60m rows in which all 120 of its
        selected features were null.

        Measured on the 2026-08-04 run: 310 contexts (tabnet 240, mlp 24,
        and 33 more) reported "no data after dropping incomplete rows", and
        Stage 5 finished with 0 predictions from 330 resolved models while
        the pipeline reported success.
        """
        ticker_rows = features_df[features_df['ticker'] == ticker]
        if timeframe:
            ticker_rows = self._rows_for_timeframe(ticker_rows, ticker, timeframe)
        ticker_df = ticker_rows.tail(50)
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

        ticker_df_clean = self.prepare_ticker_data(
            features_df, ticker, self._model_timeframe(meta, context_id)
        )
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

        # Standardise BEFORE selecting the model's columns. The imputer and
        # scaler were fitted on the full feature set in that exact order, so
        # they must see the same frame; slicing first would present a
        # different matrix to a transform that has fixed per-column means and
        # scales.
        ticker_df_clean = self._apply_training_preprocessor(
            ticker_df_clean, meta, context_id
        )

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

    #: A row with more than this share of its features imputed is refused.
    #: Training fills a missing value with the train median, so prediction now
    #: does too — but the intraday case makes the limit necessary: `ctx_1d_*`
    #: columns are absent on the newest bars because the day has not closed,
    #: and imputing a whole day's context is inventing the very thing being
    #: asked about, not tolerating a gap.
    MAX_IMPUTED_SHARE = 0.5

    def _apply_training_preprocessor(self, frame: "pd.DataFrame", meta: dict[str, Any],
                                     context_id: str) -> "pd.DataFrame":
        """Put the features back into the space the model was trained in.

        Training standardises: prepare_data_for_models fits a SimpleImputer
        and a StandardScaler on the training split and hands the models
        z-scores. Prediction did not, so a model that learned "close > 0.3"
        was asked about a close of 120, and one fitted against unit variance
        received a volume of 5e7. Measured on a real 35-feature champion,
        the same model returned [0.033, -0.023, 0.156, ...] on z-scores and
        [128288, 127314, 133867, ...] on the raw frame. Nothing raised.

        Columns are reindexed to the FIT-TIME order before transforming — a
        StandardScaler applied to the same columns in a different order is a
        different transform, and the frame assembled here has no reason to
        match by luck. Columns the scaler never saw are dropped; ones it saw
        and this frame lacks arrive as NaN for the imputer to fill, which is
        the same treatment they received during training.

        A context whose preprocessor is missing is left ALONE and said so:
        older champions were promoted before this artifact existed, and
        silently guessing a transform is how the original defect looked.
        """
        payload = self._load_preprocessor(meta, context_id)
        if not payload:
            return frame

        scaler = payload.get('scaler')
        imputer = payload.get('imputer')
        fit_columns = list(payload.get('feature_names') or [])
        if not fit_columns or (scaler is None and imputer is None):
            return frame

        try:
            import pandas as pd

            ordered = frame.reindex(columns=fit_columns)
            values = ordered.to_numpy(dtype=float)

            # SimpleImputer DROPS any column that had no observed value during
            # fit ("Skipping features without any observed values"), so its
            # output can be narrower than fit_columns and the scaler behind it
            # was fitted on that narrower matrix. Measured on this batch, 1d
            # contexts carry 398 all-NaN ctx_1d_* columns — a daily bar gets no
            # daily context, by design — so building a frame with fit_columns
            # raised "Shape of passed values is (1, 2), indices imply (1, 3)",
            # which the handler below turned into zero predictions for every
            # one of the 396 daily contexts.
            output_columns = self._surviving_columns(imputer, fit_columns)

            if imputer is not None:
                values = imputer.transform(values)
            if scaler is not None:
                values = scaler.transform(values)

            # How much of each row is about to be INVENTED, measured over the
            # columns the model actually uses. Counting the all-NaN ones would
            # mark every 1d row as mostly-missing and refuse it, when those
            # columns are not features at all — they were dropped at fit time.
            missing_share = ordered[output_columns].isna().mean(axis=1)

            transformed = pd.DataFrame(values, columns=output_columns, index=frame.index)

            mostly_invented = missing_share > self.MAX_IMPUTED_SHARE
            if mostly_invented.any():
                self.logger.warning(
                    '%s: dropping %d of %d row(s) where more than %.0f%% of the '
                    'features had to be imputed — a prediction from mostly '
                    'invented inputs is not a prediction.',
                    context_id, int(mostly_invented.sum()), len(transformed),
                    self.MAX_IMPUTED_SHARE * 100,
                )
                transformed = transformed.loc[~mostly_invented]
                frame = frame.loc[~mostly_invented]
            if missing_share.max() > 0:
                self.logger.debug(
                    f'{context_id}: imputed up to {missing_share.max():.1%} of a row'
                )
            # Carry over what the transform was never meant to touch: context
            # and identity columns. NOT the fit columns the imputer dropped —
            # those were features with no observed value at fit time, and
            # putting them back would reintroduce an all-NaN column the model
            # was never fitted on.
            fit_column_set = set(fit_columns)
            for column in frame.columns:
                if column not in transformed.columns and column not in fit_column_set:
                    transformed[column] = frame[column]
            transformed.attrs = dict(frame.attrs)
            self.logger.debug(
                f'{context_id}: applied training preprocessor over {len(fit_columns)} columns'
            )
            return transformed
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            self.logger.error(
                f'{context_id}: could not apply the training preprocessor ({e}); '
                f'refusing to predict on differently-scaled features.'
            )
            return frame.iloc[0:0]

    @staticmethod
    def _surviving_columns(imputer: Any, fit_columns: list[str]) -> list[str]:
        """The columns the imputer actually emits.

        A SimpleImputer silently drops every feature that had no observed
        value during fit, recording NaN in `statistics_` for it. The scaler
        chained behind it was therefore fitted on the narrower matrix, and any
        frame rebuilt with the original column list will not line up.
        """
        if imputer is None:
            return list(fit_columns)
        try:
            names = imputer.get_feature_names_out(fit_columns)
            return [str(name) for name in names]
        except (AttributeError, ValueError, TypeError):
            statistics = getattr(imputer, 'statistics_', None)
            if statistics is None or len(statistics) != len(fit_columns):
                return list(fit_columns)
            return [
                column
                for column, statistic in zip(fit_columns, statistics)
                if not np.isnan(statistic)
            ]

    def _load_preprocessor(self, meta: dict[str, Any], context_id: str) -> dict[str, Any] | None:
        """Load the imputer+scaler saved beside this context's champion."""
        try:
            import joblib

            from src.pipeline.constants import preprocessor_filename
            from src.utils.artifact_security import resolve_trusted_artifact_path

            model_path = meta.get('model_path')
            if not model_path:
                return None
            directory = Path(model_path).parent
            path = directory / preprocessor_filename(
                str(meta.get('ticker') or ''),
                str(meta.get('timeframe') or ''),
                str(meta.get('target') or meta.get('target_name') or ''),
            )
            if not path.exists():
                self.logger.warning(
                    f'{context_id}: no preprocessor at {path.name}; features are '
                    f'served in their raw scale, which matches training only if '
                    f'this model was trained without standardisation.'
                )
                return None
            trusted = resolve_trusted_artifact_path(path, must_exist=True)
            return joblib.load(trusted)  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
        except (OSError, ValueError, TypeError, AttributeError, ImportError) as e:
            self.logger.error(f'{context_id}: could not load preprocessor ({e})')
            return None

    def _model_timeframe(self, meta: dict[str, Any], context_id: str) -> str | None:
        """Which timeframe's rows this model was trained on.

        Declared metadata wins. Where it is absent -- Colab writes
        selected_features_*.json without one -- the features themselves say
        it: every column carries a timeframe suffix, so a model whose
        features are SMA_5_1d, EMA_20_1d ... is a 1d model. Inferred from the
        majority so one stray unsuffixed name cannot flip the answer.
        """
        declared = normalize_timeframe(meta.get('timeframe'))
        if declared and is_timeframe_token(declared):
            return declared

        suffixes: Counter[str] = Counter()
        for name in meta.get('selected_features') or []:
            _, separator, tail = str(name).rpartition('_')
            if separator and is_timeframe_token(tail):
                suffixes[normalize_timeframe(tail)] += 1

        if not suffixes:
            # Not an error: a frame with a single timeframe needs no filter.
            # Said at debug so a silent full-frame tail stays traceable.
            self.logger.debug(
                'Context %s declares no timeframe and its features carry no '
                'suffix; using every row for the ticker.', context_id,
            )
            return None

        winner, count = suffixes.most_common(1)[0]
        if len(suffixes) > 1:
            self.logger.warning(
                'Context %s mixes feature timeframes %s; using %s (%d of %d '
                'features). A model trained across timeframes cannot be '
                'sliced to one.',
                context_id, dict(suffixes), winner, count, sum(suffixes.values()),
            )
        return winner

    def _rows_for_timeframe(
        self, ticker_rows: pd.DataFrame, ticker: str, timeframe: str
    ) -> pd.DataFrame:
        """Rows of `ticker_rows` belonging to `timeframe`, or all of them."""
        column = next(
            (c for c in ('interval', 'timeframe') if c in ticker_rows.columns), None
        )
        if column is None:
            return ticker_rows

        normalized = ticker_rows[column].map(normalize_timeframe)
        matching = ticker_rows[normalized == timeframe]
        if matching.empty:
            # Falling through to the unfiltered frame would reproduce the
            # exact bug this method exists to prevent, so it does not.
            self.logger.error(
                'No %s rows for %s (available: %s); skipping rather than '
                'predicting from another timeframe.',
                timeframe, ticker, sorted(set(normalized.dropna())),
            )
        return matching

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

        original = ticker_df
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
            # Name the columns that emptied it.
            #
            # This used to report only that nothing survived. On the
            # 2026-08-09 run 313 of 660 contexts ended here -- 127 of 154 on
            # 15m, 84 of 110 on 60m -- and the message named not one feature,
            # so the cause could not be established from the artifacts
            # afterwards. A row is dropped BY a column; the column is the
            # part worth knowing.
            present = [c for c in model_feature_cols if c in original.columns]
            null_counts = original[present].isna().sum()
            blockers = null_counts[null_counts == len(original)]
            partial = null_counts[
                (null_counts > 0) & (null_counts < len(original))
            ].sort_values(ascending=False)

            self.logger.error(
                f'Context {context_id} has no data after dropping incomplete '
                f'rows; skipping prediction. '
                f'{len(blockers)} of {len(present)} required feature(s) are '
                f'null in EVERY one of the {len(original)} candidate rows'
                + (f' (e.g. {", ".join(list(blockers.index)[:5])})'
                   if len(blockers) else '')
                + (f'; {len(partial)} more are null in some'
                   f' (worst: {", ".join(list(partial.index)[:3])})'
                   if len(partial) else '')
                + '.'
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
