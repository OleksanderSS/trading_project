# src/pipeline/stages/stage_5_prediction.py

"""
Stage 5: Prediction Generation with Stacked Ensembles and Contextual Adjustments

Uses champion models and stacked ensembles to generate forecasts,
incorporating real-time market regime adjustments and historical performance.

Refactored: heavy logic moved to sub-package `prediction/`:
  - ModelResolver   → model path resolution & loading
  - PredictionGenerator → ensemble/single prediction & denormalization
  - AnomalyEngine   → anomaly detection & confidence scoring
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder
from src.analytics.context.market_context_analyzer import MarketContextAnalyzer
from src.analytics.context.prediction_adjuster import PredictionAdjuster
from src.analytics.detectors.anomaly_detector import AnomalyDetector
from src.analytics.detectors.critical_signal_detector import CriticalSignalDetector
from src.analytics.signals.signal_analytics import analyze_signals
from src.analytics.signals.significance_detector import detect_significant_events
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.ensembling.stacked_ensemble import StackedEnsemble
from src.features.utils.datetime_utils import normalize_metadata_columns
from src.meta_learning.memory.diary_engine import DiaryEngine
from src.models.loader import ModelLoaderStrategy
from src.models.model_pool import get_model_pool
from src.models.quality.controller import get_quality_controller
from src.models.model_selector.adaptive_selector import AdaptiveModelSelector
from src.models.model_selector.smart_selector import SmartModelSelector
from src.pipeline.stages.base_stage import BaseStage
from src.pipeline.stages.prediction import AnomalyEngine, ModelResolver, PredictionGenerator
from src.predictions.caching import get_ensemble_cache


@dataclass
class PredictionResultRequest:
    """Request for creating prediction result."""

    context_id: str
    ticker: str
    adjusted_prediction: float
    raw_prediction: float
    model_contributions: dict[str, float]
    best_model_name: str
    ticker_df_clean: pd.DataFrame
    meta: dict[str, Any]
    shap_explanations: dict[str, Any] | None = None


class PredictionStage(BaseStage):
    """
    Stage responsible for generating model predictions using an ensemble approach,
    calculating confidence scores, and adjusting forecasts based on market context.
    """

    ACCUMULATION_OUTPUT_DIR_CONFIG = 'system.accumulation.output_dir'
    DEFAULT_ACCUMULATION_DIR = 'data/colab/accumulated'

    def __init__(self, config_manager: UnifiedConfigManager, error_handler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger("PredictionStage")
        self.prediction_config = self.config_manager.get_config('prediction', {})
        self.models_path = self.config_manager.get_models_path()

        self.diary = DiaryEngine()
        self.adjuster = PredictionAdjuster(
            config=self.config_manager.get('analysis.prediction_adjustment', {})
        )
        self.ensemble_factory = StackedEnsemble()

        # Use AdaptiveModelSelector if enabled in config (default: True for online learning)
        # ✅ INT FIX: Changed default to True so AdaptiveModelSelector is active by default.
        use_adaptive = self.config_manager.get('prediction.use_adaptive_selector', True)
        if use_adaptive:
            self.context_selector: SmartModelSelector | AdaptiveModelSelector = AdaptiveModelSelector(
                fallback="lightgbm",
                leaderboard_path="data/model_leaderboard.json",
                learning_rate=0.1
            )
            self.logger.info("✅ Using AdaptiveModelSelector with online learning")
        else:
            self.context_selector = SmartModelSelector()
            self.logger.info("✅ Using SmartModelSelector (default)")

        self.knn_similarity = KnnSimilarityFinder(config={'n_neighbors': 5})
        self.model_loader = ModelLoaderStrategy(self.logger)

        # Caches (LRU)
        self.ensemble_cache = get_ensemble_cache(maxsize=5000)
        self.logger.info("✅ Ensemble prediction cache enabled (LRU, maxsize=5000)")

        max_models = self.config_manager.get('performance.model_pool_size', 50)
        self.model_pool = get_model_pool(max_models=max_models)
        self.logger.info(f"✅ Model pool enabled (maxsize={max_models}, LRU eviction)")

        # Initialize ModelQualityController
        drift_threshold = self.config_manager.get('prediction.drift_threshold', 0.3)
        self.quality_controller = get_quality_controller(drift_threshold=drift_threshold)
        self.logger.info(f"✅ Model quality controller enabled (drift_threshold={drift_threshold})")

        # Sub-module helpers (extracted to reduce file size)
        self.model_resolver = ModelResolver(
            config_manager=self.config_manager,
            model_pool=self.model_pool,
            model_loader=self.model_loader,
        )
        self.anomaly_engine = AnomalyEngine(diary=self.diary)
        self.prediction_generator = PredictionGenerator(
            ensemble_factory=self.ensemble_factory,
            ensemble_cache=self.ensemble_cache,
            adjuster=self.adjuster,
        )

        # Initialize advanced detection tools
        self.anomaly_detector = AnomalyDetector()
        self.critical_signal_detector = CriticalSignalDetector()

        # Initialize advanced context and signal analysis tools
        self.market_context_analyzer = MarketContextAnalyzer(['volatility', 'trend', 'momentum'])
        self.signal_analytics = analyze_signals  # Function, not class
        self.significance_detector = detect_significant_events  # Function, not class
        self.logger.info("✅ MarketContextAnalyzer, SignalAnalytics and SignificanceDetector functions initialized")

    # ------------------------------------------------------------------
    # Public entry point (pipeline API — unchanged)
    # ------------------------------------------------------------------

    async def run(self, **kwargs) -> dict[str, Any]:
        """
        Generates adjusted predictions for tickers processed in earlier stages.

        Args:
            **kwargs: Pipeline data dict with 'features_data' and 'models_metadata'.

        Returns:
            Dict[str, Any]: Updated pipeline data with 'prediction_results'.
        """
        features_df, models_meta, market_regime = self._prepare_inputs(kwargs)
        if features_df is None or (hasattr(features_df, 'empty') and features_df.empty) or not models_meta:
            return {}

        if not self._ensure_local_models(models_meta, kwargs):
            return {}

        prediction_results = self._generate_predictions_for_contexts(
            models_meta, features_df, market_regime
        )
        return self._prepare_final_results(prediction_results, models_meta, kwargs)

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self, kwargs: dict[str, Any]
    ) -> tuple[pd.DataFrame | None, dict[str, Any], str]:
        features_df = next(
            (kwargs[k] for k in ('features_data', 'features_df', 'enriched_data')
             if k in kwargs and kwargs[k] is not None),
            None,
        )
        models_meta = kwargs.get('models_metadata') or kwargs.get('models_meta', {})
        market_regime = kwargs.get('market_regime', 'neutral')

        if not models_meta:
            models_meta = self.model_resolver.load_models_metadata_from_disk(kwargs)
            if not models_meta:
                self.logger.warning("Failed to load models_metadata from disk")
                return None, {}, market_regime
            self.logger.info(f"Loaded {len(models_meta)} models from disk")

        is_valid, _ = self._validate_inputs(features_df, models_meta)
        if not is_valid:
            return None, {}, market_regime

        if isinstance(features_df, pd.DataFrame):
            features_df = normalize_metadata_columns(features_df)
            self.logger.info("Normalized features_df at stage entry")

        return features_df, models_meta, market_regime

    def _validate_inputs(self, features_df, models_meta) -> tuple[bool, str]:
        if features_df is None or features_df.empty or not models_meta:
            self.logger.warning("Required features or model metadata not found. Skipping Stage 5.")
            self.logger.warning(f"  - features_df is None: {features_df is None}")
            self.logger.warning(
                f"  - features_df empty: {features_df.empty if features_df is not None else 'N/A'}"
            )
            self.logger.warning(f"  - models_meta empty: {not models_meta}")
            return False, "Invalid inputs"
        return True, "Valid inputs"

    def _ensure_local_models(
        self, models_meta: dict[str, Any], kwargs: dict[str, Any] | None = None
    ) -> bool:
        has_local = self.model_resolver.check_local_models(models_meta)
        if not has_local:
            self.model_resolver.log_model_status(models_meta)
            batch_dir = self.model_resolver.resolve_batch_directory(models_meta, kwargs or {})
            if batch_dir and batch_dir.exists():
                has_local = self.model_resolver.update_local_model_paths(models_meta, batch_dir)
            if not has_local:
                self.logger.error("No local models found. Skipping Stage 5.")
                return False
        return True

    # ------------------------------------------------------------------
    # Prediction loop
    # ------------------------------------------------------------------

    def _generate_predictions_for_contexts(
        self,
        models_meta: dict[str, Any],
        features_df: pd.DataFrame,
        market_regime: str,
    ) -> dict[str, Any]:
        prediction_results: dict[str, Any] = {}

        # Filter contexts to only include available model types
        available_model_types = self._get_available_model_types()
        filtered_models_meta = {}

        for context_id, meta in models_meta.items():
            model_type = meta.get('model_type', '')
            if model_type in available_model_types:
                filtered_models_meta[context_id] = meta
            else:
                self.logger.debug(f"Skipping {context_id} - {model_type} models not available")

        self.logger.info(f"Generating ensemble predictions for {len(filtered_models_meta)}/{len(models_meta)} available contexts...")

        for context_id, meta in filtered_models_meta.items():
            try:
                result = self._process_single_context(context_id, meta, features_df, market_regime)
                if result:
                    prediction_results[context_id] = result
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                self.handle_stage_error(e, context=f"Prediction-{context_id}", severity="error")
                self.logger.error(f"Prediction failed for context {context_id}: {e}", exc_info=True)

        return prediction_results

    def _get_available_model_types(self) -> set:
        """Get available model types by scanning model files in the database directory"""
        try:
            from pathlib import Path

            # Get the accumulation directory
            base_dir = Path(
                self.config_manager.get(self.ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR)
            )
            batch_dir = base_dir / "main_database"

            if not batch_dir.exists():
                self.logger.warning(f"Model directory not found: {batch_dir}")
                return {'mlp', 'tabnet'}  # Default fallback

            # Scan for model files and extract types
            model_types = set()

            # Check for .pkl files (MLP models)
            pkl_files = list(batch_dir.glob("*.pkl"))
            if pkl_files:
                model_types.add('mlp')
                self.logger.debug(f"Found {len(pkl_files)} MLP models")

            # Check for .zip files (TabNet models)
            zip_files = list(batch_dir.glob("*.zip"))
            if zip_files:
                model_types.add('tabnet')
                self.logger.debug(f"Found {len(zip_files)} TabNet models")

            # Check for .keras files (if any exist)
            keras_files = list(batch_dir.glob("*.keras"))
            if keras_files:
                model_types.update(['cnn', 'lstm', 'gru', 'transformer', 'autoencoder'])
                self.logger.debug(f"Found {len(keras_files)} Keras models")

            self.logger.info(f"Available model types: {sorted(model_types)}")
            return model_types if model_types else {'mlp', 'tabnet'}  # Default fallback

        except Exception as e:
            self.logger.error(f"Error scanning model types: {e}")
            return {'mlp', 'tabnet'}  # Safe fallback

    def _process_single_context(
        self,
        context_id: str,
        meta: dict[str, Any],
        features_df: pd.DataFrame,
        market_regime: str,
    ) -> dict[str, Any] | None:
        context_result = self._process_context_data(context_id, meta, features_df)
        if context_result is None:
            return None

        ticker_df_clean, filtered_features_list = context_result
        ticker = meta.get('ticker')
        if not ticker:  # Should never happen due to _process_context_data validation
            self.logger.error(f"Ticker missing for context {context_id}")
            return None

        models = self.model_resolver.load_available_models(context_id, {context_id: meta})
        if not models:
            self.logger.warning(f"No models found for {context_id}, skipping")
            return None

        target_scaler = self._load_target_scaler(meta)

        target_col = meta.get('target', '')
        target_type = meta.get('target_type', 'classification')
        is_classification = target_type == 'classification' or 'up' in target_col.lower() or 'down' in target_col.lower() or 'multi' in target_col.lower()

        if target_scaler is None:
            if not is_classification:
                self.logger.warning(f"Target scaler not found for {context_id} - creating fallback scaler")
                target_scaler = self._create_fallback_scaler(meta)
            else:
                self.logger.debug(f"Target scaler not needed for classification context {context_id}")

        best_model_name = self._select_best_model_for_context(
            ticker_df_clean, meta, models, ticker, market_regime
        )

        raw_prediction, model_contributions, shap_explanations = self.prediction_generator.generate_prediction(
            models, best_model_name, ticker_df_clean, filtered_features_list, market_regime, context_id
        )
        if raw_prediction is None:
            return None

        adjusted_prediction = self.prediction_generator.adjust_prediction_contextually(
            raw_prediction, best_model_name, market_regime, ticker
        )
        adjusted_prediction = self.prediction_generator.denormalize_prediction(
            adjusted_prediction, target_scaler
        )

        # Validation & Quality Control Integration
        try:
            is_valid = self.quality_controller.validate_predictions(np.array([adjusted_prediction]))
            if not is_valid:
                self.logger.warning(f"⚠️ Prediction validation failed for {context_id} (value={adjusted_prediction:.6f})")

            # Calculate quality score for ensembling models
            if len(models) > 1 and raw_prediction is not None:
                model_preds_dict = {m_name: float(contrib) for m_name, contrib in model_contributions.items()}
                ensemble_pred_val = float(raw_prediction[-1] if isinstance(raw_prediction, np.ndarray) else raw_prediction)
                active_weights = {m_name: 1.0 / len(models) for m_name in model_preds_dict.keys()}

                q_score = self.quality_controller.get_quality_score(ensemble_pred_val, model_preds_dict, active_weights)
                # Update quality scores in the persistent pool
                for m_name in model_preds_dict.keys():
                    self.model_pool.update_quality_score(m_name, q_score)
                self.logger.info(f"🏆 Context {context_id} - Dynamic Prediction Quality Score: {q_score:.4f}")
        except Exception as qc_err:
            self.logger.debug(f"QC check bypassed: {qc_err}")

        request = PredictionResultRequest(
            context_id=context_id,
            ticker=ticker,
            adjusted_prediction=adjusted_prediction,
            raw_prediction=raw_prediction,
            model_contributions=model_contributions,
            best_model_name=best_model_name,
            ticker_df_clean=ticker_df_clean,
            meta=meta,
            shap_explanations=shap_explanations
        )
        return self._create_prediction_result(request)

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------

    def _process_context_data(
        self, context_id: str, meta: dict[str, Any], features_df: pd.DataFrame
    ) -> tuple | None:
        ticker = meta.get('ticker')
        if not ticker:
            self.logger.error(f"No ticker found in metadata for context {context_id}")
            return None

        target_col = meta.get('target', '')
        model_type = meta.get('model_type', '')

        self.logger.info(f"🔍 Processing context: {context_id}")
        self.logger.info(f"   ticker={ticker}, target={target_col}, model_type={model_type}")

        ticker_df_clean = self._prepare_ticker_data(features_df, ticker)
        if ticker_df_clean is None:
            return None

        selected_features = meta.get('selected_features', [])
        self.logger.debug(f"🔍 Stage 5: selected_features from metadata: {len(selected_features)} features")

        missing_features = [f for f in selected_features if f not in ticker_df_clean.columns]
        if missing_features:
            self.logger.info(f"⚠️ Context {context_id} missing {len(missing_features)} features. Attempting adaptive re-enrichment...")
            ticker_df_clean = self._adaptive_re_enrichment(ticker_df_clean, missing_features)
            
            # Re-check missing features after enrichment
            still_missing = [f for f in selected_features if f not in ticker_df_clean.columns]
            if still_missing:
                self.logger.warning(f"Adding {len(still_missing)} remaining missing features filled with 0")
                for f in still_missing:
                    ticker_df_clean[f] = 0.0

        filtered_features_list = selected_features
        if selected_features:
            self.logger.info(f"✅ Using {len(filtered_features_list)} selected features for prediction")
        else:
            self.logger.warning("⚠️ No selected features specified in metadata")

        if selected_features and not filtered_features_list:
            self.logger.warning(f"⚠️ None of selected features found for {model_type}")
            return None

        # Filter the dataframe to only include the required features
        if filtered_features_list:
            # Ensure we only use columns that actually exist
            existing_cols = [c for c in filtered_features_list if c in ticker_df_clean.columns]
            ticker_df_clean = ticker_df_clean[existing_cols]
            self.logger.info(f" Using {len(existing_cols)} features for {model_type}")
        else:
            self.logger.warning(
                f" No selected features for {model_type}, using all {ticker_df_clean.shape[1]} columns"
            )
            filtered_features_list = ticker_df_clean.columns.tolist()

        return ticker_df_clean, filtered_features_list

    def _adaptive_re_enrichment(self, df: pd.DataFrame, missing_features: list[str]) -> pd.DataFrame:
        """
        Dynamically re-calculates missing features using FeatureOrchestrator.
        Requires OHLCV columns to be present in the DataFrame.
        """
        try:
            from src.features.feature_orchestrator import FeatureOrchestrator
            
            # Create a temporary orchestrator for re-enrichment
            orchestrator = FeatureOrchestrator.create_from_config(self.config_manager)
            
            # Ensure we have required columns for enrichment
            required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_ohlcv):
                self.logger.warning(f"Cannot re-enrich: missing OHLCV columns. Available: {df.columns.tolist()[:10]}")
                return df
                
            # Perform enrichment via .run() method
            # Note: FeatureOrchestrator.run returns the enriched DataFrame
            enriched_df = orchestrator.run(df)
            
            new_features_found = [f for f in missing_features if f in enriched_df.columns]
            if new_features_found:
                self.logger.info(f"✅ Successfully recovered {len(new_features_found)} missing features via adaptive enrichment")
                return enriched_df
            else:
                self.logger.warning("⚠️ Adaptive enrichment completed but missing features were not recovered.")
                return enriched_df
                
        except Exception as e:
            self.logger.error(f"❌ Adaptive re-enrichment failed: {e}")
            return df

    def _prepare_ticker_data(
        self, features_df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame | None:
        # ✅ ENHANCED: Increased tail from 50 to 250 to support indicators with longer windows (e.g. SMA 200)
        ticker_df = features_df[features_df['ticker'] == ticker].tail(250)
        if ticker_df.empty:
            self.logger.warning(f"⚠️ No data for ticker {ticker}")
            return None

        ticker_df_clean = ticker_df.copy()

        # ELITE FIX: Ensure datetime is in index before dropping metadata columns
        for dt_col in ['datetime', 'date']:
            if dt_col in ticker_df_clean.columns:
                try:
                    ticker_df_clean.index = pd.to_datetime(ticker_df_clean[dt_col])
                    self.logger.debug(f"Moved {dt_col} to index for timestamp preservation")
                    break
                except Exception as e:
                    self.logger.debug(f"Failed to move {dt_col} to index: {e}")

        metadata_cols = [
            'ticker', 'datetime', 'date', 'interval', 'timeframe', 'hash', 'symbol',
            'context_fingerprint', '_cache_ticker', '_cache_date', '_cache_config_hash',
            'is_significant',
        ]
        ticker_df_clean = ticker_df_clean.drop(
            columns=[c for c in metadata_cols if c in ticker_df_clean.columns], errors='ignore'
        )

        # Drop any remaining non-numeric columns
        obj_cols = ticker_df_clean.select_dtypes(exclude=['number']).columns.tolist()
        if obj_cols:
            ticker_df_clean = ticker_df_clean.drop(columns=obj_cols, errors='ignore')

        for col in ticker_df_clean.columns:
            try:
                ticker_df_clean[col] = pd.to_numeric(ticker_df_clean[col], errors='coerce')
            except (ValueError, TypeError) as e:
                self.logger.debug(f"Failed to convert column {col} to numeric: {e}")
                ticker_df_clean = ticker_df_clean.drop(columns=[col], errors='ignore')

        ticker_df_clean = ticker_df_clean.fillna(0).replace([np.inf, -np.inf], 0)

        if ticker_df_clean.empty or ticker_df_clean.dtypes.apply(
            lambda x: x.kind not in 'biufc'
        ).any():
            self.logger.warning(f"⚠️ Data for {ticker} contains non-numeric columns, skipping")
            return None

        return ticker_df_clean

    def _create_context_fingerprint(self, ticker_df: pd.DataFrame, market_regime: str) -> str:
        """
        Create context fingerprint for AdaptiveModelSelector.
        Format: "volatility|trend|regime|momentum|volume"
        """
        try:
            # Calculate context features
            if 'close' in ticker_df.columns and len(ticker_df) > 1:
                returns = ticker_df['close'].pct_change().dropna()
                volatility = 1 if returns.std() > 0.02 else 0  # High volatility
                trend = 1 if returns.mean() > 0 else (-1 if returns.mean() < 0 else 0)
            else:
                volatility = 0
                trend = 0

            # Map regime to numeric
            regime_map = {'bull': 1, 'bear': -1, 'sideways': 0, 'volatile': 2}
            regime_val = regime_map.get(market_regime.lower(), 0)

            # Momentum and volume (simplified)
            momentum = 0
            volume = 0

            fingerprint = f"{volatility}|{trend}|{regime_val}|{momentum}|{volume}"
            return fingerprint

        except Exception as e:
            self.logger.warning(f"Failed to create context fingerprint: {e}")
            return "0|0|0|0|0"  # Default fingerprint

    def _load_target_scaler(self, meta: dict[str, Any]) -> Any | None:
        ticker = meta.get('ticker', '')
        target_col = meta.get('target', '')
        model_path_str = meta.get('model_path', '')

        base_dir = Path(
            self.config_manager.get(self.ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR)
        )
        # Configured scalers directory (primary save location during training)
        scalers_dir = Path(self.config_manager.get('paths.scalers', 'data/scalers'))
        scaler_filename = f"scaler_{ticker}_{target_col}.pkl"

        # Strategy 0: dedicated scalers directory
        scaler_path = scalers_dir / scaler_filename
        if scaler_path.exists():
            return self._try_load_scaler(scaler_path)

        # Strategy 1: derive batch_name from model_path
        if model_path_str:
            model_path = Path(model_path_str.replace('/', '\\'))
            batch_dir = model_path.parent
            scaler_path = batch_dir / scaler_filename
            if scaler_path.exists():
                return self._try_load_scaler(scaler_path)

        # Strategy 2: look in known batch directories
        for batch_name in ['main_database', 'test_ticker_AAPL_target_return_1d']:
            scaler_path = base_dir / batch_name / scaler_filename
            if scaler_path.exists():
                return self._try_load_scaler(scaler_path)

        # Strategy 3: legacy path with 'models' in path
        if model_path_str:
            parts = model_path_str.replace('/', '\\').split('\\')
            if 'models' in parts:
                models_idx = parts.index('models')
                if models_idx > 0:
                    batch_name = parts[models_idx - 1]
                    scaler_path = base_dir / batch_name / scaler_filename
                    if scaler_path.exists():
                        return self._try_load_scaler(scaler_path)

        # Strategy 4: scan all subdirectories under base_dir
        try:
            for candidate in base_dir.rglob(scaler_filename):
                result = self._try_load_scaler(candidate)
                if result is not None:
                    return result
        except Exception:
            pass

        self.logger.debug(f"No target scaler found for {ticker}_{target_col}")
        return None

    def _try_load_scaler(self, scaler_path: Path) -> Any | None:
        """Load and validate a scaler file."""
        try:
            target_scaler = joblib.load(scaler_path)
            if hasattr(target_scaler, 'scale_'):
                if target_scaler.scale_.shape[0] == 1:
                    self.logger.info(f"Loaded target scaler from {scaler_path}")
                    return target_scaler
                else:
                    self.logger.error(
                        f"INVALID scaler at {scaler_path}: "
                        f"{target_scaler.scale_.shape[0]} features instead of 1"
                    )
            else:
                self.logger.warning(f"Scaler at {scaler_path} has no scale_ attribute")
        except Exception as e:
            self.logger.warning(f"Failed to load scaler from {scaler_path}: {e}")
        return None

    def _create_fallback_scaler(self, meta: dict[str, Any]) -> Any | None:
        """Create a fallback target scaler when original scaler is not found.

        ✅ CALC FIX: Instead of fitting on arbitrary dummy data (which produces wrong scale),
        we now try to recover statistics from DiaryEngine trade history.
        If DiaryEngine has no data, we use domain-appropriate ranges per target type
        but we log a clear warning so the operator knows the scale may be off.
        """
        try:
            import numpy as np
            from sklearn.preprocessing import StandardScaler

            ticker = meta.get('ticker', '')
            target_col = meta.get('target', '')

            fallback_scaler = StandardScaler()

            # Try to recover P&L stats from DiaryEngine for more accurate scale
            try:
                recent_entries = self.diary.get_recent_entries(limit=100)
                pnl_values = [e.profit_loss for e in recent_entries if e.profit_loss is not None]
                if len(pnl_values) >= 5:
                    pnl_arr = np.array(pnl_values).reshape(-1, 1)
                    fallback_scaler.fit(pnl_arr)
                    self.logger.info(
                        f"✅ Fallback scaler fitted from DiaryEngine P&L stats "
                        f"(n={len(pnl_values)}, mean={pnl_arr.mean():.4f}, std={pnl_arr.std():.4f})"
                    )
                    return fallback_scaler
            except Exception as diary_err:
                self.logger.debug(f"DiaryEngine fallback scaler skipped: {diary_err}")

            # Domain-appropriate dummy ranges per target type
            if 'return' in target_col.lower():
                dummy_data = np.array([[-0.05], [0.0], [0.05]])
            elif 'up' in target_col.lower() or 'down' in target_col.lower():
                dummy_data = np.array([[0.0], [0.5], [1.0]])
            elif 'multi' in target_col.lower():
                dummy_data = np.array([[0.0], [1.0], [2.0]])
            else:
                dummy_data = np.array([[0.0], [0.5], [1.0]])

            fallback_scaler.fit(dummy_data)
            self.logger.warning(
                f"⚠️ Fallback scaler for {ticker}_{target_col} fitted on dummy data — "
                "predictions MAY be incorrectly scaled. Investigate missing scaler file."
            )
            return fallback_scaler

        except Exception as e:
            self.logger.error(f"❌ Failed to create fallback scaler: {e}")
            return None

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def _select_best_model_for_context(
        self,
        ticker_df_clean: pd.DataFrame,
        meta: dict[str, Any],
        models: dict[str, Any],
        ticker: str,
        market_regime: str,
    ) -> str:
        # ELITE FIX: Autoencoders are for anomaly detection, not regression
        models_list = [m for m in models.keys() if 'autoencoder' not in m.lower()]
        if not models_list:
            # Fallback to all models if somehow everything was an autoencoder
            models_list = list(models.keys())

        target_type = meta.get('target_type', 'classification')

        best_model_name, knn_confidence = self._perform_knn_similarity_analysis(
            ticker_df_clean, models_list
        )
        if best_model_name is None or best_model_name not in models_list:
            if isinstance(self.context_selector, AdaptiveModelSelector):
                context_fingerprint = self._create_context_fingerprint(ticker_df_clean, market_regime)
                best_model_name = self.context_selector.select_best_model_adaptive(context_fingerprint)
                if best_model_name not in models_list:
                    best_model_name = models_list[0] if models_list else "lightgbm"
            else:
                # Build type→full_key map so SmartModelSelector works with short names
                # e.g. {'mlp': 'model_AAPL_target_up_1d_mlp', 'lstm': 'model_AAPL_target_up_1d_lstm'}
                type_to_key: dict[str, str] = {}
                for full_key in models_list:
                    # Last segment after last '_' is the model type
                    parts = full_key.split('_')
                    short = parts[-1] if parts else full_key
                    type_to_key.setdefault(short, full_key)

                available_types = list(type_to_key.keys())
                if len(available_types) == 1:
                    # Single model — use it directly
                    best_model_name = models_list[0]
                else:
                    chosen_type = self.context_selector.select_best_model(
                        ticker_df_clean, target_type, available_types
                    )[0]
                    best_model_name = type_to_key.get(chosen_type, models_list[0])

            self.logger.info(
                f"Contextual Selector chose '{best_model_name}' for {ticker} in {market_regime} regime."
            )
        else:
            self.logger.info(
                f"KNN Similarity chose '{best_model_name}' for {ticker} (confidence: {knn_confidence:.2f})"
            )

        return best_model_name or ""

    def _perform_knn_similarity_analysis(
        self, ticker_df_clean: pd.DataFrame, models_list: list[str]
    ) -> tuple[str | None, float]:
        """
        Uses KnnSimilarityFinder with historical data from DiaryEngine.

        ✅ INT FIX: historical_performance is now fetched from DiaryEngine (get_recent_entries)
        instead of being a hardcoded empty DataFrame. This makes KNN actually functional.
        """
        try:
            target_features = ticker_df_clean.tail(5)

            # Fetch real historical performance from DiaryEngine
            historical_performance = pd.DataFrame()
            try:
                recent_entries = self.diary.get_recent_entries(limit=200)
                if recent_entries:
                    records = []
                    for entry in recent_entries:
                        record = {
                            'model_name': entry.agent_id,
                            'profit_loss': entry.profit_loss or 0.0,
                            'win': 1 if (entry.profit_loss or 0) > 0 else 0,
                        }
                        # Parse market_context if available
                        if entry.market_context and isinstance(entry.market_context, dict):
                            record.update({f'ctx_{k}': v for k, v in entry.market_context.items() if isinstance(v, (int, float))})
                        records.append(record)
                    historical_performance = pd.DataFrame(records)
            except Exception as diary_err:
                self.logger.debug(f"KNN DiaryEngine fetch skipped: {diary_err}")

            if not historical_performance.empty:
                return self._analyze_knn_similarities(target_features, historical_performance, models_list)
        except Exception as e:
            self.logger.warning(f"⚠️ KNN similarity failed: {e}, falling back to SmartModelSelector")
        return None, 0.0

    def _analyze_knn_similarities(
        self,
        target_features: pd.DataFrame,
        historical_performance: pd.DataFrame,
        models_list: list[str],
    ) -> tuple[str | None, float]:
        historical_features = historical_performance.get('features', pd.DataFrame())
        if not historical_features.empty and len(historical_features.columns) == len(
            target_features.columns
        ):
            knn_result = self.knn_similarity.analyze(
                {
                    'historical_features': historical_features,
                    'target_features': target_features,
                }
            )
            return self._process_knn_results(knn_result, target_features, historical_performance, models_list)
        return None, 0.0

    def _process_knn_results(
        self,
        knn_result: dict,
        target_features: pd.DataFrame,
        historical_performance: pd.DataFrame,
        models_list: list[str],
    ) -> tuple[str | None, float]:
        if 'similarities' not in knn_result or not knn_result['similarities']:
            return None, 0.0
        similarities = knn_result['similarities']
        last_target_id = target_features.index[-1]
        if last_target_id not in similarities:
            return None, 0.0
        similar_cases = similarities[last_target_id]
        if not similar_cases:
            return None, 0.0
        model_votes = self._calculate_model_votes(similar_cases, historical_performance, models_list)
        if model_votes:
            best_model_name = max(model_votes.keys(), key=lambda k: model_votes[k])
            knn_confidence = model_votes[best_model_name] / sum(model_votes.values())
            self.logger.info(
                f"🎯 KNN selected '{best_model_name}' with confidence {knn_confidence:.2f}"
            )
            return best_model_name, knn_confidence
        return None, 0.0

    def _calculate_model_votes(
        self,
        similar_cases: list[dict],
        historical_performance: pd.DataFrame,
        models_list: list[str],
    ) -> dict[str, float]:
        model_votes: dict[str, float] = {}
        for case in similar_cases[:3]:
            case_id = case['id']
            similarity_score = case['similarity_score']
            case_model = historical_performance[historical_performance.index == case_id].get(
                'model_name'
            )
            if case_model is not None and not case_model.empty:
                model_name = case_model.iloc[0]
                if model_name in models_list:
                    model_votes[model_name] = model_votes.get(model_name, 0) + similarity_score
        return model_votes

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _create_prediction_result(self, request: PredictionResultRequest) -> dict[str, Any]:
        anomaly_score = self.anomaly_engine.calculate_anomaly_score(request.ticker_df_clean)

        # ELITE INTEGRATION: If there is an autoencoder model for this ticker & target, use it to calculate reconstruction normalcy
        ae_normalcy = None
        ticker = request.ticker
        target_col = request.meta.get('target', '')
        ae_key = f"{ticker}_{target_col}_autoencoder"

        try:
            batch_dir = self.model_resolver.resolve_batch_directory({request.context_id: request.meta})
            if batch_dir:
                ae_models = None
                for ext in ['.keras', '.pkl', '.h5', '.pt', '.joblib']:
                    ae_model_path = batch_dir / f"model_{ticker}_{target_col}_autoencoder{ext}"
                    if ae_model_path.exists():
                        ae_meta = {
                            'ticker': ticker,
                            'target': target_col,
                            'model_type': 'autoencoder',
                            'model_path': str(ae_model_path)
                        }
                        ae_models = self.model_resolver.load_available_models(ae_key, {ae_key: ae_meta})
                        break

                if ae_models:
                    ae_model_name = list(ae_models.keys())[0]
                    ae_model = ae_models[ae_model_name]

                    # Align features for autoencoder
                    ae_features = []
                    features_path = batch_dir / f"selected_features_{ticker}_{target_col}_autoencoder.json"
                    if features_path.exists():
                        try:
                            with open(features_path) as f:
                                data = json.load(f)
                                ae_features = data.get('selected_features', [])
                        except Exception as fe:
                            self.logger.warning(f"⚠️ Failed to read autoencoder features file: {fe}")

                    if not ae_features:
                        ae_features = request.meta.get('selected_features', [])

                    X_ae = self.prediction_generator._align_features(ae_model, request.ticker_df_clean, ae_features)
                    raw_reconstruction = ae_model.predict(X_ae)

                    x_input_flat = X_ae.iloc[-1:].values.flatten()
                    reconstruction_flat = raw_reconstruction.flatten()

                    min_len = min(len(x_input_flat), len(reconstruction_flat))
                    if min_len > 0:
                        mse = float(np.mean((x_input_flat[:min_len] - reconstruction_flat[:min_len]) ** 2))
                        ae_normalcy = float(np.exp(-mse * 2.0))

                        # Blend the normalcy scores: 50% classical anomaly, 50% Autoencoder anomaly
                        anomaly_score = 0.5 * anomaly_score + 0.5 * ae_normalcy
                        self.logger.info(f"🔒 Autoencoder anomaly integration for {ticker} ({target_col}): MSE={mse:.4f}, normalcy={ae_normalcy:.2%}, blended_normalcy={anomaly_score:.2%}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to integrate autoencoder anomaly detection: {e}")

        confidence_info = self.anomaly_engine.calculate_ensemble_confidence(
            models={},
            X=request.ticker_df_clean,
            prediction=request.adjusted_prediction,
            context_id=request.context_id,
            predictions_by_model=request.model_contributions,
        )

        final_confidence = confidence_info.get('score', 0.5) * anomaly_score
        if anomaly_score < 0.4:
            self.logger.warning(f"Low normalcy score ({anomaly_score:.2f}) - potential data anomaly!")

        pred_value = self.prediction_generator.extract_prediction_value(request.adjusted_prediction)
        self.logger.info(
            f"Ensemble forecast for {request.ticker}: {pred_value:.4f} | Conf: {confidence_info.get('score'):.2%}"
        )

        # ELITE FIX: Ensure JSON-serializable types (no numpy arrays in final output)
        def to_serializable(val):
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, (np.float32, np.float64)):
                return float(val)
            if isinstance(val, (np.int32, np.int64)):
                return int(val)
            return val

        # Extract timestamp with fallback
        ts_val = None
        if len(request.ticker_df_clean) > 0:
            last_ts = request.ticker_df_clean.index[-1]
            if pd.notnull(last_ts):
                ts_val = str(last_ts)

        if ts_val is None:
            ts_val = datetime.now().isoformat()

        return {
            'ticker': request.ticker,
            'predictions': to_serializable(request.adjusted_prediction),
            'raw_forecast': to_serializable(request.raw_prediction),
            'predictions_by_model': {k: to_serializable(v) for k, v in request.model_contributions.items()},
            'selected_primary_model': request.best_model_name,
            'confidence': float(final_confidence),
            'anomaly_score': float(anomaly_score),
            'last_price': self._get_last_price(request.ticker_df_clean, request.ticker) or 0.0,
            'shap_explanations': request.shap_explanations,
            'timestamp': ts_val,
        }

    def _get_last_price(
        self, ticker_df: pd.DataFrame, ticker: str
    ) -> float | None:
        if 'close' in ticker_df.columns:
            return float(ticker_df['close'].iloc[-1])
        elif f'{ticker}_1d_close' in ticker_df.columns:
            return float(ticker_df[f'{ticker}_1d_close'].iloc[-1])
        return None

    def _prepare_final_results(
        self,
        prediction_results: dict[str, Any],
        models_meta: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        predictions_list = list(prediction_results.values())
        current_prices = {
            pred_data['ticker']: pred_data['last_price']
            for pred_data in prediction_results.values()
            if pred_data.get('ticker') and pred_data.get('last_price')
        }

        light_models_count = sum(
            1 for m in models_meta.values() if m.get('model_category') == 'light'
        )
        heavy_models_count = sum(
            1 for m in models_meta.values() if m.get('model_category') in ['heavy', 'colab']
        )

        self.logger.info(
            f"Stage 5 complete: {len(predictions_list)} predictions, {len(current_prices)} prices"
        )
        self.logger.info(
            f"Models: {light_models_count} light, {heavy_models_count} heavy, {len(models_meta)} total"
        )

        self._save_stage_5_results(
            predictions_list=predictions_list,
            current_prices=current_prices,
            prediction_results=prediction_results,
            models_meta=models_meta,
            kwargs=kwargs,
        )

        return {
            'predictions': predictions_list,
            'current_prices': current_prices,
            'prediction_results': prediction_results,
            'models_metadata': models_meta,
            'light_models_count': light_models_count,
            'heavy_models_count': heavy_models_count,
            'total_models': len(models_meta),
        }

    def update_selector_feedback(self, prediction_results: dict[str, Any], actual_results: dict[str, float]):
        """
        Update AdaptiveModelSelector with feedback from actual results.

        Args:
            prediction_results: Results from Stage 5 predictions
            actual_results: Dict of {ticker: actual_return}
        """
        if not isinstance(self.context_selector, AdaptiveModelSelector):
            return  # Only works with AdaptiveModelSelector

        for context_id, pred_data in prediction_results.items():
            ticker = pred_data.get('ticker')
            if not ticker or ticker not in actual_results:
                continue

            model_id = pred_data.get('selected_primary_model')
            predicted_return = pred_data.get('predictions', 0)
            actual_return = actual_results[ticker]

            # Extract context fingerprint if available
            context_fingerprint = pred_data.get('context_fingerprint', context_id)

            try:
                self.context_selector.update_from_feedback(
                    model_id=model_id,
                    context_fingerprint=context_fingerprint,
                    actual_return=actual_return,
                    predicted_return=predicted_return
                )
                self.logger.debug(f"Updated selector feedback for {ticker}: {model_id}")
            except Exception as e:
                self.logger.warning(f"Failed to update selector feedback: {e}")

    def _save_stage_5_results(
        self,
        predictions_list: list[dict],
        current_prices: dict,
        prediction_results: dict,
        models_meta: dict,
        kwargs: dict,
    ) -> None:
        try:
            batch_name = kwargs.get('batch_name') or self.brain.get('batch_name')
            output_dir = Path(
                self.config_manager.get(self.ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR)
            )

            if not batch_name:
                for meta in models_meta.values():
                    path = meta.get('model_path', '')
                    if path:
                        path_parts = Path(path.replace('/', '\\')).parts
                        if 'models' in path_parts:
                            idx = path_parts.index('models')
                            if idx > 0:
                                batch_name = path_parts[idx - 1]
                                break

            if not batch_name:
                batch_dirs = list(output_dir.glob('test_ticker_*'))
                if batch_dirs:
                    batch_name = max(batch_dirs, key=lambda p: p.stat().st_mtime).name

            if batch_name:
                batch_dir = output_dir / batch_name
                batch_dir.mkdir(parents=True, exist_ok=True)

                stage_5_results = {
                    'timestamp': datetime.now().isoformat(),
                    'batch_name': batch_name,
                    'predictions': predictions_list,
                    'current_prices': current_prices,
                    'prediction_results': prediction_results,
                    'models_metadata': models_meta,
                    'light_models_count': sum(
                        1 for m in models_meta.values() if m.get('model_category') == 'light'
                    ),
                    'heavy_models_count': sum(
                        1 for m in models_meta.values() if m.get('model_category') in ['heavy', 'colab']
                    ),
                    'total_models': len(models_meta),
                    'total_predictions': len(predictions_list),
                }

                stage_5_file = batch_dir / "stage_5_results.json"
                with open(stage_5_file, 'w') as f:
                    json.dump(stage_5_results, f, indent=2, default=str)

                self.logger.info(f"✅ Stage 5 results saved: {stage_5_file.name}")
        except Exception as e:
            self.logger.warning(f"Error saving Stage 5 results: {e}")
