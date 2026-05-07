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
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder
from src.analytics.context.prediction_adjuster import PredictionAdjuster
from src.analytics.context.market_context_analyzer import MarketContextAnalyzer
from src.analytics.signals.signal_analytics import analyze_signals
from src.analytics.signals.significance_detector import detect_significant_events
from src.analytics.detectors.anomaly_detector import AnomalyDetector
from src.analytics.detectors.critical_signal_detector import CriticalSignalDetector
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ModelLoadingError
from src.core.logging.logger import ProjectLogger
from src.ensembling.stacked_ensemble import StackedEnsemble
from src.features.utils.datetime_utils import ensure_datetime_column, normalize_metadata_columns
from src.meta_learning.memory.diary_engine import DiaryEngine
from src.models.loader import ModelLoaderStrategy
from src.models.model_selector.smart_selector import SmartModelSelector
from src.models.model_selector.adaptive_selector import AdaptiveModelSelector
from src.models.model_pool import clear_model_pool, get_model_pool, get_pool_stats
from src.pipeline.stages.base_stage import BaseStage
from src.pipeline.stages.prediction import AnomalyEngine, ModelResolver, PredictionGenerator
from src.predictions.caching import clear_all_caches, get_ensemble_cache


@dataclass
class PredictionResultRequest:
    """Request for creating prediction result."""

    context_id: str
    ticker: str
    adjusted_prediction: float
    raw_prediction: float
    model_contributions: Dict[str, float]
    best_model_name: str
    ticker_df_clean: pd.DataFrame
    meta: Dict[str, Any]


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
        
        # Use AdaptiveModelSelector if enabled in config, otherwise SmartModelSelector
        use_adaptive = self.config_manager.get('prediction.use_adaptive_selector', False)
        if use_adaptive:
            self.context_selector: Union[SmartModelSelector, AdaptiveModelSelector] = AdaptiveModelSelector(
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

    async def run(self, **kwargs) -> Dict[str, Any]:
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
        self, kwargs: Dict[str, Any]
    ) -> tuple[Optional[pd.DataFrame], Dict[str, Any], str]:
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
        self, models_meta: Dict[str, Any], kwargs: Optional[Dict[str, Any]] = None
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
        models_meta: Dict[str, Any],
        features_df: pd.DataFrame,
        market_regime: str,
    ) -> Dict[str, Any]:
        prediction_results: Dict[str, Any] = {}
        
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
        meta: Dict[str, Any],
        features_df: pd.DataFrame,
        market_regime: str,
    ) -> Optional[Dict[str, Any]]:
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
        if target_scaler is None:
            self.logger.warning(f"Target scaler not found for {context_id} - creating fallback scaler")
            target_scaler = self._create_fallback_scaler(meta)

        best_model_name = self._select_best_model_for_context(
            ticker_df_clean, meta, models, ticker, market_regime
        )

        raw_prediction, model_contributions = self.prediction_generator.generate_prediction(
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

        request = PredictionResultRequest(
            context_id=context_id,
            ticker=ticker,
            adjusted_prediction=adjusted_prediction,
            raw_prediction=raw_prediction,
            model_contributions=model_contributions,
            best_model_name=best_model_name,
            ticker_df_clean=ticker_df_clean,
            meta=meta,
        )
        return self._create_prediction_result(request)

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------

    def _process_context_data(
        self, context_id: str, meta: Dict[str, Any], features_df: pd.DataFrame
    ) -> Optional[tuple]:
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
            self.logger.debug(f"Adding {len(missing_features)} missing features filled with 0")
            for f in missing_features:
                ticker_df_clean[f] = 0.0

        filtered_features_list = selected_features
        if selected_features:
            self.logger.info(f"✅ Using {len(filtered_features_list)} selected features for prediction")
        else:
            self.logger.warning("⚠️ No selected features specified in metadata")

        if selected_features and not filtered_features_list:
            self.logger.warning(f"⚠️ None of selected features found for {model_type}")
            return None

        if filtered_features_list:
            ticker_df_clean = ticker_df_clean[filtered_features_list]
            self.logger.info(f" Using {len(filtered_features_list)} features for {model_type}")
        else:
            self.logger.warning(
                f" No selected features for {model_type}, using all {ticker_df_clean.shape[1]} columns"
            )
            filtered_features_list = ticker_df_clean.columns.tolist()

        return ticker_df_clean, filtered_features_list

    def _prepare_ticker_data(
        self, features_df: pd.DataFrame, ticker: str
    ) -> Optional[pd.DataFrame]:
        ticker_df = features_df[features_df['ticker'] == ticker].tail(50)
        if ticker_df.empty:
            self.logger.warning(f"⚠️ No data for ticker {ticker}")
            return None

        ticker_df_clean = ticker_df.copy()
        metadata_cols = ['ticker', 'datetime', 'date', 'interval', 'timeframe', 'hash', 'symbol']
        ticker_df_clean = ticker_df_clean.drop(
            columns=[c for c in metadata_cols if c in ticker_df_clean.columns], errors='ignore'
        )

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

    def _load_target_scaler(self, meta: Dict[str, Any]) -> Optional[Any]:
        ticker = meta.get('ticker', '')
        target_col = meta.get('target', '')
        model_path_str = meta.get('model_path', '')
        if not model_path_str:
            return None

        model_path_str = model_path_str.replace('/', '\\')
        parts = model_path_str.split('\\')
        if 'models' not in parts:
            return None

        models_idx = parts.index('models')
        if models_idx <= 0:
            return None

        batch_name = parts[models_idx - 1]
        base_dir = Path(
            self.config_manager.get(self.ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR)
        )
        scaler_path = base_dir / batch_name / f"scaler_{ticker}_{target_col}.pkl"

        if scaler_path.exists():
            target_scaler = joblib.load(scaler_path)
            if hasattr(target_scaler, 'scale_'):
                if target_scaler.scale_.shape[0] == 1:
                    self.logger.info(f"✅ Loaded target scaler from {scaler_path}")
                    return target_scaler
                else:
                    self.logger.error(
                        f"❌ INVALID scaler! Has {target_scaler.scale_.shape[0]} features instead of 1"
                    )
            else:
                self.logger.warning("⚠️ Scaler has no scale_ attribute")
        else:
            self.logger.debug(f"⚠️ No target scaler found at {scaler_path}")
        return None

    def _create_fallback_scaler(self, meta: Dict[str, Any]) -> Optional[Any]:
        """Create a fallback target scaler when original scaler is not found"""
        try:
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            ticker = meta.get('ticker', '')
            target_col = meta.get('target', '')
            
            # Create a simple StandardScaler with default parameters
            # This assumes the target was normalized using StandardScaler during training
            fallback_scaler = StandardScaler()
            
            # Fit the scaler with some dummy data to make it functional
            # We use typical ranges for different target types
            if 'return' in target_col.lower():
                # Returns are typically small (-0.1 to 0.1)
                dummy_data = np.array([[-0.05], [0.0], [0.05]])
            elif 'up' in target_col.lower() or 'down' in target_col.lower():
                # Binary targets (0/1)
                dummy_data = np.array([[0.0], [0.5], [1.0]])
            elif 'multi' in target_col.lower():
                # Multi-class targets (0, 1, 2...)
                dummy_data = np.array([[0.0], [1.0], [2.0]])
            else:
                # Default case - assume normalized data
                dummy_data = np.array([[0.0], [0.5], [1.0]])
            
            fallback_scaler.fit(dummy_data)
            
            self.logger.info(f"✅ Created fallback scaler for {ticker}_{target_col}")
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
        meta: Dict[str, Any],
        models: Dict[str, Any],
        ticker: str,
        market_regime: str,
    ) -> str:
        models_list = list(models.keys())
        target_type = meta.get('target_type', 'classification')

        best_model_name, knn_confidence = self._perform_knn_similarity_analysis(
            ticker_df_clean, models_list
        )
        if best_model_name is None or best_model_name not in models_list:
            # Use appropriate selector method based on type
            if isinstance(self.context_selector, AdaptiveModelSelector):
                # AdaptiveModelSelector uses fingerprint-based selection
                context_fingerprint = self._create_context_fingerprint(ticker_df_clean, market_regime)
                best_model_name = self.context_selector.select_best_model_adaptive(context_fingerprint)
                # Validate model is in available list
                if best_model_name not in models_list:
                    best_model_name = models_list[0] if models_list else "lightgbm"
            else:
                # SmartModelSelector uses context-based selection
                best_model_name = self.context_selector.select_best_model(
                    ticker_df_clean,
                    target_type,
                    models_list
                )[0]
            self.logger.info(
                f"Contextual Selector chose '{best_model_name}' for {ticker} in {market_regime} regime."
            )
        else:
            self.logger.info(
                f"KNN Similarity chose '{best_model_name}' for {ticker} (confidence: {knn_confidence:.2f})"
            )
        return best_model_name or ""

    def _perform_knn_similarity_analysis(
        self, ticker_df_clean: pd.DataFrame, models_list: List[str]
    ) -> tuple[Optional[str], float]:
        try:
            target_features = ticker_df_clean.tail(5)
            historical_performance = pd.DataFrame()
            if not historical_performance.empty:
                return self._analyze_knn_similarities(target_features, historical_performance, models_list)
        except Exception as e:
            self.logger.warning(f"⚠️ KNN similarity failed: {e}, falling back to SmartModelSelector")
        return None, 0.0

    def _analyze_knn_similarities(
        self,
        target_features: pd.DataFrame,
        historical_performance: pd.DataFrame,
        models_list: List[str],
    ) -> tuple[Optional[str], float]:
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
        knn_result: Dict,
        target_features: pd.DataFrame,
        historical_performance: pd.DataFrame,
        models_list: List[str],
    ) -> tuple[Optional[str], float]:
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
        similar_cases: List[Dict],
        historical_performance: pd.DataFrame,
        models_list: List[str],
    ) -> Dict[str, float]:
        model_votes: Dict[str, float] = {}
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

    def _create_prediction_result(self, request: PredictionResultRequest) -> Dict[str, Any]:
        anomaly_score = self.anomaly_engine.calculate_anomaly_score(request.ticker_df_clean)
        confidence_info = self.anomaly_engine.calculate_ensemble_confidence(
            models={},
            X=request.ticker_df_clean,
            prediction=request.adjusted_prediction,
            context_id=request.context_id,
        )

        final_confidence = confidence_info.get('score', 0.5) * anomaly_score
        if anomaly_score < 0.8:
            self.logger.warning(f"Low anomaly score ({anomaly_score:.2f}) - potential data anomaly!")

        pred_value = self.prediction_generator.extract_prediction_value(request.adjusted_prediction)
        self.logger.info(
            f"Ensemble forecast for {request.ticker}: {pred_value:.4f} | Conf: {confidence_info.get('score'):.2%}"
        )

        return {
            'ticker': request.ticker,
            'predictions': request.adjusted_prediction,
            'raw_forecast': request.raw_prediction,
            'predictions_by_model': request.model_contributions,
            'selected_primary_model': request.best_model_name,
            'confidence': final_confidence,
            'anomaly_score': anomaly_score,
            'last_price': self._get_last_price(request.ticker_df_clean, request.ticker),
            'timestamp': (
                request.ticker_df_clean.index[-1]
                if isinstance(request.ticker_df_clean.index, pd.DatetimeIndex)
                else None
            ),
        }

    def _get_last_price(
        self, ticker_df: pd.DataFrame, ticker: str
    ) -> Optional[float]:
        if 'close' in ticker_df.columns:
            return float(ticker_df['close'].iloc[-1])
        elif f'{ticker}_1d_close' in ticker_df.columns:
            return float(ticker_df[f'{ticker}_1d_close'].iloc[-1])
        return None

    def _prepare_final_results(
        self,
        prediction_results: Dict[str, Any],
        models_meta: Dict[str, Any],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
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
    
    def update_selector_feedback(self, prediction_results: Dict[str, Any], actual_results: Dict[str, float]):
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
        predictions_list: List[Dict],
        current_prices: Dict,
        prediction_results: Dict,
        models_meta: Dict,
        kwargs: Dict,
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
