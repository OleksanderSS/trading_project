# src/pipeline/stages/stage_5_prediction.py

"""
Stage 5: Prediction Generation with Stacked Ensembles and Contextual Adjustments

Uses champion models and stacked ensembles to generate forecasts, 
incorporating real-time market regime adjustments and historical performance.
"""

import os
import json
from typing import Optional, Any, Dict, List
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.meta_learning.memory.diary_engine import DiaryEngine
from src.ensembling.stacked_ensemble import StackedEnsemble
from src.analytics.context.prediction_adjuster import PredictionAdjuster
from src.models.model_selector.smart_selector import SmartModelSelector
from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder
from src.features.utils.datetime_utils import ensure_datetime_column, normalize_metadata_columns
from src.models.loader import ModelLoaderStrategy
from src.core.error_handling.error_handler import ModelLoadingError
from src.predictions.caching import get_ensemble_cache, clear_all_caches
from src.models.model_pool import get_model_pool, clear_model_pool, get_pool_stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

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
    
    # Constants to avoid duplication
    ACCUMULATION_OUTPUT_DIR_CONFIG = 'system.accumulation.output_dir'
    DEFAULT_ACCUMULATION_DIR = 'data/colab/accumulated'
    def __init__(self, config_manager: UnifiedConfigManager, error_handler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger("PredictionStage")
        self.prediction_config = self.config_manager.get_config('prediction', {})
        
        # Use centralized path getter method
        self.models_path = self.config_manager.get_models_path()
        
        self.diary = DiaryEngine()
        self.adjuster = PredictionAdjuster(config=self.config_manager.get('analysis.prediction_adjustment', {}))
        self.ensemble_factory = StackedEnsemble()
        self.context_selector = SmartModelSelector()
        self.knn_similarity = KnnSimilarityFinder(config={'n_neighbors': 5})
        self.model_loader = ModelLoaderStrategy(self.logger)
        
        # ✅ Phase 3 Optimization: Initialize ensemble cache (40-70% speedup)
        self.ensemble_cache = get_ensemble_cache(maxsize=5000)
        self.logger.info("✅ Ensemble prediction cache enabled (LRU, maxsize=5000)")
        
        # ✅ Phase 3 Optimization: Initialize model pool for lazy loading (30-40% speedup)
        max_models = self.config_manager.get('performance.model_pool_size', 50)
        self.model_pool = get_model_pool(max_models=max_models)
        self.logger.info(f"✅ Model pool enabled (maxsize={max_models}, LRU eviction)")
        
        # ✅ NEW: Cache for anomaly estimators (IsolationForest/LOF) to prevent per-call fitting
        self._anomaly_estimators_cache = {}

    def _validate_inputs(self, features_df, models_meta) -> tuple[bool, str]:
        """Validate inputs for the prediction stage."""
        if features_df is None or features_df.empty or not models_meta:
            self.logger.warning("Required features or model metadata not found. Skipping Stage 5.")
            self.logger.warning(f"  - features_df is None: {features_df is None}")
            self.logger.warning(f"  - features_df empty: {features_df.empty if features_df is not None else 'N/A'}")
            self.logger.warning(f"  - models_meta empty: {not models_meta}")
            return False, "Invalid inputs"
        return True, "Valid inputs"

    def _check_local_models(self, models_meta: Dict[str, Any]) -> bool:
        """Check if models are available locally."""
        has_local_models = False
        for context_id, meta in models_meta.items():
            model_path = meta.get('model_path', '')
            if model_path and '/content/drive' not in model_path and ('data\\' in model_path or 'data/' in model_path):
                has_local_models = True
                self.logger.debug(f"✅ Found local model: {context_id} -> {model_path}")
                break
        return has_local_models

    def _log_model_status(self, models_meta: Dict[str, Any]) -> None:
        """Log model status information."""
        self.logger.warning("⚠️ All models are from Colab (not available locally).")
        self.logger.warning("   Checked models:")
        for context_id, meta in list(models_meta.items())[:5]:
            model_path = meta.get('model_path', '')
            model_type = meta.get('model_type', '')
            self.logger.warning(f"   - {context_id}: model_path='{model_path}', model_type='{model_type}'")

    def _resolve_batch_directory(self, models_meta: Dict[str, Any], kwargs: Dict[str, Any] = None) -> Optional[Path]:
        """Resolve batch directory from kwargs batch_name or model paths."""
        base_dir = Path(self.config_manager.get(self.ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR))
        
        # Priority 1: batch_name from kwargs
        if kwargs:
            batch_name = kwargs.get('batch_name')
            if batch_name:
                batch_dir = base_dir / batch_name
                if batch_dir.exists():
                    self.logger.info(f"✅ Resolved batch_dir from batch_name: {batch_dir}")
                    return batch_dir
        
        # Priority 2: extract from model_path (if non-empty)
        for context_id, meta in models_meta.items():
            model_path = meta.get('model_path', '')
            if model_path:
                model_path_str = model_path.replace('/', os.sep)
                parts = Path(model_path_str).parts
                for i, part in enumerate(parts):
                    if part == 'accumulated' and i + 1 < len(parts):
                        batch_dir = base_dir / parts[i + 1]
                        if batch_dir.exists():
                            self.logger.info(f"✅ Resolved batch_dir from model_path: {batch_dir}")
                            return batch_dir
        
        # Priority 3: use most recently modified subdir
        if base_dir.exists():
            subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
            if subdirs:
                chosen = max(subdirs, key=lambda p: p.stat().st_mtime)
                self.logger.info(f"✅ Using most recent batch_dir: {chosen}")
                return chosen
        
        return None

    def _update_local_model_paths(self, models_meta: Dict[str, Any], batch_dir: Path) -> bool:
        """Update model paths to use local files found in batch_dir."""
        # Build an index of available model files: {stem_lower -> full_path}
        model_extensions = {'.keras', '.pkl', '.h5', '.pt', '.joblib'}
        available_files = {}
        for f in batch_dir.iterdir():
            if f.is_file() and f.suffix in model_extensions:
                available_files[f.stem.lower()] = f
        
        if not available_files:
            self.logger.warning(f"⚠️ No model files found in: {batch_dir}")
            return False
        
        self.logger.info(f"✅ Found {len(available_files)} model files in {batch_dir}")
        has_local_models = False
        
        for context_id, meta in models_meta.items():
            ticker = meta.get('ticker', '')
            target = meta.get('target', '')
            model_type = meta.get('model_type', '')
            
            if not ticker or not model_type:
                continue
            
            # Files are named: model_{TICKER}_{TARGET}*_{MODEL_TYPE}
            # Search by matching ticker + model_type in filename
            search_key = f"{ticker}_{target}".lower().replace('-', '_')
            
            matched = None
            for stem, fpath in available_files.items():
                stem_lower = stem.lower()
                if (ticker.lower() in stem_lower and 
                    model_type.lower() in stem_lower and
                    target.lower().replace('-', '_') in stem_lower):
                    matched = fpath
                    break
            
            if matched:
                meta['model_path'] = str(matched)
                has_local_models = True
                self.logger.debug(f"✅ Mapped {context_id} -> {matched.name}")
            # Don't warn for every miss — too noisy with 1638 models
        
        mapped = sum(1 for m in models_meta.values() if m.get('model_path'))
        self.logger.info(f"📊 Mapped model paths: {mapped}/{len(models_meta)}")
        return has_local_models

    def _prepare_ticker_data(self, features_df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Prepare and clean ticker data for prediction."""
        ticker_df = features_df[features_df['ticker'] == ticker].tail(50)
        if ticker_df.empty:
            self.logger.warning(f"⚠️ No data for ticker {ticker}")
            return None

        ticker_df_clean = ticker_df.copy()
        metadata_cols = ['ticker', 'datetime', 'date', 'interval', 'timeframe', 'hash', 'symbol']
        ticker_df_clean = ticker_df_clean.drop(columns=[c for c in metadata_cols if c in ticker_df_clean.columns], errors='ignore')
        
        for col in ticker_df_clean.columns:
            try:
                ticker_df_clean[col] = pd.to_numeric(ticker_df_clean[col], errors='coerce')
            except (ValueError, TypeError) as e:
                self.logger.debug(f"Failed to convert column {col} to numeric: {e}")
                ticker_df_clean = ticker_df_clean.drop(columns=[col], errors='ignore')
        
        ticker_df_clean = ticker_df_clean.fillna(0)
        ticker_df_clean = ticker_df_clean.replace([np.inf, -np.inf], 0)
        
        if ticker_df_clean.empty or ticker_df_clean.dtypes.apply(lambda x: x.kind not in 'biufc').any():
            self.logger.warning(f"⚠️ Data for {ticker} contains non-numeric columns, skipping")
            return None
            
        return ticker_df_clean

    def _get_filtered_features(self, selected_features: List[str], ticker_df_clean: pd.DataFrame) -> List[str]:
        """Get filtered features list based on selected features."""
        # Always use all selected features as expected by the model
        filtered_features_list = selected_features
        if filtered_features_list:
            self.logger.info(f"✅ Using {len(filtered_features_list)} selected features for prediction")
        else:
            self.logger.warning("⚠️ No selected features specified in metadata")
        return filtered_features_list

    def _process_context_data(self, context_id: str, meta: Dict[str, Any], features_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process and prepare data for a specific context."""
        ticker = meta.get('ticker')
        target_col = meta.get('target', '')
        model_type = meta.get('model_type', '')
        
        self.logger.info(f"🔍 Processing context: {context_id}")
        self.logger.info(f"   ticker={ticker}, target={target_col}, model_type={model_type}")
        
        # Prepare ticker data
        ticker_df_clean = self._prepare_ticker_data(features_df, ticker)
        if ticker_df_clean is None:
            return None
        
        # Get filtered features
        selected_features = meta.get('selected_features', [])
        self.logger.debug(f"🔍 Stage 5: context_id={context_id}")
        self.logger.debug(f"🔍 Stage 5: selected_features from metadata: {len(selected_features)} features")
        
        # Ensure all expected features are present in the DataFrame (fill missing with 0)
        missing_features = [f for f in selected_features if f not in ticker_df_clean.columns]
        if missing_features:
            self.logger.debug(f"Adding {len(missing_features)} missing features to DataFrame filled with 0")
            for f in missing_features:
                ticker_df_clean[f] = 0.0
                
        filtered_features_list = self._get_filtered_features(selected_features, ticker_df_clean)
        if selected_features and not filtered_features_list:
            self.logger.warning(f"⚠️ None of selected features found for {model_type}")
            return None
        
        # Apply feature filtering
        if filtered_features_list:
            ticker_df_clean = ticker_df_clean[filtered_features_list]
            self.logger.info(f" Using {len(filtered_features_list)} features for {model_type}")
        else:
            self.logger.warning(f" No selected features found for {model_type}, using all {ticker_df_clean.shape[1]} columns")
            filtered_features_list = ticker_df_clean.columns.tolist()
        
        return ticker_df_clean, filtered_features_list

    def _load_target_scaler(self, meta: Dict[str, Any]) -> Optional[Any]:
        """Load target scaler for denormalization."""
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
        base_dir = Path(self.config_manager.get(self.ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR))
        batch_dir = base_dir / batch_name
        
        scaler_path = batch_dir / f"scaler_{ticker}_{target_col}.pkl"
        if scaler_path.exists():
            target_scaler = joblib.load(scaler_path)
            if hasattr(target_scaler, 'scale_'):
                if target_scaler.scale_.shape[0] == 1:
                    self.logger.info(f"✅ Loaded target scaler from {scaler_path}")
                    return target_scaler
                else:
                    self.logger.error(f"❌ INVALID scaler! Has {target_scaler.scale_.shape[0]} features instead of 1")
            else:
                self.logger.warning("⚠️ Scaler has no scale_ attribute")
        else:
            self.logger.debug(f"⚠️ No target scaler found at {scaler_path}")
        
        return None

    def _perform_knn_similarity_analysis(self, ticker_df_clean: pd.DataFrame, models_list: List[str]) -> tuple[Optional[str], float]:
        """Perform KNN similarity analysis for model selection."""
        try:
            target_features = ticker_df_clean.tail(5)
            historical_performance = pd.DataFrame()
            
            if not historical_performance.empty:
                return self._analyze_knn_similarities(target_features, historical_performance, models_list)
        except Exception as e:
            self.logger.warning(f"⚠️ KNN similarity failed: {e}, falling back to SmartModelSelector")
        
        return None, 0.0

    def _analyze_knn_similarities(self, target_features: pd.DataFrame, historical_performance: pd.DataFrame, models_list: List[str]) -> tuple[Optional[str], float]:
        """Analyze KNN similarities and select best model."""
        historical_features = historical_performance.get('features', pd.DataFrame())
        if not historical_features.empty and len(historical_features.columns) == len(target_features.columns):
            knn_result = self.knn_similarity.analyze({
                'historical_features': historical_features,
                'target_features': target_features
            })
            return self._process_knn_results(knn_result, target_features, historical_performance, models_list)
        
        return None, 0.0

    def _process_knn_results(self, knn_result: Dict, target_features: pd.DataFrame, historical_performance: pd.DataFrame, models_list: List[str]) -> tuple[Optional[str], float]:
        """Process KNN results to find best model."""
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
            best_model_name = max(model_votes, key=model_votes.get)
            knn_confidence = model_votes[best_model_name] / sum(model_votes.values())
            self.logger.info(f"🎯 KNN selected '{best_model_name}' with confidence {knn_confidence:.2f}")
            return best_model_name, knn_confidence
            
        return None, 0.0

    def _calculate_model_votes(self, similar_cases: List[Dict], historical_performance: pd.DataFrame, models_list: List[str]) -> Dict[str, float]:
        """Calculate model votes from similar cases."""
        model_votes = {}
        for case in similar_cases[:3]:
            case_id = case['id']
            similarity_score = case['similarity_score']
            case_model = historical_performance[historical_performance.index == case_id].get('model_name')
            if case_model is not None and not case_model.empty:
                model_name = case_model.iloc[0]
                if model_name in models_list:
                    model_votes[model_name] = model_votes.get(model_name, 0) + similarity_score
        return model_votes

    def _select_best_model(self, ticker_df_clean: pd.DataFrame, ticker: str, target_type: str, models_list: List[str]) -> str:
        """Select the best model using context selector."""
        best_model_name, _ = self.context_selector.select_best_model(
            df=ticker_df_clean,
            target_type=target_type,
            available_models=models_list
        )
        return best_model_name

    def _generate_ensemble_prediction(self, models: Dict[str, Any], ticker_df_clean: pd.DataFrame, filtered_features_list: List[str], market_regime: str, context_id: str) -> tuple[float, Dict[str, float]]:
        """Generate ensemble prediction from multiple models."""
        model_preds = {}
        for m_name, m_inst in models.items():
            feature_cols = filtered_features_list if filtered_features_list else ticker_df_clean.columns.tolist()
            model_features = ticker_df_clean[feature_cols] if all(c in ticker_df_clean.columns for c in feature_cols) else ticker_df_clean
            self.logger.debug(f"   {m_name}: X shape={model_features.shape}, features={len(feature_cols)}")
            
            if 'autoencoder' in m_name.lower():
                self.logger.debug("   ⏭️ Skipping autoencoder for prediction (used only for anomaly detection)")
                continue
            
            model_preds[m_name] = self.ensemble_cache.get_or_compute_model_prediction(
                features=model_features,
                model_id=m_name,
                model_fn=lambda features=model_features, model=m_inst: model.predict(features)
            )
        
        if not model_preds:
            self.logger.warning(f"⚠️ No models for prediction (only autoencoder), skipping {context_id}")
            return None, {}
        
        preds_df = pd.DataFrame(model_preds)
        ensemble_result = self.ensemble_factory.predict(
            X=preds_df,
            context_params={"ticker": ticker_df_clean.get('ticker', 'unknown'), "regime": market_regime}
        )
        raw_prediction = ensemble_result.final_signal
        model_contributions = ensemble_result.active_weights
        
        return raw_prediction, model_contributions

    def _generate_single_model_prediction(self, models: Dict[str, Any], best_model_name: str, ticker_df_clean: pd.DataFrame, filtered_features_list: List[str]) -> tuple[float, Dict[str, float]]:
        """Generate prediction from a single selected model."""
        selected_model = models.get(best_model_name, list(models.values())[0])
        if 'autoencoder' in best_model_name.lower():
            self.logger.warning("⚠️ Autoencoder not suitable for regression prediction")
            return None, {}
        
        feature_cols = filtered_features_list if filtered_features_list else ticker_df_clean.columns.tolist()
        X = ticker_df_clean[feature_cols] if all(c in ticker_df_clean.columns for c in feature_cols) else ticker_df_clean
        self.logger.debug(f"   {best_model_name}: X shape={X.shape}, features={len(feature_cols)}")
        
        raw_prediction = selected_model.predict(X)
        pred_value = raw_prediction[-1] if isinstance(raw_prediction, np.ndarray) else raw_prediction
        model_contributions = {best_model_name: pred_value}
        
        return raw_prediction, model_contributions

    def _adjust_prediction_contextually(self, raw_prediction, best_model_name: str, market_regime: str, ticker: str) -> float:
        """Adjust prediction based on market context."""
        adjustment_result = self.adjuster.analyze(
            data={
                'predictions': {best_model_name: raw_prediction[-1] if isinstance(raw_prediction, np.ndarray) else raw_prediction},
                'market_regime': market_regime,
                'ticker': ticker
            }
        )
        return adjustment_result.get('enhanced_predictions', {}).get(best_model_name, raw_prediction)

    def _denormalize_prediction(self, adjusted_prediction, target_scaler) -> float:
        """Denormalize prediction using target scaler."""
        if target_scaler is None:
            return adjusted_prediction
            
        try:
            if isinstance(adjusted_prediction, np.ndarray):
                if adjusted_prediction.ndim == 1:
                    pred_to_denorm = adjusted_prediction[-1:].reshape(-1, 1)
                else:
                    pred_to_denorm = adjusted_prediction.reshape(-1, 1)
            else:
                pred_to_denorm = np.array([[adjusted_prediction]])
            
            if hasattr(target_scaler, 'scale_') and target_scaler.scale_.shape[0] != 1:
                raise ValueError(f"Scaler has wrong number of features: {target_scaler.scale_.shape[0]} instead of 1")
            
            denormalized = target_scaler.inverse_transform(pred_to_denorm)
            result = float(denormalized.flatten()[-1])
            self.logger.info(f"✅ Denormalized prediction: {result:.6f}")
            return result
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to denormalize prediction: {e}")
            return adjusted_prediction

    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Generates adjusted predictions for tickers processed in earlier stages.

        Args:
            **kwargs: Dictionary containing 'features_data' and 'models_metadata'.

        Returns:
            Dict[str, Any]: Updated pipeline data with 'prediction_results'.
        """
        # Extract and validate inputs
        features_df, models_meta, market_regime = self._prepare_inputs(kwargs)
        if features_df is None or (hasattr(features_df, 'empty') and features_df.empty) or not models_meta:
            return {}
        
        # Check and handle local models
        if not self._ensure_local_models(models_meta, kwargs):
            return {}

        # Generate predictions for all contexts
        prediction_results = self._generate_predictions_for_contexts(
            models_meta, features_df, market_regime
        )

        # Prepare and return results
        return self._prepare_final_results(prediction_results, models_meta, kwargs)

    def _prepare_inputs(self, kwargs: Dict[str, Any]) -> tuple[Optional[pd.DataFrame], Dict[str, Any], str]:
        """Prepare and validate inputs for prediction stage."""
        features_df = next(
            (kwargs[k] for k in ('features_data', 'features_df', 'enriched_data')
             if k in kwargs and kwargs[k] is not None),
            None
        )
        models_meta = kwargs.get('models_metadata') or kwargs.get('models_meta', {})
        market_regime = kwargs.get('market_regime', 'neutral')
        
        # Load models metadata if not provided
        if not models_meta:
            models_meta = self._load_models_metadata_from_disk(kwargs)
            if not models_meta:
                self.logger.warning("Failed to load models_metadata from disk")
                return None, {}, market_regime
            else:
                self.logger.info(f"Loaded {len(models_meta)} models from disk")
        
        # Validate inputs
        is_valid, _ = self._validate_inputs(features_df, models_meta)
        if not is_valid:
            return None, {}, market_regime
        
        # Normalize features
        if isinstance(features_df, pd.DataFrame):
            features_df = normalize_metadata_columns(features_df)
            self.logger.info("Normalized features_df at stage entry")
        
        return features_df, models_meta, market_regime

    def _ensure_local_models(self, models_meta: Dict[str, Any], kwargs: Dict[str, Any] = None) -> bool:
        """Ensure local models are available."""
        has_local_models = self._check_local_models(models_meta)
        if not has_local_models:
            self._log_model_status(models_meta)
            batch_dir = self._resolve_batch_directory(models_meta, kwargs)
            if batch_dir and batch_dir.exists():
                has_local_models = self._update_local_model_paths(models_meta, batch_dir)
            
            if not has_local_models:
                self.logger.error("No local models found. Skipping Stage 5.")
                return False
        
        return True

    def _generate_predictions_for_contexts(self, models_meta: Dict[str, Any], features_df: pd.DataFrame, market_regime: str) -> Dict[str, Any]:
        """Generate predictions for all contexts."""
        prediction_results = {}
        self.logger.info(f"Generating ensemble predictions for {len(models_meta)} contexts...")

        for context_id, meta in models_meta.items():
            try:
                result = self._process_single_context(context_id, meta, features_df, market_regime)
                if result:
                    prediction_results[context_id] = result
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                self.handle_stage_error(e, context=f"Prediction-{context_id}", severity="error")
                self.logger.error(f"Prediction failed for context {context_id}: {e}", exc_info=True)

        return prediction_results

    def _process_single_context(self, context_id: str, meta: Dict[str, Any], features_df: pd.DataFrame, market_regime: str) -> Optional[Dict[str, Any]]:
        """Process a single context and generate prediction."""
        # Process context data
        context_result = self._process_context_data(context_id, meta, features_df)
        if context_result is None:
            return None
        
        ticker_df_clean, filtered_features_list = context_result
        ticker = meta.get('ticker')

        # Load models and scaler
        models = self._load_available_models(context_id, {context_id: meta})
        if not models:
            self.logger.warning(f"No models found for {context_id}, skipping")
            return None
        
        target_scaler = self._load_target_scaler(meta)
        if target_scaler is None:
            self.logger.warning(f"Target scaler not found for {context_id} - prediction remains normalized")

        # Select best model
        best_model_name = self._select_best_model_for_context(ticker_df_clean, meta, models, ticker, market_regime)

        # Generate prediction
        raw_prediction, model_contributions = self._generate_prediction_for_context(
            models, best_model_name, ticker_df_clean, filtered_features_list, market_regime, context_id
        )
        if raw_prediction is None:
            return None

        # Adjust and denormalize prediction
        adjusted_prediction = self._adjust_and_denormalize_prediction(
            raw_prediction, best_model_name, market_regime, ticker, target_scaler
        )

        # Calculate confidence and create result
        prediction_request = PredictionResultRequest(
            context_id=context_id,
            ticker=ticker,
            adjusted_prediction=adjusted_prediction,
            raw_prediction=raw_prediction,
            model_contributions=model_contributions,
            best_model_name=best_model_name,
            ticker_df_clean=ticker_df_clean,
            meta=meta
        )
        return self._create_prediction_result(prediction_request)

    def _select_best_model_for_context(self, ticker_df_clean: pd.DataFrame, meta: Dict[str, Any], models: Dict[str, Any], ticker: str, market_regime: str) -> str:
        """Select the best model for the current context."""
        models_list = list(models.keys())
        target_type = meta.get('target_type', 'classification')
        
        # Try KNN similarity analysis first
        best_model_name, knn_confidence = self._perform_knn_similarity_analysis(ticker_df_clean, models_list)
        
        # Fall back to context selector if KNN fails
        if best_model_name is None or best_model_name not in models_list:
            best_model_name = self._select_best_model(ticker_df_clean, ticker, target_type, models_list)
            self.logger.info(f"Contextual Selector chose '{best_model_name}' for {ticker} in {market_regime} regime.")
        else:
            self.logger.info(f"KNN Similarity chose '{best_model_name}' for {ticker} (confidence: {knn_confidence:.2f})")
        
        return best_model_name

    def _generate_prediction_for_context(self, models: Dict[str, Any], best_model_name: str, ticker_df_clean: pd.DataFrame, filtered_features_list: List[str], market_regime: str, context_id: str) -> tuple[Optional[float], Dict[str, float]]:
        """Generate prediction for the context."""
        if len(models) > 1:
            return self._generate_ensemble_prediction(models, ticker_df_clean, filtered_features_list, market_regime, context_id)
        else:
            return self._generate_single_model_prediction(models, best_model_name, ticker_df_clean, filtered_features_list)

    def _adjust_and_denormalize_prediction(self, raw_prediction: float, best_model_name: str, market_regime: str, ticker: str, target_scaler) -> float:
        """Adjust prediction contextually and denormalize."""
        adjusted_prediction = self._adjust_prediction_contextually(raw_prediction, best_model_name, market_regime, ticker)
        return self._denormalize_prediction(adjusted_prediction, target_scaler)

    def _create_prediction_result(self, request: PredictionResultRequest) -> Dict[str, Any]:
        """Create prediction result dictionary."""
        # Calculate anomaly score and confidence
        anomaly_score = self._calculate_anomaly_score(request.ticker_df_clean)
        confidence_info = self._calculate_ensemble_confidence(
            models={},  # Will be populated by caller if needed
            X=request.ticker_df_clean,
            prediction=request.adjusted_prediction,
            context_id=request.context_id
        )
        
        final_confidence = confidence_info.get('score', 0.5) * anomaly_score
        if anomaly_score < 0.8:
            self.logger.warning(f"Low anomaly score ({anomaly_score:.2f}) - potential data anomaly!")

        pred_value = self._extract_prediction_value(request.adjusted_prediction)
        self.logger.info(f"Ensemble forecast for {request.ticker}: {pred_value:.4f} | Conf: {confidence_info.get('score'):.2%}")

        return {
            'ticker': request.ticker,
            'predictions': request.adjusted_prediction,
            'raw_forecast': request.raw_prediction,
            'predictions_by_model': request.model_contributions,
            'selected_primary_model': request.best_model_name,
            'confidence': final_confidence,
            'anomaly_score': anomaly_score,
            'last_price': self._get_last_price(request.ticker_df_clean, request.ticker),
            'timestamp': request.ticker_df_clean.index[-1] if isinstance(request.ticker_df_clean.index, pd.DatetimeIndex) else None
        }

    def _prepare_final_results(self, prediction_results: Dict[str, Any], models_meta: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare final results for return."""
        predictions_list = list(prediction_results.values())
        
        current_prices = {}
        for context_id, pred_data in prediction_results.items():
            ticker = pred_data.get('ticker')
            last_price = pred_data.get('last_price')
            if ticker and last_price:
                current_prices[ticker] = last_price
        
        light_models_count = sum(1 for m in models_meta.values() if m.get('model_category') == 'light')
        heavy_models_count = sum(1 for m in models_meta.values() if m.get('model_category') in ['heavy', 'colab'])
        
        self.logger.info(f"Stage 5 complete: {len(predictions_list)} predictions, {len(current_prices)} prices")
        self.logger.info(f"Models: {light_models_count} light, {heavy_models_count} heavy, {len(models_meta)} total")
        
        # Save Stage 5 results to disk
        self._save_stage_5_results(
            predictions_list=predictions_list,
            current_prices=current_prices,
            prediction_results=prediction_results,
            models_meta=models_meta,
            kwargs=kwargs
        )        
        return {
            'predictions': predictions_list,
            'current_prices': current_prices,
            'prediction_results': prediction_results,
            'models_metadata': models_meta,
            'light_models_count': light_models_count,
            'heavy_models_count': heavy_models_count,
            'total_models': len(models_meta)
        }

    def _save_stage_5_results(self, predictions_list: List[Dict], current_prices: Dict, prediction_results: Dict, models_meta: Dict, kwargs: Dict) -> None:
        """Saves Stage 5 results to disk for flexible runs."""
        try:
            # Try multiple sources for batch_name
            batch_name = kwargs.get('batch_name') or self.brain.get('batch_name')
            output_dir = Path(self.config_manager.get(self.ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR))
            
            if not batch_name:
                # Try to extract from model paths in models_meta
                for meta in models_meta.values():
                    path = meta.get('model_path', '')
                    if path:
                        path_parts = Path(path.replace('/', '\\')).parts
                        if 'models' in path_parts:
                            idx = path_parts.index('models')
                            if idx > 0:
                                batch_name = path_parts[idx-1]
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
                    'light_models_count': sum(1 for m in models_meta.values() if m.get('model_category') == 'light'),
                    'heavy_models_count': sum(1 for m in models_meta.values() if m.get('model_category') in ['heavy', 'colab']),
                    'total_models': len(models_meta),
                    'total_predictions': len(predictions_list)
                }
                
                stage_5_file = batch_dir / "stage_5_results.json"
                with open(stage_5_file, 'w') as f:
                    json.dump(stage_5_results, f, indent=2, default=str)
                
                self.logger.info(f"✅ Stage 5 results saved: {stage_5_file.name}")
        except Exception as e:
            self.logger.warning(f"Error saving Stage 5 results: {e}")

    def _load_models_metadata_from_disk(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Loads models_metadata from disk if not provided."""
        models_metadata = {}
        batch_dir = self._resolve_batch_directory_from_kwargs(kwargs)
        
        if batch_dir:
            self._load_light_models_from_disk(batch_dir, models_metadata)
            self._load_heavy_models_from_disk(batch_dir, models_metadata)
        
        return models_metadata

    def _resolve_batch_directory_from_kwargs(self, kwargs: Dict[str, Any]) -> Optional[Path]:
        """Resolve batch directory from kwargs."""
        batch_name = kwargs.get('batch_name')
        output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
        
        if not batch_name:
            batch_dirs = list(output_dir.glob('test_ticker_*'))
            if batch_dirs:
                batch_name = max(batch_dirs, key=lambda p: p.stat().st_mtime).name
                self.logger.info(f"Found latest batch: {batch_name}")
        
        if batch_name:
            return output_dir / batch_name
        return None

    def _load_light_models_from_disk(self, batch_dir: Path, models_metadata: Dict[str, Any]) -> None:
        """Load light models from disk."""
        light_results_files = list(batch_dir.glob("light_models_results_*.json"))
        if light_results_files:
            latest_light = max(light_results_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_light, 'r') as f:
                    light_results = json.load(f)
                    light_meta = light_results.get('models_metadata', {})
                    models_metadata.update(light_meta)
                    self.logger.info(f"Loaded {len(light_meta)} light models from {latest_light.name}")
            except Exception as e:
                self.logger.warning(f"Error loading light models: {e}")

    def _load_heavy_models_from_disk(self, batch_dir: Path, models_metadata: Dict[str, Any]) -> None:
        """Load heavy models from disk."""
        colab_summary_file = batch_dir / "colab_results_summary.json"
        if colab_summary_file.exists():
            try:
                with open(colab_summary_file, 'r') as f:
                    colab_results = json.load(f)
                    if 'models_metadata' in colab_results:
                        heavy_meta = colab_results['models_metadata']
                        models_metadata.update(heavy_meta)
                        self.logger.info(f"Loaded {len(heavy_meta)} heavy models from {colab_summary_file.name}")
                    else:
                        self._process_ticker_results_from_colab(colab_results, models_metadata)
            except Exception as e:
                self.logger.warning(f"Error loading colab models: {e}")

    def _process_ticker_results_from_colab(self, colab_results: Dict[str, Any], models_metadata: Dict[str, Any]) -> None:
        """Process ticker results from colab results."""
        ticker_results = colab_results.get('ticker_results', {})
        for ticker, ticker_data in ticker_results.items():
            timeframes = ticker_data.get('timeframes', {})
            for tf, tf_data in timeframes.items():
                results = tf_data.get('results', {})
                for target, target_data in results.items():
                    models = target_data.get('models', {})
                    for model_type, model_data in models.items():
                        context_key = f"{ticker}_{target}_{model_type}"
                        models_metadata[context_key] = {
                            'ticker': ticker,
                            'target': target,
                            'winner': model_type,
                            'model_type': model_type,
                            'model_category': 'heavy',
                            'metrics': model_data.get('metrics', {}),
                            'selected_features': model_data.get('selected_features', [])
                        }

    def _load_available_models(self, context_id: str, models_meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """Loads all available models for a context."""
        models_meta = models_meta or {}
        
        # Try direct model loading first
        direct_result = self._try_load_direct_model(context_id, models_meta)
        if direct_result:
            return direct_result
        
        # Search for models in batch directories
        batch_dir = self._resolve_batch_dir_from_context(context_id, models_meta)
        search_patterns = self._get_model_search_patterns(context_id)
        
        return self._search_and_load_models(batch_dir, search_patterns, context_id, models_meta)

    def _try_load_direct_model(self, context_id: str, models_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try to load model directly from metadata path."""
        if context_id not in models_meta:
            return None
            
        model_path_str = models_meta[context_id].get('model_path', '')
        if not model_path_str:
            return None
        
        direct_path = Path(model_path_str.replace('/', os.sep))
        if not direct_path.exists():
            return None
        
        try:
            model_name = direct_path.stem
            model_meta = self._create_model_meta(context_id, models_meta, model_name, str(direct_path))
            loaded_model = self.model_pool.get_model(
                model_name,
                loader_fn=lambda path=str(direct_path), meta=model_meta: self.model_loader.load_path(path, meta)
            )
            if loaded_model is not None:
                return {model_name: loaded_model}
        except Exception as e:
            self.logger.warning(f"Failed to load model via direct path {direct_path}: {e}")
        
        return None

    def _resolve_batch_dir_from_context(self, context_id: str, models_meta: Dict[str, Any]) -> Path:
        """Resolve batch directory from context metadata."""
        if context_id in models_meta:
            model_path_str = models_meta[context_id].get('model_path', '')
            if model_path_str:
                batch_dir = self._extract_batch_dir_from_path(model_path_str)
                if batch_dir:
                    return batch_dir
        
        return Path(self.config_manager.get(self.ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR))

    def _extract_batch_dir_from_path(self, model_path_str: str) -> Optional[Path]:
        """Extract batch directory from model path."""
        model_path_str = model_path_str.replace('/', '\\')
        parts = model_path_str.split('\\')
        if 'models' in parts:
            models_idx = parts.index('models')
            if models_idx > 0:
                batch_name = parts[models_idx - 1]
                base_dir = Path(self.config_manager.get(self.ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR))
                return base_dir / batch_name
        return None

    def _get_model_search_patterns(self, context_id: str) -> List[str]:
        """Get model search patterns based on context ID format."""
        parts = context_id.split('_')
        if len(parts) >= 4:
            ticker = parts[0]
            target = '_'.join(parts[1:-1]) 
            model_name = parts[-1] 
            
            return [
                # New standard Colab naming
                f"model_{ticker}_{target}*_{model_name}.keras",
                f"model_{ticker}_{target}*_{model_name}.pkl",
                f"model_{ticker}_{target}*_{model_name}.h5",
                f"model_{ticker}_{target}*_{model_name}.pt",
                f"model_{ticker}_{target}*_{model_name}.joblib",
                # Legacy / Generic match
                f"*{ticker}*{target}*{model_name}*.*",
                f"{model_name}_{ticker}_{target}.pt",
                f"CHAMP_{context_id}*.joblib",
                f"MODEL_{context_id}*.joblib",
                f"*{context_id}*.pt",
                f"*{context_id}*.pkl"
            ]
        return [
            f"*{context_id}*.keras",
            f"*{context_id}*.pkl",
            f"*{context_id}*.pt", 
            f"*{context_id}*.joblib"
        ]
    def _search_and_load_models(self, batch_dir: Path, patterns: List[str], context_id: str, models_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Search and load models from batch directory."""
        loaded_models = {}
        
        # Read runtime params if available
        self._read_runtime_params_if_exists(batch_dir)
        
        # Define search paths and iterate
        models_search_paths = self._get_models_search_paths(batch_dir)
        
        for search_path in models_search_paths:
            if not search_path.exists():
                continue
            
            self._search_patterns_in_path(search_path, patterns, context_id, models_meta, loaded_models)

        return loaded_models

    def _get_models_search_paths(self, batch_dir: Path) -> List[Path]:
        """Get model search paths."""
        return [
            batch_dir / 'models',
            batch_dir,
            self.models_path / 'models',
            self.models_path
        ]

    def _search_patterns_in_path(self, search_path: Path, patterns: List[str], context_id: str, models_meta: Dict[str, Any], loaded_models: Dict[str, Any]) -> None:
        """Search for patterns in a specific path and load models."""
        for pattern in patterns:
            for path in search_path.glob(pattern):
                self._try_load_model_from_path(path, context_id, models_meta, loaded_models)

    def _try_load_model_from_path(self, path: Path, context_id: str, models_meta: Dict[str, Any], loaded_models: Dict[str, Any]) -> None:
        """Try to load a model from a specific path."""
        try:
            cur_model_name = path.stem.replace(f"_{context_id}", "")
            model_meta = self._create_model_meta(context_id, models_meta, cur_model_name, str(path))
            
            loaded_model = self.model_pool.get_model(
                cur_model_name,
                loader_fn=lambda path=str(path), meta=model_meta: self.model_loader.load_path(path, meta)
            )
            if loaded_model is not None:
                loaded_models[cur_model_name] = loaded_model
        except Exception as e:
            self.logger.warning(f"Failed to load model from {path}: {e}")

    def _create_model_meta(self, context_id: str, models_meta: Dict[str, Any], model_name: str, model_path: str) -> Dict[str, Any]:
        """Create model metadata dictionary."""
        return {
            'model_id': model_name,
            'model_path': model_path,
            'model_type': models_meta.get(context_id, {}).get('model_type', model_name),
            'ticker': models_meta.get(context_id, {}).get('ticker'),
            'target': models_meta.get(context_id, {}).get('target')
        }

    def _read_runtime_params_if_exists(self, batch_dir: Path) -> None:
        """Read runtime params if file exists."""
        runtime_params_path = batch_dir / "runtime_params.json"
        if runtime_params_path.exists():
            try:
                with open(runtime_params_path, 'r') as f:
                    runtime_params = json.load(f)
                    test_mode = runtime_params.get('test_mode', {})
                    _ = test_mode.get('enabled', False)
            except Exception as e:
                self.logger.warning(f"Could not read runtime_params.json: {e}")

    def _calculate_anomaly_score(self, X: pd.DataFrame, context_id: str = "default") -> float:
        """Calculates anomaly score from 0 to 1."""
        try:
            if X.empty or len(X) < 2:
                return 0.5

            current_row, historical_data = self._prepare_anomaly_data(X)
            cache_key = f"{context_id}_{X.shape[1]}"
            
            # Calculate different anomaly scores
            z_score = self._calculate_zscore_anomaly(current_row, historical_data)
            iso_score = self._calculate_isolation_forest_anomaly(current_row, historical_data, cache_key)
            lof_score = self._calculate_lof_anomaly(current_row, historical_data, cache_key)
            
            # Combine scores
            final_anomaly = (z_score * 0.4 + iso_score * 0.4 + lof_score * 0.2)
            return float(np.clip(final_anomaly, 0, 1))
            
        except Exception as e:
            self.logger.warning(f"Anomaly detection failure: {e}")
            return 0.5

    def _prepare_anomaly_data(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for anomaly detection."""
        current_row = X.iloc[-1:].values
        historical_data = X.iloc[:-1].values if len(X) > 1 else X.values
        return current_row, historical_data

    def _calculate_zscore_anomaly(self, current_row: np.ndarray, historical_data: np.ndarray) -> float:
        """Calculate Z-score based anomaly score."""
        try:
            mean = np.mean(historical_data, axis=0)
            std = np.std(historical_data, axis=0)
            z_scores = np.abs((current_row - mean) / (std + 1e-6))
            return np.clip(float(np.mean(z_scores)) / 3.0, 0, 1)
        except Exception:
            return 0.5

    def _calculate_isolation_forest_anomaly(self, current_row: np.ndarray, historical_data: np.ndarray, cache_key: str) -> float:
        """Calculate Isolation Forest based anomaly score."""
        try:
            if len(historical_data) <= 10:
                return 0.5
                
            iso_key = f"iso_{cache_key}"
            if iso_key not in self._anomaly_estimators_cache:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                iso_forest.fit(historical_data)
                self._anomaly_estimators_cache[iso_key] = iso_forest
            
            iso_pred = self._anomaly_estimators_cache[iso_key].predict(current_row)
            return 1.0 if iso_pred[0] == -1 else 0.0
        except Exception:
            return 0.5

    def _calculate_lof_anomaly(self, current_row: np.ndarray, historical_data: np.ndarray, cache_key: str) -> float:
        """Calculate Local Outlier Factor based anomaly score."""
        try:
            if len(historical_data) <= 10:
                return 0.5
                
            lof_key = f"lof_{cache_key}"
            if lof_key not in self._anomaly_estimators_cache:
                lof = LocalOutlierFactor(n_neighbors=min(20, len(historical_data)-1), novelty=True)
                lof.fit(historical_data)
                self._anomaly_estimators_cache[lof_key] = lof
            
            lof_pred = self._anomaly_estimators_cache[lof_key].predict(current_row)
            return 1.0 if lof_pred[0] == -1 else 0.0
        except Exception:
            return 0.5
    
    def _get_last_price(self, ticker_df: pd.DataFrame, ticker: str) -> Optional[float]:
        """Get the last price from ticker dataframe with fallback options."""
        if 'close' in ticker_df.columns:
            return ticker_df['close'].iloc[-1]
        elif f'{ticker}_1d_close' in ticker_df.columns:
            return ticker_df[f'{ticker}_1d_close'].iloc[-1]
        else:
            return None

    def _extract_prediction_value(self, adjusted_prediction) -> float:
        """Extract prediction value from various prediction formats."""
        if hasattr(adjusted_prediction, '__len__') and len(adjusted_prediction) > 0:
            return adjusted_prediction[-1] if hasattr(adjusted_prediction, '__getitem__') else float(adjusted_prediction)
        else:
            return float(adjusted_prediction)

    def _calculate_ensemble_confidence(self, models: Dict[str, Any], X: pd.DataFrame, prediction: float, context_id: str) -> Dict[str, float]:
        """Calculates multi-factor confidence score."""
        try:
            if not models:
                return {'score': 0.5}

            raw_preds = []
            for m_name, model in models.items():
                try:
                    p = model.predict(X)
                    val = float(p[-1]) if hasattr(p, '__len__') else float(p)
                    raw_preds.append(val)
                except (ValueError, TypeError, AttributeError):
                    continue

            if not raw_preds:
                return {'score': 0.5}

            consensus_score = 0.5
            dispersion_score = 0.5
            
            if len(raw_preds) > 1:
                final_dir = (prediction > 0)
                agreement = sum(1 for p in raw_preds if (p > 0) == final_dir)
                consensus_score = agreement / len(raw_preds)
                variance = np.var(raw_preds)
                dispersion_score = 1.0 / (1.0 + variance * 5)
            
            accuracy_score = 0.5
            try:
                perf = self.diary.get_recent_performance(context=context_id, window=20)
                accuracy_score = perf.get('accuracy', 0.5)
            except Exception:
                pass

            volatility_factor = 0.5
            try:
                if len(X) > 5:
                    vol = np.std(X.iloc[-10:, 0].values)
                    volatility_factor = 1.0 / (1.0 + vol * 20)
            except Exception:
                pass

            final_score = (
                consensus_score * 0.35 + 
                dispersion_score * 0.25 + 
                accuracy_score * 0.25 + 
                volatility_factor * 0.15
            )
            
            return {'score': float(np.clip(final_score, 0, 1))}
        except Exception as e:
            self.logger.warning(f"⚠️ Confidence calculation failure: {e}")
            return {'score': 0.5}
