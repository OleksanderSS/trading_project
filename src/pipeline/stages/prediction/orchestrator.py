# audit-ignore: ARCHITECTURAL_USAGE
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
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.context.prediction_adjuster import PredictionAdjuster
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.ensembling.caching import get_ensemble_cache
from src.ensembling.stacked_ensemble import StackedEnsemble
from src.meta_learning.memory.diary_engine import DiaryEngine
from src.models.calibration.adaptive_confidence_calibrator import get_confidence_calibrator
from src.models.calibration.prediction_ledger import record_prediction
from src.models.loader import ModelLoaderStrategy
from src.models.model_pool import get_model_pool
from src.models.model_selector.adaptive_selector import AdaptiveModelSelector
from src.models.model_selector.smart_selector import PerformanceHistorySelector
from src.pipeline.stages.base_stage import BaseStage
from src.pipeline.stages.prediction import AnomalyEngine, ModelResolver, PredictionGenerator
from src.pipeline.stages.prediction.data_preparation_service import DataPreparationService
from src.pipeline.stages.prediction.lineage import (
    prediction_observed_at,
    prediction_timeframe,
    prediction_timeframe_lineage,
    trusted_context_fingerprint,
)
from src.pipeline.stages.prediction.model_selection_service import ModelSelectionService
from src.pipeline.stages.prediction.output_contract import (
    build_model_output_contract,
    infer_classification_predict_semantics,
)
from src.pipeline.stages.prediction.prediction_context_manager import PredictionContextManager
from src.pipeline.stages.prediction.scaler_service import ScalerService


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
    models: dict[str, Any] = None
    model_output_contract: dict[str, Any] | None = None


class PredictionStage(BaseStage):
    """
    Stage responsible for generating model predictions using an ensemble approach,
    calculating confidence scores, and adjusting forecasts based on market context.
    """
    ACCUMULATION_OUTPUT_DIR_CONFIG = 'system.accumulation.output_dir'
    DEFAULT_ACCUMULATION_DIR = 'data/colab/accumulated'

    def __init__(self, config_manager: UnifiedConfigManager, error_handler,
        **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger('PredictionStage')
        self.context_manager = PredictionContextManager()
        self.prediction_config = self.config_manager.get_config('prediction',
            {})
        self.models_path = self.config_manager.get_models_path()
        self.diary = DiaryEngine()
        self.adjuster = PredictionAdjuster(config=self.config_manager.get(
            'analysis.prediction_adjustment', {}))
        self.ensemble_factory = StackedEnsemble()
        use_adaptive = self.config_manager.get(
            'prediction.use_adaptive_selector', False)
        if use_adaptive:
            self.context_selector: PerformanceHistorySelector | AdaptiveModelSelector = AdaptiveModelSelector(fallback=
                'lightgbm', leaderboard_path='data/model_leaderboard.json',
                learning_rate=0.1)
            self.logger.info(
                '✅ Using AdaptiveModelSelector with online learning')
        else:
            self.context_selector = PerformanceHistorySelector()
            self.logger.info('✅ Using PerformanceHistorySelector (default)')
        self.model_loader = ModelLoaderStrategy(self.logger)
        self.ensemble_cache = get_ensemble_cache(maxsize=5000)
        self.logger.info(
            '✅ Ensemble prediction cache enabled (LRU, maxsize=5000)')
        max_models = self.config_manager.get('performance.model_pool_size', 50)
        self.model_pool = get_model_pool(max_models=max_models)
        self.logger.info(
            f'✅ Model pool enabled (maxsize={max_models}, LRU eviction)')
        self.model_resolver = ModelResolver(config_manager=self.
            config_manager, model_pool=self.model_pool, model_loader=self.
            model_loader)
        self.anomaly_engine = AnomalyEngine(diary=self.diary)

        # Initialize extracted services to reduce coupling
        self.data_preparation_service = DataPreparationService()
        self.model_selection_service = ModelSelectionService(self.config_manager)
        self.scaler_service = ScalerService(self.config_manager)

        self.prediction_generator = PredictionGenerator(ensemble_factory=
            self.ensemble_factory, ensemble_cache=self.ensemble_cache,
            adjuster=self.adjuster)

    async def run(self, **kwargs) ->dict[str, Any]:
        """
        Generates adjusted predictions for tickers processed in earlier stages.

        Args:
            **kwargs: Pipeline data dict with 'features_data' and 'models_metadata'.

        Returns:
            Dict[str, Any]: Updated pipeline data with 'prediction_results'.
        """
        features_df, models_meta, market_regime = self._prepare_inputs(kwargs)
        if features_df is None or hasattr(features_df, 'empty'
            ) and features_df.empty or not models_meta:
            # DataPreparationService._validate_inputs already logs the
            # detailed reason (features_df is None/empty, models_meta
            # empty) when it's the one that rejected these inputs -- this
            # check is defense-in-depth in case that contract ever changes,
            # so it must never itself return {} silently. Also: since the
            # champion filter (ResultsProcessor.build_models_metadata ->
            # filter_to_champions) now hard-drops any (ticker, target)
            # group with no comparable metric, an empty models_meta here is
            # a realistic, expected-to-happen case, not just a data bug.
            self.logger.warning(
                'PredictionStage.run: no usable inputs after prepare_inputs '
                f'(features_df is None: {features_df is None}, models_meta '
                f'empty: {not models_meta}) -- Stage 5 will produce no '
                'predictions. If models_meta is empty, check whether the '
                'champion filter dropped every (ticker, target) group '
                '(see ResultsProcessor.build_models_metadata logs).'
            )
            return {}
        if not self._ensure_local_models(models_meta, kwargs):
            return {}
        prediction_results = self._generate_predictions_for_contexts(
            models_meta,
            features_df,
            market_regime,
            news_data=kwargs.get('news_data'),
        )
        return self._prepare_final_results(prediction_results, models_meta,
            kwargs)

    def _prepare_inputs(self, kwargs: dict[str, Any]) ->tuple[pd.DataFrame | None, dict[str, Any], str]:
        return self.data_preparation_service.prepare_inputs(kwargs, self.model_resolver)

    def _validate_inputs(self, features_df, models_meta) ->tuple[bool, str]:
        if features_df is None or features_df.empty or not models_meta:
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

    def _ensure_local_models(self, models_meta: dict[str, Any], kwargs:
        dict[str, Any] | None=None) ->bool:
        has_local = self.model_resolver.check_local_models(models_meta)
        if not has_local:
            self.model_resolver.log_model_status(models_meta)
            batch_dir = self.model_resolver.resolve_batch_directory(models_meta
                , kwargs or {})
            if batch_dir and batch_dir.exists():
                has_local = self.model_resolver.update_local_model_paths(
                    models_meta, batch_dir)
            if not has_local:
                self.logger.error('No local models found. Skipping Stage 5.')
                return False
        return True

    def _generate_predictions_for_contexts(self, models_meta: dict[str, Any
        ], features_df: pd.DataFrame, market_regime: str,
        news_data: Any=None) ->dict[str, Any]:
        prediction_results: dict[str, Any] = {}
        available_model_types = self._get_available_model_types()
        filtered_models_meta = {}
        for context_id, meta in models_meta.items():
            model_type = meta.get('model_type', '')
            if model_type in available_model_types:
                filtered_models_meta[context_id] = meta
            else:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f'Skipping {context_id} - {model_type} models not available'
                        )
        self.logger.info(
            f'Generating ensemble predictions for {len(filtered_models_meta)}/{len(models_meta)} available contexts...'
            )
        for context_id, meta in filtered_models_meta.items():
            try:
                result = self._process_single_context(context_id, meta,
                    features_df, market_regime, news_data=news_data)
                if result:
                    prediction_results[context_id] = result
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                self.handle_stage_error(e, context=
                    f'Prediction-{context_id}', severity='error')
                self.logger.error(
                    f'Prediction failed for context {context_id}: {e}',
                    exc_info=True)
        return prediction_results

    def _get_available_model_types(self) ->set:
        """Get available model types by scanning model files in the database directory"""
        return self.model_selection_service.get_available_model_types()

    def _process_single_context(self, context_id: str, meta: dict[str, Any],
        features_df: pd.DataFrame, market_regime: str,
        news_data: Any=None) ->dict[str, Any] | None:
        context_result = self._process_context_data(context_id, meta,
            features_df)
        if context_result is None:
            return None
        ticker_df_clean, filtered_features_list = context_result
        ticker = meta.get('ticker')
        if not ticker:
            self.logger.error(f'Ticker missing for context {context_id}')
            return None

        state = self._extract_context_state(ticker_df_clean)

        # Unpack state values for use below
        current_pattern = state['pattern']
        current_pattern_seq = state['seq']
        current_fingerprint = state['fingerprint']
        champion_state = state['champion']
        context_velocity = state['velocity']

        # 1. Шукаємо ЕКСПЕРТНУ модель для цього патерна
        expert_context_id = f"{ticker}_{meta.get('target')}_{current_pattern}"
        models = self.model_resolver.load_available_models(expert_context_id, {
            expert_context_id: meta})

        if not models:
            self.logger.info(f"ℹ️ No expert model for pattern {current_pattern}, using general champion")
            models = self.model_resolver.load_available_models(context_id, {
                context_id: meta})

        if not models:
            self.logger.warning(f'No models found for {context_id}, skipping')
            return None

        target_scaler = self._load_target_scaler(meta)
        # If no scaler found, denormalize_prediction will return raw value
        # (scale_target=False by default, so no denormalization needed)

        best_model_name = self._select_best_model_for_context(ticker_df_clean,
            meta, models, ticker, market_regime)
        # Generate model output before optional contextual overlays.
        raw_prediction, model_contributions = (
            self.prediction_generator.generate_prediction(
                models,
                best_model_name,
                ticker_df_clean,
                filtered_features_list,
                market_regime,
                context_id,
                ticker,
                meta.get('timeframe'),
            )
        )
        if raw_prediction is None:
            return None

        confidence_adjustment = 1.0
        if champion_state != 0:
            last_raw_pred = (
                raw_prediction[-1]
                if isinstance(raw_prediction, np.ndarray)
                else raw_prediction
            )
            if np.sign(last_raw_pred) != np.sign(champion_state):
                confidence_adjustment = self.prediction_config.get(
                    'champion_contradiction_penalty',
                    0.7,
                )
                self.logger.info(
                    'Contradiction with Champion detected for %s. '
                    'Penalizing confidence by %.0f%%.',
                    ticker,
                    confidence_adjustment * 100,
                )

        adjusted_prediction = (
            self.prediction_generator.adjust_prediction_contextually(
                raw_prediction,
                best_model_name,
                market_regime,
                ticker,
            )
        )
        nlp_adjustment_applied = False
        if news_data:
            from src.patterns.pattern_recognition_adjustment import adjust_predictions_with_patterns
            base_pred_dict = {ticker: adjusted_prediction}
            pattern_adj_dict = adjust_predictions_with_patterns(base_pred_dict, news_data)
            if ticker in pattern_adj_dict:
                self.logger.info(f"📰 Applied NLP pattern adjustment: {adjusted_prediction:.4f} -> {pattern_adj_dict[ticker]:.4f}")
                adjusted_prediction = pattern_adj_dict[ticker]
                nlp_adjustment_applied = True

        adjusted_prediction = self.prediction_generator.denormalize_prediction(
            adjusted_prediction, target_scaler)
        timeframe_lineage = prediction_timeframe_lineage(
            ticker_df_clean,
            declared_timeframe=meta.get("timeframe"),
        )
        resolved_timeframe = timeframe_lineage.get(
            "resolved_timeframe"
        )
        resolved_context_fingerprint = trusted_context_fingerprint(
            current_fingerprint,
            meta.get("context_fingerprint"),
        )
        model_output_contract = build_model_output_contract(
            target_name=meta.get('target_name') or meta.get('target'),
            target_type=meta.get('target_type'),
            model_count=sum(
                'autoencoder' not in str(name).lower()
                for name in models
            ),
            contextual_adjustment_applied=True,
            nlp_adjustment_applied=nlp_adjustment_applied,
            target_scaler_applied=target_scaler is not None,
            classification_predict_semantics=(
                infer_classification_predict_semantics(models)
            ),
        )

        request_meta = dict(meta)
        request_meta["timeframe"] = resolved_timeframe
        request_meta["_timeframe_lineage"] = timeframe_lineage
        request_meta["_timeframe_lineage_source"] = (
            "model_and_feature_cadence_verified"
            if timeframe_lineage.get("status")
            == "timeframe_cadence_verified"
            else (
                "feature_frame_metadata"
                if resolved_timeframe
                else "invalid_or_missing"
            )
        )
        request_meta["context_fingerprint"] = (
            resolved_context_fingerprint
        )
        request_meta["_context_fingerprint_lineage_source"] = (
            "model_or_feature_context_fingerprint"
            if resolved_context_fingerprint
            else "missing"
        )
        request = PredictionResultRequest(context_id=context_id, ticker=
            ticker, adjusted_prediction=adjusted_prediction, raw_prediction
            =raw_prediction, model_contributions=model_contributions,
            best_model_name=best_model_name, ticker_df_clean=
            ticker_df_clean, meta=request_meta, models=models,
            model_output_contract=model_output_contract)

        result = self._create_prediction_result(request)
        if result:
            result['confidence'] *= confidence_adjustment
            result['context_fingerprint'] = (
                resolved_context_fingerprint
            )
            result['context_pattern_id'] = current_pattern
            result['context_pattern_seq'] = current_pattern_seq
            try:
                result['context_velocity'] = float(context_velocity)
            except (TypeError, ValueError):
                result['context_velocity'] = 0.0

        return result

    def _extract_context_state(self, ticker_df: pd.DataFrame) -> dict[str, Any]:
        """Extracts context state from dataframe."""
        return {
            'pattern': ticker_df['context_pattern_id'].iloc[-1] if 'context_pattern_id' in ticker_df.columns and len(ticker_df) > 0 else 'normal',
            'seq': ticker_df['context_pattern_seq'].iloc[-1] if 'context_pattern_seq' in ticker_df.columns and len(ticker_df) > 0 else None,
            'fingerprint': ticker_df['context_fingerprint'].iloc[-1] if 'context_fingerprint' in ticker_df.columns and len(ticker_df) > 0 else None,
            'champion': ticker_df['state_champion'].iloc[-1] if 'state_champion' in ticker_df.columns and len(ticker_df) > 0 else 0,
            'velocity': ticker_df['context_velocity'].iloc[-1] if 'context_velocity' in ticker_df.columns and len(ticker_df) > 0 else 0
        }

    def _process_context_data(self, context_id: str, meta: dict[str, Any],
        features_df: pd.DataFrame) ->tuple | None:
        return self.data_preparation_service.prepare_context_data(context_id, meta, features_df)

    def _prepare_ticker_data(self, features_df: pd.DataFrame, ticker: str
        ) ->pd.DataFrame | None:
        return self.data_preparation_service.prepare_ticker_data(features_df, ticker)

    def _create_context_fingerprint(self, ticker_df: pd.DataFrame,
        market_regime: str) ->str:
        """
        🎯 SMART FINGERPRINT:
        Використовує наш новий context_pattern_id замість старої логіки.
        """
        return self.data_preparation_service.create_context_fingerprint(ticker_df, market_regime)

    def _load_target_scaler(self, meta: dict[str, Any]) ->Any | None:
        return self.scaler_service.load_target_scaler(meta)

    # _create_fallback_scaler REMOVED: scale_target=False by default,
    # so denormalization is not needed when scaler is missing

    def _select_best_model_for_context(self, ticker_df_clean: pd.DataFrame,
        meta: dict[str, Any], models: dict[str, Any], ticker: str,
        market_regime: str) ->str:
        if not hasattr(self, 'model_selection_service'):
            self.model_selection_service = ModelSelectionService(getattr(self, 'config_manager', None))
        return self.model_selection_service.select_best_model_for_context(
            ticker_df_clean, meta, models, ticker, market_regime,
            self.context_selector, diary=getattr(self, "diary", None)
        )

    # Helper methods removed - now in ModelSelectionService
    # _prediction_model_candidates, _build_model_alias_map,
    # _resolve_model_selection, _model_type_alias

    def _create_prediction_result(self, request: PredictionResultRequest
        ) ->dict[str, Any]:
        anomaly_score = self.anomaly_engine.calculate_anomaly_score(request
            .ticker_df_clean)
        confidence_info = self.anomaly_engine.calculate_ensemble_confidence(
            models=request.models or {}, X=request.ticker_df_clean, prediction=request.
            adjusted_prediction, context_id=request.context_id)
        raw_confidence = confidence_info.get('score', 0.5) * anomaly_score
        try:
            final_confidence = float(get_confidence_calibrator().calibrate(raw_confidence))
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.warning(f'Confidence calibration unavailable, using raw score: {e}')
            final_confidence = raw_confidence
        if anomaly_score < 0.8:
            self.logger.warning(
                f'Low anomaly score ({anomaly_score:.2f}) - potential data anomaly!'
                )
        pred_value = self.prediction_generator.extract_prediction_value(request
            .adjusted_prediction)
        self.logger.info(
            f"Ensemble forecast for {request.ticker}: {pred_value:.4f} | Conf: {confidence_info.get('score'):.2%}"
            )
        observed_at = prediction_observed_at(request.ticker_df_clean)
        resolved_timeframe = (
            request.meta.get("timeframe")
            or prediction_timeframe(request.ticker_df_clean)
        )
        target_name = request.meta.get('target_name') or request.meta.get('target')
        last_price = self._get_last_price(request.ticker_df_clean, request.ticker)
        self._record_prediction_for_calibration(
            ticker=request.ticker,
            target_name=target_name,
            timeframe=resolved_timeframe,
            predicted_value=pred_value,
            last_price=last_price,
            raw_confidence=raw_confidence,
            calibrated_confidence=final_confidence,
        )
        return {'ticker': request.ticker,
            'model_context_id': request.context_id,
            'target_name': target_name,
            'model_type': request.meta.get('model_type') or request.best_model_name,
            'timeframe': resolved_timeframe,
            'timeframe_lineage': request.meta.get(
                '_timeframe_lineage',
                prediction_timeframe_lineage(
                    request.ticker_df_clean,
                    declared_timeframe=request.meta.get("timeframe"),
                ),
            ),
            'context_fingerprint': trusted_context_fingerprint(
                request.meta.get('context_fingerprint')
            ),
            'model_output_contract': request.model_output_contract,
            'predictions': request.
            adjusted_prediction, 'raw_forecast': request.raw_prediction,
            'predictions_by_model': request.model_contributions,
            'selected_primary_model': request.best_model_name, 'confidence':
            final_confidence, 'raw_confidence': raw_confidence, 'anomaly_score': anomaly_score, 'last_price':
            last_price,
            'timestamp': observed_at,
            'lineage_sources': {
                'timeframe': (
                    request.meta.get(
                        '_timeframe_lineage_source',
                        'model_metadata'
                        if request.meta.get('timeframe')
                        else 'missing',
                    )
                ),
                'prediction_as_of': (
                    'feature_frame_metadata'
                    if observed_at
                    else 'missing'
                ),
                'context_fingerprint': (
                    request.meta.get(
                        '_context_fingerprint_lineage_source',
                        'model_or_feature_context_fingerprint'
                        if request.meta.get('context_fingerprint')
                        else 'missing',
                    )
                ),
            }}

    def _get_last_price(self, ticker_df: pd.DataFrame, ticker: str) ->float | None:
        if ticker_df.empty:
            return None
        if 'close' in ticker_df.columns:
            return float(ticker_df['close'].iloc[-1])
        elif f'{ticker}_1d_close' in ticker_df.columns:
            return float(ticker_df[f'{ticker}_1d_close'].iloc[-1])
        return None

    def _record_prediction_for_calibration(
        self,
        *,
        ticker: str,
        target_name: str | None,
        timeframe: str | None,
        predicted_value: float,
        last_price: float | None,
        raw_confidence: float,
        calibrated_confidence: float,
    ) -> None:
        """Best-effort ledger write for the confidence-calibrator outcome
        feedback loop (see src/models/calibration/prediction_ledger.py).
        Never blocks prediction generation — a ledger write failure is a
        lost calibration data point, not a reason to fail the pipeline."""
        try:
            record_prediction(
                ticker=ticker,
                target_name=target_name or 'unknown',
                timeframe=timeframe or 'unknown',
                predicted_value=predicted_value,
                last_price=last_price,
                raw_confidence=raw_confidence,
                calibrated_confidence=calibrated_confidence,
            )
        except (ValueError, TypeError, AttributeError, KeyError, OSError) as e:
            self.logger.warning(f'Failed to record prediction to calibration ledger: {e}')

    def _prepare_final_results(self, prediction_results: dict[str, Any],
        models_meta: dict[str, Any], kwargs: dict[str, Any]) ->dict[str, Any]:
        predictions_list = list(prediction_results.values())
        current_prices = {pred_data['ticker']: pred_data['last_price'] for
            pred_data in prediction_results.values() if pred_data.get(
            'ticker') and pred_data.get('last_price')}
        light_models_count = sum(1 for m in models_meta.values() if m.get(
            'model_category') == 'light')
        heavy_models_count = sum(1 for m in models_meta.values() if m.get(
            'model_category') in ['heavy', 'colab'])
        self.logger.info(
            f'Stage 5 complete: {len(predictions_list)} predictions, {len(current_prices)} prices'
            )
        self.logger.info(
            f'Models: {light_models_count} light, {heavy_models_count} heavy, {len(models_meta)} total'
            )
        self._save_stage_5_results(predictions_list=predictions_list,
            current_prices=current_prices, prediction_results=
            prediction_results, models_meta=models_meta, kwargs=kwargs)
        return {'predictions': predictions_list, 'current_prices':
            current_prices, 'prediction_results': prediction_results,
            'models_metadata': models_meta, 'light_models_count':
            light_models_count, 'heavy_models_count': heavy_models_count,
            'total_models': len(models_meta)}

    def update_selector_feedback(self, prediction_results: dict[str, Any],
        actual_results: dict[str, float]):
        """
        Update AdaptiveModelSelector with feedback from actual results.

        Args:
            prediction_results: Results from Stage 5 predictions
            actual_results: Dict of {ticker: actual_return}
        """
        if not isinstance(self.context_selector, AdaptiveModelSelector):
            return
        for context_id, pred_data in prediction_results.items():
            ticker = pred_data.get('ticker')
            if not ticker or ticker not in actual_results:
                continue
            model_id = pred_data.get('selected_primary_model')
            predicted_return = pred_data.get('predictions', 0)
            actual_return = actual_results[ticker]
            context_fingerprint = pred_data.get('context_fingerprint',
                context_id)
            try:
                self.context_selector.update_from_feedback(model_id=
                    model_id, context_fingerprint=context_fingerprint,
                    actual_return=actual_return, predicted_return=
                    predicted_return)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f'Updated selector feedback for {ticker}: {model_id}')
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                self.logger.warning(f'Failed to update selector feedback: {e}')

    def _save_stage_5_results(self, predictions_list: list[dict],
        current_prices: dict, prediction_results: dict, models_meta: dict,
        kwargs: dict) ->None:
        try:
            batch_name = kwargs.get('batch_name') or self.brain.get(
                'batch_name')
            output_dir = Path(self.config_manager.get(self.
                ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR))
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
                    batch_name = max(batch_dirs, key=lambda p: p.stat().
                        st_mtime).name
            if batch_name:
                batch_dir = output_dir / batch_name
                batch_dir.mkdir(parents=True, exist_ok=True)
                stage_5_results = {'timestamp': datetime.now().isoformat(),
                    'batch_name': batch_name, 'predictions':
                    predictions_list, 'current_prices': current_prices,
                    'prediction_results': prediction_results,
                    'models_metadata': models_meta, 'light_models_count':
                    sum(1 for m in models_meta.values() if m.get(
                    'model_category') == 'light'), 'heavy_models_count':
                    sum(1 for m in models_meta.values() if m.get(
                    'model_category') in ['heavy', 'colab']),
                    'total_models': len(models_meta), 'total_predictions':
                    len(predictions_list)}
                stage_5_file = batch_dir / 'stage_5_results.json'
                with open(stage_5_file, 'w') as f:
                    json.dump(stage_5_results, f, indent=2, default=str)
                self.logger.info(
                    f'✅ Stage 5 results saved: {stage_5_file.name}')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Error saving Stage 5 results: {e}')
