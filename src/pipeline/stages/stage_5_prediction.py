"""
Stage 5: Prediction Generation with Stacked Ensembles and Contextual Adjustments

Uses champion models and stacked ensembles to generate forecasts, 
incorporating real-time market regime adjustments and historical performance.
"""

import os
from typing import Optional, Any, Dict, List
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.meta_learning.memory.diary_engine import ExperienceDiaryEngine
from src.ensembling.ensemble.ensemble_model import StackedEnsemble
from src.analytics.context.prediction_adjuster import PredictionAdjuster
from src.analytics.context.contextual_model_selector import ContextualModelSelector

class PredictionStage(BaseStage):
    """
    Stage responsible for generating model predictions using an ensemble approach,
    calculating confidence scores, and adjusting forecasts based on market context.
    """
    def __init__(self, config_manager: UnifiedConfigManager, brain: Dict[str, Any], **kwargs):
        super().__init__(config_manager, brain)
        self.logger = ProjectLogger.get_logger("PredictionStage")
        self.prediction_config = self.config_manager.get_config('prediction', {})
        self.models_path = Path(self.config_manager.get_config('system', {}).get('models_path', 'src/trained_models'))
        
        self.diary = ExperienceDiaryEngine()
        self.adjuster = PredictionAdjuster(config=self.config_manager.get_specific_config('analysis', 'prediction_adjustment', {}))
        self.ensemble_factory = StackedEnsemble()
        self.context_selector = ContextualModelSelector(config_manager=config_manager)

    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Generates adjusted predictions for tickers processed in earlier stages.

        Args:
            **kwargs: Dictionary containing 'features_data' and 'models_metadata'.

        Returns:
            Dict[str, Any]: Updated pipeline data with 'prediction_results'.
        """
        features_df = kwargs.get('features_data')
        models_meta = kwargs.get('models_metadata', {})
        market_regime = self.brain.get('market_regime', 'neutral')

        if features_df is None or features_df.empty or not models_meta:
            self.logger.warning("Required features or model metadata not found. Skipping Stage 5.")
            return {}

        prediction_results = {}
        self.logger.info(f"Generating ensemble predictions for {len(models_meta)} contexts...")

        for context_id, meta in models_meta.items():
            try:
                ticker = meta.get('ticker')
                # Filter features for this specific ticker
                ticker_df = features_df[features_df['ticker'] == ticker].tail(50) # Use recent window
                if ticker_df.empty:
                    continue

                # 1. Load All Available Models for this Context (Ensemble)
                models = self._load_available_models(context_id)
                if not models:
                    continue

                # 2. Contextual Model Selection
                models_list = list(models.keys())
                best_model_name = self.context_selector.select_best_model(models_list, market_regime)
                self.logger.info(f"Contextual Selector chose '{best_model_name}' for {ticker} in {market_regime} regime.")

                # 3. Generate Ensemble Prediction
                if len(models) > 1:
                    raw_prediction, model_contributions = self.ensemble_factory.predict_with_weights(
                        models=models, 
                        X=ticker_df,
                        context_id=context_id,
                        diary=self.diary
                    )
                else:
                    selected_model = models.get(best_model_name, list(models.values())[0])
                    feature_cols = getattr(selected_model, 'feature_names_in_', ticker_df.columns.tolist())
                    X = ticker_df[feature_cols] if all(c in ticker_df.columns for c in feature_cols) else ticker_df
                    raw_prediction = selected_model.predict(X)
                    model_contributions = {best_model_name: 1.0}

                # 4. Contextual Prediction Adjustment (Market Regime Awareness)
                adjusted_prediction = self.adjuster.adjust(
                    prediction=raw_prediction,
                    regime=market_regime,
                    ticker=ticker
                )

                # 5. Calculate Final Confidence
                confidence_info = self._calculate_ensemble_confidence(
                    models=models, 
                    X=ticker_df, 
                    prediction=adjusted_prediction, 
                    context_id=context_id
                )

                prediction_results[context_id] = {
                    'ticker': ticker,
                    'predictions': adjusted_prediction,
                    'raw_forecast': raw_prediction,
                    'predictions_by_model': model_contributions,
                    'selected_primary_model': best_model_name,
                    'confidence': confidence_info.get('score'),
                    'last_price': ticker_df['close'].iloc[-1] if 'close' in ticker_df.columns else None,
                    'timestamp': ticker_df.index[-1] if isinstance(ticker_df.index, pd.DatetimeIndex) else None
                }
                
                self.logger.info(f"Ensemble forecast for {ticker}: {adjusted_prediction[-1]:.4f} | Conf: {confidence_info.get('score'):.2%}")

            except Exception as e:
                self.logger.error(f"Prediction failed for context {context_id}: {e}", exc_info=True)

        return {'prediction_results': prediction_results}

    def _load_available_models(self, context_id: str) -> Dict[str, Any]:
        """Loads all available model versions (Light/Heavy) for the context."""
        loaded_models = {}
        patterns = [f"CHAMP_{context_id}*.joblib", f"MODEL_{context_id}*.joblib"]
        
        for pattern in patterns:
            for path in self.models_path.glob(pattern):
                try:
                    model_name = path.stem.replace(f"_{context_id}", "")
                    loaded_models[model_name] = joblib.load(path)
                except Exception as e:
                    self.logger.warning(f"Failed to load model from {path}: {e}")
        
        return loaded_models

    def _calculate_ensemble_confidence(self, models: Dict[str, Any], X: pd.DataFrame, prediction: np.ndarray, context_id: str) -> Dict[str, Any]:
        """Calculates confidence based on model consensus and historical diary data."""
        scores = []
        
        # 1. Internal Consistency (Model Disagreement)
        if len(models) > 1:
            preds = [m.predict(X)[-1] for m in models.values()]
            agreement = 1.0 - (np.std(preds) / (np.mean(np.abs(preds)) + 1e-6))
            scores.append(np.clip(agreement, 0, 1))
        
        # 2. Historical performance from Diary
        try:
            perf = self.diary.get_recent_performance(context=context_id, window=30)
            scores.append(perf.get('accuracy', 0.5))
        except:
            scores.append(0.5)

        return {'score': np.mean(scores)}