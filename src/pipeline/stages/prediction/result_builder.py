import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

from .result_request import PredictionResultRequest


class PredictionResultBuilder:
    """Builds Stage 5 prediction payloads and saves stage outputs."""
    ACCUMULATION_OUTPUT_DIR_CONFIG = 'system.accumulation.output_dir'
    DEFAULT_ACCUMULATION_DIR = 'data/colab/accumulated'

    def __init__(self, anomaly_engine: Any, model_resolver: Any,
        prediction_generator: Any, brain: Any):
        self.logger = ProjectLogger.get_logger('PredictionResultBuilder')
        self.anomaly_engine = anomaly_engine
        self.model_resolver = model_resolver
        self.prediction_generator = prediction_generator
        self.brain = brain

    def _load_autoencoder_model(self, batch_dir: Path, ticker: str, target_col: str) -> tuple[Any, list[str]] | None:
        """Load autoencoder model and features."""
        ae_key = f'{ticker}_{target_col}_autoencoder'
        ae_models = None

        for ext in ['.keras', '.pkl', '.h5', '.pt', '.joblib']:
            ae_model_path = batch_dir / f'model_{ticker}_{target_col}_autoencoder{ext}'
            if ae_model_path.exists():
                ae_meta = {'ticker': ticker, 'target': target_col, 'model_type': 'autoencoder', 'model_path': str(ae_model_path)}
                ae_models = self.model_resolver.load_available_models(ae_key, {ae_key: ae_meta})
                break

        if not ae_models:
            return None

        ae_model_name = list(ae_models.keys())[0]
        ae_model = ae_models[ae_model_name]
        ae_features = self._load_autoencoder_features(batch_dir, ticker, target_col)

        if not ae_features:
            ae_features = []

        return ae_model, ae_features

    def _load_autoencoder_features(self, batch_dir: Path, ticker: str, target_col: str) -> list[str] | None:
        """Load autoencoder selected features from file."""
        features_path = batch_dir / f'selected_features_{ticker}_{target_col}_autoencoder.json'
        if features_path.exists():
            try:
                with open(features_path) as f:
                    data = json.load(f)
                    return data.get('selected_features', [])
            except Exception as fe:
                self.logger.error(f'Виникла помилка: {fe}', exc_info=True)
                self.logger.warning(f'⚠️ Failed to read autoencoder features file: {fe}')
                raise
        return None

    def _calculate_autoencoder_normalcy(self, ae_model: Any, X_ae: pd.DataFrame, ticker: str, target_col: str) -> float:
        """Calculate autoencoder normalcy score."""
        raw_reconstruction = ae_model.predict(X_ae)
        x_input_flat = X_ae.iloc[-1:].values.flatten()
        reconstruction_flat = raw_reconstruction.flatten()
        min_len = min(len(x_input_flat), len(reconstruction_flat))

        if min_len > 0:
            mse = float(np.mean((x_input_flat[:min_len] - reconstruction_flat[:min_len]) ** 2))
            ae_normalcy = float(np.exp(-mse * 2.0))
            self.logger.info(f'🔒 Autoencoder anomaly integration for {ticker} ({target_col}): MSE={mse:.4f}, normalcy={ae_normalcy:.2%}')
            return ae_normalcy

        return 0.5

    def _integrate_autoencoder_anomaly(self, request: PredictionResultRequest, anomaly_score: float) -> float:
        """Integrate autoencoder anomaly detection into anomaly score."""
        ticker = request.ticker
        target_col = request.meta.get('target', '')

        try:
            batch_dir = self.model_resolver.resolve_batch_directory({request.context_id: request.meta})
            if not batch_dir:
                return anomaly_score

            ae_result = self._load_autoencoder_model(batch_dir, ticker, target_col)
            if not ae_result:
                return anomaly_score

            ae_model, ae_features = ae_result
            if not ae_features:
                ae_features = request.meta.get('selected_features', [])

            X_ae = self.prediction_generator._align_features(ae_model, request.ticker_df_clean, ae_features)
            ae_normalcy = self._calculate_autoencoder_normalcy(ae_model, X_ae, ticker, target_col)

            blended_normalcy = 0.5 * anomaly_score + 0.5 * ae_normalcy
            self.logger.info(f'Blended normalcy for {ticker} ({target_col}): {blended_normalcy:.2%}')
            return blended_normalcy

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'⚠️ Failed to integrate autoencoder anomaly detection: {e}')
            raise

    def _to_serializable(self, val: Any) -> Any:
        """Convert numpy types to serializable Python types."""
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, (np.float32, np.float64)):
            return float(val)
        if isinstance(val, (np.int32, np.int64)):
            return int(val)
        return val

    def _get_timestamp(self, ticker_df_clean: pd.DataFrame) -> str:
        """Get timestamp from DataFrame or current time."""
        if len(ticker_df_clean) > 0:
            last_ts = ticker_df_clean.index[-1]
            if pd.notnull(last_ts):
                return str(last_ts)
        return datetime.now().isoformat()

    def build_result(self, request: PredictionResultRequest) ->dict[str, Any]:
        anomaly_score = self.anomaly_engine.calculate_anomaly_score(request.ticker_df_clean)

        # Integrate autoencoder anomaly detection
        try:
            anomaly_score = self._integrate_autoencoder_anomaly(request, anomaly_score)
        except Exception:
            pass  # Fall back to original anomaly score

        # Calculate ensemble confidence
        confidence_info = self.anomaly_engine.calculate_ensemble_confidence(
            models={}, X=request.ticker_df_clean, prediction=request.adjusted_prediction,
            context_id=request.context_id, predictions_by_model=request.model_contributions)
        final_confidence = confidence_info.get('score', 0.5) * anomaly_score

        if anomaly_score < 0.4:
            self.logger.warning(f'Low normalcy score ({anomaly_score:.2f}) - potential data anomaly!')

        pred_value = self.prediction_generator.extract_prediction_value(request.adjusted_prediction)
        self.logger.info(f"Ensemble forecast for {request.ticker}: {pred_value:.4f} | Conf: {confidence_info.get('score'):.2%}")

        ts_val = self._get_timestamp(request.ticker_df_clean)

        return {
            'ticker': request.ticker,
            'predictions': self._to_serializable(request.adjusted_prediction),
            'raw_forecast': self._to_serializable(request.raw_prediction),
            'predictions_by_model': {k: self._to_serializable(v) for k, v in request.model_contributions.items()},
            'selected_primary_model': request.best_model_name,
            'confidence': float(final_confidence),
            'anomaly_score': float(anomaly_score),
            'last_price': self._get_last_price(request.ticker_df_clean, request.ticker) or 0.0,
            'shap_explanations': request.shap_explanations,
            'timestamp': ts_val
        }

    def _get_last_price(self, ticker_df: pd.DataFrame, ticker: str) ->(float |
        None):
        if ticker_df.empty:
            return None
        if 'close' in ticker_df.columns:
            return float(ticker_df['close'].iloc[-1])
        elif f'{ticker}_1d_close' in ticker_df.columns:
            return float(ticker_df[f'{ticker}_1d_close'].iloc[-1])
        return None

    def prepare_final_results(self, prediction_results: dict[str, Any],
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

    def _save_stage_5_results(self, predictions_list: list[dict],
        current_prices: dict[str, Any], prediction_results: dict[str, Any],
        models_meta: dict[str, Any], kwargs: dict[str, Any]) ->None:
        try:
            batch_name = kwargs.get('batch_name') or self.brain.get(
                'batch_name')
            output_dir = Path(self.model_resolver.config_manager.get(self.
                ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR))
            if not batch_name:
                for meta in models_meta.values():
                    path = meta.get('model_path', '')
                    if path:
                        path_parts = Path(path.replace('/', os.sep)).parts
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
            raise
