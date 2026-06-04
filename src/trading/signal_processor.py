"""Signal processor for trading orchestrator."""
import logging
from typing import Any, cast

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger


class SignalProcessor:
    def __init__(self, consensus_engine: Any = None, post_filter: Any = None, knn_finder: Any = None, logger=None):
        self.consensus_engine = consensus_engine
        self.post_filter = post_filter
        self.knn_finder = knn_finder
        self.logger = logger or ProjectLogger.get_logger(__name__)

    def prepare_predictions(self, raw_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sanitize and optionally filter raw predictions before consensus processing."""
        if not self.post_filter:
            return raw_predictions
        sanitized_preds = [
            {k: v for k, v in prediction.items() if isinstance(v, (int, float, str))} for prediction in raw_predictions
        ]
        filtered_df = self.post_filter.apply(pd.DataFrame(sanitized_preds))
        return cast(list[dict[str, Any]], filtered_df.to_dict("records"))

    def generate_consensus_signals(
        self,
        predictions_to_process: list[dict[str, Any]],
        regime: str = "neutral",
        enriched_data: (pd.DataFrame | None) = None,
    ) -> list[dict[str, Any]]:
        """Build consensus signals from sanitized predictions."""
        if self.consensus_engine is None:
            self.logger.warning("Consensus engine unavailable. No signals will be generated.")
            return []
        consensus_signals: list[dict[str, Any]] = []
        for prediction in predictions_to_process:
            ticker = prediction.get("ticker")
            if not ticker:
                self.logger.warning("Skipping prediction payload missing 'ticker'.")
                continue
            signal = self._build_signal(prediction, ticker, regime, enriched_data)
            if signal:
                consensus_signals.append(signal)
        return consensus_signals

    def _build_signal(
        self, prediction: dict[str, Any], ticker: str, regime: str, enriched_data: (pd.DataFrame | None)
    ) -> dict[str, Any] | None:
        pred_value = self._extract_prediction_value(prediction)
        model_predictions = self._build_model_predictions(prediction, pred_value)
        context_data = self._build_context_data(prediction, ticker)
        context_data["regime"] = regime
        knn_results = self._run_knn_analysis(ticker, enriched_data)
        try:
            report = self.consensus_engine.generate_consensus(
                model_predictions=model_predictions, context_data=context_data, knn_results=knn_results
            )
            if getattr(report, "final_signal", "HOLD") != "HOLD":
                return {
                    "ticker": ticker,
                    "final_signal": report.final_signal,
                    "confidence": getattr(report, "confidence", 0.0),
                    "report": report,
                }
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Consensus for {ticker} resulted in HOLD. Skipping execution.")
        except Exception as e:
            self.logger.error(f"Consensus synthesis failed for {ticker}: {e}", exc_info=True)
            raise
        return None

    def _run_knn_analysis(self, ticker: str, enriched_data: (pd.DataFrame | None)) -> list[Any] | None:
        if not self.knn_finder or enriched_data is None:
            return None
        try:
            historical_features = enriched_data[enriched_data["ticker"] == ticker]
            target_features = historical_features.tail(1)
            if target_features.empty:
                return None
            analysis = self.knn_finder.analyze(
                {"historical_features": historical_features, "target_features": target_features}
            )
            return analysis.get("similarities", {}).get(target_features.index[-1], [])
        except Exception as e:
            self.logger.error(f"Виникла помилка: {e}", exc_info=True)
            self.logger.warning(f"KNN analysis failed for {ticker}: {e}")
            raise RuntimeError(f"KNN analysis failed for {ticker}") from e

    def _extract_prediction_value(self, prediction: dict[str, Any]) -> float:
        pred_value = prediction.get("predictions")
        if isinstance(pred_value, (list, tuple, np.ndarray)):
            return float(pred_value[-1]) if len(pred_value) > 0 else 0.0
        if pred_value is not None and hasattr(pred_value, "item"):
            return float(pred_value.item())
        return float(pred_value) if pred_value is not None else 0.0

    def _build_model_predictions(self, prediction: dict[str, Any], pred_value: float) -> dict[str, float]:
        predictions_by_model = prediction.get("predictions_by_model", {})
        if predictions_by_model:
            return {model_name: float(pred) for model_name, pred in predictions_by_model.items()}
        primary_model = prediction.get("selected_primary_model", "unknown")
        return {primary_model: pred_value}

    def _build_context_data(self, prediction: dict[str, Any], ticker: str) -> dict[str, Any]:
        return {
            "ticker": ticker,
            "fingerprint": prediction.get("context_fingerprint", "0|0|0"),
            "regime": prediction.get("market_regime", "neutral"),
            "tf": prediction.get("timeframe", "1d"),
            "last_price": prediction.get("last_price"),
            "anomaly_score": prediction.get("anomaly_score", 0.0),
            "timestamp": prediction.get("timestamp"),
        }
