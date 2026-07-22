from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.pipeline.stages.prediction.output_contract import (
    build_model_output_contract,
)
from src.pipeline.stages.prediction.data_preparation_service import (
    DataPreparationService,
)
from src.pipeline.stages.prediction.lineage import (
    prediction_timeframe,
    prediction_timeframe_lineage,
)
from src.pipeline.stages.stage_5_prediction import PredictionStage


class _Classifier:
    classes_ = np.array([0, 1])


class _ModelResolver:
    def load_available_models(self, context_id, metadata):
        return {"random_forest": _Classifier()}


class _PredictionGenerator:
    def generate_prediction(self, *args, **kwargs):
        return np.array([1]), {"random_forest": 1}

    def adjust_prediction_contextually(self, *args, **kwargs):
        return 0.8

    def denormalize_prediction(self, value, scaler):
        return float(value)

    def extract_prediction_value(self, value):
        return float(value)


class _AnomalyEngine:
    def calculate_anomaly_score(self, frame):
        return 1.0

    def calculate_ensemble_confidence(self, **kwargs):
        return {"score": 0.75}


def test_classification_predict_output_is_not_probability():
    contract = build_model_output_contract(
        target_name="target_intraday_up_15m",
        target_type="classification",
        model_count=1,
        contextual_adjustment_applied=True,
        nlp_adjustment_applied=False,
        target_scaler_applied=False,
        classification_predict_semantics="class_label",
    )

    assert contract["status"] == "model_output_contract_ready"
    assert contract["raw_output"]["scale"] == "class_label_from_predict"
    assert contract["final_output"]["scale"] == (
        "adjusted_classification_score"
    )
    assert contract["final_output"]["model_output_scale_known"] is True
    assert contract["final_output"]["positive_class_probability"] is False
    assert contract["final_output"]["directional_inference_allowed"] is False


def test_unknown_target_type_keeps_output_contract_fail_closed():
    contract = build_model_output_contract(
        target_name="target_unknown",
        target_type=None,
        model_count=2,
        contextual_adjustment_applied=True,
        nlp_adjustment_applied=False,
        target_scaler_applied=False,
    )

    assert contract["status"] == "model_output_contract_partial"
    assert contract["final_output"]["model_output_scale_known"] is False
    assert contract["issues"] == ["target_type_missing_or_unsupported"]


def test_unverified_classification_predict_scale_stays_partial():
    contract = build_model_output_contract(
        target_name="target_intraday_up_15m",
        target_type="classification",
        model_count=1,
        contextual_adjustment_applied=True,
        nlp_adjustment_applied=False,
        target_scaler_applied=False,
    )

    assert contract["status"] == "model_output_contract_partial"
    assert contract["raw_output"]["scale"] == (
        "classification_predict_output_unknown_scale"
    )
    assert contract["final_output"]["model_output_scale_known"] is False
    assert contract["issues"] == [
        "classification_predict_scale_unverified"
    ]


def test_stage5_context_path_generates_prediction_and_contract():
    stage = object.__new__(PredictionStage)
    stage.logger = logging.getLogger("test-stage5-output-contract")
    stage.model_resolver = _ModelResolver()
    stage.prediction_generator = _PredictionGenerator()
    stage.anomaly_engine = _AnomalyEngine()
    stage.prediction_config = {"champion_contradiction_penalty": 0.7}
    stage._process_context_data = lambda context_id, meta, features: (
        features,
        ["feature"],
    )
    stage._load_target_scaler = lambda meta: None
    stage._select_best_model_for_context = (
        lambda frame, meta, models, ticker, regime: "random_forest"
    )
    frame = pd.DataFrame(
        {
            "ticker": ["NVDA"],
            "close": [100.0],
            "feature": [0.5],
            "context_pattern_id": ["normal"],
            "context_fingerprint": ["ctx-nvda-15m"],
            "state_champion": [1],
            "context_velocity": [0.1],
        },
        index=pd.DatetimeIndex(["2026-06-29T12:00:00Z"]),
    )

    result = stage._process_single_context(
        "NVDA_15m_target_intraday_up_15m_normal",
        {
            "ticker": "NVDA",
            "timeframe": "15m",
            "target": "target_intraday_up_15m",
            "target_type": "classification",
            "model_type": "random_forest",
            "context_fingerprint": "ctx-nvda-15m",
        },
        frame,
        "neutral",
    )

    assert result is not None
    assert result["predictions"] == 0.8
    assert result["raw_forecast"].tolist() == [1]
    assert result["model_output_contract"]["status"] == (
        "model_output_contract_ready"
    )
    assert result["model_output_contract"]["final_output"][
        "positive_class_probability"
    ] is False


def test_stage5_preserves_feature_frame_time_and_timeframe_lineage():
    stage = object.__new__(PredictionStage)
    stage.logger = logging.getLogger("test-stage5-source-lineage")
    stage.model_resolver = _ModelResolver()
    stage.prediction_generator = _PredictionGenerator()
    stage.anomaly_engine = _AnomalyEngine()
    stage.prediction_config = {"champion_contradiction_penalty": 0.7}
    stage.data_preparation_service = DataPreparationService()
    stage._process_context_data = (
        stage.data_preparation_service.prepare_context_data
    )
    stage._load_target_scaler = lambda meta: None
    stage._select_best_model_for_context = (
        lambda frame, meta, models, ticker, regime: "random_forest"
    )
    frame = pd.DataFrame(
        {
            "ticker": ["NVDA"],
            "datetime": [pd.Timestamp("2026-06-29T12:00:00Z")],
            "interval": ["15m"],
            "close": [100.0],
            "feature": [0.5],
        }
    )

    result = stage._process_single_context(
        "NVDA_15m_target_intraday_up_15m_normal",
        {
            "ticker": "NVDA",
            "target": "target_intraday_up_15m",
            "target_type": "classification",
            "model_type": "random_forest",
            "selected_features": ["feature"],
        },
        frame,
        "neutral",
    )

    assert result is not None
    assert result["timeframe"] == "15m"
    assert result["timestamp"] == "2026-06-29T12:00:00+00:00"
    assert result["context_fingerprint"] is None
    assert result["lineage_sources"]["timeframe"] == (
        "feature_frame_metadata"
    )


def test_stage5_rejects_declared_timeframe_that_conflicts_with_cadence():
    timestamps = pd.date_range(
        "2026-06-29T13:30:00Z",
        periods=12,
        freq="15min",
    )
    frame = pd.DataFrame(
        {
            "ticker": ["NVDA"] * len(timestamps),
            "datetime": timestamps,
            "interval": ["1d"] * len(timestamps),
            "feature": range(len(timestamps)),
        }
    )

    prepared = DataPreparationService().prepare_ticker_data(
        frame,
        "NVDA",
    )

    assert prepared is not None
    assert prediction_timeframe(prepared) is None
    lineage = prediction_timeframe_lineage(prepared)
    assert lineage["status"] == "timeframe_cadence_mismatch"
    assert lineage["declared_timeframe"] == "1d"
    assert lineage["observed_timeframe"] == "15m"
