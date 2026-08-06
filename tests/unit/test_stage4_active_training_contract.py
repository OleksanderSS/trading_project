from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import src.pipeline.stages.modeling.orchestrator as stage4_orchestrator_module
from src.models.adapters.data_preparation import prepare_data_for_models
from src.models.artifact_store import ModelArtifactStore
from src.pipeline.stages.evaluation.pipeline_control_artifacts import (
    build_evaluation_metric_candidate,
)
from src.pipeline.stages.modeling.pipeline_control_artifacts import (
    build_split_evaluation_window,
)
from src.pipeline.stages.stage_4_modeling import ModelingStage
from src.pipeline.stages.prediction.orchestrator import PredictionResultRequest
from src.pipeline.stages.stage_5_prediction import PredictionStage
from src.training.batch_trainer import BatchTrainer


class _FakeTrainingManager:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.received = None

    def execute_unified_training(self, *, tickers, data_context):
        self.received = data_context
        return {
            "tickers_results": {
                tickers[0]: {
                    "status": "success",
                    "winner": "random_forest",
                    "winner_metrics": {
                        "score": 0.62,
                        "accuracy": 0.64,
                    },
                    "model_path": str(self.model_path),
                }
            }
        }


class _FakeAnomalyEngine:
    def calculate_anomaly_score(self, frame):
        return 1.0

    def calculate_ensemble_confidence(self, **kwargs):
        return {"score": 0.8}


class _FakePredictionGenerator:
    def extract_prediction_value(self, value):
        return float(value)


def _prepared_data():
    train_index = pd.date_range(
        "2026-01-01",
        periods=12,
        freq="15min",
        tz="UTC",
    )
    validation_index = pd.date_range(
        "2026-01-02",
        periods=4,
        freq="15min",
        tz="UTC",
    )
    holdout_index = pd.date_range(
        "2026-01-03",
        periods=4,
        freq="15min",
        tz="UTC",
    )
    return {
        "light_models": {
            "X_train": pd.DataFrame(
                {
                    "momentum": np.linspace(0.0, 1.0, 12),
                    "volume_ratio": np.linspace(1.0, 2.0, 12),
                },
                index=train_index,
            ),
            "y_train": np.array(
                [[0], [1]] * 6,
            ),
            "X_val": pd.DataFrame(
                {
                    "momentum": [0.2, 0.4, 0.6, 0.8],
                    "volume_ratio": [1.2, 1.4, 1.6, 1.8],
                },
                index=validation_index,
            ),
            "y_val": np.array([[0], [1], [0], [1]]),
            "X_test": pd.DataFrame(
                {
                    "momentum": [0.1, 0.3, 0.5, 0.7],
                    "volume_ratio": [1.1, 1.3, 1.5, 1.7],
                },
                index=holdout_index,
            ),
            "y_test": np.array([[1], [1], [0], [0]]),
            "feature_names": ["momentum", "volume_ratio"],
        }
    }


def test_active_stage4_adapts_nested_splits_and_reserves_holdout():
    stage = object.__new__(ModelingStage)
    prepared = _prepared_data()

    context = stage._build_unified_training_context(
        prepared,
        target_name="target_intraday_up_15m",
        context_fingerprint="ctx-1",
    )

    assert context["X_train"] is prepared["light_models"]["X_train"]
    assert context["X_test"] is prepared["light_models"]["X_val"]
    assert context["y_test"] is prepared["light_models"]["y_val"]
    assert context["X_test"] is not prepared["light_models"]["X_test"]
    assert context["selection_split_role"] == "validation"
    assert context["prepared_holdout_reserved"] is True
    assert context["target_type"] == "classification"


def test_model_preparation_preserves_timestamp_split_lineage():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2026-01-01T14:30:00Z",
                periods=80,
                freq="15min",
            ),
            "ticker": "NVDA",
            "interval": "15m",
            "close": np.linspace(100.0, 105.0, 80),
            "volume": np.arange(80) + 1_000,
            "target_intraday_up_15m": [0, 1] * 40,
        }
    )

    prepared = prepare_data_for_models(
        frame,
        ticker="NVDA",
        timeframe="15m",
        target_cols=["target_intraday_up_15m"],
        gap_size=2,
        val_size=0.2,
        test_size=0.2,
    )

    validation = prepared["light_models"]["X_val"]
    window = build_split_evaluation_window(
        validation,
        source="stage4_validation_feature_index",
    )
    assert isinstance(validation.index, pd.DatetimeIndex)
    assert str(validation.index.tz) == "UTC"
    assert window["start"].startswith("2026-")
    assert window["end"].startswith("2026-")
    assert window["sample_count"] == len(validation)


def test_stage4_builds_deterministic_context_fingerprint_when_missing():
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2026-01-01T14:30:00Z",
                periods=2,
                freq="15min",
            ),
            "ticker": ["NVDA", "NVDA"],
            "interval": ["15m", "15m"],
            "momentum": [0.1, 0.2],
            "volume_ratio": [1.0, 1.1],
        }
    )
    prepared = _prepared_data()

    first = ModelingStage._build_context_fingerprint(
        frame=frame,
        prepared_data=prepared,
        ticker="NVDA",
        timeframe="15m",
        target_name="target_intraday_up_15m",
        current_pattern="normal",
    )
    second = ModelingStage._build_context_fingerprint(
        frame=frame.copy(),
        prepared_data=prepared,
        ticker="NVDA",
        timeframe="15m",
        target_name="target_intraday_up_15m",
        current_pattern="normal",
    )
    changed = frame.copy()
    changed.loc[changed.index[-1], "momentum"] = 0.9
    third = ModelingStage._build_context_fingerprint(
        frame=changed,
        prepared_data=prepared,
        ticker="NVDA",
        timeframe="15m",
        target_name="target_intraday_up_15m",
        current_pattern="normal",
    )

    assert len(first) == 64
    assert first == second
    assert first != third


def test_active_stage4_emits_honest_partial_evidence_and_prediction_metadata(
    tmp_path,
    monkeypatch,
):
    prepared = _prepared_data()
    monkeypatch.setattr(
        stage4_orchestrator_module,
        "prepare_data_for_models",
        lambda **kwargs: prepared,
    )
    stage = object.__new__(ModelingStage)
    stage.modeling_config = {"test_size": 0.2}
    stage.training_manager = _FakeTrainingManager(
        tmp_path / "CHAMP_NVDA_target_intraday_up_15m.joblib"
    )
    stage.diary_path = tmp_path / "diary.csv"
    stage._log_expert_to_diary = lambda *args, **kwargs: None
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2026-01-01",
                periods=20,
                freq="15min",
                tz="UTC",
            ),
            "ticker": "NVDA",
            "interval": "15m",
            "context_pattern_id": "normal",
            "context_fingerprint": "ctx-nvda-15m",
            "market_regime": "neutral",
            "volatility_regime": "normal",
            "feature": np.arange(20),
            "target_intraday_up_15m": [0, 1] * 10,
        }
    )
    champions = {}
    artifacts = []

    asyncio.run(
        stage._process_ticker_with_async(
            "NVDA",
            frame,
            champions,
            "normal",
            timeframe="15m",
            metric_artifacts=artifacts,
            metric_artifact_dir=tmp_path / "stage4_artifacts",
        )
    )

    received = stage.training_manager.received
    assert received["X_test"] is prepared["light_models"]["X_val"]
    assert received["prepared_holdout_reserved"] is True
    champion = champions[
        "NVDA_15m_target_intraday_up_15m_normal"
    ]
    assert champion["model_type"] == "random_forest"
    assert champion["target_name"] == "target_intraday_up_15m"
    assert champion["target_type"] == "classification"
    assert champion["timeframe"] == "15m"
    assert champion["context_fingerprint"] == "ctx-nvda-15m"
    assert champion["selected_features"] == [
        "momentum",
        "volume_ratio",
    ]
    assert len(artifacts) == 1

    model_candidate = json.loads(
        Path(artifacts[0]["model_evaluation_json"]).read_text(
            encoding="utf-8"
        )
    )
    feature_candidate = json.loads(
        Path(artifacts[0]["feature_stability_report"]).read_text(
            encoding="utf-8"
        )
    )
    assert model_candidate["metrics"]["train_score"] is None
    assert model_candidate["metrics"]["validation_score"] == 0.62
    assert model_candidate["metrics"]["test_score"] is None
    assert model_candidate["metrics"]["test_sample_count"] == 0
    assert set(model_candidate["missing_for_locked_model_evaluation"]) == {
        "max_drawdown",
        "train_score",
    }
    assert model_candidate["evaluation_window"]["source"] == (
        "stage4_validation_feature_index"
    )
    assert feature_candidate["stability_signal_status"] == "measured"
    assert feature_candidate["feature_importance"] == {}
    assert feature_candidate["missing_for_locked_feature_stability"] == [
        "feature_importance"
    ]


def test_base_trainer_persists_candidates_and_promotes_actual_winner(tmp_path):
    trainer = object.__new__(BatchTrainer)
    trainer.output_dir = tmp_path
    trainer.logger = logging.getLogger("test-base-trainer-persistence")
    trainer.artifact_store = ModelArtifactStore()
    first = trainer._save_model_candidate(
        {"name": "first"},
        ticker="NVDA",
        timeframe="1d",
        target="target_up",
        model_type="linear",
    )
    winner = trainer._save_model_candidate(
        {"name": "winner"},
        ticker="NVDA",
        timeframe="1d",
        target="target_up",
        model_type="random_forest",
    )

    result = trainer._finalize_ticker_results(
        {
            "ticker": "NVDA",
            "timeframe": "1d",
            "target_name": "target_up",
            "models": [
                {"model_type": "linear", "model_path": str(first)},
                {
                    "model_type": "random_forest",
                    "model_path": str(winner),
                },
            ],
            "metrics": {
                "linear": {"score": 0.55},
                "random_forest": {"score": 0.62},
            },
        },
        "random_forest",
        0.62,
    )

    assert first.exists()
    assert winner.exists()
    assert first != winner
    assert result["winner_model_path"] == str(winner)
    champion_path = Path(result["model_path"])
    # The timeframe is part of the champion filename. Stage 4 runs this
    # suite once per (ticker, timeframe) into one output directory, so
    # without it the 15m, 60m and 1d champions were three writes to one
    # path -- see test_light_model_files_are_per_timeframe.py.
    assert champion_path.name == "CHAMP_NVDA_1d_target_up.joblib"
    assert joblib.load(champion_path) == {"name": "winner"}


def test_stage5_result_carries_stage4_lineage_into_stage7_candidate(tmp_path):
    stage = object.__new__(PredictionStage)
    stage.anomaly_engine = _FakeAnomalyEngine()
    stage.prediction_generator = _FakePredictionGenerator()
    stage.logger = logging.getLogger("test-stage5-lineage")
    frame = pd.DataFrame(
        {"close": [100.0]},
        index=pd.DatetimeIndex(["2026-01-02T15:00:00Z"]),
    )
    request = PredictionResultRequest(
        context_id="NVDA_15m_target_intraday_up_15m_normal",
        ticker="NVDA",
        adjusted_prediction=1.0,
        raw_prediction=0.8,
        model_contributions={"random_forest": 0.8},
        best_model_name="random_forest",
        ticker_df_clean=frame,
        meta={
            "target_name": "target_intraday_up_15m",
            "model_type": "random_forest",
            "timeframe": "15m",
            "context_fingerprint": "ctx-nvda-15m",
        },
    )

    result = stage._create_prediction_result(request)
    signals = pd.DataFrame(
        [
            {
                **result,
                "context_fingerprint": "ctx-nvda-15m",
                "signal": "BUY",
                "price": 100.0,
            }
        ],
        index=pd.DatetimeIndex(["2026-01-02T15:00:00Z"]),
    )
    candidate = build_evaluation_metric_candidate(
        financial_metrics={
            "max_drawdown": 0.05,
            "total_return": 0.02,
        },
        backtest_results={},
        evaluation_summary={},
        signals_df=signals,
        portfolio_history=pd.DataFrame(
            {"total_value": [100_000.0]},
            index=signals.index,
        ),
        summary_path=tmp_path / "summary.json",
    )

    assert result["model_context_id"] == request.context_id
    assert result["target_name"] == "target_intraday_up_15m"
    assert result["model_type"] == "random_forest"
    assert result["timeframe"] == "15m"
    assert candidate["ticker"] == "NVDA"
    assert candidate["target_name"] == "target_intraday_up_15m"
    assert candidate["model_type"] == "random_forest"
    assert candidate["timeframe"] == "15m"
    assert candidate["context_fingerprint"] == "ctx-nvda-15m"
