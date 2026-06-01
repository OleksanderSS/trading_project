import logging

import duckdb
import pandas as pd

from src.features.enrichers.context_map_enricher import ContextMapEnricher
from src.meta_learning.memory.diary_engine import DiaryEngine
from src.pipeline.stages.prediction.data_preparation_service import (
    DataPreparationService,
)
from src.pipeline.stages.prediction.model_selection_service import ModelSelectionService


class _DiaryDataManager:
    def __init__(self):
        self.con = duckdb.connect(database=":memory:")
        self.con.execute(
            """
            CREATE TABLE experience_diary (
                agent_id VARCHAR,
                decision_timestamp BIGINT,
                ticker VARCHAR,
                decision_type VARCHAR,
                context_fingerprint VARCHAR,
                context_pattern_seq VARCHAR,
                model_prediction DOUBLE,
                outcome VARCHAR,
                profit_loss DOUBLE
            )
            """
        )


class _Config:
    def get(self, key, default=None):
        return default


class _FallbackSelector:
    def select_best_model(self, df, target_type, available_models):
        return available_models[0], 0.5


def test_context_pattern_sequence_uses_only_current_and_past_fingerprints():
    enricher = object.__new__(ContextMapEnricher)
    enricher.pattern_length = 3
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "context_fingerprint": ["a", "b", "c", "d"],
        }
    )

    enricher._generate_pattern_sequences(df)

    assert df["context_pattern_seq"].tolist() == [
        "a>>START>>START",
        "b>>a>>START",
        "c>>b>>a",
        "d>>c>>b",
    ]
    assert df.loc[2, "context_pattern_id"] != df.loc[3, "context_pattern_id"]


def test_data_preparation_preserves_pattern_sequence_for_stage5_selection():
    service = DataPreparationService()
    features = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "close": [100.0, 101.0],
            "context_pattern_id": ["old", "new"],
            "context_pattern_seq": ["0|1>>START", "1|1>>0|1"],
            "context_fingerprint": ["0|1", "1|1"],
        }
    )

    prepared = service.prepare_ticker_data(features, "AAPL")

    assert prepared["context_pattern_seq"].iloc[-1] == "1|1>>0|1"
    assert prepared["context_fingerprint"].iloc[-1] == "1|1"


def test_diary_knn_sequence_weights_can_select_model_by_pattern_similarity():
    manager = _DiaryDataManager()
    manager.con.execute(
        """
        INSERT INTO experience_diary VALUES
        ('catboost', 1, 'AAPL', 'training', 'fp_a', '1|1>>1|0>>0|0', 0.90, 'neutral', 0.20),
        ('catboost', 2, 'AAPL', 'training', 'fp_a', '1|1>>1|0>>0|0', 0.80, 'neutral', 0.30),
        ('lightgbm', 3, 'AAPL', 'training', 'fp_b', '-1|-1>>-1|0>>0|0', 0.40, 'neutral', 0.10),
        ('lightgbm', 4, 'AAPL', 'training', 'fp_b', '-1|-1>>-1|0>>0|0', 0.50, 'neutral', 0.10)
        """
    )
    diary = object.__new__(DiaryEngine)
    diary.data_manager = manager
    diary.logger = logging.getLogger("test_diary_knn")

    weights = diary.get_knn_contextual_model_weights(
        "no_exact_fingerprint",
        context_pattern_seq="1|1>>1|0>>0|1",
        n_neighbors=1,
        min_neighbors=1,
    )

    assert weights["catboost"] > weights.get("lightgbm", 0.0)


def test_model_selection_uses_diary_pattern_alias_weights():
    service = ModelSelectionService(_Config())
    ticker_df = pd.DataFrame(
        {
            "close": [100.0],
            "context_pattern_id": ["no_exact_fingerprint"],
            "context_pattern_seq": ["1|1>>1|0>>0|1"],
        }
    )

    class Diary:
        def get_knn_contextual_model_weights(self, context_fingerprint, **kwargs):
            assert kwargs["context_pattern_seq"] == "1|1>>1|0>>0|1"
            return {"catboost": 0.8, "lightgbm": 0.2}

    selected = service.select_best_model_for_context(
        ticker_df,
        {"target_type": "regression"},
        {
            "model_AAPL_target_return_1d_lightgbm": object(),
            "model_AAPL_target_return_1d_catboost": object(),
        },
        "AAPL",
        "bull",
        _FallbackSelector(),
        diary=Diary(),
    )

    assert selected == "model_AAPL_target_return_1d_catboost"
