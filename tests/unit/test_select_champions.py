"""Tests for scripts/colab/select_champions.py.

Context: the user wants a champion picked per (ticker, target) -- target
already encodes horizon (target_up_1d vs target_up_5d vs
target_weekly_up_1w), comparing real training-run metrics, never the static
priors in src/models/model_selector/model_competence_map.json (left
untouched by design). Entries with no real metric ({'info': 'already_exists'}
skipped models, or {'error': ...} failed runs) must never win a group by
default -- a group where nothing is comparable must come back as
`no_champion`, not a fabricated winner.
"""
from __future__ import annotations

from src.pipeline.hybrid.champion_selector import select_champions


def _entry(ticker, target, model_type, metrics):
    return {"ticker": ticker, "target": target, "model_type": model_type, "metrics": metrics, "model_path": f"model_{ticker}_{target}_{model_type}.keras"}


class TestClassificationChampion:
    def test_binary_prefers_auc_over_accuracy_keras_naming(self):
        """Keras trainers (cnn/lstm/gru/transformer) report val_accuracy/val_auc."""
        metadata = {
            "a": _entry("AAPL", "target_up_1d", "cnn", {"val_accuracy": 0.9, "val_auc": 0.55}),
            "b": _entry("AAPL", "target_up_1d", "lstm", {"val_accuracy": 0.6, "val_auc": 0.70}),
        }
        champions = select_champions(metadata, {"target_up_1d": "classification_binary"})
        result = champions["AAPL::target_up_1d"]
        assert result["status"] == "champion_selected"
        # lstm wins on AUC (0.70 > 0.55) despite cnn having higher raw accuracy.
        assert result["champion_model_type"] == "lstm"
        assert result["selection_metric"] == "auc"

    def test_binary_recognizes_unprefixed_auc_from_sklearn_tabnet(self):
        """mlp/tabnet trainers report plain accuracy/auc (no val_ prefix)."""
        metadata = {
            "a": _entry("AAPL", "target_up_1d", "mlp", {"accuracy": 0.9, "auc": 0.55}),
            "b": _entry("AAPL", "target_up_1d", "tabnet", {"accuracy": 0.6, "auc": 0.70}),
        }
        champions = select_champions(metadata, {"target_up_1d": "classification_binary"})
        result = champions["AAPL::target_up_1d"]
        assert result["champion_model_type"] == "tabnet"
        assert result["selection_metric"] == "auc"

    def test_mixed_keras_and_sklearn_naming_in_same_group(self):
        """A real champion comparison spans Keras (val_-prefixed) and
        sklearn/TabNet (unprefixed) models for the same (ticker, target) --
        both conventions must be comparable in one ranking."""
        metadata = {
            "cnn": _entry("AAPL", "target_up_1d", "cnn", {"val_accuracy": 0.5, "val_auc": 0.50}),
            "mlp": _entry("AAPL", "target_up_1d", "mlp", {"accuracy": 0.5, "auc": 0.80}),
        }
        champions = select_champions(metadata, {"target_up_1d": "classification_binary"})
        result = champions["AAPL::target_up_1d"]
        assert result["champion_model_type"] == "mlp"
        assert len(result["ranking"]) == 2

    def test_falls_back_to_accuracy_when_no_auc(self):
        metadata = {
            "a": _entry("AAPL", "target_multi_1d", "mlp", {"accuracy": 0.5}),
            "b": _entry("AAPL", "target_multi_1d", "cnn", {"val_accuracy": 0.4}),
        }
        # classification_multiclass has no 'auc' concept in this selector.
        champions = select_champions(metadata, {"target_multi_1d": "classification_multiclass"})
        # Neither entry uses the exact key "accuracy" for cnn (val_accuracy) --
        # only mlp's "accuracy" key is recognized by _score for multiclass.
        result = champions["AAPL::target_multi_1d"]
        assert result["status"] == "champion_selected"
        assert result["champion_model_type"] == "mlp"


class TestRegressionChampion:
    def test_lower_mse_wins(self):
        metadata = {
            "a": _entry("AAPL", "target_return_1d", "mlp", {"mse": 0.05}),
            "b": _entry("AAPL", "target_return_1d", "tabnet", {"mse": 0.9}),
        }
        champions = select_champions(metadata, {"target_return_1d": "regression"})
        result = champions["AAPL::target_return_1d"]
        assert result["champion_model_type"] == "mlp"
        assert result["selection_metric"] == "mse"
        assert result["selection_score"] == 0.05  # unnegated in the reported field


class TestNoComparableMetric:
    def test_only_already_exists_entries_yields_no_champion(self):
        metadata = {
            "a": _entry("AAPL", "target_up_1d", "cnn", {"info": "already_exists"}),
            "b": _entry("AAPL", "target_up_1d", "lstm", {"info": "already_exists"}),
        }
        champions = select_champions(metadata, {"target_up_1d": "classification_binary"})
        result = champions["AAPL::target_up_1d"]
        assert result["status"] == "no_champion"
        assert result["candidates_considered"] == 2

    def test_error_entries_are_excluded_but_dont_block_real_winner(self):
        metadata = {
            "a": _entry("AAPL", "target_return_1d", "cnn", {"error": "OOM"}),
            "b": _entry("AAPL", "target_return_1d", "mlp", {"mse": 0.1}),
        }
        champions = select_champions(metadata, {"target_return_1d": "regression"})
        result = champions["AAPL::target_return_1d"]
        assert result["status"] == "champion_selected"
        assert result["champion_model_type"] == "mlp"
        assert result["candidates_considered"] == 2
        assert result["candidates_comparable"] == 1


class TestModelPathKey:
    """colab_clean_cell.py's models_metadata used to write 'path' while
    every real downstream consumer (model_resolver.py, prediction/
    orchestrator.py, scaler_service.py, data_preparer.py, result_builder.py)
    reads 'model_path' -- select_champions.py must read/report the same
    'model_path' key those consumers use, not the stale 'path' name."""

    def test_champion_payload_reports_model_path_from_source_entry(self):
        metadata = {
            "a": _entry("AAPL", "target_return_1d", "mlp", {"mse": 0.1}),
        }
        champions = select_champions(metadata, {"target_return_1d": "regression"})
        result = champions["AAPL::target_return_1d"]
        assert result["model_path"] == "model_AAPL_target_return_1d_mlp.keras"

    def test_missing_model_path_key_does_not_crash(self):
        entry = {"ticker": "AAPL", "target": "target_return_1d", "model_type": "mlp", "metrics": {"mse": 0.1}}
        champions = select_champions({"a": entry}, {"target_return_1d": "regression"})
        assert champions["AAPL::target_return_1d"]["model_path"] is None


class TestGroupingAndDefaults:
    def test_groups_are_keyed_by_ticker_and_target_not_model_type(self):
        metadata = {
            "a": _entry("AAPL", "target_up_1d", "cnn", {"accuracy": 0.5}),
            "b": _entry("MSFT", "target_up_1d", "cnn", {"accuracy": 0.6}),
        }
        champions = select_champions(metadata, {"target_up_1d": "classification_binary"})
        assert "AAPL::target_up_1d" in champions
        assert "MSFT::target_up_1d" in champions
        assert champions["AAPL::target_up_1d"]["champion_model_type"] == "cnn"

    def test_unknown_target_defaults_to_regression_comparison(self):
        metadata = {
            "a": _entry("AAPL", "target_unregistered", "mlp", {"mse": 1.0}),
        }
        champions = select_champions(metadata, {})  # empty registry
        result = champions["AAPL::target_unregistered"]
        assert result["target_type"] == "regression"
        assert result["status"] == "champion_selected"

    def test_ranking_is_sorted_best_first(self):
        metadata = {
            "a": _entry("AAPL", "target_return_1d", "mlp", {"mse": 0.5}),
            "b": _entry("AAPL", "target_return_1d", "cnn", {"mse": 0.1}),
            "c": _entry("AAPL", "target_return_1d", "lstm", {"mse": 0.3}),
        }
        champions = select_champions(metadata, {"target_return_1d": "regression"})
        ranking = champions["AAPL::target_return_1d"]["ranking"]
        assert [r["model_type"] for r in ranking] == ["cnn", "lstm", "mlp"]
