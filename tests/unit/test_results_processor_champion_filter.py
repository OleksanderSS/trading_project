"""Tests for ResultsProcessor.build_models_metadata's champion-filter wiring.

Context: build_models_metadata used to hand Stage 5 every trained model_type
for every (ticker, target) -- prediction could end up running whichever
architecture happened to be resolved first, not the one that actually
performed best. It now hard-filters through
src/pipeline/hybrid/champion_selector.filter_to_champions so only the
empirical champion per (ticker, target) survives.
"""
from __future__ import annotations

from src.pipeline.hybrid.results_processor import ResultsProcessor


def _entry(ticker, target, model_type, metrics):
    """Mirrors colab_clean_cell.py's real models_metadata shape: caller
    keys the dict by f"{ticker}_{target}_{model_type}" (this helper
    returns only the value; see each test for the keying)."""
    return {
        "ticker": ticker,
        "target": target,
        "model_type": model_type,
        "metrics": metrics,
        "model_path": f"model_{ticker}_{target}_{model_type}.keras",
    }


def _key(ticker, target, model_type):
    return f"{ticker}_{target}_{model_type}"


class TestBuildModelsMetadataChampionFilter:
    def test_only_the_champion_model_type_survives(self):
        colab_results = {
            "models_metadata": {
                _key("AAPL", "target_return_1d", "mlp"): _entry("AAPL", "target_return_1d", "mlp", {"mse": 0.5}),
                _key("AAPL", "target_return_1d", "cnn"): _entry("AAPL", "target_return_1d", "cnn", {"mse": 0.1}),
            }
        }
        processor = ResultsProcessor()
        result = processor.build_models_metadata(colab_results, None)
        assert set(result.keys()) == {"AAPL_target_return_1d_cnn"}

    def test_multiple_ticker_target_groups_each_keep_their_own_champion(self):
        colab_results = {
            "models_metadata": {
                _key("AAPL", "target_return_1d", "mlp"): _entry("AAPL", "target_return_1d", "mlp", {"mse": 0.5}),
                _key("AAPL", "target_return_1d", "cnn"): _entry("AAPL", "target_return_1d", "cnn", {"mse": 0.1}),
                _key("MSFT", "target_return_1d", "mlp"): _entry("MSFT", "target_return_1d", "mlp", {"mse": 0.05}),
                _key("MSFT", "target_return_1d", "cnn"): _entry("MSFT", "target_return_1d", "cnn", {"mse": 0.2}),
            }
        }
        processor = ResultsProcessor()
        result = processor.build_models_metadata(colab_results, None)
        assert set(result.keys()) == {"AAPL_target_return_1d_cnn", "MSFT_target_return_1d_mlp"}

    def test_group_with_no_comparable_metric_is_dropped_not_kept_arbitrarily(self):
        colab_results = {
            "models_metadata": {
                _key("AAPL", "target_return_1d", "mlp"): _entry("AAPL", "target_return_1d", "mlp", {"info": "already_exists"}),
                _key("AAPL", "target_return_1d", "cnn"): _entry("AAPL", "target_return_1d", "cnn", {"error": "OOM"}),
            }
        }
        processor = ResultsProcessor()
        result = processor.build_models_metadata(colab_results, None)
        assert result == {}

    def test_light_results_models_are_included_in_the_champion_pool(self):
        colab_results = {
            "models_metadata": {
                _key("AAPL", "target_return_1d", "cnn"): _entry("AAPL", "target_return_1d", "cnn", {"mse": 0.5}),
            }
        }
        light_results = {
            "models_metadata": {
                _key("AAPL", "target_return_1d", "lgbm"): _entry("AAPL", "target_return_1d", "lgbm", {"mse": 0.05}),
            }
        }
        processor = ResultsProcessor()
        result = processor.build_models_metadata(colab_results, light_results)
        assert set(result.keys()) == {"AAPL_target_return_1d_lgbm"}

    def test_filter_uses_actual_source_key_not_a_reconstructed_one(self):
        """Regression: filter_to_champions must key off the winning entry's
        real dict key, not a "{ticker}_{target}_{model_type}"-formatted
        guess -- a models_metadata keyed some other way must still work."""
        colab_results = {
            "models_metadata": {
                "arbitrary_key_1": _entry("AAPL", "target_return_1d", "mlp", {"mse": 0.5}),
                "arbitrary_key_2": _entry("AAPL", "target_return_1d", "cnn", {"mse": 0.1}),
            }
        }
        processor = ResultsProcessor()
        result = processor.build_models_metadata(colab_results, None)
        assert set(result.keys()) == {"arbitrary_key_2"}

    def test_no_models_metadata_at_all_returns_empty_without_crashing(self):
        processor = ResultsProcessor()
        result = processor.build_models_metadata({}, None)
        assert result == {}
