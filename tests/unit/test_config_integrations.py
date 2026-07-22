import importlib
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize(
    "module_path,class_name",
    [
        ("src.analytics.detectors.regime_detector", "MarketRegimeDetector"),
        ("src.features.enrichers.significance_features_enricher", "SignificanceFeaturesEnricher"),
        ("src.data_sources.local_file_data_source", "LocalFileDataSource"),
        ("src.features.transformers.transformers", "StandardScalerTransformer"),
        ("src.features.transformers.transformers", "MinMaxScalerTransformer"),
    ],
)
def test_config_referenced_classes_are_importable(module_path, class_name):
    module = importlib.import_module(module_path)

    assert hasattr(module, class_name)


def test_yaml_config_paths_match_real_classes():
    analysis = _load_yaml("src/config/analysis.yaml")
    enrichment = _load_yaml("src/config/enrichment.yaml")
    data_sources = _load_yaml("src/config/data_sources.yaml")
    transformers = _load_yaml("src/config/transformers.yaml")

    refs = [
        analysis["calculators_config"]["market_regime_calc"],
        enrichment["enrichment"]["significance_features"],
        data_sources["data_sources"][0],
        *transformers["transformers"],
    ]

    for ref in refs:
        module = importlib.import_module(ref["module"])
        assert hasattr(module, ref["class"])


def test_local_file_data_source_loads_csv():
    from src.data_sources.local_file_data_source import LocalFileDataSource

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "prices.csv"
        path.write_text("date,close\n2024-01-01,100\n", encoding="utf-8")

        df = LocalFileDataSource(path, date_col="date").load()

        assert df.loc[0, "close"] == 100
        assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_transformer_config_compatibility_class_scales_dataframe():
    from src.features.transformers.transformers import StandardScalerTransformer

    df = pd.DataFrame({"close": [100.0, 110.0]})

    transformed = StandardScalerTransformer(columns=["close"]).fit_transform(df)

    assert transformed["close"].round(6).tolist() == [-1.0, 1.0]


def test_processing_handler_unwraps_filters_and_normalizes_nested_prices():
    from src.pipeline.stages.processing.data_handler import ProcessingDataHandler
    from src.processing.normalization_manager import NormalizationManager

    class StubFilter:
        def filter_quality_data(self, raw_data):
            return {
                "filtered_data": {
                    "prices": {
                        "1d": {
                            "data": raw_data["prices"]["1d"],
                            "quality": {"status": "accepted"},
                        }
                    }
                },
                "quality_report": {"prices": {"1d": {"status": "accepted"}}},
            }

    df = pd.DataFrame({"close": [100.0, 110.0], "volume": [10.0, 20.0]})
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = NormalizationManager(scaler_dir=str(tmp_dir))
        handler = ProcessingDataHandler(manager, StubFilter())

        filtered = handler.apply_intelligent_filtering({"prices": {"1d": df}})
        handler.apply_normalization(
            filtered,
            features_to_normalize=[
                {"feature": "close", "scaler_type": "standard"},
                {"feature": "volume", "scaler_type": "min_max"},
            ],
        )

        normalized = filtered["prices"]["1d"]
        assert normalized["close"].round(6).tolist() == [-1.0, 1.0]
        assert normalized["volume"].tolist() == [0.0, 1.0]


def test_legacy_ensemble_selector_import_uses_active_implementation():
    from src.analytics.context.ensemble_selector import EnsembleContext, EnsembleSelector
    from src.integration.ensemble_selector import EnsembleSelector as LegacySelector

    context = EnsembleContext(
        data_size=100,
        has_real_time_data=False,
        model_count=1,
        market_regime="normal",
        volatility_level=0.1,
        prediction_frequency="batch",
        computational_resources="low",
        latency_requirement="high",
    )

    result = LegacySelector().select_best_ensemble(context, ["model_a"])
    simple_average = EnsembleSelector().create_ensemble_instance("simple_average")

    assert result["selected_ensemble"] == "simple_average"
    assert simple_average({"a": [1.0], "b": [3.0]}).tolist() == [2.0]
