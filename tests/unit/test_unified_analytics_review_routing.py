from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from src.analytics.interfaces import IAnalyzer
from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine


class _ResultsManager:
    def __init__(self):
        self.cached = None

    def get_cached_analysis(self, data_hash):
        return None

    def cache_analysis(self, data_hash, results):
        self.cached = results


class _DataFrameAnalyzer(IAnalyzer):
    def analyze(self, data, **kwargs):
        return pd.DataFrame({"value": [1.0, 2.0]})


class _FailingAnalyzer(IAnalyzer):
    def analyze(self, data, **kwargs):
        raise RuntimeError("isolated analyzer failure")


def _engine() -> UnifiedAnalyticsEngine:
    engine = object.__new__(UnifiedAnalyticsEngine)
    engine.analyzers = {
        "good": _DataFrameAnalyzer(),
        "missing": _DataFrameAnalyzer(),
        "failing": _FailingAnalyzer(),
    }
    engine.analyzer_data_map = {
        "good": ["good_input"],
        "missing": ["missing_input"],
        "failing": ["failing_input"],
    }
    engine.analyzer_configs = [{}, {}, {}, {"enabled": False}]
    engine.analyzer_registration_report = {
        "good": {"status": "registered"},
        "missing": {"status": "registered"},
        "failing": {"status": "registered"},
        "disabled": {"status": "disabled"},
    }
    engine.results_manager = _ResultsManager()
    engine.thread_pool = ThreadPoolExecutor(max_workers=2)
    return engine


def test_engine_isolates_missing_inputs_and_analyzer_failures():
    engine = _engine()
    try:
        result = engine.run_full_analysis({
            "good_input": pd.DataFrame({"x": [1, 2]}),
            "failing_input": pd.DataFrame({"x": [1, 2]}),
        })
    finally:
        engine.thread_pool.shutdown(wait=True)

    assert result["good"]["status"] == "completed"
    assert result["good"]["output_type"] == "dataframe_summary"
    assert result["good"]["row_count"] == 2
    assert result["missing"]["status"] == "skipped_missing_inputs"
    assert result["missing"]["missing_inputs"] == ["missing_input"]
    assert result["failing"]["status"] == "failed"

    coverage = result["_analysis_coverage"]
    assert coverage["executed"] == ["good"]
    assert coverage["skipped_missing_inputs"] == ["missing"]
    assert coverage["failed"] == ["failing"]
    assert coverage["disabled"] == ["disabled"]
    assert len(coverage["analysis_contract_hash"]) == 64
    assert coverage["can_promote_model"] is False
    assert coverage["can_trade"] is False


def test_cache_fingerprint_changes_with_analyzer_suite_contract():
    engine = _engine()
    try:
        data_map = {"good_input": pd.DataFrame({"x": [1, 2]})}
        original_hash = engine._generate_data_hash(data_map)
        engine.analyzer_data_map["good"] = ["different_input"]
        changed_hash = engine._generate_data_hash(data_map)
    finally:
        engine.thread_pool.shutdown(wait=True)

    assert original_hash != changed_hash


def test_cache_fingerprint_changes_when_late_data_changes():
    engine = _engine()
    try:
        original = pd.DataFrame({"x": range(20)})
        changed = original.copy()
        changed.loc[19, "x"] = 999

        original_hash = engine._generate_data_hash({"good_input": original})
        changed_hash = engine._generate_data_hash({"good_input": changed})
    finally:
        engine.thread_pool.shutdown(wait=True)

    assert original_hash != changed_hash


def test_stage7_analyzer_suite_has_one_canonical_config_source():
    root = Path(__file__).resolve().parents[2]
    analysis_config = yaml.safe_load(
        (root / "src/config/analysis.yaml").read_text(encoding="utf-8")
    )
    unified_config = yaml.safe_load(
        (root / "src/config/unified_config.yaml").read_text(encoding="utf-8")
    )

    engine_config = analysis_config["analysis"]["engine"]
    enabled = sorted(
        analyzer["name"]
        for analyzer in engine_config["analyzers"]
        if analyzer.get("enabled", True)
    )

    assert enabled == ["critical_signals", "market_regime"]
    assert "engine" not in unified_config["analysis"]
