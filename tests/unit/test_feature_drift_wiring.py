"""Feature drift monitoring must report real numbers, not constants.

Four faults kept this from working, and two of them produced plausible-looking
output rather than an error, which is worse:

- DriftAnalyzer called `detect_drift`; the monitor only has `check_drift`.
- nothing ever set reference data, so the corrected call still raised.
- the dataset-drift result was read at a hardcoded index `metrics[1]`, which
  is DataDriftTable (no `drift_share` key) rather than DatasetDriftMetric, so
  the score defaulted to 0.0 forever.
- per-column drift was looked for in standalone ColumnDriftMetric entries,
  which DataDriftPreset does not emit, so every report said "0/0 features".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.analyzers.drift_analyzer import DriftAnalyzer
from src.monitoring.feature_drift_monitor import EVIDENTLY_AVAILABLE

pytestmark = pytest.mark.skipif(
    not EVIDENTLY_AVAILABLE, reason="Evidently AI not importable"
)


@pytest.fixture(autouse=True)
def isolated_baseline(tmp_path, monkeypatch):
    """Never touch the real reports/drift baseline.

    Without this, a baseline written by any previous run (or by ad-hoc
    verification) makes the first-frame test see an existing reference and
    fail -- the same on-disk-state coupling this suite exists to prevent.
    """
    monkeypatch.setattr(
        DriftAnalyzer, "__init__",
        _init_with_default(tmp_path / "reference_features.parquet"),
    )


def _init_with_default(default_path):
    original = DriftAnalyzer.__init__

    def patched(self, threshold=0.05, config=None, baseline_path=None):
        original(self, threshold, config, baseline_path or str(default_path))

    return patched


def _frame(shift=0.0, n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "f1": rng.normal(0 + shift, 1, n),
        "f2": rng.normal(5 + shift, 2, n),
        "f3": rng.normal(-1, 1, n),
    })


def test_evidently_is_importable_on_this_install():
    """The 0.7 line moved the classic API under `evidently.legacy`; importing
    only the pre-0.7 paths reported an installed package as missing."""
    assert EVIDENTLY_AVAILABLE is True


def test_first_frame_becomes_the_baseline():
    analyzer = DriftAnalyzer()
    result = analyzer.analyze({"features_data": _frame()})
    assert result["status"] == "baseline_set"


def test_clean_data_reports_zero_drift_not_a_constant():
    analyzer = DriftAnalyzer()
    analyzer.analyze({"features_data": _frame(seed=1)})
    result = analyzer.analyze({"features_data": _frame(seed=2)})

    assert result["drift_detected"] is False
    assert result["drift_score"] == 0.0, (
        "a constant 0.5 here means the THRESHOLD is being reported as the score"
    )
    assert result["drifted_features_count"] == 0


def test_shifted_features_are_counted_individually():
    analyzer = DriftAnalyzer()
    analyzer.analyze({"features_data": _frame(seed=1)})

    moved = _frame(seed=2)
    moved["f1"] += 4.0
    moved["f2"] += 7.0
    result = analyzer.analyze({"features_data": moved})

    assert result["drift_detected"] is True
    assert result["total_features"] == 3, "per-column accounting must not be 0"
    assert result["drifted_features_count"] == 2
    assert result["drift_score"] == pytest.approx(2 / 3, abs=0.01)


def test_empty_frame_is_skipped_not_raised():
    analyzer = DriftAnalyzer()
    assert analyzer.analyze({"features_data": pd.DataFrame()})["status"] == "skipped"
    assert analyzer.analyze(None)["status"] == "skipped"


def test_a_monitoring_adapter_never_raises():
    """It observes the analytics pass; throwing would take that pass down."""
    analyzer = DriftAnalyzer()
    analyzer.analyze({"features_data": _frame()})

    # Disjoint columns -> the monitor raises internally; the adapter must not.
    result = analyzer.analyze({"features_data": pd.DataFrame({"zzz": [1.0, 2.0]})})
    assert result["status"] == "unavailable"


def test_drift_analyzer_is_registered_in_config():
    """Registration is what makes UnifiedAnalyticsEngine build it at all."""
    import yaml

    config = yaml.safe_load(
        open("src/config/analysis.yaml", encoding="utf-8")
    )
    analyzers = config["stage7_analysis_engine"]["analyzers"]
    entry = next((a for a in analyzers if a.get("name") == "feature_drift"), None)

    assert entry is not None, "feature_drift missing from analysis.yaml"
    assert entry["enabled"] is True
    assert entry["class"] == "DriftAnalyzer"


def test_baseline_survives_a_new_analyzer_instance(tmp_path):
    """A pipeline run builds the analyzer fresh. An in-memory baseline would
    make every run report 'baseline_set' and never compare anything."""
    path = tmp_path / "reference.parquet"

    first = DriftAnalyzer(baseline_path=str(path))
    assert first.analyze({"features_data": _frame(seed=1)})["status"] == "baseline_set"
    assert path.exists()

    # New process, new object -- must pick the baseline back up and compare.
    second = DriftAnalyzer(baseline_path=str(path))
    moved = _frame(seed=2)
    moved["f1"] += 4.0
    moved["f2"] += 7.0
    result = second.analyze({"features_data": moved})

    assert result["status"] != "baseline_set"
    assert result["drift_detected"] is True
    assert result["drifted_features_count"] == 2
