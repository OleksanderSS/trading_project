"""Every analyzer the config enables must actually load.

UnifiedAnalyticsEngine is live -- Stage 7 constructs it -- and builds its
registry dynamically from analysis.yaml. A configured analyzer whose module
is missing is logged and skipped, so the stage runs on fewer analyzers than
the config says with nothing failing.

That is what had happened to CriticalSignalDetector. Commit dabe5540
archived it as having "zero callers", which was true of the one path it
looked at: PredictionStage.__init__ built it and never called it. The
config-driven path was missed -- analysis.yaml lists it with enabled: true
and data_mapping ['price_data'] -- even though the same commit correctly
names that path "the actual live registration path". Static reachability
cannot see config-driven loading; it is the blind spot documented in
diagnostic_reports/AUDIT_GUIDE.md, and it cost Stage 7 a working analyzer
(engine went from 3 registered back to 2, silently).

Restored to src/analytics/detectors/. This test is what keeps the config and
the tree from drifting apart again.
"""
from __future__ import annotations

import importlib

import pytest
import yaml

from src.analytics.interfaces import IAnalyzer
from src.config.unified_config_manager import get_current_config


def _configured_analyzers():
    with open("src/config/analysis.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config["analysis"]["engine"]["analyzers"]


ENABLED = [
    entry for entry in _configured_analyzers()
    if entry.get("enabled", True)
]
ALL = _configured_analyzers()


@pytest.mark.parametrize("entry", ENABLED, ids=lambda e: e.get("name", "?"))
def test_an_enabled_analyzer_can_be_imported(entry):
    module = importlib.import_module(entry["module"])
    assert hasattr(module, entry["class"]), (
        f"{entry['class']} is not in {entry['module']}"
    )


@pytest.mark.parametrize("entry", ENABLED, ids=lambda e: e.get("name", "?"))
def test_an_enabled_analyzer_satisfies_the_interface(entry):
    """The engine drops anything that is not an IAnalyzer, with a warning."""
    module = importlib.import_module(entry["module"])
    analyzer_class = getattr(module, entry["class"])

    assert issubclass(analyzer_class, IAnalyzer)


@pytest.mark.parametrize("entry", ALL, ids=lambda e: e.get("name", "?"))
def test_a_disabled_analyzer_is_disabled_on_purpose(entry):
    """A disabled entry pointing at a module that no longer exists is a
    leftover, not a decision. Kept separate from the enabled checks so the
    two failures read differently."""
    if entry.get("enabled", True):
        return
    try:
        importlib.import_module(entry["module"])
    except ModuleNotFoundError:
        pytest.skip(
            f"{entry['name']} is disabled AND its module is gone; remove the "
            f"entry or restore {entry['module']}"
        )


def test_the_engine_registers_every_enabled_analyzer():
    from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine

    engine = UnifiedAnalyticsEngine(get_current_config())
    report = engine.analyzer_registration_report

    failed = {
        name: info for name, info in report.items()
        if info.get("status") not in ("registered", "disabled")
    }
    assert not failed, f"analyzers the config enables but the engine dropped: {failed}"


def test_critical_signals_is_among_them():
    """The specific regression: archived while the config still enabled it."""
    from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine

    engine = UnifiedAnalyticsEngine(get_current_config())

    assert "critical_signals" in engine.analyzers
