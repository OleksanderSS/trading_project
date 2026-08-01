"""Every config section the code asks for must exist.

`get_config('X')` returns None for an absent key and callers almost always
write `or {}` straight after, so a missing section degrades into hardcoded
defaults with nothing said. A sweep of every get_config call found 10 of 24
requested keys did not exist -- including `processing`, which left
IntelligentDataFilter running unconfigured on every run while deciding what
data enters the pipeline at all, and `modeling`, which made four training
settings unreachable.

This is a change-detector: a new get_config for a section nobody created will
fail here rather than silently doing nothing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config.unified_config_manager import get_current_config

ROOT = Path(__file__).resolve().parents[2]
SKIP_PARTS = {"__pycache__", "archive", "draft", "audit"}
CALL_RE = re.compile(r"""get_config\(\s*["']([A-Za-z0-9_.]+)["']""")

#: Lookups that are legitimately optional or belong to another config domain.
ALLOWED_ABSENT = {
    # Obsolete lookups in modules that are themselves not on a live path.
    "asset_universe",   # auto_accumulator docstring only; the code no longer uses it
    "experiments",      # src/scripts/experiments/compare_layers.py, known-broken script
    "unified",          # scripts/check_enrichers_integration.py, dev helper
    "simulation",       # monster_test / synthetic generation, opt-in modes
    "cache",            # cli/pipeline_executor optional cache tuning
    "notifications",    # notifier: absent means "no notifications configured"
    "secrets",          # notifier: secrets come from the env, not YAML
}


def _requested_sections() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for root in ("src", "dean_os", "scripts"):
        base = ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if SKIP_PARTS & set(path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for key in CALL_RE.findall(text):
                found.setdefault(key, set()).add(str(path.relative_to(ROOT)))
    return found


def test_every_requested_config_section_exists():
    config = get_current_config()
    requested = _requested_sections()

    missing = {
        key: sorted(files)
        for key, files in requested.items()
        if key not in ALLOWED_ABSENT and config.get_config(key) is None
    }

    assert not missing, (
        "Config sections requested by code but absent from every YAML — the "
        "caller will silently use hardcoded defaults:\n"
        + "\n".join(f"  '{k}' asked by {v[:3]}" for k, v in sorted(missing.items()))
    )


@pytest.mark.parametrize("section", ["processing", "prediction", "backtest", "modeling"])
def test_previously_missing_sections_are_present(section):
    """Regression guard for the four sections found absent."""
    assert get_current_config().get_config(section) is not None


def test_data_filter_parameters_are_configurable():
    """IntelligentDataFilter decides what data enters the pipeline at all.

    Its eleven thresholds used to be unreachable because `processing` did not
    exist.
    """
    filtering = (get_current_config().get_config("processing") or {}).get("filtering", {})

    for key in (
        "min_data_quality_score",
        "anomaly_std_dev_threshold",
        "min_cadence_match_ratio",
        "max_extreme_return_ratio",
    ):
        assert key in filtering, f"{key} is not configurable"


def test_absent_section_is_reported_not_silent():
    """The root cause was silence, not the missing keys themselves."""
    config = get_current_config()
    type(config)._absent_sections_reported.discard("definitely_not_a_section")

    import logging

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture()
    logging.getLogger().addHandler(handler)
    try:
        config.get_config("definitely_not_a_section")
    finally:
        logging.getLogger().removeHandler(handler)

    assert any("definitely_not_a_section" in m for m in records)
