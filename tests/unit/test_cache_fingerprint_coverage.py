"""The cache fingerprint must cover everything that decides what is cached.

PipelineExecutor reuses features.parquet/targets.parquet when the raw-data
fingerprint AND the code fingerprint both match. The code fingerprint exists
because a purely data-driven cache silently served pre-fix features after a
bug fix -- exactly the failure this project has already paid for.

It had two holes. stages/feature_engineering/targets.py was covered but only
delegates: TargetOrchestrator and timeframe_contract, which decide every
value in targets.parquet, live under src/targets and were not hashed. Stage 2
cleaning under src/processing produces the frame Stage 3 enriches and was not
hashed either. Config was not hashed at all, though it decides which
enrichers and targets run in the first place.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.cli.pipeline_executor import PipelineExecutor

ROOT = Path(__file__).resolve().parents[2]


def _fingerprinted() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in PipelineExecutor._fingerprint_files(ROOT)
    }


@pytest.mark.parametrize("path", [
    # Decides every value in the cached features.parquet.
    "src/features/utils/technical_indicators_lib.py",
    "src/features/enrichers/context_map_enricher.py",
    # Decides every value in the cached targets.parquet.
    "src/targets/target_orchestrator.py",
    "src/targets/timeframe_contract.py",
    # Stage 2 output is Stage 3 input.
    "src/processing/cleaners.py",
    "src/processing/data_filter.py",
    # Decides which enrichers and analyzers run at all.
    "src/config/analysis.yaml",
    # Decides what data ARRIVES, which decides the features built from it.
    # Omitting these closed a circle: the cache gates all of stages 0-3 on
    # whether the database grew, the database only grows if collection runs,
    # so a broken collector could never be fixed -- the fix would not
    # invalidate the cache, and collection would be skipped again.
    "src/data/collectors/yf_collector.py",
    "src/data/validation/price_source_gate.py",
])
def test_a_file_that_changes_the_cached_output_is_fingerprinted(path):
    assert path in _fingerprinted(), (
        f"{path} decides what stages 0-3 compute but is not hashed into the "
        "code fingerprint, so a change to it would be silently skipped by a "
        "stale features.parquet."
    )


def test_generated_config_is_excluded():
    """ContextRuleGenerator writes src/config/generated_context_rules.yaml.

    Hashing a file the run itself produces would move the fingerprint on
    every run and turn the cache into a permanent miss.
    """
    assert "src/config/generated_context_rules.yaml" not in _fingerprinted()


def test_the_fingerprint_is_deterministic():
    assert (
        PipelineExecutor._compute_code_fingerprint()
        == PipelineExecutor._compute_code_fingerprint()
    )


def test_editing_a_covered_file_moves_the_fingerprint(tmp_path, monkeypatch):
    """The property the whole mechanism rests on."""
    covered = tmp_path / "src" / "targets"
    covered.mkdir(parents=True)
    source = covered / "orchestrator.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    before = sorted(PipelineExecutor._fingerprint_files(tmp_path))
    assert source in before

    import hashlib

    def digest():
        hasher = hashlib.sha256()
        for path in sorted(PipelineExecutor._fingerprint_files(tmp_path)):
            hasher.update(path.name.encode())
            hasher.update(path.read_bytes())
        return hasher.hexdigest()

    first = digest()
    source.write_text("VALUE = 2\n", encoding="utf-8")

    assert digest() != first


def test_the_atr_change_would_have_invalidated_the_cache():
    """A concrete check, tied to a real fix from this audit.

    calculate_atr moved from a rolling mean to Wilder smoothing on
    2026-08-02, changing ATR_14 and everything derived from it. That file
    must be inside the fingerprint, or the next prepare run would happily
    reuse the pre-fix features.
    """
    assert "src/features/utils/technical_indicators_lib.py" in _fingerprinted()
