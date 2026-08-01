"""Context vulnerability and success zones, as fed to compare_agents.

Live: dual_loops.py:184 calls compare_agents, which calls both of these per
agent.

_analyze_fingerprint_components decomposes a fingerprint into per-driver
tri-state counts. That works for the Context Map form ('1|-1|0|1'), and the
pipeline writes a SHA-256 (ModelingStage._build_context_fingerprint) or the
literal 'normal'. For those, every token falls outside {-1, 0, 1}, and the
function used to return

    {0: {'-1': 0.0, '0': 0.0, '1': 0.0}}

which reads as "no driver is implicated" when it means "this fingerprint
cannot be read". Same silent shape as the leakage report that defaulted to
'clean', and the empty stage output that counted as success.

Two more things fixed here:

- get_context_vulnerability had no HAVING threshold while
  get_context_success_analysis required two wins, so one loss made a
  "failure pattern" but one win made nothing. compare_agents prints both
  side by side.
- the docstring now records that this counts RAW OCCURRENCES, not rates: a
  driver value present in most trades tops the table whatever its loss rate,
  because there is no base-rate denominator.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.meta_learning.memory.diary_engine import DiaryEngine

HASH = "67a9e31d5fb39bf5212a44f8ff79c9e423d088acc1c18f695f51c59f38091385"


@pytest.fixture()
def engine():
    instance = object.__new__(DiaryEngine)
    instance.logger = logging.getLogger("diary-context-test")
    return instance


def _patterns(fingerprint, count=7, column="loss_count"):
    return pd.DataFrame([{"context_fingerprint": fingerprint, column: count}])


def test_a_tristate_fingerprint_decomposes(engine):
    stats = engine._analyze_fingerprint_components(_patterns("1|-1|0|1"))

    assert stats[0]["1"] == 7.0
    assert stats[1]["-1"] == 7.0
    assert stats[2]["0"] == 7.0
    assert stats[3]["1"] == 7.0


def test_the_time_suffix_is_stripped_before_decoding(engine):
    stats = engine._analyze_fingerprint_components(_patterns("1|0__mon|14"))

    assert stats[0]["1"] == 7.0
    assert stats[1]["0"] == 7.0
    assert len(stats) == 2, "the time part must not become drivers"


@pytest.mark.parametrize("fingerprint", [HASH, "normal", "elevated"])
def test_an_undecodable_fingerprint_returns_nothing_not_zeros(engine, fingerprint):
    """The regression: zero-filled counts read as a clean bill of health."""
    assert engine._analyze_fingerprint_components(_patterns(fingerprint)) == {}


def test_counts_accumulate_across_rows(engine):
    frame = pd.DataFrame([
        {"context_fingerprint": "1|0", "loss_count": 3},
        {"context_fingerprint": "1|1", "loss_count": 4},
    ])

    stats = engine._analyze_fingerprint_components(frame)

    assert stats[0]["1"] == 7.0
    assert stats[1]["0"] == 3.0
    assert stats[1]["1"] == 4.0


def test_the_win_count_column_is_honoured(engine):
    stats = engine._analyze_fingerprint_components(
        _patterns("1|0", count=5, column="win_count"), col="win_count"
    )

    assert stats[0]["1"] == 5.0


def test_a_mixed_fingerprint_keeps_only_the_readable_positions(engine):
    """Junk between drivers must not shift the positions that do decode."""
    stats = engine._analyze_fingerprint_components(_patterns("1|abc|0"))

    assert stats[0]["1"] == 7.0
    assert 2 in stats and stats[2]["0"] == 7.0
    assert 1 not in stats


def test_callers_can_tell_undecodable_from_clean():
    """The flag that makes the two distinguishable at the call site."""
    import inspect

    for method in (
        DiaryEngine.get_context_vulnerability,
        DiaryEngine.get_context_success_analysis,
    ):
        assert "components_decoded" in inspect.getsource(method)


def test_both_analyses_use_the_same_evidence_threshold():
    """compare_agents shows them side by side, so one loss must not count
    for more than one win."""
    import inspect

    losses = inspect.getsource(DiaryEngine.get_context_vulnerability)
    wins = inspect.getsource(DiaryEngine.get_context_success_analysis)

    assert "HAVING loss_count >= 2" in losses
    assert "HAVING win_count >= 2" in wins
