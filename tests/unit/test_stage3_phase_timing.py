"""Stage 3 must report where its five hours went.

Written after a healthy rebuild was killed on a guess: asked whether the run
was behaving abnormally, the only evidence available was per-enricher lines,
so the phases *around* them -- redundancy detection, feature selection,
combining timeframes -- were invisible and had to be reconstructed by diffing
timestamps out of a 31 MB log after the fact.
"""

import logging

import pytest

from src.pipeline.stages.feature_engineering.orchestrator import FeatureEngineeringStage


class _Recorder(logging.Logger):
    def __init__(self):
        super().__init__('recorder')
        self.lines: list[str] = []

    def info(self, msg, *args, **kwargs):  # noqa: D102
        self.lines.append(msg % args if args else msg)


@pytest.fixture
def stage():
    inst = FeatureEngineeringStage.__new__(FeatureEngineeringStage)
    inst.logger = _Recorder()
    inst._phase_seconds = {}
    inst._phase_memory = {}
    return inst


def test_phase_accumulates_repeated_names(stage):
    """Three timeframes enriched under one label must sum, not overwrite."""
    for _ in range(3):
        with stage._phase('enrich'):
            pass
    assert list(stage._phase_seconds) == ['enrich']
    assert stage._phase_seconds['enrich'] >= 0.0


def test_breakdown_is_sorted_longest_first(stage):
    stage._phase_seconds = {'quick': 1.0, 'slow': 600.0, 'middling': 60.0}
    stage._log_phase_breakdown()
    body = [line for line in stage.logger.lines if 'min' in line and '%' in line]
    assert [line.split('%')[1].strip() for line in body] == ['slow', 'middling', 'quick']
    assert '10.0 min' in body[0] and '90.8%' in body[0]  # 600 / 661


def test_breakdown_survives_a_crash(stage):
    """The numbers matter most when the stage dies -- don't lose them."""
    with pytest.raises(ValueError):
        with stage._phase('doomed'):
            raise ValueError('boom')
    assert 'doomed' in stage._phase_seconds
    assert any('phase breakdown' in line for line in stage.logger.lines)


def test_silent_when_nothing_was_timed(stage):
    stage._log_phase_breakdown()
    assert stage.logger.lines == []


def test_memory_is_recorded_at_every_phase_boundary(stage):
    """The run that died left no record of what it was holding."""
    with stage._phase('enrich 1d'):
        pass
    assert 'enrich 1d (start)' in stage._phase_memory
    assert 'enrich 1d (end)' in stage._phase_memory
    assert all(v > 0 for v in stage._phase_memory.values())


def test_the_breakdown_names_the_peak(stage):
    stage._phase_seconds = {'a': 1.0}
    stage._phase_memory = {'quiet (start)': 1.5, 'heavy (end)': 9.25}
    stage._log_phase_breakdown()
    peak = [line for line in stage.logger.lines if 'Peak memory' in line]
    assert peak and '9.25 GiB' in peak[0] and 'heavy (end)' in peak[0]


def test_instrumentation_never_ends_a_run(stage, monkeypatch):
    """A measurement that can kill the thing it measures is worse than none."""
    import builtins
    real_import = builtins.__import__

    def _no_psutil(name, *args, **kwargs):
        if name == 'psutil':
            raise ImportError('gone')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', _no_psutil)
    with stage._phase('enrich 1d'):
        pass
    assert stage._phase_memory == {}
    assert stage._phase_seconds['enrich 1d'] >= 0.0
