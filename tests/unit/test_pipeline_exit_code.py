"""A failed stage must never leave a zero behind.

Twice in two days a run ended with

    RuntimeError: Stage FeatureEngineeringStage execution failed

and neither "Pipeline completed" nor "Pipeline failed" reached the log, so
main() had not got as far as reporting. One of those runs still returned 0 to
the shell and the other returned 1, from the same invocation shape. The
mechanism was not reproduced, so `_run` is a GUARANTEE rather than a diagnosis:
anything escaping main() is reported and exits non-zero whatever the cause.

The cost of the alternative is measured, not hypothetical. A rebuild that
"succeeded" left the batch untouched at its previous timestamp, and the next
step was planned on data that had never changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import run_hybrid_pipeline as rhp  # noqa: E402


@pytest.fixture
def patched(monkeypatch):
    def install(behaviour):
        async def fake_main():
            return behaviour()
        monkeypatch.setattr(rhp, 'main', fake_main)
    return install


class TestFailuresSurfaceAsNonZero:
    def test_a_raising_stage_exits_non_zero(self, patched):
        def boom():
            raise RuntimeError('Stage FeatureEngineeringStage execution failed')
        patched(boom)
        assert rhp._run() == 1

    def test_a_memory_error_exits_non_zero(self, patched):
        # The actual failure both times: 4.22 GiB for a (2192, 258397) array.
        def boom():
            raise MemoryError('Unable to allocate 4.22 GiB')
        patched(boom)
        assert rhp._run() == 1

    def test_a_keyboard_interrupt_is_not_reported_as_success(self, patched):
        # BaseException, not Exception: a run someone stopped is not a run
        # that worked, and the shell should not be told otherwise.
        def boom():
            raise KeyboardInterrupt()
        patched(boom)
        assert rhp._run() == 1

    def test_the_failure_is_logged_with_its_type(self, patched, caplog):
        import logging

        def boom():
            raise ValueError('something specific')
        patched(boom)
        with caplog.at_level(logging.CRITICAL):
            rhp._run()
        assert any('ValueError' in r.message or 'ValueError' in str(r.args)
                   for r in caplog.records)


class TestSuccessAndDeliberateExitsAreUntouched:
    def test_a_clean_run_exits_zero(self, patched):
        patched(lambda: None)
        assert rhp._run() == 0

    def test_sys_exit_keeps_its_own_code(self, patched):
        # The reporting block calls sys.exit(1) for a result marked failed.
        # Swallowing that into a generic 1 would work by accident; re-raising
        # keeps whatever code the caller chose.
        def boom():
            raise SystemExit(3)
        patched(boom)
        with pytest.raises(SystemExit) as caught:
            rhp._run()
        assert caught.value.code == 3
