"""Tests for get_confidence_calibrator() — the singleton/persistence wiring
around src/models/calibration/adaptive_confidence_calibrator.py.

That module existed but had zero callers anywhere in the active codebase
before this change (found during a duplication/dead-code audit); its
calibration methodology was separately verified sound by a same-session
correctness audit. This only tests the wiring (singleton behavior, load
when a file exists, graceful no-op when it doesn't) — the calibration math
itself already has its own tests elsewhere.
"""
import pytest

import src.models.calibration.adaptive_confidence_calibrator as calibrator_module
from src.models.calibration.adaptive_confidence_calibrator import (
    AdaptiveConfidenceCalibrator,
    get_confidence_calibrator,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """The module-level singleton must not leak state between tests."""
    calibrator_module._calibrator_instance = None
    yield
    calibrator_module._calibrator_instance = None


def test_returns_same_instance_on_repeated_calls():
    a = get_confidence_calibrator("nonexistent/path/does_not_matter.joblib")
    b = get_confidence_calibrator("nonexistent/path/does_not_matter.joblib")
    assert a is b


def test_uncalibrated_instance_behaves_as_identity_clip():
    """Before any outcomes are recorded, calibrate() must be a safe no-op
    (clip to [0.01, 0.99]) — this is what makes wiring it into the
    prediction path zero-risk even with no persisted state on disk."""
    calibrator = get_confidence_calibrator("nonexistent/path/does_not_matter.joblib")
    assert calibrator.calibrate(0.5) == pytest.approx(0.5, abs=1e-6)
    assert calibrator.calibrate(1.5) == pytest.approx(0.99, abs=1e-6)
    assert calibrator.calibrate(-0.5) == pytest.approx(0.01, abs=1e-6)


def test_loads_persisted_state_when_file_exists():
    """resolve_trusted_artifact_path (used by .load()) only accepts paths
    under data/models/trained_models/artifacts/checkpoints — pytest's
    tmp_path lives outside the project and is correctly rejected, so this
    test must use a path inside one of the trusted roots instead."""
    import os

    path = "data/models/_test_confidence_calibrator_wiring.joblib"
    os.makedirs("data/models", exist_ok=True)
    try:
        saved = AdaptiveConfidenceCalibrator()
        for _ in range(60):
            saved.update_with_outcome(raw_confidence=0.9, actual_outcome=1)
            saved.update_with_outcome(raw_confidence=0.5, actual_outcome=0)
        saved.save(path)

        loaded = get_confidence_calibrator(path)

        assert loaded.calibration_history  # non-empty: history was restored
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_missing_file_does_not_raise():
    # Must degrade gracefully — this is called from the hot prediction path.
    calibrator = get_confidence_calibrator("this/path/definitely/does/not/exist.joblib")
    assert isinstance(calibrator, AdaptiveConfidenceCalibrator)
