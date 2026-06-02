from pathlib import Path

import pytest

from src.core.error_handling.error_handler import ModelLoadingError
from src.models.loader import ModelLoaderStrategy
from src.utils.artifact_security import resolve_trusted_artifact_path


def test_resolve_trusted_artifact_allows_project_data_path():
    path = resolve_trusted_artifact_path("data/trained_models/model.joblib")

    assert path.name == "model.joblib"
    assert "data" in path.parts


def test_resolve_trusted_artifact_rejects_outside_project(tmp_path):
    outside = tmp_path / "model.joblib"

    with pytest.raises(ValueError, match="outside trusted roots"):
        resolve_trusted_artifact_path(outside)


def test_resolve_trusted_artifact_allows_explicit_env_root(tmp_path, monkeypatch):
    outside = tmp_path / "model.joblib"
    monkeypatch.setenv("TRADING_TRUSTED_ARTIFACT_ROOTS", str(tmp_path))

    assert resolve_trusted_artifact_path(outside) == outside.resolve()


def test_resolve_trusted_artifact_rejects_unsupported_suffix():
    with pytest.raises(ValueError, match="Unsupported artifact suffix"):
        resolve_trusted_artifact_path("data/trained_models/model.txt")


def test_model_loader_rejects_untrusted_path_before_deserialization(tmp_path, monkeypatch):
    outside = tmp_path / "model.joblib"
    outside.write_bytes(b"not a real joblib artifact")
    called = False

    def fake_load(path):
        nonlocal called
        called = True
        raise AssertionError(f"joblib.load should not be called for {path}")

    monkeypatch.setattr("src.models.loader.joblib.load", fake_load)

    with pytest.raises(ModelLoadingError, match="Unsafe or missing model artifact path"):
        ModelLoaderStrategy().load_path(str(outside), {})

    assert called is False
