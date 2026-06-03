"""Security helpers for loading serialized model artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


TRUSTED_ARTIFACT_ROOTS = (
    "data",
    "models",
    "trained_models",
    "artifacts",
    "checkpoints",
)
MODEL_ARTIFACT_SUFFIXES = {
    ".joblib",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".keras",
    ".h5",
    ".zip",
    ".meta",
    ".npy",
}


def get_project_root() -> Path:
    """Return the repository root for project-relative artifact paths."""
    return Path(__file__).resolve().parents[2]


def _as_iterable(values: Iterable[str | Path] | None) -> list[str | Path]:
    if values is None:
        return list(TRUSTED_ARTIFACT_ROOTS)
    return list(values)


def _env_trusted_roots() -> list[Path]:
    raw = os.environ.get("TRADING_TRUSTED_ARTIFACT_ROOTS", "")
    if not raw.strip():
        return []
    return [Path(part) for part in raw.split(os.pathsep) if part.strip()]


def _resolve_root(root: str | Path, project_root: Path) -> Path:
    root_path = Path(root).expanduser()
    if not root_path.is_absolute():
        root_path = project_root / root_path
    return root_path.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_trusted_artifact_path(
    artifact_path: str | Path,
    *,
    allowed_roots: Iterable[str | Path] | None = None,
    allowed_suffixes: Iterable[str] | None = MODEL_ARTIFACT_SUFFIXES,
    must_exist: bool = False,
    project_root: str | Path | None = None,
) -> Path:
    """
    Resolve and validate a serialized artifact path before loading it.

    Pickle/joblib/torch/Keras artifacts can execute code during deserialization,
    so callers should load only from known artifact directories.
    """
    if artifact_path is None:
        raise ValueError("Artifact path cannot be None")

    artifact_text = str(artifact_path)
    if not artifact_text.strip():
        raise ValueError("Artifact path cannot be empty")
    if "\0" in artifact_text:
        raise ValueError("Null byte detected in artifact path")

    root = Path(project_root).resolve(strict=False) if project_root else get_project_root()
    candidate = Path(artifact_text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)

    suffixes = {suffix.lower() for suffix in allowed_suffixes or ()}
    if suffixes and resolved.suffix.lower() not in suffixes:
        raise ValueError(f"Unsupported artifact suffix: {resolved.suffix}")

    trusted_roots = [
        _resolve_root(entry, root)
        for entry in _as_iterable(allowed_roots)
    ]
    trusted_roots.extend(_resolve_root(entry, root) for entry in _env_trusted_roots())

    if not any(_is_within(resolved, trusted_root) for trusted_root in trusted_roots):
        roots = ", ".join(str(path) for path in trusted_roots)
        raise ValueError(f"Artifact path is outside trusted roots: {resolved} not in {roots}")

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Artifact file not found: {resolved}")

    return resolved


def is_trusted_artifact_path(
    artifact_path: str | Path,
    *,
    allowed_roots: Iterable[str | Path] | None = None,
    allowed_suffixes: Iterable[str] | None = MODEL_ARTIFACT_SUFFIXES,
    must_exist: bool = False,
    project_root: str | Path | None = None,
) -> bool:
    """Return True when an artifact path passes trust validation."""
    try:
        resolve_trusted_artifact_path(
            artifact_path,
            allowed_roots=allowed_roots,
            allowed_suffixes=allowed_suffixes,
            must_exist=must_exist,
            project_root=project_root,
        )
        return True
    except (OSError, ValueError):
        return False
