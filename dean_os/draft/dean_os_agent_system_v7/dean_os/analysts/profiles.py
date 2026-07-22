from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import yaml

from .schemas import DomainProfile

_PACKAGE_PROFILES_DIR = Path(__file__).resolve().parent.parent / "config" / "domain_profiles"
_LEGACY_PROFILES_DIR = Path(__file__).resolve().parents[2] / "config" / "domain_profiles"


def _candidate_profile_dirs(extra_dirs: Iterable[str | Path] | None = None) -> list[Path]:
    """Return domain-profile directories in precedence order.

    Package-local profiles are canonical for portable DEAN-OS distributions.
    A repository-level legacy directory remains supported for compatibility.
    An explicit ``DEAN_DOMAIN_PROFILES_DIR`` override is checked first.
    """

    candidates: list[Path] = []
    env_dir = os.getenv("DEAN_DOMAIN_PROFILES_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    if extra_dirs:
        candidates.extend(Path(item).expanduser() for item in extra_dirs)
    candidates.extend([_PACKAGE_PROFILES_DIR, _LEGACY_PROFILES_DIR])

    resolved: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return resolved


def _load_profile(yaml_path: Path) -> DomainProfile:
    with yaml_path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    profile = DomainProfile.model_validate(data)
    if profile.domain_id != yaml_path.stem:
        raise ValueError(
            f"Domain profile filename/domain_id mismatch: {yaml_path.name} != {profile.domain_id}"
        )
    return profile


def _load_all_profiles(extra_dirs: Iterable[str | Path] | None = None) -> dict[str, DomainProfile]:
    profiles: dict[str, DomainProfile] = {}
    sources: dict[str, Path] = {}
    for profiles_dir in _candidate_profile_dirs(extra_dirs):
        if not profiles_dir.is_dir():
            continue
        for yaml_path in sorted(profiles_dir.glob("*.yaml")):
            profile = _load_profile(yaml_path)
            if profile.domain_id in profiles:
                # Earlier directories have higher precedence. Duplicate profiles
                # are tolerated only when their content is identical.
                if profiles[profile.domain_id] != profile:
                    raise ValueError(
                        "Conflicting domain profiles for "
                        f"{profile.domain_id}: {sources[profile.domain_id]} and {yaml_path}"
                    )
                continue
            profiles[profile.domain_id] = profile
            sources[profile.domain_id] = yaml_path
    return profiles


_PROFILES: dict[str, DomainProfile] = _load_all_profiles()


def reload_domain_profiles(
    extra_dirs: Iterable[str | Path] | None = None,
) -> dict[str, DomainProfile]:
    """Reload profiles for tests, plugin installation, or operator review."""

    global _PROFILES
    _PROFILES = _load_all_profiles(extra_dirs)
    return dict(_PROFILES)


def get_domain_profile(domain_id: str) -> DomainProfile:
    try:
        return _PROFILES[domain_id]
    except KeyError as exc:
        searched = ", ".join(str(path) for path in _candidate_profile_dirs())
        raise KeyError(
            f"Unknown domain profile: {domain_id}. Searched: {searched}"
        ) from exc


def list_domain_profiles() -> list[str]:
    return sorted(_PROFILES)


__all__ = [
    "get_domain_profile",
    "list_domain_profiles",
    "reload_domain_profiles",
]
