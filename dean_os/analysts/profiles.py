from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schemas import DomainProfile

_PROFILES_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "domain_profiles"


def _load_profile(yaml_path: Path) -> DomainProfile:
    with open(yaml_path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return DomainProfile.model_validate(data)


def _load_all_profiles() -> dict[str, DomainProfile]:
    profiles_dir = _PROFILES_DIR
    if not profiles_dir.is_dir():
        return {}
    profiles: dict[str, DomainProfile] = {}
    for yaml_path in sorted(profiles_dir.glob("*.yaml")):
        profile = _load_profile(yaml_path)
        profiles[profile.domain_id] = profile
    return profiles


_PROFILES: dict[str, DomainProfile] = _load_all_profiles()


def get_domain_profile(domain_id: str) -> DomainProfile:
    try:
        return _PROFILES[domain_id]
    except KeyError as exc:
        raise KeyError(f"Unknown domain profile: {domain_id}") from exc


def list_domain_profiles() -> list[str]:
    return sorted(_PROFILES)
