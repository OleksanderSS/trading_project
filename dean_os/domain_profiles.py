"""Compatibility access to the canonical YAML-backed domain profiles.

Domain definitions already live in ``config/domain_profiles/*.yaml`` and are
loaded by :mod:`dean_os.analysts.profiles`.  Keep one source of truth: this
module only exposes the short names used by the domain-orchestrator scaffold
and a small, explicitly opt-in profile-agent routing map.
"""
from __future__ import annotations

from dean_os.analysts.profiles import (
    get_domain_profile,
    list_domain_profiles,
)
from dean_os.analysts.schemas import DomainProfile


PROFILE_AGENTS_BY_DOMAIN: dict[str, tuple[str, ...]] = {
    "semiconductor_ai_infrastructure": ("value_screening", "sector_cycle"),
    "agriculture": ("sector_cycle",),
    "energy": ("sector_cycle", "value_screening"),
    "geopolitics": ("sector_cycle",),
    "liquidity_credit": ("value_screening",),
    "logistics": ("sector_cycle",),
    "macro_policy": ("macro_policy",),
    "real_estate": ("value_screening",),
}


def get_profile(domain_id: str) -> DomainProfile:
    """Return the canonical, validated YAML-backed domain profile."""
    return get_domain_profile(domain_id)


def list_domain_ids() -> list[str]:
    """Return all canonical domain ids."""
    return list_domain_profiles()


def get_profile_agents(domain_id: str) -> tuple[str, ...]:
    """Return optional profile agents; execution still requires explicit opt-in."""
    return PROFILE_AGENTS_BY_DOMAIN.get(domain_id, ())
