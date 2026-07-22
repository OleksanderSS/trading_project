from __future__ import annotations

from pathlib import Path
from typing import Any

PROFILES_DIR = Path(__file__).resolve().parent.parent / "config" / "domain_profiles"


def create_domain_profile(
    domain_id: str,
    display_name: str,
    *,
    description: str | None = None,
    sector_keywords: list[str] | None = None,
    ticker_universe: list[str] | None = None,
) -> Path:
    profiles_dir = PROFILES_DIR
    profiles_dir.mkdir(parents=True, exist_ok=True)
    path = profiles_dir / f"{domain_id}.yaml"
    if path.exists():
        raise FileExistsError(f"Profile already exists: {path}")
    content = _generate_profile_yaml(
        domain_id=domain_id,
        display_name=display_name,
        description=description or f"Analyzes the {display_name} sector.",
        sector_keywords=sector_keywords or [],
        ticker_universe=ticker_universe or [],
    )
    path.write_text(content, encoding="utf-8")
    return path


def _generate_profile_yaml(
    domain_id: str,
    display_name: str,
    description: str,
    sector_keywords: list[str],
    ticker_universe: list[str],
) -> str:
    kw_lines = "\n".join(f"  - {kw}" for kw in sector_keywords) if sector_keywords else "  # TODO"
    tu_lines = "\n".join(f"  - {t}" for t in ticker_universe) if ticker_universe else "  # TODO"

    return f"""domain_id: {domain_id}
display_name: {display_name}
sector_label: {domain_id}
macro_evidence_type: macro_context
description: >-
  {description}
horizon_days_default: 180
allowed_horizons: [30, 90, 180, 365]
core_questions:
  - What is the current supply/demand balance?
  - Are prices trending up or down?
  - What are the key risks and catalysts?
  - Are fundamentals supportive of investment?
required_evidence_types:
  - supply
  - demand
  - policy_or_geopolitical
  - market_confirmation
useful_evidence_types:
  - earnings_guidance
  - valuation_context
  - relative_strength
  - analyst_revision
sector_keywords:
{kw_lines}
ticker_universe_hint:
{tu_lines}
contradiction_rules:
  - Positive demand but supply disruption reduces confidence.
  - Strong price momentum without fundamental evidence is market evidence only.
direct_ticker_evidence_rules:
  - Company-specific earnings or guidance evidence.
  - Ticker-specific price or relative-strength confirmation.
blocked_if_missing:
  - No evidence timestamp or as_of.
  - No supply or demand evidence.
  - Contradicting evidence exists but is not addressed.
"""


def generate_registry_entry(domain_id: str) -> str:
    return f"""  {domain_id}_analyst:
    class_path: dean_os.agents.domain_analyst:DomainAnalystAgent
    branch: pipeline
    veto_level: none
    enabled: true
    error_behavior: skip
    timeout_seconds: 30
    domain_id: {domain_id}
    horizon_days: 180
    agent_role: standalone_domain_analysis
    decision_influence: false
    execution_group: {domain_id}_domain_analysis
    run_phases:
      - pre_trade
"""


__all__ = ["create_domain_profile", "generate_registry_entry"]
