from __future__ import annotations

from dean_os.schemas import PipelineReport
from dean_os.world_state import WorldStateBuilder


def _report(agent_name: str, verdict: str = "neutral", confidence: float = 0.6) -> PipelineReport:
    return PipelineReport(
        agent_name=agent_name,
        agent_version="0.1.0",
        verdict=verdict,
        confidence=confidence,
        data_quality_score=0.7,
    )


def test_geopolitics_and_liquidity_credit_get_sector_entries():
    # geopolitics_analyst and liquidity_credit_analyst are real, enabled
    # DomainAnalystAgent entries in agent_registry.yaml -- they used to be
    # entirely absent from DOMAIN_SECTOR_MAP/_sector_id_from_agent, so their
    # stance/confidence/thesis were silently dropped from the Stage 7 world
    # state snapshot instead of appearing as their own sector.
    reports = [
        _report("geopolitics_analyst", verdict="bearish"),
        _report("liquidity_credit_analyst", verdict="bullish"),
    ]
    snapshot = WorldStateBuilder().build(reports, as_of="2026-07-24T00:00:00+00:00")

    assert "geopolitics" in snapshot.sectors
    assert snapshot.sectors["geopolitics"].stance == "bearish"
    assert "liquidity_credit" in snapshot.sectors
    assert snapshot.sectors["liquidity_credit"].stance == "bullish"


def test_macro_analyst_never_creates_a_sector_entry():
    # macro_analyst deliberately returns early into global_state.macro_stance
    # instead of a per-sector entry -- confirm it never leaks a "macro_policy"
    # (or any other) sector key, now that the dead DOMAIN_SECTOR_MAP/
    # _sector_id_from_agent entries for it have been removed.
    reports = [_report("macro_analyst", verdict="bearish")]
    snapshot = WorldStateBuilder().build(reports, as_of="2026-07-24T00:00:00+00:00")

    assert snapshot.sectors == {}
    assert snapshot.global_state.macro_stance == "bearish"
