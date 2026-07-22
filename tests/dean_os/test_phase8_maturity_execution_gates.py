from __future__ import annotations

import json

import pytest

from dean_os.execution.execution_gateway import (
    ExecutionGateway,
    OrderDecision,
    OrderRequest,
)
from dean_os.execution.maturity_gates import (
    PAPER_GATE_CHECKS,
    REPLAY_GATE_CHECKS,
    SHADOW_GATE_CHECKS,
    SUPERVISED_LIVE_GATE_CHECKS,
    run_promotion_pipeline,
    verify_gate_receipt,
)
from dean_os.risk.risk_engine import (
    KillSwitchTrigger,
    PortfolioState,
    RiskEngine,
)
from dean_os.strategies.strategy_playbook import (
    MaturityLevel,
    PromotionPolicy,
    StrategyDescription,
    StrategyPlaybook,
)
from dean_os.strategies.strategy_registry import StrategyRegistry


def _evidence(tmp_path, name: str) -> dict[str, str]:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"artifact": name, "status": "verified"}), encoding="utf-8")
    return {name: str(path)}


def _all(checks: list[str]) -> dict[str, bool]:
    return {check: True for check in checks}


def _approved_chain(tmp_path, strategy_id: str = "strategy_1"):
    replay = run_promotion_pipeline(
        strategy_id,
        "replay",
        _all(REPLAY_GATE_CHECKS),
        approver="operator_a",
        current_level="research",
        evidence_artifacts=_evidence(tmp_path, "replay_packet"),
    )
    paper = run_promotion_pipeline(
        strategy_id,
        "paper",
        _all(PAPER_GATE_CHECKS),
        approver="operator_a",
        current_level="replay",
        previous_receipt=replay["receipt"],
        evidence_artifacts=_evidence(tmp_path, "paper_packet"),
    )
    shadow = run_promotion_pipeline(
        strategy_id,
        "shadow",
        _all(SHADOW_GATE_CHECKS),
        approver="operator_a",
        current_level="paper",
        previous_receipt=paper["receipt"],
        evidence_artifacts=_evidence(tmp_path, "shadow_packet"),
    )
    return replay, paper, shadow


def test_replay_receipt_requires_evidence_and_operator(tmp_path):
    review = run_promotion_pipeline(
        "strategy_1",
        "replay",
        _all(REPLAY_GATE_CHECKS),
        evidence_artifacts=_evidence(tmp_path, "replay_review"),
    )
    assert review["result"]["decision"] == "review_required"

    approved = run_promotion_pipeline(
        "strategy_1",
        "replay",
        _all(REPLAY_GATE_CHECKS),
        approver="operator_a",
        evidence_artifacts=_evidence(tmp_path, "replay_approved"),
    )
    ok, failures = verify_gate_receipt(
        approved["receipt"],
        expected_strategy_id="strategy_1",
        expected_target_gate="replay",
    )
    assert approved["result"]["decision"] == "approved"
    assert ok is True
    assert failures == []


def test_paper_cannot_use_bare_previous_gate_boolean(tmp_path):
    result = run_promotion_pipeline(
        "strategy_1",
        "paper",
        _all(PAPER_GATE_CHECKS),
        approver="operator_a",
        current_level="replay",
        evidence_artifacts=_evidence(tmp_path, "paper_without_receipt"),
    )
    assert result["result"]["decision"] == "blocked"
    assert "maturity_receipt_missing" in result["result"]["checks_failed"]


def test_gate_chain_rejects_jump_and_tampering(tmp_path):
    replay, paper, shadow = _approved_chain(tmp_path)
    assert [item["result"]["decision"] for item in (replay, paper, shadow)] == [
        "approved", "approved", "approved"
    ]

    jump = run_promotion_pipeline(
        "strategy_1",
        "shadow",
        _all(SHADOW_GATE_CHECKS),
        approver="operator_a",
        current_level="replay",
        previous_receipt=replay["receipt"],
        evidence_artifacts=_evidence(tmp_path, "jump"),
    )
    assert jump["result"]["decision"] == "blocked"
    assert any(item.startswith("non_sequential_transition") for item in jump["result"]["checks_failed"])

    tampered = dict(paper["receipt"])
    tampered["strategy_id"] = "attacker_strategy"
    invalid = run_promotion_pipeline(
        "strategy_1",
        "shadow",
        _all(SHADOW_GATE_CHECKS),
        approver="operator_a",
        previous_receipt=tampered,
        evidence_artifacts=_evidence(tmp_path, "tampered"),
    )
    assert invalid["result"]["decision"] == "blocked"
    assert "maturity_receipt_hash_invalid" in invalid["result"]["checks_failed"]


def test_receipt_is_invalid_after_bound_evidence_changes(tmp_path):
    evidence = _evidence(tmp_path, "mutable_packet")
    result = run_promotion_pipeline(
        "strategy_1", "replay", _all(REPLAY_GATE_CHECKS),
        approver="operator_a", evidence_artifacts=evidence,
    )
    evidence_path = tmp_path / "mutable_packet.json"
    evidence_path.write_text("changed after approval", encoding="utf-8")
    ok, failures = verify_gate_receipt(result["receipt"])
    assert ok is False
    assert "maturity_receipt_evidence_hash_invalid:mutable_packet" in failures


def test_supervised_live_remains_hard_disabled(tmp_path):
    _, _, shadow = _approved_chain(tmp_path)
    live = run_promotion_pipeline(
        "strategy_1",
        "supervised_live",
        _all(SUPERVISED_LIVE_GATE_CHECKS),
        approver="operator_a",
        previous_receipt=shadow["receipt"],
        evidence_artifacts=_evidence(tmp_path, "live_packet"),
    )
    assert live["result"]["decision"] == "blocked"
    assert "supervised_live_disabled_by_system_policy" in live["result"]["checks_failed"]


def test_gateway_requires_receipt_and_portfolio_state(tmp_path):
    _, paper, _ = _approved_chain(tmp_path)
    gateway = ExecutionGateway(allowed_assets={"NVDA"})
    base = dict(
        strategy_id="strategy_1",
        asset="NVDA",
        direction="buy",
        size_pct=0.02,
        decision_lineage_id="lineage_1",
        mode="paper",
    )
    missing_receipt = gateway.submit(OrderRequest(**base), PortfolioState())
    assert missing_receipt.decision is OrderDecision.BLOCKED_HARD
    assert "maturity_receipt_missing" in missing_receipt.hard_blocks_triggered

    missing_state = gateway.submit(OrderRequest(**base, maturity_receipt=paper["receipt"]))
    assert "portfolio_state_missing" in missing_state.hard_blocks_triggered

    approved = gateway.submit(
        OrderRequest(**base, maturity_receipt=paper["receipt"]),
        PortfolioState(),
    )
    assert approved.decision is OrderDecision.APPROVED_SIMULATED
    assert approved.maturity_receipt_sha256 == paper["receipt"]["receipt_sha256"]


def test_gateway_enforces_size_and_market_quality(tmp_path):
    _, paper, _ = _approved_chain(tmp_path)
    gateway = ExecutionGateway(allowed_assets={"NVDA"})
    order = OrderRequest(
        strategy_id="strategy_1", asset="NVDA", size_pct=0.06,
        decision_lineage_id="lineage_1", mode="paper", maturity_receipt=paper["receipt"],
    )
    assert "invalid_position_size" in gateway.submit(order, PortfolioState()).hard_blocks_triggered

    risk_order = OrderRequest(
        strategy_id="strategy_1", asset="NVDA", size_pct=0.02,
        decision_lineage_id="lineage_2", mode="paper", maturity_receipt=paper["receipt"],
    )
    rejected = gateway.submit(
        risk_order,
        PortfolioState(estimated_slippage_bps=70, asset_liquidity_usd=100_000),
    )
    assert rejected.decision is OrderDecision.REJECTED
    assert any(item.startswith("slippage_limit_breached") for item in rejected.risk_violations)
    assert any(item.startswith("liquidity_minimum_breached") for item in rejected.risk_violations)


def test_gateway_blocks_llm_direct_and_all_live_requests(tmp_path):
    _, paper, _ = _approved_chain(tmp_path)
    gateway = ExecutionGateway(allowed_assets={"NVDA"})
    llm_order = OrderRequest(
        strategy_id="strategy_1", asset="NVDA", source="llm_direct",
        decision_lineage_id="lineage_1", maturity_receipt=paper["receipt"], mode="paper",
    )
    assert "llm_direct_order" in gateway.submit(llm_order, PortfolioState()).hard_blocks_triggered

    live_order = OrderRequest(
        strategy_id="strategy_1", asset="NVDA", decision_lineage_id="lineage_2",
        maturity_receipt=paper["receipt"], mode="supervised_live",
    )
    assert "supervised_live_disabled" in gateway.submit(live_order, PortfolioState()).hard_blocks_triggered


def test_registry_accepts_only_matching_approved_receipt(tmp_path):
    _, paper, _ = _approved_chain(tmp_path)
    playbook = StrategyPlaybook(
        strategy_id="strategy_1",
        description=StrategyDescription(name="Test", thesis="Test thesis"),
        promotion_policy=PromotionPolicy(
            current_maturity_level=MaturityLevel.REPLAY,
            next_allowed_level=MaturityLevel.PAPER,
            approval_required=True,
        ),
    )
    registry = StrategyRegistry(tmp_path / "registry")
    registry.register(playbook)
    blocked = registry.request_promotion("strategy_1", MaturityLevel.PAPER, approver="operator_a")
    assert blocked["decision"]["status"] == "blocked"

    approved = registry.request_promotion(
        "strategy_1", MaturityLevel.PAPER,
        approver="operator_a", gate_receipt=paper["receipt"],
    )
    assert approved["decision"]["status"] == "approved"
    assert approved["decision"]["gate_receipt_sha256"] == paper["receipt"]["receipt_sha256"]


def test_kill_switch_requires_identified_operator():
    engine = RiskEngine()
    engine.kill_switch.activate(KillSwitchTrigger.OPERATOR_MANUAL_KILL)
    with pytest.raises(ValueError):
        engine.kill_switch.deactivate_manual("  ")
    assert engine.kill_switch.is_active is True
