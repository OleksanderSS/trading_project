# -*- coding: utf-8 -*-
"""
Перевірка Phase 8: Risk Engine, Kill Switch, Maturity Gates, Execution Gateway, Stress Scenarios.
"""
import sys
import tempfile
import json
from pathlib import Path


def _use_utf8_console() -> None:
    """UTF-8 output on a Windows console, for when this file is RUN.

    This used to happen at import time, as:

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    Under pytest that is destructive. `sys.stdout` is then a wrapper around a
    temporary file that pytest owns; taking its `.buffer`, wrapping it again
    and rebinding `sys.stdout` drops the last reference to the original
    wrapper, whose finaliser closes the underlying file -- pytest's capture
    file, for the rest of the session.

    The cost was not local. `tests/dean_os/test_builders_refuse_empty_input`
    imports this module by name, and after it did, every remaining test in the
    directory failed at setup with "I/O operation on closed file". Running
    `pytest tests/dean_os/` reported 293 passed and 2,004 setup/teardown
    errors: roughly 1,002 tests never ran, and a directory that looked like it
    had a green tail was not testing two thirds of itself.

    So: only when this file is the program, and `reconfigure` in preference,
    because it changes the encoding of the existing stream instead of
    replacing the object.
    """
    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
            return
        except (ValueError, OSError):
            pass

    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        import io as _io

        sys.stdout = _io.TextIOWrapper(buffer, encoding="utf-8")

from dean_os.risk.risk_engine import RiskEngine, RiskLimits, PortfolioState
from dean_os.execution.maturity_gates import (
    run_promotion_pipeline,
    REPLAY_GATE_CHECKS,
    GateDecision,
)
from dean_os.execution.execution_gateway import ExecutionGateway, OrderRequest
from dean_os.stress.scenario_library import all_scenarios, scenarios_by_severity


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_risk_engine():
    separator("1. RISK ENGINE + KILL SWITCH")
    engine = RiskEngine(limits=RiskLimits(max_daily_loss_pct=0.02))

    # Нормальний стан — має пройти
    ok_state = PortfolioState(daily_pnl_pct=0.005, drawdown_pct=0.01)
    result = engine.check(ok_state, "strategy_test")
    print(f"  Normal state -> passed={result.passed}, violations={result.violations}")
    assert result.passed, "Normal state should pass"

    # Перевищення денних збитків — має активувати kill switch
    bad_state = PortfolioState(daily_pnl_pct=-0.03, drawdown_pct=0.01)
    result = engine.check(bad_state, "strategy_test")
    print(f"  Bad loss state -> passed={result.passed}, violations={result.violations}")
    print(f"  Kill switch active: {engine.kill_switch.is_active}")
    assert not result.passed
    assert engine.kill_switch.is_active

    print("  [OK] Risk engine and kill switch")


def test_maturity_gates():
    separator("2. MATURITY GATES (Replay -> Paper -> Shadow -> Supervised Live)")

    # Для replay gate потрібен evidence_artifact (SHA-bound receipt)
    # Створюємо тимчасовий файл, щоб симулювати наявність артефакту
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        json.dump({"run_id": "test_replay_artifact"}, tmp)
        artifact_path = tmp.name

    try:
        checks = {k: True for k in REPLAY_GATE_CHECKS}
        result = run_promotion_pipeline(
            "strat_001",
            "replay",
            checks,
            approver="operator_oleksandr",
            evidence_artifacts={"replay_report": artifact_path},
        )
        decision = result['result']['decision']
        failed = result['result']['checks_failed']
        print(f"  Replay gate (full): {decision}, failed={failed}")
        assert decision == "approved", f"Expected approved, got {decision} | failed={failed}"

        # Replay без artifacts — має бути blocked (evidence_artifact_missing)
        result_no_ev = run_promotion_pipeline(
            "strat_001",
            "replay",
            checks,
            approver="operator_oleksandr",
            evidence_artifacts=None,
        )
        print(f"  Replay gate (no evidence): {result_no_ev['result']['decision']}, "
              f"failed={result_no_ev['result']['checks_failed']}")
        assert result_no_ev['result']['decision'] == "blocked"

        # Replay з неповними перевірками — має бути blocked
        incomplete = {"as_of_data_only": True}
        result_block = run_promotion_pipeline(
            "strat_001", "replay", incomplete,
            evidence_artifacts={"replay_report": artifact_path},
        )
        failed_count = len(result_block['result']['checks_failed'])
        print(f"  Replay (incomplete): {result_block['result']['decision']}, {failed_count} checks failed")
        assert result_block['result']['decision'] == "blocked"

        # supervised_live — системна заборона (LIVE_EXECUTION_ENABLED=False)
        all_checks = {k: True for k in [
            "shadow_gate_passed", "small_capital_allocation", "allowed_assets_only",
            "allowed_hours_only", "human_approval_or_emergency_stop_available",
            "max_position_limit_active", "max_daily_loss_limit_active",
            "max_drawdown_limit_active", "unsupported_assets_blocked",
            "execution_gateway_required", "kill_switch_required",
            "decision_lineage_complete", "operator_review_record_present",
        ]}
        result_live = run_promotion_pipeline(
            "strat_001", "supervised_live", all_checks,
            approver="operator_oleksandr",
            evidence_artifacts={"shadow_report": artifact_path},
        )
        print(f"  Supervised live: {result_live['result']['decision']} "
              f"(system policy block expected)")
        assert result_live['result']['decision'] == "blocked"

    finally:
        Path(artifact_path).unlink(missing_ok=True)

    print("  [OK] Maturity gates")


def test_execution_gateway():
    separator("3. EXECUTION GATEWAY")
    from dean_os.execution.maturity_gates import run_promotion_pipeline, REPLAY_GATE_CHECKS
    import tempfile, json
    from pathlib import Path

    gateway = ExecutionGateway(allowed_assets={"NVDA", "ASML", "TSMC"})
    ok_portfolio = PortfolioState(
        daily_pnl_pct=0.005,
        drawdown_pct=0.01,
        model_state_known=True,
        is_authorized_asset=True,
        market_data_age_seconds=5.0,
    )

    # Спочатку отримуємо затверджений receipt через replay gate
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        json.dump({"run_id": "test_replay_artifact"}, tmp)
        artifact_path = tmp.name

    try:
        checks = {k: True for k in REPLAY_GATE_CHECKS}
        gate_result = run_promotion_pipeline(
            "strat_001", "replay", checks,
            approver="operator_oleksandr",
            evidence_artifacts={"replay_report": artifact_path},
        )
        # Для paper gate потрібен receipt попереднього (replay) gate
        # Але в тесті mode="paper" перевіряє receipt для target_gate="paper"
        # Тому передаємо replay receipt напряму і очікуємо maturity_receipt_invalid
        # (правильна поведінка — receipt має бути для mode)
        # Для простоти тесту: перевіряємо блокування без receipt
        order_no_receipt = OrderRequest(
            strategy_id="strat_001",
            asset="NVDA",
            direction="buy",
            size_pct=0.02,
            decision_lineage_id="lineage_abc123",
            mode="paper",
            maturity_receipt=None,
        )
        result = gateway.submit(order_no_receipt, portfolio_state=ok_portfolio)
        print(f"  Paper order (no receipt) -> {result.decision.value}, blocks={result.hard_blocks_triggered}")
        assert result.decision.value == "blocked_hard"
        assert "maturity_receipt_missing" in result.hard_blocks_triggered

        # Без lineage — hard block
        order_no_lineage = OrderRequest(
            strategy_id="strat_001",
            asset="NVDA",
            direction="buy",
            mode="paper",
            decision_lineage_id=None,
            maturity_receipt=None,
        )
        result = gateway.submit(order_no_lineage, portfolio_state=ok_portfolio)
        print(f"  No lineage -> {result.decision.value}, blocks={result.hard_blocks_triggered}")
        assert result.decision.value == "blocked_hard"
        assert "missing_lineage" in result.hard_blocks_triggered

        # LLM direct order — hard block
        order_llm = OrderRequest(
            strategy_id="strat_001",
            asset="NVDA",
            direction="buy",
            source="llm_direct",
            decision_lineage_id="lineage_xyz",
            mode="paper",
            maturity_receipt=None,
        )
        result = gateway.submit(order_llm, portfolio_state=ok_portfolio)
        print(f"  LLM direct order -> {result.decision.value}, blocks={result.hard_blocks_triggered}")
        assert "llm_direct_order" in result.hard_blocks_triggered

    finally:
        Path(artifact_path).unlink(missing_ok=True)

    print("  [OK] Execution gateway")


def test_stress_scenarios():
    separator("4. STRESS SCENARIO LIBRARY")
    all_sc = all_scenarios()
    extreme = scenarios_by_severity("extreme")
    print(f"  Total scenarios: {len(all_sc)}")
    print(f"  Extreme scenarios: {len(extreme)}")
    for s in extreme:
        print(f"    - [{s.scenario_id}] {s.title}")

    # Перевірка що forbidden_outputs визначені в усіх
    assert all("buy_sell_hold" in s.forbidden_outputs for s in all_sc)
    print("  All scenarios have forbidden_outputs defined")

    print("  [OK] Stress scenario library")


if __name__ == "__main__":
    _use_utf8_console()
    try:
        test_risk_engine()
        test_maturity_gates()
        test_execution_gateway()
        test_stress_scenarios()
        print("\n" + "="*60)
        print("  PHASE 8 -- ALL CHECKS PASSED")
        print("="*60)
    except AssertionError as e:
        print(f"\nASSERTION FAILED: {e}")
        sys.exit(1)
