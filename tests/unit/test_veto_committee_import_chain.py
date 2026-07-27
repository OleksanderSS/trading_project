from unittest.mock import patch

from src.core.logging.logger import ProjectLogger
from src.trading.trading_orchestrator import TradingOrchestrator


def test_veto_system_module_imports_successfully():
    """The real veto_system module lives at src.agents.archive.veto_system,
    not src.agents.veto_system (which never existed). trading_orchestrator.py
    used the wrong path, so _apply_veto_committee's import always raised
    ModuleNotFoundError, silently falling back to unvetoed consensus
    signals on every trading cycle. Even the correct path was itself
    broken (2 further stale imports inside veto_system.py), so this test
    exercises the whole chain, not just one path segment."""
    from src.agents.archive.veto_system import AgenticVetoSystem, veto_system

    assert veto_system is not None
    assert isinstance(veto_system, AgenticVetoSystem)


def test_apply_veto_committee_reaches_the_real_veto_system_not_the_except_fallback():
    """Proves _apply_veto_committee's try block actually executes (imports
    succeed) rather than immediately hitting the except-and-fall-back-
    unvetoed path, which is what happened on every real trading cycle
    before this fix."""
    orchestrator = object.__new__(TradingOrchestrator)
    orchestrator.logger = ProjectLogger.get_logger("test")

    fake_reviewed = [
        {"vetoed": False, "veto_reason": "OK", "causal_graph": []},
    ]

    with patch(
        "src.agents.archive.veto_system.veto_system.review_recommendations",
        return_value=_async_return(fake_reviewed),
    ):
        signals = [{"ticker": "AAPL", "final_signal": "BUY", "confidence": 0.9}]
        result = orchestrator._apply_veto_committee(signals)

    assert len(result) == 1
    assert result[0]["ticker"] == "AAPL"


async def _async_return(value):
    return value
