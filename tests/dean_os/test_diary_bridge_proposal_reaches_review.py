"""The last link of the feedback loop was a file nobody had to read.

`DiaryBridgeAgent` inspects paper-trade outcomes and proposes bridging them
into the pipeline's experience diary. It deliberately does not write there
itself -- it says so in its own report, and that boundary is the one every
agent in this system honours.

The proposal it produced went into `reports/dean_os/diary_bridge/latest.json`
and nowhere else. `agent_lab` queues proposals when it is given a store, but
DiaryBridgeAgent is not in the registry, so it never runs through agent_lab
and never got one. Paper trading accumulated outcomes, the agent noticed, and
the notice evaporated with the process.

Gemini's §15.1 called for the agent to INSERT results into the diary directly.
That removes the boundary instead of closing the gap. The operation queue is
where a proposal waits for a person, which is what it is for.
"""

from __future__ import annotations

import asyncio

import pytest

import run_agent_diary_bridge as cli
from dean_os.operation_queue import OperationQueue
from dean_os.schemas import MarketContext, PipelineActionProposal


def _args(tmp_path, **overrides):
    parser = cli.build_parser()
    args = parser.parse_args([])
    args.output_dir = str(tmp_path / "reports")
    args.output = None
    args.operations_store = str(tmp_path / "queue.sqlite")
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_the_cli_offers_somewhere_for_the_proposal_to_go():
    """It had no such option, which is why the proposal went nowhere."""
    args = cli.build_parser().parse_args([])
    assert hasattr(args, "operations_store")
    assert args.operations_store, "a default is needed; nobody passes flags to a cron"


def test_a_proposal_is_queued_for_review(tmp_path, monkeypatch):
    proposal = PipelineActionProposal(
        agent_name="diary_bridge",
        action_type="report",
        target="experience_diary",
        reason="bridge 3 paper outcomes into the diary",
    )

    async def _fake_run(self, context: MarketContext):
        context.action_proposals.append(proposal)
        context.metadata["diary_bridge"] = {"status": "bridge_proposal_ready"}
        from dean_os.schemas import PipelineReport

        return PipelineReport(
            agent_name="diary_bridge", agent_version="test", verdict="caution",
            confidence=0.5, data_quality_score=0.5, signal_strength=0.0,
        )

    monkeypatch.setattr(cli.DiaryBridgeAgent, "run", _fake_run)
    args = _args(tmp_path)

    payload = asyncio.run(cli.main_async(args))

    assert payload["queued_proposal_ids"], "the proposal did not reach the queue"
    stored = OperationQueue(args.operations_store).get_proposal(
        payload["queued_proposal_ids"][0]
    )
    assert stored is not None


def test_nothing_is_queued_when_there_is_nothing_to_propose(tmp_path, monkeypatch):
    async def _fake_run(self, context: MarketContext):
        from dean_os.schemas import PipelineReport

        return PipelineReport(
            agent_name="diary_bridge", agent_version="test", verdict="clear",
            confidence=0.5, data_quality_score=0.5, signal_strength=0.0,
        )

    monkeypatch.setattr(cli.DiaryBridgeAgent, "run", _fake_run)
    payload = asyncio.run(cli.main_async(_args(tmp_path)))
    assert payload["queued_proposal_ids"] == []


def test_queueing_can_be_turned_off(tmp_path, monkeypatch):
    """An inspection run should be able to inspect without proposing anything."""
    proposal = PipelineActionProposal(
        agent_name="diary_bridge", action_type="report",
        target="experience_diary", reason="x",
    )

    async def _fake_run(self, context: MarketContext):
        context.action_proposals.append(proposal)
        from dean_os.schemas import PipelineReport

        return PipelineReport(
            agent_name="diary_bridge", agent_version="test", verdict="caution",
            confidence=0.5, data_quality_score=0.5, signal_strength=0.0,
        )

    monkeypatch.setattr(cli.DiaryBridgeAgent, "run", _fake_run)
    payload = asyncio.run(cli.main_async(_args(tmp_path, operations_store="")))
    assert payload["queued_proposal_ids"] == []
    assert payload["operations_store"] is None


def test_the_agent_still_refuses_to_write_the_diary_itself():
    """The boundary is the point, not an oversight to be fixed."""
    import inspect

    from dean_os.agents.diary_bridge import DiaryBridgeAgent

    source = inspect.getsource(DiaryBridgeAgent)
    assert "does not write to the pipeline experience diary" in source
