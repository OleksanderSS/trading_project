from __future__ import annotations

import hashlib
import json

from dean_os.current_cycle_journal import CurrentCycleJournalBuilder
from dean_os.hypothesis_learning_review import HYPOTHESIS_LEARNING_REVIEW_CONTRACT
from dean_os.world_model_event_learning import WORLD_MODEL_EVENT_LEARNING_CONTRACT
from dean_os.world_model_replay_review_gate import (
    WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT,
)


NOW = "2026-07-13T10:00:00+00:00"


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _binding(path):
    return {"path": str(path), "sha256": _sha(path)}


def test_current_cycle_import_is_complete_hash_bound_and_idempotent(tmp_path):
    news_path = tmp_path / "news.json"
    cycle_path = tmp_path / "cycle.json"
    world_path = tmp_path / "world.json"
    gate_path = tmp_path / "gate.json"
    learning_path = tmp_path / "learning.json"
    closure_path = tmp_path / "closure.json"
    journal_path = tmp_path / "journal.jsonl"
    _write(
        news_path,
        {
            "run_id": "news_1",
            "created_at": NOW,
            "producer_contract": "news_v1",
            "summary": {"accepted_news_record_count": 1},
            "candidates": [
                {
                    "candidate_sha256": "a" * 64,
                    "published_at": NOW,
                    "title": "Capex warning",
                    "summary": "Funding concerns remain.",
                    "source": "Source A",
                    "source_identity": "source_a",
                    "source_tier": "tier_2_strong_context",
                    "source_locator": "https://example.test/capex",
                    "evidence_type": "capex_cycle",
                }
            ],
        },
    )
    cycle = {
        "run_id": "cycle_1",
        "created_at": NOW,
        "contract": "dean_full_system_review_cycle_v1",
        "inputs": {
            "artifacts": {
                "news": {
                    "path": str(news_path),
                    "exists": True,
                    "sha256": _sha(news_path),
                }
            }
        },
        "summary": {"cycle_status": "ready", "evidence_count": 1, "hypothesis_count": 1},
    }
    _write(cycle_path, cycle)
    world = {
        "run_id": "world_1",
        "created_at": NOW,
        "contract": WORLD_MODEL_EVENT_LEARNING_CONTRACT,
        "cycle_binding_contract": "dean_full_system_cycle_world_model_bridge_v1",
        "upstream_bindings": {
            "full_system_review_cycle": {"run_id": "cycle_1", "sha256": _sha(cycle_path)}
        },
        "summary": {
            "domain_id": "semiconductors",
            "as_of": NOW,
            "downstream_hash_binding_ready": True,
        },
        "classified_events": [
            {
                "event_id": "e1",
                "evidence_id": "e1",
                "title": "Capex warning",
                "event_class": "capex_signal",
                "source_evidence_type": "capex_cycle",
                "source_id": "https://example.test/capex",
                "source_type": "news",
                "provenance": {"published_at": NOW, "source_tier": "tier_2_strong_context"},
            }
        ],
        "hypotheses": [
            {
                "hypothesis_id": "h1",
                "as_of": NOW,
                "hypothesis": "Capex grows",
                "trigger_evidence_ids": ["e1"],
                "horizons_to_check": [1, 5, 20],
            }
        ],
    }
    _write(world_path, world)
    gate = {
        "run_id": "gate_1",
        "created_at": NOW,
        "contract": WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT,
        "source_packet": {"run_id": "world_1", "sha256": _sha(world_path)},
        "operator_decision": {"reviewer": "reviewer"},
        "hypothesis_review": [
            {
                "hypothesis_id": "h1",
                "hypothesis": "Capex grows",
                "disposition": "reformulate",
                "rationale": "Wrong polarity",
                "proposed_hypothesis": "Capex expectations weaken",
                "source_assessment": "credible_context_source_but_trigger_polarity_conflicts_with_generated_claim",
            }
        ],
    }
    _write(gate_path, gate)
    learning = {
        "run_id": "learning_1",
        "created_at": NOW,
        "contract": HYPOTHESIS_LEARNING_REVIEW_CONTRACT,
        "inputs": {"review_gate": {**_binding(gate_path), "run_id": "gate_1"}},
        "learning_proposals": [
            {
                "proposal_id": "p1",
                "pattern_key": "b" * 64,
                "error_code": "trigger_polarity_mismatch",
                "current_case_ids": ["gate_1:h1"],
                "production_rule_update_performed": False,
            }
        ],
    }
    _write(learning_path, learning)
    closure = {
        "run_id": "closure_1",
        "created_at": NOW,
        "inputs": {
            "cycle": _binding(cycle_path),
            "world_model": _binding(world_path),
            "replay_review_gate": _binding(gate_path),
        },
        "summary": {
            "closure_status": "reformulation_required",
            "current_cycle_decision_state": "reformulation_required",
            "can_register_new_replay_tasks": False,
            "can_write_learning_memory": False,
            "can_trade": False,
        },
    }
    _write(closure_path, closure)
    builder = CurrentCycleJournalBuilder(tmp_path / "reports")
    kwargs = {
        "cycle_json": cycle_path,
        "world_model_json": world_path,
        "review_gate_json": gate_path,
        "closure_json": closure_path,
        "learning_review_json": learning_path,
        "journal_path": journal_path,
        "apply": True,
        "save": False,
    }

    first = builder.build(**kwargs)
    second = builder.build(**kwargs)

    assert first["summary"]["new_journal_event_count"] == 9
    assert first["summary"]["journal_chain_valid"] is True
    assert first["summary"]["news_event_count"] == 1
    assert first["summary"]["action_proposal_count"] == 1
    assert first["summary"]["actions_executed"] is False
    assert second["summary"]["new_journal_event_count"] == 0
    assert second["summary"]["idempotent_existing_event_count"] == 9

    reasoning_path = tmp_path / "reasoning.json"
    delta = {"module_name": "hypothesis_ledger", "module_version": "0.1.0"}
    proposal = {
        "proposal_id": "proposal_h1",
        "hypothesis_id": "h1",
        "proposal_type": "candidate_contradiction",
        "suggested_status": "weakened",
        "as_of": NOW,
        "requires_manual_review": True,
        "requires_outcome_evidence": True,
        "status_changed": False,
    }
    receipt = {
        "contract": "dean_analyst_reasoning_receipt_v1",
        "receipt_id": "reasoning_receipt_1",
        "as_of": NOW,
        "analysis_output_sha256": "c" * 64,
        "delta_sha256s": [_json_sha(delta)],
        "proposal_ids": ["proposal_h1"],
    }
    _write(
        reasoning_path,
        {
            "run_id": "reasoning_1",
            "created_at": NOW,
            "contract": "dean_analyst_core_reasoning_snapshot_v1",
            "status": "reasoning_snapshot_ready_for_review",
            "inputs": {
                "domain_id": "semiconductors",
                "as_of": NOW,
                "runtime_json": str(tmp_path / "not_present.json"),
                "runtime_sha256": "d" * 64,
            },
            "reasoning_receipt": receipt,
            "reasoning_delta_trail": [delta],
            "hypothesis_ledger": [],
            "hypothesis_review_proposals": [proposal],
            "review_checks": [],
        },
    )
    with_reasoning = builder.build(
        **{**kwargs, "reasoning_snapshot_json": reasoning_path}
    )
    with_reasoning_again = builder.build(
        **{**kwargs, "reasoning_snapshot_json": reasoning_path}
    )

    assert with_reasoning["summary"]["new_journal_event_count"] == 2
    assert with_reasoning["summary"]["reasoning_receipt_count"] == 1
    assert with_reasoning["summary"]["machine_review_proposal_count"] == 1
    assert with_reasoning_again["summary"]["new_journal_event_count"] == 0
