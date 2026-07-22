from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.world_state_outcomes import (
    LearningPromotionGate,
    OutcomeCalibrationService,
    OutcomeReviewDecisionBuilder,
    OutcomeSnapshotBuilder,
    SQLiteOutcomeStore,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.world_state_store import SQLiteWorldStateStore


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input JSON must contain an object")
    return payload


def _print(payload: Any) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate and review fixed-horizon DEAN-OS World-State outcomes."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--world-state-store", required=True)
    evaluate.add_argument("--outcome-store", required=True)
    evaluate.add_argument("--snapshot-id", required=True)
    evaluate.add_argument("--input-json", required=True)

    list_cmd = sub.add_parser("list")
    list_cmd.add_argument("--outcome-store", required=True)
    list_cmd.add_argument("--domain")
    list_cmd.add_argument("--horizon", type=int)
    list_cmd.add_argument("--world-state-snapshot-id")
    list_cmd.add_argument("--limit", type=int, default=100)

    review = sub.add_parser("review")
    review.add_argument("--outcome-store", required=True)
    review.add_argument("--outcome-snapshot-id", required=True)
    review.add_argument("--decision", choices=["approved", "rejected"], required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--rationale", required=True)
    review.add_argument("--decided-at")

    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--outcome-store", required=True)
    calibrate.add_argument("--domain", required=True)
    calibrate.add_argument("--horizon", required=True, type=int)
    calibrate.add_argument("--min-approved-samples", type=int, default=20)

    promotion = sub.add_parser("promotion-gate")
    promotion.add_argument("--outcome-store", required=True)
    promotion.add_argument("--domain", required=True)
    promotion.add_argument("--horizon", required=True, type=int)
    promotion.add_argument("--min-approved-samples", type=int, default=20)
    promotion.add_argument("--reviewer-decision", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "evaluate":
        world_store = SQLiteWorldStateStore(args.world_state_store)
        outcome_store = SQLiteOutcomeStore(args.outcome_store)
        world_state = world_store.get(args.snapshot_id)
        if world_state is None:
            raise SystemExit(f"Unknown world-state snapshot: {args.snapshot_id}")
        payload = _load_json(args.input_json)
        outcome = OutcomeSnapshotBuilder().build(
            world_state=world_state,
            horizon_days=int(payload["horizon_days"]),
            evaluation_as_of=str(payload["evaluation_as_of"]),
            evidence=list(payload.get("evidence", [])),
            scenario_resolutions=list(payload.get("scenario_resolutions", [])),
            hypothesis_resolutions=list(payload.get("hypothesis_resolutions", [])),
            evidence_gaps=list(payload.get("evidence_gaps", [])),
        )
        append_result = outcome_store.append(outcome)
        _print(
            {
                "append_result": append_result.model_dump(mode="json"),
                "outcome": outcome.model_dump(mode="json"),
            }
        )
        return

    outcome_store = SQLiteOutcomeStore(args.outcome_store)
    if args.command == "list":
        _print(
            [
                item.model_dump(mode="json")
                for item in outcome_store.list_outcomes(
                    domain_id=args.domain,
                    horizon_days=args.horizon,
                    world_state_snapshot_id=args.world_state_snapshot_id,
                    limit=args.limit,
                )
            ]
        )
        return

    if args.command == "review":
        review = OutcomeReviewDecisionBuilder().build(
            outcome_snapshot_id=args.outcome_snapshot_id,
            decision=args.decision,
            reviewer=args.reviewer,
            rationale=args.rationale,
            decided_at=args.decided_at,
        )
        _print(outcome_store.append_review(review))
        return

    proposal = OutcomeCalibrationService().propose(
        outcome_store=outcome_store,
        domain_id=args.domain,
        horizon_days=args.horizon,
        min_approved_samples=args.min_approved_samples,
    )
    if args.command == "calibrate":
        _print(proposal)
        return

    if args.command == "promotion-gate":
        _print(
            LearningPromotionGate().evaluate(
                calibration_proposal=proposal,
                reviewer_decision=args.reviewer_decision,
            )
        )
        return


if __name__ == "__main__":
    main()
