from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.recommendation_memory import RecommendationMemoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage DEAN-OS recommendation memory cases.")
    parser.add_argument("--store", default="data/dean_os/recommendation_memory.sqlite")
    parser.add_argument("--print-json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("summary")

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--agent-name", default=None)
    list_parser.add_argument("--context-tag", default=None)
    list_parser.add_argument("--outcome-label", default=None)

    add = sub.add_parser("add-case")
    add.add_argument("--source-type", default="manual_case")
    add.add_argument("--source-id", required=True)
    add.add_argument("--agent-name", required=True)
    add.add_argument("--topic", required=True)
    add.add_argument("--thesis", required=True)
    add.add_argument("--expected-direction", choices=["bullish", "bearish", "neutral"], required=True)
    add.add_argument("--context-tags", nargs="*", default=None)
    add.add_argument("--tickers", nargs="*", default=None)
    add.add_argument("--sectors", nargs="*", default=None)
    add.add_argument("--outcome-label", choices=["hit", "miss", "inconclusive", "pending"], default="pending")
    add.add_argument("--realized-return", type=float, default=None)
    add.add_argument("--lesson", default="")
    add.add_argument("--confidence-before", type=float, default=None)
    add.add_argument("--confidence-after", type=float, default=None)
    add.add_argument("--outcome-at", default=None)

    update = sub.add_parser("update-outcome")
    update.add_argument("memory_id")
    update.add_argument("--outcome-label", choices=["hit", "miss", "inconclusive", "pending"], required=True)
    update.add_argument("--realized-return", type=float, default=None)
    update.add_argument("--lesson", default=None)
    update.add_argument("--confidence-after", type=float, default=None)
    update.add_argument("--outcome-at", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = RecommendationMemoryStore(args.store)
    if args.command == "summary":
        payload = store.summary()
    elif args.command == "list":
        records = store.list_records(agent_name=args.agent_name, context_tag=args.context_tag, outcome_label=args.outcome_label)
        payload = {"record_count": len(records), "records": [record.model_dump(mode="json") for record in records]}
    elif args.command == "add-case":
        record = store.add_case(
            source_type=args.source_type,
            source_id=args.source_id,
            agent_name=args.agent_name,
            topic=args.topic,
            thesis=args.thesis,
            expected_direction=args.expected_direction,
            context_tags=args.context_tags or [],
            tickers=[ticker.upper() for ticker in args.tickers or []],
            sectors=args.sectors or [],
            outcome_label=args.outcome_label,
            realized_return=args.realized_return,
            lesson=args.lesson,
            confidence_before=args.confidence_before,
            confidence_after=args.confidence_after,
            outcome_at=args.outcome_at,
        )
        payload = {"record": record.model_dump(mode="json")}
    else:
        record = store.update_outcome(
            args.memory_id,
            outcome_label=args.outcome_label,
            realized_return=args.realized_return,
            lesson=args.lesson,
            confidence_after=args.confidence_after,
            outcome_at=args.outcome_at,
        )
        payload = {"record": record.model_dump(mode="json")}

    if args.print_json:
        print_json(payload)
        return
    if args.command == "summary":
        print(f"Records: {payload.get('record_count')} | completed={payload.get('completed_count')} | hit_rate={payload.get('hit_rate')}")
        print(f"Outcomes: {payload.get('records_by_outcome')}")
    elif args.command == "list":
        print(f"Records: {payload['record_count']}")
        for record in payload["records"][-10:]:
            print(f"- {record['memory_id']} | {record['agent_name']} | {record['topic']} | {record['outcome_label']}")
    else:
        record = payload["record"]
        print(f"Memory record: {record['memory_id']} | {record['outcome_label']}")


if __name__ == "__main__":
    main()

