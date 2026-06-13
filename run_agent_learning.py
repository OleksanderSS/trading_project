from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.learning import LearningStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or update DEAN-OS learning records.")
    parser.add_argument("--store", default="data/dean_os/agent_learning.sqlite")
    parser.add_argument("--print-json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--agent-name", default=None)

    score_parser = sub.add_parser("score")
    score_parser.add_argument("agent_name")

    update_parser = sub.add_parser("update")
    update_parser.add_argument("record_id")
    update_parser.add_argument("--realized-return", type=float, required=True)
    update_parser.add_argument("--outcome-at", default=None)
    update_parser.add_argument("--neutral-band", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = LearningStore(args.store)
    if args.command == "list":
        records = [record.model_dump(mode="json") for record in store.list_records(agent_name=args.agent_name)]
        payload = {"store": args.store, "records": records, "record_count": len(records)}
    elif args.command == "score":
        payload = store.score_agent(args.agent_name)
    else:
        record = store.update_outcome(
            args.record_id,
            realized_return=args.realized_return,
            outcome_at=args.outcome_at,
            neutral_band=args.neutral_band,
        )
        payload = {"updated": record.model_dump(mode="json")}

    if args.print_json:
        print_json(payload)
        return
    if args.command == "list":
        print(f"Records: {payload['record_count']}")
        for record in payload["records"][-10:]:
            print(f"- {record['record_id']} | {record['agent_name']} | {record['expected_direction']} | {record.get('outcome_label')}")
    elif args.command == "score":
        print(f"Agent: {payload['agent_name']} | completed={payload['record_count']} | hit_rate={payload['hit_rate']}")
        print(f"Suggested weight: {payload['suggested_weight']}")
    else:
        updated = payload["updated"]
        print(f"Updated: {updated['record_id']} | outcome={updated.get('outcome_label')} | return={updated.get('realized_return')}")


if __name__ == "__main__":
    main()

