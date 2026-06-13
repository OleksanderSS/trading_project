from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.review_actions import ReviewActionStore, validate_review_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record human review lifecycle actions for DEAN-OS.")
    parser.add_argument("--store", default="data/dean_os/review_actions.sqlite")
    parser.add_argument("--operations-store", default="data/dean_os/operation_queue.sqlite")
    parser.add_argument("--log-path", default="logs/dean_os/events.jsonl")
    parser.add_argument("--reports-dir", default="reports/dean_os/agent_lab")
    parser.add_argument("--learning-store", default="data/dean_os/agent_learning.sqlite")
    parser.add_argument("--print-json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--source-type", default=None)
    list_parser.add_argument("--action-type", default=None)

    for name in ("mark-reviewed", "needs-more-data", "promote-watchlist"):
        item = sub.add_parser(name)
        item.add_argument("--source-type", required=True)
        item.add_argument("--source-id", required=True)
        item.add_argument("--notes", default="")
        item.add_argument("--reviewer", default="human")
        if name == "needs-more-data":
            item.add_argument("--data-request", required=True)
        if name == "promote-watchlist":
            item.add_argument("--tickers", nargs="*", default=None)
            item.add_argument("--thesis", required=True)
            item.add_argument("--reason", required=True)

    void_parser = sub.add_parser("void-action")
    void_parser.add_argument("action_id")
    void_parser.add_argument("--reason", default="")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if getattr(args, "source_type", None) and getattr(args, "source_id", None):
        validate_review_source(
            args.source_type,
            args.source_id,
            reports_dir=args.reports_dir,
            learning_path=args.learning_store,
            operations_path=args.operations_store,
        )


def main() -> None:
    args = build_parser().parse_args()
    store = ReviewActionStore(args.store, operations_path=args.operations_store, event_log_path=args.log_path)
    if args.command == "list":
        actions = store.list_actions(source_type=args.source_type, action_type=args.action_type)
        payload = {"action_count": len(actions), "actions": [item.model_dump(mode="json") for item in actions]}
    elif args.command == "mark-reviewed":
        _validate(args)
        payload = {"action": store.mark_reviewed(args.source_type, args.source_id, notes=args.notes, reviewer=args.reviewer).model_dump(mode="json")}
    elif args.command == "needs-more-data":
        _validate(args)
        payload = {
            "action": store.needs_more_data(
                args.source_type,
                args.source_id,
                data_request=args.data_request,
                notes=args.notes,
                reviewer=args.reviewer,
            ).model_dump(mode="json")
        }
    elif args.command == "promote-watchlist":
        _validate(args)
        payload = {
            "action": store.promote_to_watchlist_proposal(
                args.source_type,
                args.source_id,
                tickers=[ticker.upper() for ticker in args.tickers or []],
                thesis=args.thesis,
                reason=args.reason,
                notes=args.notes,
                reviewer=args.reviewer,
            ).model_dump(mode="json")
        }
    else:
        payload = {"action": store.void_action(args.action_id, reason=args.reason).model_dump(mode="json")}

    if args.print_json:
        print_json(payload)
        return
    if args.command == "list":
        print(f"Actions: {payload['action_count']}")
        for action in payload["actions"][-10:]:
            print(f"- {action['action_id']} | {action['action_type']} | {action['source_type']}:{action['source_id']} | {action['status']}")
    else:
        action = payload["action"]
        print(f"Action: {action['action_id']} | {action['action_type']} | status={action['status']}")


if __name__ == "__main__":
    main()

