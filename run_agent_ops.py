from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.operation_queue import OperationQueue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review DEAN-OS operation proposals without executing the pipeline.")
    parser.add_argument("--store", default="data/dean_os/operation_queue.sqlite")
    parser.add_argument("--log-path", default="logs/dean_os/events.jsonl")
    parser.add_argument("--print-json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    import_parser = sub.add_parser("import-report")
    import_parser.add_argument("report_path")

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--status", default=None)
    list_parser.add_argument("--action-type", default=None)

    for name in ("approve", "reject", "dry-run"):
        item = sub.add_parser(name)
        item.add_argument("proposal_id")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    queue = OperationQueue(args.store, event_log_path=args.log_path)
    if args.command == "import-report":
        payload = {"imported_proposal_ids": queue.import_agent_lab_report(args.report_path)}
    elif args.command == "list":
        proposals = queue.list_proposals(status=args.status, action_type=args.action_type)
        payload = {"proposal_count": len(proposals), "proposals": [item.model_dump(mode="json") for item in proposals]}
    elif args.command == "approve":
        payload = {"proposal": queue.approve(args.proposal_id).model_dump(mode="json")}
    elif args.command == "reject":
        payload = {"proposal": queue.reject(args.proposal_id).model_dump(mode="json")}
    else:
        payload = queue.dry_run(args.proposal_id)

    if args.print_json:
        print_json(payload)
        return
    if args.command == "list":
        print(f"Proposals: {payload['proposal_count']}")
        for proposal in payload["proposals"][-10:]:
            print(f"- {proposal['proposal_id']} | {proposal['status']} | {proposal['action_type']} -> {proposal['target']}")
    elif args.command == "import-report":
        print(f"Imported proposals: {len(payload['imported_proposal_ids'])}")
    elif args.command == "dry-run":
        print(f"Dry run: {payload.get('proposal_id')} | ready={payload.get('ready_for_manual_execution')}")
        print(payload.get("command_preview") or "No command preview.")
    else:
        proposal = payload["proposal"]
        print(f"{args.command}: {proposal['proposal_id']} -> {proposal['status']}")


if __name__ == "__main__":
    main()

