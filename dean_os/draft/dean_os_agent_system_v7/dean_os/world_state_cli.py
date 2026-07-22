from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.world_state_store import (
    HistoricalWorldStateRetriever,
    SQLiteWorldStateStore,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect immutable DEAN-OS world-state snapshots."
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("reports/dean_os/world_state/world_states.sqlite3"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List snapshots for a domain")
    list_parser.add_argument("--domain", required=True)
    list_parser.add_argument("--before-as-of")
    list_parser.add_argument("--knowledge-cutoff")
    list_parser.add_argument("--limit", type=int, default=20)

    get_parser = subparsers.add_parser("get", help="Read one snapshot")
    get_parser.add_argument("--snapshot-id", required=True)

    as_of_parser = subparsers.add_parser(
        "as-of", help="Read the latest admissible snapshot as of a decision time"
    )
    as_of_parser.add_argument("--domain", required=True)
    as_of_parser.add_argument("--as-of", required=True)
    as_of_parser.add_argument("--knowledge-cutoff")

    analog_parser = subparsers.add_parser(
        "analogs", help="Find prior world-state analogs for a stored snapshot"
    )
    analog_parser.add_argument("--snapshot-id", required=True)
    analog_parser.add_argument("--limit", type=int, default=5)
    analog_parser.add_argument("--min-similarity", type=float, default=0.0)
    analog_parser.add_argument("--candidate-limit", type=int, default=500)
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    store = SQLiteWorldStateStore(args.store)
    if args.command == "list":
        return [
            item.model_dump(mode="json")
            for item in store.list_snapshots(
                domain_id=args.domain,
                before_as_of=args.before_as_of,
                knowledge_cutoff=args.knowledge_cutoff,
                limit=args.limit,
            )
        ]
    if args.command == "get":
        snapshot = store.get(args.snapshot_id)
        return snapshot.model_dump(mode="json") if snapshot else {"status": "not_found"}
    if args.command == "as-of":
        snapshot = store.get_as_of(
            domain_id=args.domain,
            as_of=args.as_of,
            knowledge_cutoff=args.knowledge_cutoff,
        )
        return snapshot.model_dump(mode="json") if snapshot else {"status": "not_found"}
    if args.command == "analogs":
        snapshot = store.get(args.snapshot_id)
        if snapshot is None:
            return {"status": "not_found", "snapshot_id": args.snapshot_id}
        return [
            item.model_dump(mode="json")
            for item in HistoricalWorldStateRetriever(store).find_analogs(
                snapshot,
                limit=args.limit,
                min_similarity=args.min_similarity,
                candidate_limit=args.candidate_limit,
            )
        ]
    raise ValueError(f"unsupported command: {args.command}")


def main() -> int:
    args = _parser().parse_args()
    print(json.dumps(_run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
