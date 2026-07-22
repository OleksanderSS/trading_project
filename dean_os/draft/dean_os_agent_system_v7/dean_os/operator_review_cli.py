from __future__ import annotations

import argparse
import json

from dean_os.draft.dean_os_agent_system_v7.dean_os.operator_review_inbox_v2 import SQLiteOperatorReviewInbox


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the DEAN-OS review-only operator inbox.")
    parser.add_argument("--store", required=True, help="Path to operator-review SQLite store")
    parser.add_argument("--domain-id", default=None)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    items = SQLiteOperatorReviewInbox(args.store).list_pending(domain_id=args.domain_id, limit=args.limit)
    print(json.dumps([item.model_dump(mode="json") for item in items], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
