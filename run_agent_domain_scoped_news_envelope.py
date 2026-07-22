from __future__ import annotations

import argparse
import json

from dean_os.domain_scoped_news_envelope import DomainScopedNewsEnvelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one saved news artifact as a review-only, trigger-evidence "
            "domain binding candidate."
        )
    )
    parser.add_argument("domain_id")
    parser.add_argument("--as-of", required=True)
    parser.add_argument(
        "--source-path",
        default=(
            "reports/dean_os/"
            "saved_semiconductor_news_evidence_producer_current/latest.json"
        ),
    )
    parser.add_argument(
        "--dispatch-path",
        default="reports/dean_os/domain_binding_task_dispatch_current/latest.json",
    )
    parser.add_argument(
        "--journal-path", default="data/dean_os/system_journal.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        default="reports/dean_os/domain_scoped_news_envelope_current",
    )
    parser.add_argument("--apply-journal", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = DomainScopedNewsEnvelope(args.output_dir).build(
        domain_id=args.domain_id,
        as_of=args.as_of,
        source_path=args.source_path,
        dispatch_path=args.dispatch_path,
        journal_path=args.journal_path,
        apply_journal=args.apply_journal,
        save=not args.no_save,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
