from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.domain_macro_collection_execution_gate import DomainMacroCollectionExecutionGate

# Build parameters not exposed on this CLI (non-scalar types); call the
# builder directly if you need them:
#   credential_present_override: bool | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DomainMacroCollectionExecutionGate (domain_macro_collection_execution_gate).")
    parser.add_argument("--domain-id", default='energy')
    parser.add_argument("--request-path", default='reports/dean_os/domain_macro_collection_request_current/latest.json')
    parser.add_argument("--registry-path", default='dean_os/config/macro_series_registry.yaml')
    parser.add_argument("--evaluated-at", default=None)
    parser.add_argument("--journal-path", default='data/dean_os/system_journal.jsonl')
    parser.add_argument("--apply-journal", dest="apply_journal", action="store_true")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/domain_macro_collection_execution_gate_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = DomainMacroCollectionExecutionGate(output_dir=args.output_dir).build(
        domain_id=args.domain_id,
        request_path=args.request_path,
        registry_path=args.registry_path,
        evaluated_at=args.evaluated_at,
        journal_path=args.journal_path,
        apply_journal=args.apply_journal,
        save=args.save,
    )
    if args.print_json:
        print_json(payload)
        return
    print(f"Run ID: {payload.get('run_id')}")
    for key, value in (payload.get("summary") or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            print(f"{key}: {value}")
    saved = payload.get("saved_paths") or {}
    if saved:
        print(f"Report JSON: {saved.get('latest_json') or saved.get('json')}")


if __name__ == "__main__":
    main()
