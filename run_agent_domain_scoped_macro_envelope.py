from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.domain_scoped_macro_envelope import DomainScopedMacroEnvelopeCeremony


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DomainScopedMacroEnvelopeCeremony (domain_scoped_macro_envelope).")
    parser.add_argument("--domain-id", default='energy')
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--registry-path", default='dean_os/config/macro_series_registry.yaml')
    parser.add_argument("--dispatch-path", default='reports/dean_os/domain_binding_task_dispatch_current/latest.json')
    parser.add_argument("--execution-gate-path", default=None)
    parser.add_argument("--journal-path", default='data/dean_os/system_journal.jsonl')
    parser.add_argument("--apply-journal", dest="apply_journal", action="store_true")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/domain_scoped_macro_envelope_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = DomainScopedMacroEnvelopeCeremony(output_dir=args.output_dir).build(
        domain_id=args.domain_id,
        source_path=args.source_path,
        as_of=args.as_of,
        registry_path=args.registry_path,
        dispatch_path=args.dispatch_path,
        execution_gate_path=args.execution_gate_path,
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
