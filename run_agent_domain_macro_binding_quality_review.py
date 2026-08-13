from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.domain_macro_binding_quality_review import DomainMacroBindingQualityReview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DomainMacroBindingQualityReview (domain_macro_binding_quality_review).")
    parser.add_argument("--domain-id", default='energy')
    parser.add_argument("--candidate-path", default='reports/dean_os/domain_scoped_macro_envelope_current/latest.json')
    parser.add_argument("--binding-plan-path", default='reports/dean_os/domain_analyst_binding_plan_current/latest.json')
    parser.add_argument("--review-as-of", default=None)
    parser.add_argument("--journal-path", default='data/dean_os/system_journal.jsonl')
    parser.add_argument("--apply-journal", dest="apply_journal", action="store_true")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/domain_macro_binding_quality_review_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = DomainMacroBindingQualityReview(output_dir=args.output_dir).build(
        domain_id=args.domain_id,
        candidate_path=args.candidate_path,
        binding_plan_path=args.binding_plan_path,
        review_as_of=args.review_as_of,
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
