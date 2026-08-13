from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.unknown_voi_candidate_proposal import UnknownValueOfInformationCandidateProposalBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UnknownValueOfInformationCandidateProposalBuilder (unknown_voi_candidate_proposal).")
    parser.add_argument("--evidence-plan-path", default=None)
    parser.add_argument("--voi-review-path", default=None)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/unknown_voi_candidate_proposal_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = UnknownValueOfInformationCandidateProposalBuilder(output_dir=args.output_dir).build(
        evidence_plan_path=args.evidence_plan_path,
        voi_review_path=args.voi_review_path,
        max_candidates=args.max_candidates,
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
