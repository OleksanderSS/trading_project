from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.fundamental_input_readiness_gate import (
    DEFAULT_FUNDAMENTALS_JSON,
    FundamentalInputReadinessGate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check caller-supplied fundamental inputs before value-agent review."
        )
    )
    parser.add_argument("--fundamentals-json", default=DEFAULT_FUNDAMENTALS_JSON)
    parser.add_argument(
        "--output-dir",
        default="reports/dean_os/fundamental_input_readiness_gate_current",
    )
    parser.add_argument(
        "--as-of",
        help=(
            "Timezone-aware analysis cutoff. Without it the gate may support "
            "manual inspection but cannot authorize value-screen input."
        ),
    )
    parser.add_argument("--print-json", action="store_true")
    return parser


def print_summary(payload: dict) -> None:
    summary = payload.get("summary", {})
    guidance = payload.get("decision_guidance", {})
    saved = payload.get("saved_paths", {})
    print(f"Run ID: {payload.get('run_id')}")
    print(f"Readiness: {summary.get('readiness_status')}")
    print(f"Recommended action: {summary.get('recommended_action')}")
    print(
        "Fundamentals: "
        f"metrics={summary.get('metric_count')} | "
        f"tickers={summary.get('ticker_count')} | "
        f"missing_citations={summary.get('source_citation_missing_count')} | "
        f"missing_periods={summary.get('period_missing_count')}"
    )
    print(
        "Checks: "
        f"pass={guidance.get('pass_count')} "
        f"warn={guidance.get('warning_count')} "
        f"fail={guidance.get('fail_count')}"
    )
    if saved:
        print(f"Report JSON: {saved.get('latest_json') or saved.get('json')}")
        print(
            "Report Markdown: "
            f"{saved.get('latest_markdown') or saved.get('markdown')}"
        )


def main() -> None:
    args = build_parser().parse_args()
    payload = FundamentalInputReadinessGate(output_dir=args.output_dir).build(
        fundamentals_json=args.fundamentals_json,
        as_of=args.as_of,
    )
    if args.print_json:
        print_json(payload)
        return
    print_summary(payload)


if __name__ == "__main__":
    main()
