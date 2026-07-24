from __future__ import annotations

import argparse
import json
import sys

from dean_os.analyst_core.domain_analyst_template_decision_packet import DomainAnalystTemplateDecisionPacket, render_domain_analyst_template_decision_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the domain analyst template decision packet.")
    parser.add_argument("--vertical-slice-json", default=None)
    parser.add_argument("--template-standardization-json", default=None)
    parser.add_argument("--forecast-review-json", default=None)
    parser.add_argument("--case-registry-json", default=None)
    parser.add_argument("--portability-review-json", default=None)
    parser.add_argument("--architecture-map-json", default=None)
    parser.add_argument("--decision", default=None)
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--rationale", default=None)
    parser.add_argument("--required-followups", nargs="+", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/domain_analyst_template_decision_packet_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "vertical_slice_json": args.vertical_slice_json,
        "template_standardization_json": args.template_standardization_json,
        "forecast_review_json": args.forecast_review_json,
        "case_registry_json": args.case_registry_json,
        "portability_review_json": args.portability_review_json,
        "architecture_map_json": args.architecture_map_json,
        "decision": args.decision,
        "reviewer": args.reviewer,
        "rationale": args.rationale,
        "required_followups": args.required_followups,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = DomainAnalystTemplateDecisionPacket(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_domain_analyst_template_decision_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
