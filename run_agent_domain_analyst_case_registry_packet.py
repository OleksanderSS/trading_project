from __future__ import annotations

import argparse
import json
import sys

from dean_os.analyst_core.domain_analyst_case_registry_packet import DomainAnalystCaseRegistryPacket, render_domain_analyst_case_registry_packet_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the domain analyst case registry packet.")
    parser.add_argument("--domain-thesis-review-json", default=None)
    parser.add_argument("--domain-template-standardization-json", default=None)
    parser.add_argument("--domain-forecast-review-json", default=None)
    parser.add_argument("--outcome-evaluation-json", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/domain_analyst_case_registry_packet")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "domain_thesis_review_json": args.domain_thesis_review_json,
        "domain_template_standardization_json": args.domain_template_standardization_json,
        "domain_forecast_review_json": args.domain_forecast_review_json,
        "outcome_evaluation_json": args.outcome_evaluation_json,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = DomainAnalystCaseRegistryPacket(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_domain_analyst_case_registry_packet_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
