from __future__ import annotations

import argparse
import json
import sys

from dean_os.analyst_core.domain_analyst_template_standardization_packet import DomainAnalystTemplateStandardizationPacket, render_domain_analyst_template_standardization_packet_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the domain analyst template standardization packet.")
    parser.add_argument("--domain-instance-contract-json", default=None)
    parser.add_argument("--domain-thesis-review-json", default=None)
    parser.add_argument("--regime-scenario-json", default=None)
    parser.add_argument("--architecture-map-json", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/domain_analyst_template_standardization_packet")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "domain_instance_contract_json": args.domain_instance_contract_json,
        "domain_thesis_review_json": args.domain_thesis_review_json,
        "regime_scenario_json": args.regime_scenario_json,
        "architecture_map_json": args.architecture_map_json,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = DomainAnalystTemplateStandardizationPacket(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_domain_analyst_template_standardization_packet_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
