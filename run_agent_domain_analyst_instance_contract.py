from __future__ import annotations

import argparse
import json
import sys

from dean_os.analyst_core.domain_analyst_instance_contract import DomainAnalystInstanceContract, render_domain_analyst_instance_contract_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the domain analyst instance contract.")
    parser.add_argument("--evidence-pack-json", default=None)
    parser.add_argument("--source-gate-json", default=None)
    parser.add_argument("--domain-intake-json", default=None)
    parser.add_argument("--architecture-map-json", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/domain_analyst_instance_contract")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "evidence_pack_json": args.evidence_pack_json,
        "source_gate_json": args.source_gate_json,
        "domain_intake_json": args.domain_intake_json,
        "architecture_map_json": args.architecture_map_json,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = DomainAnalystInstanceContract(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_domain_analyst_instance_contract_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
