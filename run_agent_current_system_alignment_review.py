from __future__ import annotations

import argparse
import json
import sys

from dean_os.current_system_alignment_review import CurrentSystemAlignmentReview, render_current_system_alignment_review_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the current system alignment review report.")
    parser.add_argument("--evidence-pack-json", default=None)
    parser.add_argument("--source-gate-json", default=None)
    parser.add_argument("--agent-lab-path", default=None)
    parser.add_argument("--dropzone-inventory-json", default=None)
    parser.add_argument("--fundamental-gate-json", default=None)
    parser.add_argument("--architecture-map-json", default=None)
    parser.add_argument("--domain-analyst-intake-json", default=None)
    parser.add_argument("--domain-analyst-instance-contract-json", default=None)
    parser.add_argument("--domain-analyst-thesis-review-json", default=None)
    parser.add_argument("--domain-analyst-template-standardization-json", default=None)
    parser.add_argument("--domain-analyst-case-registry-json", default=None)
    parser.add_argument("--pipeline-metric-input-readiness-json", default=None)
    parser.add_argument("--pipeline-control-instance-contract-json", default=None)
    parser.add_argument("--pipeline-control-caution-review-json", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/current_system_alignment_review")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "evidence_pack_json": args.evidence_pack_json,
        "source_gate_json": args.source_gate_json,
        "agent_lab_path": args.agent_lab_path,
        "dropzone_inventory_json": args.dropzone_inventory_json,
        "fundamental_gate_json": args.fundamental_gate_json,
        "architecture_map_json": args.architecture_map_json,
        "domain_analyst_intake_json": args.domain_analyst_intake_json,
        "domain_analyst_instance_contract_json": args.domain_analyst_instance_contract_json,
        "domain_analyst_thesis_review_json": args.domain_analyst_thesis_review_json,
        "domain_analyst_template_standardization_json": args.domain_analyst_template_standardization_json,
        "domain_analyst_case_registry_json": args.domain_analyst_case_registry_json,
        "pipeline_metric_input_readiness_json": args.pipeline_metric_input_readiness_json,
        "pipeline_control_instance_contract_json": args.pipeline_control_instance_contract_json,
        "pipeline_control_caution_review_json": args.pipeline_control_caution_review_json,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = CurrentSystemAlignmentReview(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_current_system_alignment_review_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
