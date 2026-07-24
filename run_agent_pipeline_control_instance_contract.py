from __future__ import annotations

import argparse
import sys

from dean_os.pipeline_control.pipeline_control_instance_contract import (
    PipelineControlInstanceContract,
    render_pipeline_control_instance_contract_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the pipeline control instance contract.")
    parser.add_argument("--pipeline-surface-json", default=None)
    parser.add_argument("--architecture-map-json", default=None)
    parser.add_argument("--domain-instance-contract-json", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_control_instance_contract")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "pipeline_surface_json": args.pipeline_surface_json,
        "architecture_map_json": args.architecture_map_json,
        "domain_instance_contract_json": args.domain_instance_contract_json,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = PipelineControlInstanceContract(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_pipeline_control_instance_contract_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
