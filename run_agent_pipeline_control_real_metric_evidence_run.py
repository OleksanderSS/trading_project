from __future__ import annotations

import argparse
import sys

from dean_os.pipeline_control.pipeline_control_real_metric_evidence_run import (
    PipelineControlRealMetricEvidenceRun,
    render_pipeline_control_real_metric_evidence_run_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pipeline-control gates from real locked metric artifacts.")
    parser.add_argument("--model-evaluation-json", default=None)
    parser.add_argument("--feature-stability-report", default=None)
    parser.add_argument("--replay-batch-json", default=None)
    parser.add_argument("--data-quality-json", default=None)
    parser.add_argument("--constraints-path", default=None)
    parser.add_argument("--architecture-map-json", default=None)
    parser.add_argument("--domain-instance-contract-json", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_control_real_metric_evidence_run")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "model_evaluation_json": args.model_evaluation_json,
        "feature_stability_report": args.feature_stability_report,
        "replay_batch_json": args.replay_batch_json,
        "data_quality_json": args.data_quality_json,
        "constraints_path": args.constraints_path,
        "architecture_map_json": args.architecture_map_json,
        "domain_instance_contract_json": args.domain_instance_contract_json,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = PipelineControlRealMetricEvidenceRun(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_pipeline_control_real_metric_evidence_run_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
