from __future__ import annotations

import argparse
import sys

from dean_os.pipeline_control.pipeline_control_caution_review_packet import (
    PipelineControlCautionReviewPacket,
    render_pipeline_control_caution_review_packet_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the pipeline control caution review packet.")
    parser.add_argument("--pipeline-metric-input-readiness-json", default=None)
    parser.add_argument("--pipeline-control-instance-json", default=None)
    parser.add_argument("--model-performance-report-json", default=None)
    parser.add_argument("--feature-report-json", default=None)
    parser.add_argument("--data-quality-json", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_control_caution_review_packet")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "pipeline_metric_input_readiness_json": args.pipeline_metric_input_readiness_json,
        "pipeline_control_instance_json": args.pipeline_control_instance_json,
        "model_performance_report_json": args.model_performance_report_json,
        "feature_report_json": args.feature_report_json,
        "data_quality_json": args.data_quality_json,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = PipelineControlCautionReviewPacket(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_pipeline_control_caution_review_packet_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
