from __future__ import annotations

import argparse
import sys

from dean_os.pipeline_metric_input_readiness_gate import (
    PipelineMetricInputReadinessGate,
    render_pipeline_metric_input_readiness_gate_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the pipeline metric input readiness gate.")
    parser.add_argument("--model-performance", dest="model_performance_path", default=None)
    parser.add_argument("--replay-batch", dest="replay_batch_path", default=None)
    parser.add_argument("--feature-report", dest="feature_report_path", default=None)
    parser.add_argument("--data-quality", dest="data_quality_path", default=None)
    parser.add_argument("--constraints-path", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_metric_input_readiness_gate")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "model_performance_path": args.model_performance_path,
        "replay_batch_path": args.replay_batch_path,
        "feature_report_path": args.feature_report_path,
        "data_quality_path": args.data_quality_path,
        "constraints_path": args.constraints_path,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = PipelineMetricInputReadinessGate(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_pipeline_metric_input_readiness_gate_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
