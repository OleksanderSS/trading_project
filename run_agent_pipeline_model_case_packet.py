from __future__ import annotations

import argparse
import sys

from dean_os.packets.pipeline_model_case_packet import (
    PipelineModelCasePacket,
    render_pipeline_model_case_packet_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a review-only case from one locked pipeline evaluation chain.")
    parser.add_argument("--real-metric-evidence-json", default=None)
    parser.add_argument("--model-evaluation-json", default=None)
    parser.add_argument("--feature-stability-json", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_model_case_packet_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "real_metric_evidence_json": args.real_metric_evidence_json,
        "model_evaluation_json": args.model_evaluation_json,
        "feature_stability_json": args.feature_stability_json,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = PipelineModelCasePacket(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_pipeline_model_case_packet_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
