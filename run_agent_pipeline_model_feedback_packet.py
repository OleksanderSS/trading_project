from __future__ import annotations

import argparse
import sys

from dean_os.packets.pipeline_model_feedback_packet import (
    PipelineModelFeedbackPacket,
    render_pipeline_model_feedback_packet_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize human feedback for a pipeline model case.")
    parser.add_argument("--pipeline-model-case-json", default=None)
    parser.add_argument("--manual-feedback-json", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_model_feedback_packet_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "pipeline_model_case_json": args.pipeline_model_case_json,
        "manual_feedback_json": args.manual_feedback_json,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = PipelineModelFeedbackPacket(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_pipeline_model_feedback_packet_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
