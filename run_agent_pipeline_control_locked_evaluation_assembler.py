from __future__ import annotations

import argparse
import sys

from dean_os.pipeline_control.pipeline_control_locked_evaluation_assembler import (
    PipelineControlLockedEvaluationAssembler,
    render_pipeline_control_locked_evaluation_assembler_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble a locked model-evaluation artifact from joined real candidates.")
    parser.add_argument("--training-candidate-json", default=None)
    parser.add_argument("--evaluation-candidate-json", default=None)
    parser.add_argument("--no-write-artifact", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_control_locked_evaluation_assembler_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "training_candidate_json": args.training_candidate_json,
        "evaluation_candidate_json": args.evaluation_candidate_json,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["write_artifact"] = not args.no_write_artifact
    kwargs["save"] = not args.no_save

    payload = PipelineControlLockedEvaluationAssembler(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_pipeline_control_locked_evaluation_assembler_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
