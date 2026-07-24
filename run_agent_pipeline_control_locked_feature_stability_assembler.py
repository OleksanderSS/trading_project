from __future__ import annotations

import argparse
import sys

from dean_os.pipeline_control.pipeline_control_locked_feature_stability_assembler import (
    PipelineControlLockedFeatureStabilityAssembler,
    render_pipeline_control_locked_feature_stability_assembler_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble a locked feature-stability report from a measured candidate.")
    parser.add_argument("--feature-stability-candidate-json", default=None)
    parser.add_argument("--no-write-artifact", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_control_locked_feature_stability_assembler_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    kwargs = {}
    if args.feature_stability_candidate_json is not None:
        kwargs["feature_stability_candidate_json"] = args.feature_stability_candidate_json
    kwargs["write_artifact"] = not args.no_write_artifact
    kwargs["save"] = not args.no_save

    payload = PipelineControlLockedFeatureStabilityAssembler(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_pipeline_control_locked_feature_stability_assembler_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
