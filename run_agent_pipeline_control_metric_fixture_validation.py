from __future__ import annotations

import argparse
import sys

from dean_os.pipeline_control.pipeline_control_metric_fixture_validation import (
    PipelineControlMetricFixtureValidation,
    render_pipeline_control_metric_fixture_validation_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic control-flow validation for pipeline metric gates.")
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_control_metric_fixture_validation")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    payload = PipelineControlMetricFixtureValidation(args.output_dir).build(save=not args.no_save)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_pipeline_control_metric_fixture_validation_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
