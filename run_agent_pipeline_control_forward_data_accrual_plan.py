from __future__ import annotations

import argparse
import sys

from dean_os.pipeline_control.pipeline_control_forward_data_accrual_plan import (
    PipelineControlForwardDataAccrualPlan,
    render_forward_data_accrual_plan_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a prospective boundary for the next development refresh.")
    parser.add_argument("--walk-forward-json", required=True)
    parser.add_argument("--acknowledge-development-refresh-only", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_control_forward_data_accrual_plan_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    payload = PipelineControlForwardDataAccrualPlan(args.output_dir).build(
        walk_forward_json=args.walk_forward_json,
        acknowledge_development_refresh_only=args.acknowledge_development_refresh_only,
        save=not args.no_save,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_forward_data_accrual_plan_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
