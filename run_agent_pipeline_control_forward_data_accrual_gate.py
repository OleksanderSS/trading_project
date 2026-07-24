from __future__ import annotations

import argparse
import sys

from dean_os.pipeline_control.pipeline_control_forward_data_accrual_gate import (
    PipelineControlForwardDataAccrualGate,
    render_forward_data_accrual_gate_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a saved source against a registered forward-development boundary.")
    parser.add_argument("--accrual-plan-json", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_control_forward_data_accrual_gate_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    payload = PipelineControlForwardDataAccrualGate(args.output_dir).build(
        accrual_plan_json=args.accrual_plan_json,
        source_path=args.source_path,
        save=not args.no_save,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_forward_data_accrual_gate_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
