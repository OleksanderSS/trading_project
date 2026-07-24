from __future__ import annotations

import argparse
import sys

from dean_os.review_only_automation_run import (
    DeanOSReviewOnlyAutomationRun,
    render_review_only_automation_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the safe DEAN-OS review chain without starting the trading pipeline.")
    parser.add_argument("--candidate-paths", nargs="+", default=None)
    parser.add_argument("--training-candidate-json", default=None)
    parser.add_argument("--evaluation-candidate-json", default=None)
    parser.add_argument("--feature-stability-candidate-json", default=None)
    parser.add_argument("--replay-batch-json", default=None)
    parser.add_argument("--data-quality-json", default=None)
    parser.add_argument("--constraints-path", default=None)
    parser.add_argument("--domain-instance-contract-json", default=None)
    parser.add_argument("--no-real-metric-run", action="store_true")
    parser.add_argument("--output-dir", default="reports/dean_os/review_only_automation_run_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "candidate_paths": args.candidate_paths,
        "training_candidate_json": args.training_candidate_json,
        "evaluation_candidate_json": args.evaluation_candidate_json,
        "feature_stability_candidate_json": args.feature_stability_candidate_json,
        "replay_batch_json": args.replay_batch_json,
        "data_quality_json": args.data_quality_json,
        "constraints_path": args.constraints_path,
        "domain_instance_contract_json": args.domain_instance_contract_json,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["run_real_metric_when_ready"] = not args.no_real_metric_run
    kwargs["save"] = not args.no_save

    payload = DeanOSReviewOnlyAutomationRun(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_review_only_automation_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
