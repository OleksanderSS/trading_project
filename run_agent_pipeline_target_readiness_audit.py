from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.pipeline_target_readiness_audit import PipelineTargetReadinessAudit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PipelineTargetReadinessAudit (pipeline_target_readiness_audit).")
    parser.add_argument("--targets-path", default=None)
    parser.add_argument("--tickers", action="append", dest="tickers", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--batch-metadata-path", default=None)
    parser.add_argument("--target-registry-path", default='src/config/targets.yaml')
    parser.add_argument("--minimum-non-null-ratio", type=float, default=0.5)
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_target_readiness_audit_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = PipelineTargetReadinessAudit(output_dir=args.output_dir).build(
        targets_path=args.targets_path,
        tickers=args.tickers,
        timeframe=args.timeframe,
        features_path=args.features_path,
        batch_metadata_path=args.batch_metadata_path,
        target_registry_path=args.target_registry_path,
        minimum_non_null_ratio=args.minimum_non_null_ratio,
        save=args.save,
    )
    if args.print_json:
        print_json(payload)
        return
    print(f"Run ID: {payload.get('run_id')}")
    for key, value in (payload.get("summary") or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            print(f"{key}: {value}")
    saved = payload.get("saved_paths") or {}
    if saved:
        print(f"Report JSON: {saved.get('latest_json') or saved.get('json')}")


if __name__ == "__main__":
    main()
