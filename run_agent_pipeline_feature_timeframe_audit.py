from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.pipeline_feature_timeframe_audit import PipelineFeatureTimeframeAudit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PipelineFeatureTimeframeAudit (pipeline_feature_timeframe_audit).")
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--stage5-path", default=None)
    parser.add_argument("--tickers", action="append", dest="tickers", default=None)
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_feature_timeframe_audit_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = PipelineFeatureTimeframeAudit(output_dir=args.output_dir).build(
        features_path=args.features_path,
        stage5_path=args.stage5_path,
        tickers=args.tickers,
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
