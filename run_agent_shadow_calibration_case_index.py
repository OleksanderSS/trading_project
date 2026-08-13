from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.schemas import utc_now_iso
from dean_os.shadow_calibration_case_index import ShadowCalibrationCaseIndexBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ShadowCalibrationCaseIndexBuilder (shadow_calibration_case_index).")
    parser.add_argument("--prediction-review-path", required=True)
    parser.add_argument("--outcome-source-path", required=True)
    parser.add_argument("--output-dir", default="reports/dean_os/shadow_calibration_case_index_current")
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="Build the payload without writing report files.")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = ShadowCalibrationCaseIndexBuilder(
        prediction_review_path=args.prediction_review_path,
        outcome_source_path=args.outcome_source_path,
        output_dir=args.output_dir,
    )
    payload = builder.build(
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
