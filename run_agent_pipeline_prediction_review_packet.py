from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.schemas import utc_now_iso
from dean_os.packets.pipeline_prediction_review_packet import PipelinePredictionReviewPacket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PipelinePredictionReviewPacket (pipeline_prediction_review_packet).")
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_prediction_review_packet_current")
    parser.add_argument("--pipeline-result", required=True,
                        help="Path to a saved JSON artifact.")
    parser.add_argument("--requested-tickers", action="append", dest="requested_tickers", default=None)
    parser.add_argument("--requested-timeframes", action="append", dest="requested_timeframes", default=None)
    parser.add_argument("--filter-to-requested-scope", dest="filter_to_requested_scope", action="store_true")
    parser.add_argument("--source-artifact-path", default=None)
    parser.add_argument("--sector-to-ticker-review-path", default=None)
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="Build the payload without writing report files.")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = PipelinePredictionReviewPacket(
        output_dir=args.output_dir,
    )
    payload = builder.build(
        pipeline_result=args.pipeline_result,
        requested_tickers=args.requested_tickers,
        requested_timeframes=args.requested_timeframes,
        filter_to_requested_scope=args.filter_to_requested_scope,
        source_artifact_path=args.source_artifact_path,
        sector_to_ticker_review_path=args.sector_to_ticker_review_path,
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
