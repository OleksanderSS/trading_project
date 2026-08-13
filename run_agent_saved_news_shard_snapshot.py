from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.schemas import utc_now_iso
from dean_os.saved_news_shard_snapshot import SavedNewsShardSnapshotBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SavedNewsShardSnapshotBuilder (saved_news_shard_snapshot).")
    parser.add_argument("--output-dir", default="reports/dean_os/saved_news_shard_snapshot_current")
    parser.add_argument("--database-path", required=True)
    parser.add_argument("--output-parquet-path", required=True)
    parser.add_argument("--as-of", default=None,
                        help="ISO-8601 timestamp; defaults to now in UTC.")
    parser.add_argument("--include-parquet-paths", action="append", dest="include_parquet_paths", default=None)
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="Build the payload without writing report files.")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = SavedNewsShardSnapshotBuilder(
        output_dir=args.output_dir,
    )
    payload = builder.build(
        database_path=args.database_path,
        output_parquet_path=args.output_parquet_path,
        as_of=args.as_of or utc_now_iso(),
        include_parquet_paths=args.include_parquet_paths,
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
