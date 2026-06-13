from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json, run_id, save_latest_json
from dean_os.paper_trading import PaperTradeEvaluationRunner, PaperTradeStore, create_paper_trade_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record, inspect, and evaluate DEAN-OS paper-only decisions.")
    parser.add_argument("--store", default="data/dean_os/paper_trades.sqlite")
    parser.add_argument("--print-json", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record")
    record.add_argument("--action", choices=["watchlist", "paper_trade_only", "candidate_long", "candidate_short", "no_trade"], required=True)
    record.add_argument("--tickers", nargs="*", default=None)
    record.add_argument("--expected-direction", choices=["bullish", "bearish", "neutral"], default=None)
    record.add_argument("--source-type", default="manual")
    record.add_argument("--source-id", default="")
    record.add_argument("--agent-name", default="chief_review")
    record.add_argument("--horizon-days", type=int, default=30)
    record.add_argument("--thesis", default="")
    record.add_argument("--confidence", type=float, default=0.0)
    record.add_argument("--context-tags", nargs="*", default=None)
    record.add_argument("--regime-tags", nargs="*", default=None)

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--status", default=None)
    list_parser.add_argument("--agent-name", default=None)

    sub.add_parser("summary")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--market-data-path", default=None)
    evaluate.add_argument("--latest-processed-prices", default=None)
    evaluate.add_argument("--tickers", nargs="*", default=None)
    evaluate.add_argument("--as-of", default=None)
    evaluate.add_argument("--close-col", default="close")
    evaluate.add_argument("--datetime-col", default="datetime")
    evaluate.add_argument("--allow-early", action="store_true")
    evaluate.add_argument("--apply", action="store_true")
    evaluate.add_argument("--neutral-band", type=float, default=0.01)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--output", default=None)
    evaluate.add_argument("--output-dir", default="reports/dean_os/paper_trades")

    void = sub.add_parser("void")
    void.add_argument("trade_id")
    void.add_argument("--reason", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = PaperTradeStore(args.store)
    if args.command == "record":
        record = create_paper_trade_record(
            action=args.action,
            tickers=[ticker.upper() for ticker in args.tickers or []],
            expected_direction=args.expected_direction,
            source_type=args.source_type,
            source_id=args.source_id,
            agent_name=args.agent_name,
            horizon_days=args.horizon_days,
            thesis=args.thesis,
            confidence=args.confidence,
            context_tags=args.context_tags or [],
            regime_tags=args.regime_tags or [],
        )
        store.add_record(record)
        payload = {"record": record.model_dump(mode="json")}
    elif args.command == "list":
        records = store.list_records(status=args.status, agent_name=args.agent_name)
        payload = {"record_count": len(records), "records": [record.model_dump(mode="json") for record in records]}
    elif args.command == "summary":
        payload = store.summary()
    elif args.command == "evaluate":
        result = PaperTradeEvaluationRunner(args.store).evaluate(
            market_data_path=args.market_data_path,
            latest_processed_prices=args.latest_processed_prices,
            tickers=[ticker.upper() for ticker in args.tickers or []],
            as_of=args.as_of,
            close_col=args.close_col,
            datetime_col=args.datetime_col,
            allow_early=args.allow_early,
            apply_updates=args.apply,
            neutral_band=args.neutral_band,
            limit=args.limit,
        )
        payload = save_latest_json(args.output, args.output_dir, {"run_id": run_id("paper_trades_evaluation"), "inputs": vars(args), **result})
    else:
        record = store.void_record(args.trade_id, reason=args.reason)
        payload = {"record": record.model_dump(mode="json")}

    if args.print_json:
        print_json(payload)
        return
    if args.command == "record":
        record = payload["record"]
        print(f"Recorded paper decision: {record['trade_id']} | {record['action']} | {', '.join(record['tickers']) or 'no tickers'}")
    elif args.command == "list":
        print(f"Records: {payload['record_count']}")
        for record in payload["records"][-10:]:
            print(f"- {record['trade_id']} | {record['status']} | {record['action']} | {record['outcome_label']}")
    elif args.command == "summary":
        print(f"Records: {payload.get('record_count')} | pending={payload.get('pending_count')} | evaluated={payload.get('evaluated_count')}")
        print(f"Hit rate: {payload.get('hit_rate')}")
    elif args.command == "evaluate":
        print(f"Pending checked: {payload.get('pending_record_count')} | evaluable={payload.get('evaluable_count')} | updated={payload.get('updated_count')}")
        print(f"Status counts: {payload.get('status_counts')}")
        print(f"Report JSON: {payload.get('saved_paths', {}).get('latest_json') or payload.get('saved_paths', {}).get('json')}")
    else:
        record = payload["record"]
        print(f"Voided: {record['trade_id']} | status={record['status']}")


if __name__ == "__main__":
    main()

