from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime

from dean_os.analyst_core.domain_analyst_lifecycle_profile import (
    DomainAnalystLifecycleProfileCompiler,
)
from dean_os.clean_yahoo_market_snapshot import CleanYahooMarketSnapshot


def _aware(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("--end-date must be timezone-aware")
    return parsed


def _domain_tickers(domain_id: str) -> list[str]:
    profile = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
    measurement = (profile.get("domain_overlay") or {}).get(
        "market_measurement"
    ) or {}
    tickers = {
        str(item).strip().upper()
        for item in measurement.get("primary_universe") or []
        if str(item).strip()
    }
    benchmark = str(measurement.get("benchmark_ticker") or "").strip().upper()
    if benchmark:
        tickers.add(benchmark)
    if not tickers:
        raise ValueError("domain market scope is empty")
    return sorted(tickers)


async def _run(args: argparse.Namespace) -> dict:
    tickers = _domain_tickers(args.domain_id) if args.domain_id else sorted(
        {str(item).strip().upper() for item in args.tickers or [] if str(item).strip()}
    )
    if not tickers:
        raise ValueError("provide --domain-id or at least one --ticker")
    return await CleanYahooMarketSnapshot(
        artifact_dir=args.artifact_dir,
        report_dir=args.output_dir,
    ).build(
        tickers=tickers,
        config_path=args.config_path,
        timeframes=args.timeframes or ["15m"],
        end_date=args.end_date,
        max_download_attempts=args.max_download_attempts,
        save=not args.no_save,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one bounded Yahoo snapshot outside the legacy database. "
            "Network access is performed; no learning or trading action runs."
        )
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--domain-id")
    scope.add_argument("--ticker", action="append", dest="tickers")
    parser.add_argument("--timeframe", action="append", dest="timeframes")
    parser.add_argument("--end-date", required=True, type=_aware)
    parser.add_argument("--max-download-attempts", type=int, default=2)
    parser.add_argument("--config-path", default="src/config/collectors.yaml")
    parser.add_argument(
        "--artifact-dir", default="data/dean_os/clean_market_snapshots"
    )
    parser.add_argument(
        "--output-dir", default="reports/dean_os/clean_market_snapshot_current"
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    payload = asyncio.run(_run(args))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
