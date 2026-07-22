from __future__ import annotations

import argparse
import json
import sys

from dean_os.analyst_core.domain_analyst_lifecycle_profile import (
    DomainAnalystLifecycleProfileCompiler,
)
from dean_os.analysts._producers.sector_market import (
    SavedSectorMarketEvidenceProducer,
)


def _market_scope(domain_id: str) -> tuple[list[str], str]:
    profile = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
    measurement = (profile.get("domain_overlay") or {}).get(
        "market_measurement"
    ) or {}
    universe = sorted(
        {
            str(item).strip().upper()
            for item in measurement.get("primary_universe") or []
            if str(item).strip()
        }
    )
    benchmark = str(measurement.get("benchmark_ticker") or "").strip().upper()
    if not universe or not benchmark:
        raise ValueError("domain sector-market scope is incomplete")
    return universe, benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build review-only sector market evidence from one verified saved "
            "price-repair artifact."
        )
    )
    parser.add_argument("repair_artifact")
    parser.add_argument("--domain-id", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--lookback-sessions", type=int, default=20)
    parser.add_argument("--min-source-bars-per-day", type=int, default=24)
    parser.add_argument("--max-staleness-days", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        default="reports/dean_os/saved_sector_market_evidence_producer_current",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    universe, benchmark = _market_scope(args.domain_id)
    payload = SavedSectorMarketEvidenceProducer(args.output_dir).build(
        repair_artifact_path=args.repair_artifact,
        as_of=args.as_of,
        sector_tickers=universe,
        benchmark=benchmark,
        lookback_sessions=args.lookback_sessions,
        min_source_bars_per_day=args.min_source_bars_per_day,
        max_staleness_days=args.max_staleness_days,
        save=not args.no_save,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
