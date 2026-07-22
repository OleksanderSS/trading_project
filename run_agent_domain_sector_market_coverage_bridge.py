from __future__ import annotations

import argparse
import json
import sys

from dean_os.domain_sector_market_coverage_bridge import (
    DomainSectorMarketCoverageBridge,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one clean market snapshot against one domain's exact "
            "universe and emit a repair-compatible coverage candidate."
        )
    )
    parser.add_argument("domain_id")
    parser.add_argument("--analysis-cutoff", required=True)
    parser.add_argument("--snapshot-manifest", required=True)
    parser.add_argument("--min-rows", type=int, default=180)
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--max-abs-return", type=float, default=0.25)
    parser.add_argument("--min-cadence-ratio", type=float, default=0.75)
    parser.add_argument(
        "--output-dir",
        default="reports/dean_os/domain_sector_market_coverage_bridge_current",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    payload = DomainSectorMarketCoverageBridge(args.output_dir).build(
        domain_id=args.domain_id,
        analysis_cutoff=args.analysis_cutoff,
        snapshot_manifest_path=args.snapshot_manifest,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        max_abs_return=args.max_abs_return,
        min_cadence_ratio=args.min_cadence_ratio,
        save=not args.no_save,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
