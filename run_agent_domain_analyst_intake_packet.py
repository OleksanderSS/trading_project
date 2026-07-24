from __future__ import annotations

import argparse
import json
import sys

from dean_os.analyst_core.domain_analyst_intake_packet import DomainAnalystIntakePacket, render_domain_analyst_intake_packet_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the domain analyst intake packet.")
    parser.add_argument("--evidence-pack-json", default=None)
    parser.add_argument("--source-gate-json", default=None)
    parser.add_argument("--domain-id", default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--sectors", nargs="+", default=None)
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/domain_analyst_intake_packet")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "evidence_pack_json": args.evidence_pack_json,
        "source_gate_json": args.source_gate_json,
        "domain_id": args.domain_id,
        "tickers": args.tickers,
        "sectors": args.sectors,
        "horizon_days": args.horizon_days,
        "as_of": args.as_of,
        "max_items": args.max_items,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = DomainAnalystIntakePacket(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_domain_analyst_intake_packet_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
