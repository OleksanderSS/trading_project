from __future__ import annotations

import argparse
import json
import sys

from dean_os.analyst_core.domain_analyst_event_interpretation_packet import DomainAnalystEventInterpretationPacket, render_domain_analyst_event_interpretation_packet_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the domain analyst event interpretation packet.")
    parser.add_argument("--evidence-pack-json", default=None)
    parser.add_argument("--pipeline-context-json", default=None)
    parser.add_argument("--domain-id", default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/domain_analyst_event_interpretation_packet_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "evidence_pack_json": args.evidence_pack_json,
        "pipeline_context_json": args.pipeline_context_json,
        "domain_id": args.domain_id,
        "max_events": args.max_events,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = DomainAnalystEventInterpretationPacket(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_domain_analyst_event_interpretation_packet_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
