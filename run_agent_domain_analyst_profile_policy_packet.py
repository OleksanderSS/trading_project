from __future__ import annotations

import argparse
import json
import sys

from dean_os.analyst_core.domain_analyst_profile_policy_packet import DomainAnalystProfilePolicyPacket, render_domain_analyst_profile_policy_packet_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the domain analyst profile policy packet.")
    parser.add_argument("--output-dir", default="reports/dean_os/domain_analyst_profile_policy_packet_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    payload = DomainAnalystProfilePolicyPacket(args.output_dir).build(save=not args.no_save)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_domain_analyst_profile_policy_packet_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
