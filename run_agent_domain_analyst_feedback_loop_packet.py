from __future__ import annotations

import argparse
import json
import sys

from dean_os.analyst_core.domain_analyst_feedback_loop_packet import DomainAnalystFeedbackLoopPacket, render_domain_analyst_feedback_loop_packet_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the domain analyst feedback loop packet.")
    parser.add_argument("--case-registry-json", default=None)
    parser.add_argument("--forecast-review-json", default=None)
    parser.add_argument("--profile-policy-json", default=None)
    parser.add_argument("--template-decision-json", default=None)
    parser.add_argument("--manual-feedback-json", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/domain_analyst_feedback_loop_packet_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "case_registry_json": args.case_registry_json,
        "forecast_review_json": args.forecast_review_json,
        "profile_policy_json": args.profile_policy_json,
        "template_decision_json": args.template_decision_json,
        "manual_feedback_json": args.manual_feedback_json,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = DomainAnalystFeedbackLoopPacket(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_domain_analyst_feedback_loop_packet_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
