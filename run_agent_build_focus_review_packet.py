from __future__ import annotations

import argparse
import sys

from dean_os.packets.build_focus_review_packet import BuildFocusReviewPacket, render_build_focus_review_packet_markdown


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the focus review packet from alignment/template/case-registry/pipeline-control inputs."
    )
    parser.add_argument("--alignment-review-json", default=None)
    parser.add_argument("--template-standardization-json", default=None)
    parser.add_argument("--case-registry-json", default=None)
    parser.add_argument("--pipeline-control-instance-json", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/build_focus_review_packet")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    kwargs = {"save": not args.no_save}
    if args.alignment_review_json:
        kwargs["alignment_review_json"] = args.alignment_review_json
    if args.template_standardization_json:
        kwargs["template_standardization_json"] = args.template_standardization_json
    if args.case_registry_json:
        kwargs["case_registry_json"] = args.case_registry_json
    if args.pipeline_control_instance_json:
        kwargs["pipeline_control_instance_json"] = args.pipeline_control_instance_json

    payload = BuildFocusReviewPacket(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_build_focus_review_packet_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
