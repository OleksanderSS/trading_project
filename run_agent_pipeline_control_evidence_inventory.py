from __future__ import annotations

import argparse
import sys

from dean_os.pipeline_control.pipeline_control_evidence_inventory import (
    PipelineControlEvidenceInventory,
    render_pipeline_control_evidence_inventory_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory real local pipeline outputs as metric evidence candidates.")
    parser.add_argument("--candidate-paths", nargs="+", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/pipeline_control_evidence_inventory_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    kwargs = {"save": not args.no_save}
    if args.candidate_paths is not None:
        kwargs["candidate_paths"] = args.candidate_paths

    payload = PipelineControlEvidenceInventory(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_pipeline_control_evidence_inventory_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
