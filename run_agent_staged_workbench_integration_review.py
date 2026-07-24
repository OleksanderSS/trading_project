from __future__ import annotations

import argparse
import sys

from dean_os.staged_workbench_integration_review import (
    StagedWorkbenchIntegrationReview,
    render_staged_workbench_integration_review_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Review-only audit of staged web-bot workbench material.")
    parser.add_argument("--draft-bundle", default=None)
    parser.add_argument("--dropzone", default=None)
    parser.add_argument("--output-dir", default="reports/dean_os/staged_workbench_integration_review_current")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    field_map = {
        "draft_bundle": args.draft_bundle,
        "dropzone": args.dropzone,
    }
    kwargs = {key: value for key, value in field_map.items() if value is not None}
    kwargs["save"] = not args.no_save

    payload = StagedWorkbenchIntegrationReview(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(render_staged_workbench_integration_review_markdown(payload).replace("`", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
