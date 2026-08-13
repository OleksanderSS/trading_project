from __future__ import annotations

import argparse

from dean_os.cli_helpers import print_json
from dean_os.analyst_core.domain_analyst_lifecycle_profile import DomainAnalystLifecycleProfileReport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DomainAnalystLifecycleProfileReport (domain_analyst_lifecycle_profile).")
    parser.add_argument("--source-domain-id", default='semiconductor_ai_infrastructure')
    parser.add_argument("--clone-domain-id", default='energy')
    parser.add_argument("--template-path", default='dean_os/config/domain_analyst_lifecycle.template.json')
    parser.add_argument("--no-save", dest="save", action="store_false", help="Build the payload without writing report files.")
    parser.add_argument("--output-dir", default="reports/dean_os/domain_analyst_lifecycle_profile_current")
    parser.add_argument("--print-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = DomainAnalystLifecycleProfileReport(output_dir=args.output_dir).build(
        source_domain_id=args.source_domain_id,
        clone_domain_id=args.clone_domain_id,
        template_path=args.template_path,
        save=args.save,
    )
    if args.print_json:
        print_json(payload)
        return
    print(f"Run ID: {payload.get('run_id')}")
    for key, value in (payload.get("summary") or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            print(f"{key}: {value}")
    saved = payload.get("saved_paths") or {}
    if saved:
        print(f"Report JSON: {saved.get('latest_json') or saved.get('json')}")


if __name__ == "__main__":
    main()
