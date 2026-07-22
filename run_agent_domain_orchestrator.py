from __future__ import annotations

import argparse
import json

from dean_os.domain_orchestrator import DomainOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the review-only DEAN-OS domain orchestrator."
    )
    parser.add_argument("domain_id")
    parser.add_argument("--as-of", dest="as_of")
    parser.add_argument("--ticker", action="append", dest="tickers")
    parser.add_argument("--include-profile-agents", action="store_true")
    parser.add_argument("--context-set", dest="context_set_path")
    parser.add_argument(
        "--legacy-unbound-diagnostic",
        action="store_true",
        help="Run the pre-existing unbound diagnostic path explicitly.",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = DomainOrchestrator().run_sync(
        args.domain_id,
        as_of=args.as_of,
        tickers=args.tickers,
        include_profile_agents=args.include_profile_agents,
        context_set_path=args.context_set_path,
        allow_legacy_unbound_context=args.legacy_unbound_diagnostic,
        save=not args.no_save,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
