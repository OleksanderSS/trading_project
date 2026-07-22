from __future__ import annotations

import argparse
import json
import sys

from dean_os.domain_context_set import DomainContextSetAssembler


DEFAULTS = {
    "news": "reports/dean_os/domain_scoped_news_envelope_current/latest.json",
    "official_policy": "reports/dean_os/domain_scoped_official_policy_envelope_current/latest.json",
    "macro": "reports/dean_os/domain_scoped_macro_envelope_current/latest.json",
    "fundamentals": "reports/dean_os/domain_scoped_fundamentals_envelope_current/latest.json",
    "sector_market": "reports/dean_os/domain_scoped_sector_market_envelope_current/latest.json",
    "pipeline_context": "reports/dean_os/domain_scoped_pipeline_context_envelope_current/latest.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively verify six explicit domain context envelopes and emit "
            "a review-only complete or partial DomainContextSet."
        )
    )
    parser.add_argument("domain_id")
    parser.add_argument("--analysis-cutoff", required=True)
    for family, default in DEFAULTS.items():
        parser.add_argument(
            "--" + family.replace("_", "-") + "-path",
            default=default,
        )
    parser.add_argument(
        "--journal-path", default="data/dean_os/system_journal.jsonl"
    )
    parser.add_argument(
        "--output-dir", default="reports/dean_os/domain_context_set_current"
    )
    parser.add_argument("--apply-journal", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    candidates = {
        family: getattr(args, family + "_path") for family in DEFAULTS
    }
    payload = DomainContextSetAssembler(args.output_dir).build(
        domain_id=args.domain_id,
        analysis_cutoff=args.analysis_cutoff,
        candidate_artifacts=candidates,
        journal_path=args.journal_path,
        apply_journal=args.apply_journal,
        save=not args.no_save,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
