from __future__ import annotations

import argparse
import json
import sys

from dean_os.analysts._producers.macro import SavedMacroEvidenceProducer


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build review-only macro evidence from a saved long-form FRED "
            "snapshot (series_id/datetime/value/realtime_start rows -- e.g. "
            "the FredCollector output at data/processed/features/macro_data.parquet)."
        )
    )
    parser.add_argument("source_path")
    parser.add_argument("--as-of", required=True)
    parser.add_argument(
        "--registry-path",
        default=None,
        help="Override dean_os/config/macro_series_registry.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/dean_os/saved_macro_evidence_producer_current",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    kwargs = {"source_path": args.source_path, "as_of": args.as_of, "save": not args.no_save}
    if args.registry_path:
        kwargs["registry_path"] = args.registry_path

    payload = SavedMacroEvidenceProducer(args.output_dir).build(**kwargs)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
