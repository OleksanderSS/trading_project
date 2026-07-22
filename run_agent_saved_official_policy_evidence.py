import argparse
import sys
from pathlib import Path

from dean_os.analysts._producers.policy import (
    SavedOfficialPolicyEvidenceProducer,
    DEFAULT_REGISTRY,
)

def main():
    parser = argparse.ArgumentParser(
        description="Bind one official policy source to independent news corroboration."
    )
    parser.add_argument(
        "snapshot_artifact_path",
        help="Path to the official policy snapshot artifact",
    )
    parser.add_argument(
        "corroborating_news_artifact_path",
        help="Path to the corroborating news artifact",
    )
    parser.add_argument(
        "--as-of",
        required=True,
        help="Time boundary (ISO 8601 timezone-aware)",
    )
    parser.add_argument(
        "--registry-path",
        default=DEFAULT_REGISTRY,
        help="Path to the official policy evidence registry",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/dean_os/saved_official_policy_evidence_producer",
        help="Output directory for the artifact",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without saving the artifact",
    )

    args = parser.parse_args()

    producer = SavedOfficialPolicyEvidenceProducer(output_dir=args.output_dir)
    try:
        payload = producer.build(
            snapshot_artifact_path=args.snapshot_artifact_path,
            corroborating_news_artifact_path=args.corroborating_news_artifact_path,
            as_of=args.as_of,
            registry_path=args.registry_path,
            save=not args.no_save,
        )
        if not args.no_save:
            print(f"Successfully generated official policy artifact.")
            print(f"Run ID: {payload.get('run_id')}")
        else:
            print("Successfully verified official policy (no-save).")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
