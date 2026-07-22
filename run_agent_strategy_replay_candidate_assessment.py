import argparse
import json
import sys
from dean_os.strategy_maturity_operations import StrategyReplayCandidateAssessment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one real reviewed hypothesis as a research-only strategy candidate.")
    parser.add_argument("--review-gate", required=True, help="Path to the world model replay review gate JSON")
    parser.add_argument("--hypothesis-id", help="Optional specific hypothesis ID to evaluate")
    parser.add_argument("--no-save", action="store_true", help="Run in dry-run mode without saving outputs")
    
    args = parser.parse_args()
    
    assessment = StrategyReplayCandidateAssessment()
    try:
        result = assessment.build(
            review_gate_path=args.review_gate,
            hypothesis_id=args.hypothesis_id,
            apply_ledger=False,
            apply_journal=False,
            save=not args.no_save,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
