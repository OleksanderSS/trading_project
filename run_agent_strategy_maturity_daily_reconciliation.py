import argparse
import json
import sys
from dean_os.strategy_maturity_operations import StrategyMaturityDailyReconciler


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile a candidate playbook with the verified maturity-decision ledger.")
    parser.add_argument("--assessment", required=True, help="Path to the strategy replay candidate assessment JSON")
    parser.add_argument("--risk-snapshot", help="Optional path to the strategy risk snapshot JSON")
    parser.add_argument("--no-save", action="store_true", help="Run in dry-run mode without saving outputs")
    
    args = parser.parse_args()
    
    reconciler = StrategyMaturityDailyReconciler()
    try:
        result = reconciler.build(
            candidate_assessment_path=args.assessment,
            risk_snapshot_path=args.risk_snapshot,
            apply_journal=False,
            save=not args.no_save,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
