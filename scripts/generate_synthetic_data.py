#!/usr/bin/env python3
"""
CLI script for generating synthetic market data for stress testing.

Usage:
    # Generate all built-in scenarios (1000 paths each)
    python scripts/generate_synthetic_data.py

    # Generate specific scenarios
    python scripts/generate_synthetic_data.py --scenarios typical flash_crash black_swan

    # Generate augmentation data from real CSV
    python scripts/generate_synthetic_data.py --augment data/raw/AAPL_1d.csv --ratio 0.1

    # Custom parameters
    python scripts/generate_synthetic_data.py --scenarios flash_crash --paths 500 --horizon 126 --seed 123

Available scenarios:
    typical          - Normal market conditions (GBM)
    high_volatility  - 2x elevated volatility regime
    flash_crash      - 10% drop over 5 bars, partial recovery
    black_swan       - 20% crash in 1 bar, extreme vol
    liquidity_crisis - Gradual 15% decline over 20 bars
    bull_run         - Strong upward momentum, low vol
    bear_market      - Sustained downtrend, high vol
    sector_rotation  - Choppy sideways, vol spikes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.data.collectors.synthetic_generator import (
    BUILTIN_SCENARIOS,
    GeneratorConfig,
    SyntheticGenerator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic market data for stress testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        choices=list(BUILTIN_SCENARIOS.keys()),
        help="Scenarios to generate (default: all)",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=1000,
        help="Number of Monte Carlo paths per scenario (default: 1000)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=252,
        help="Simulation horizon in trading days (default: 252 = 1 year)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--augment",
        type=str,
        default=None,
        help="Path to a real CSV file to generate augmentation data from",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="Augmentation ratio (fraction of real data size, default: 0.1)",
    )
    parser.add_argument(
        "--calibrate-from",
        type=str,
        default=None,
        help="Path to a real CSV file to calibrate mu/sigma from",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GeneratorConfig(
        n_paths=args.paths,
        horizon_days=args.horizon,
        random_seed=args.seed,
        synthetic_ratio=args.ratio,
    )

    # Standalone mode — no BaseCollector dependencies needed
    generator = SyntheticGenerator(config=config)

    # Load calibration data if provided
    base_df = None
    if args.calibrate_from:
        calibration_path = Path(args.calibrate_from)
        if calibration_path.exists():
            base_df = pd.read_csv(calibration_path)
            print(f"📊 Loaded calibration data from {calibration_path} ({len(base_df)} rows)")
        else:
            print(f"⚠️  Calibration file not found: {calibration_path}")

    # --- Mode 1: Augmentation ---
    if args.augment:
        augment_path = Path(args.augment)
        if not augment_path.exists():
            print(f"❌ File not found: {augment_path}")
            sys.exit(1)

        real_df = pd.read_csv(augment_path)
        print(f"📊 Loaded real data: {len(real_df)} rows from {augment_path}")

        synthetic_df = generator.generate_augmentation_data(real_df, ratio=args.ratio)
        out_path = output_dir / f"{augment_path.stem}_augmented.csv"
        synthetic_df.to_csv(out_path, index=False)
        print(f"✅ Augmentation data saved: {out_path} ({len(synthetic_df)} rows)")
        return

    # --- Mode 2: Scenario generation (Monte Carlo stress test) ---
    scenario_names = args.scenarios or list(BUILTIN_SCENARIOS.keys())
    print(f"\n🎲 Generating {args.paths} Monte Carlo paths for {len(scenario_names)} scenarios")
    print(f"   Horizon: {args.horizon} trading days | Seed: {args.seed}\n")

    all_results = generator.generate_scenarios(
        scenario_names=scenario_names,
        base_df=base_df,
    )

    summary_report = {}

    for scenario_name, paths in all_results.items():
        # Save combined CSV (all paths concatenated)
        combined_df = pd.concat(paths, ignore_index=True)
        csv_path = output_dir / f"scenario_{scenario_name}.csv"
        combined_df.to_csv(csv_path, index=False)

        # Compute summary
        stats = generator.summarise_paths(paths)
        summary_report[scenario_name] = stats

        print(f"  📁 {scenario_name}:")
        print(f"     Paths: {stats['n_paths']}")
        print(f"     Mean Return: {stats['mean_return']:.2%}")
        print(f"     VaR 95%:     {stats['var_95']:.2%}")
        print(f"     VaR 99%:     {stats['var_99']:.2%}")
        print(f"     Worst DD:    {stats['worst_max_drawdown']:.2%}")
        print(f"     P(loss):     {stats['prob_loss']:.1%}")
        print(f"     Saved to:    {csv_path}")
        print()

    # Save summary report as JSON
    report_path = output_dir / "stress_test_report.json"
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"📋 Summary report saved: {report_path}")

    print("\n✅ Synthetic data generation complete!")
    print(f"   Files are in: {output_dir.resolve()}")
    print(f"\n   To load into pipeline via CSV Collector, configure:")
    print(f'   collectors_config.json → {{"type": "custom_csv", "file_path": "<path_to_csv>"}}')


if __name__ == "__main__":
    main()
