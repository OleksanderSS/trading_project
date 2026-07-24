"""Standalone CLI for the empirical champion selector.

The actual selection logic (`select_champions`) lives in
src/pipeline/hybrid/champion_selector.py, shared with the live pipeline --
ResultsProcessor.build_models_metadata() calls that same module's
`filter_to_champions` to hard-filter Stage 5's models_metadata down to one
champion per (ticker, target) before prediction ever runs. This script is
only for offline inspection of a colab_results.json without running the
full pipeline (e.g. right after a Colab training batch comes back, before
deciding whether to run `--mode continue`).

See champion_selector.py's module docstring for the full rationale: why
this is separate from src/models/model_selector/model_competence_map.json
(a static prior, not empirical), the comparison rule per target_type, and
why a (ticker, target) group with no comparable metric is reported as
`no_champion` rather than given a fabricated winner.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.target_type_registry import load_target_types  # noqa: E402
from src.pipeline.hybrid.champion_selector import select_champions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pick an empirical per-(ticker, target) champion model from a Colab training run's colab_results.json."
    )
    parser.add_argument("results_path", help="Path to colab_results.json")
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the champion registry JSON (default: <results_dir>/champion_selection.json)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_path)
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    models_metadata = results.get("models_metadata", {})
    if not models_metadata:
        print("⚠️ No models_metadata found in results file")
        return 1

    target_types = load_target_types()
    champions = select_champions(models_metadata, target_types)

    selected = sum(1 for c in champions.values() if c["status"] == "champion_selected")
    unresolved = len(champions) - selected
    print(f"📊 {len(champions)} (ticker, target) groups: {selected} champion(s) selected, {unresolved} without a comparable metric")

    output_path = Path(args.output) if args.output else results_path.parent / "champion_selection.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_results": str(results_path),
                "batch_name": results.get("batch_name"),
                "results_timestamp": results.get("timestamp"),
                "champions": champions,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"✅ Written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
