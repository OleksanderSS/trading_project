#!/usr/bin/env python3
"""
Clear accumulated pipeline DATA so the next prepare run starts fresh.

This used to shutil.rmtree the whole batch directory, which does not hold
only data. It also holds every trained model, its metrics sidecar,
colab_results.json and light_models_results.json -- and those went with it,
silently, from a script whose name says "data".

That is what happened between 2026-08-04 and 2026-08-06: the training
results from two Colab runs were gone by the time anyone looked for them,
and the only surviving copy was a backup from May.

The asymmetry is the whole point. features.parquet and targets.parquet are
DERIVED -- `--mode prepare` rebuilds them in about an hour. Trained models
are not. They cost hours of Colab GPU time, and their metrics cannot be
recomputed at all without retraining, because a metric is a measurement of
one particular fit.

So training artifacts are MOVED ASIDE, never deleted, and the script says
what it did with each.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

#: Things that took real compute to produce. Matched against file names in
#: each directory being cleared.
TRAINING_ARTIFACT_PATTERNS = (
    "model_*",                     # the models themselves
    "*.metrics.json",              # their measured quality -- unrecoverable
    "colab_results*.json",
    "light_models_results*.json",
    "final_results_*.json",
    "selected_features_*.json",
)

DATA_PATHS = [
    "data/colab/accumulated/main_database",
    "data/colab/accumulated/processed",
    "data/colab/accumulated/raw",
    "data/colab/accumulated/temp",
]


def _rescue_training_artifacts(path: Path, rescue_root: Path) -> int:
    """Move anything expensive out of `path` before it is removed."""
    rescued = 0
    for pattern in TRAINING_ARTIFACT_PATTERNS:
        for item in path.glob(pattern):
            destination = rescue_root / path.name / item.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(destination))
            rescued += 1
    return rescued


def clear_old_data(confirm: bool = True) -> None:
    """Remove derived data; preserve anything that cost compute."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rescue_root = Path("data/colab/rescued") / stamp

    present = [Path(p) for p in DATA_PATHS if Path(p).exists()]
    if not present:
        print("Nothing to clear.")
        return

    print("About to clear:")
    for path in present:
        # Counted as a SET. A file can match more than one pattern --
        # model_X.keras.metrics.json is both `model_*` and `*.metrics.json`
        # -- and summing the globs reported six artifacts where five
        # existed. A number shown to justify a destructive action has to be
        # the real one.
        artifacts = len({
            item for pattern in TRAINING_ARTIFACT_PATTERNS
            for item in path.glob(pattern)
        })
        note = f"  ({artifacts} training artifact(s) will be MOVED, not deleted)" if artifacts else ""
        print(f"  {path}{note}")

    if confirm and sys.stdin is not None and sys.stdin.isatty():
        # Asked only when a human is there to answer, so a piped or
        # scheduled run keeps the protective behaviour without hanging.
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted. Nothing was removed.")
            return

    total_rescued = 0
    for path in present:
        rescued = _rescue_training_artifacts(path, rescue_root)
        total_rescued += rescued
        shutil.rmtree(path)
        print(f"  removed {path}" + (f", rescued {rescued} artifact(s)" if rescued else ""))

    cache_path = Path("data/cache")
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"  removed {cache_path}")

    print("\n✅ Derived data cleared.")
    if total_rescued:
        print(f"📦 {total_rescued} training artifact(s) kept in {rescue_root}")
        print("   Move them back into the batch directory if you want the next")
        print("   Colab run to skip retraining those models.")
    print("\nNext steps:")
    print("1. Run: python run_hybrid_pipeline.py --mode prepare")
    print("2. Verify data spans only 60 days")
    print("3. Check 15m data has sufficient rows")


if __name__ == "__main__":
    clear_old_data(confirm="--yes" not in sys.argv)
