"""
Ablation experiment runner skeleton.

This file intentionally does NOT assume your pipeline API.
You connect it by implementing `run_pipeline_with_components`.

Goal:
- compare baseline vs with/without component
- measure metric_delta, runtime_delta, stability
- produce component_ablation_results.csv

Usage:
    1. Edit run_pipeline_with_components() to call your pipeline on a small offline dataset.
    2. Define components in components_to_test list or load from component_engagement.csv.
    3. Run:
        python diagnostics/ablation_experiment_runner.py --out diagnostic_reports
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class AblationResult:
    component: str
    baseline_metric: float | None
    with_component_metric: float | None
    without_component_metric: float | None
    delta_with_vs_baseline: float | None
    delta_without_vs_baseline: float | None
    runtime_seconds: float
    status: str
    notes: str


def run_pipeline_with_components(enabled_components: list[str], disabled_components: list[str]) -> dict[str, Any]:
    """Project-specific adapter.

    Replace this with a call to your actual offline pipeline:
    - use tiny local dataset
    - disable network/API/secrets
    - use deterministic model/dummy model
    - return {"primary_metric": float, "metrics": {...}}

    Until implemented, this raises NotImplementedError.
    """
    raise NotImplementedError("Connect this adapter to your offline minimal pipeline")


def evaluate_component(component: str) -> AblationResult:
    start = time.perf_counter()
    try:
        baseline = run_pipeline_with_components(enabled_components=[], disabled_components=[])
        with_comp = run_pipeline_with_components(enabled_components=[component], disabled_components=[])
        without_comp = run_pipeline_with_components(enabled_components=[], disabled_components=[component])

        b = baseline.get("primary_metric")
        w = with_comp.get("primary_metric")
        wo = without_comp.get("primary_metric")

        return AblationResult(
            component=component,
            baseline_metric=b,
            with_component_metric=w,
            without_component_metric=wo,
            delta_with_vs_baseline=(w - b) if w is not None and b is not None else None,
            delta_without_vs_baseline=(wo - b) if wo is not None and b is not None else None,
            runtime_seconds=time.perf_counter() - start,
            status="DONE",
            notes="",
        )
    except NotImplementedError as exc:
        return AblationResult(component, None, None, None, None, None, time.perf_counter() - start, "ADAPTER_NOT_IMPLEMENTED", str(exc))
    except Exception as exc:
        return AblationResult(component, None, None, None, None, None, time.perf_counter() - start, "FAILED", repr(exc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", nargs="*", default=[])
    ap.add_argument("--component-csv", default="diagnostic_reports/component_engagement.csv")
    ap.add_argument("--out", default="diagnostic_reports")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    components = list(args.components)
    csv_path = Path(args.component_csv)
    if not components and csv_path.exists():
        import csv as _csv
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(_csv.DictReader(f))
        components = [
            r["component"] for r in rows
            if r.get("category") in {"enricher", "calculator", "analyzer", "algorithm", "context_map", "selector"}
            and r.get("status") in {"ACTIVE_NEEDS_RUNTIME_CONFIRMATION", "ACTIVE_OUTPUT_UNTESTED", "UNUSED_POTENTIALLY_VALUABLE"}
        ][:50]

    results = [evaluate_component(c) for c in components]

    with (out / "component_ablation_results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(asdict(results[0]).keys()) if results else [field for field in AblationResult.__dataclass_fields__]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    summary = {"components": len(results), "status_counts": {}}
    for r in results:
        summary["status_counts"][r.status] = summary["status_counts"].get(r.status, 0) + 1
    (out / "component_ablation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
