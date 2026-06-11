"""
Component harness runner.

Purpose:
- try to instantiate component classes with no args or simple config
- run expected methods on a tiny deterministic dataframe
- capture added/removed/modified columns, row count changes, NaN/inf, target_* leakage

This will not run every component. Many need dependencies/configs.
Skipped components are reported with reason.

Run:
    python diagnostics/component_harness_runner.py --root src --out diagnostic_reports
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


METHOD_CANDIDATES = ["enrich", "calculate", "compute", "analyze", "detect", "select", "validate", "transform", "run"]


@dataclass
class HarnessResult:
    component: str
    module: str
    class_name: str
    method: str
    status: str
    row_count_before: int
    row_count_after: int
    added_columns: str
    removed_columns: str
    modified_columns_count: int
    warnings: str
    error: str


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["A"] * 8 + ["B"] * 8,
        "timestamp": list(pd.date_range("2024-01-01", periods=8, freq="D")) * 2,
        "open": [100,101,102,103,104,105,106,107, 200,199,198,197,196,195,194,193],
        "high": [102,103,104,105,106,107,108,109, 202,201,200,199,198,197,196,195],
        "low": [99,100,101,102,103,104,105,106, 198,197,196,195,194,193,192,191],
        "close": [101,102,103,104,105,106,107,108, 199,198,197,196,195,194,193,192],
        "volume": [1000,1100,1200,1300,1400,1500,1600,1700, 2000,2100,2200,2300,2400,2500,2600,2700],
    }).sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def module_name_from_path(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ".".join(("src", *rel.parts)) if root.name == "src" else ".".join(rel.parts)


def discover_classes(root: Path):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        mod_name = module_name_from_path(root, path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        # fast filter
        if not any(token in text for token in ["class ", "Enricher", "Calculator", "Analyzer", "Detector", "Selector", "Validator"]):
            continue
        try:
            import ast
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if any(token in node.name for token in ["Enricher", "Calculator", "Analyzer", "Detector", "Selector", "Validator", "Guard", "Algorithm", "Context", "Map"]):
                    yield mod_name, node.name


def instantiate(cls):
    # Try no args, then common config patterns.
    attempts = [
        lambda: cls(),
        lambda: cls(config={}),
        lambda: cls({}),
    ]
    last = None
    for fn in attempts:
        try:
            return fn()
        except Exception as exc:
            last = exc
    raise last or RuntimeError("Cannot instantiate")


def call_method(obj, method_name: str, df: pd.DataFrame):
    method = getattr(obj, method_name)
    sig = inspect.signature(method)
    params = sig.parameters

    kwargs = {}
    args = []

    # Most component methods accept df/data as first arg.
    if len(params) == 0:
        return method()

    first = next(iter(params.values()))
    if first.kind in (first.POSITIONAL_ONLY, first.POSITIONAL_OR_KEYWORD):
        args.append(df.copy())
    elif "df" in params:
        kwargs["df"] = df.copy()
    elif "data" in params:
        kwargs["data"] = df.copy()

    # common optional params
    if "base_col" in params:
        kwargs["base_col"] = "close"
    if "target_col" in params:
        kwargs["target_col"] = "close"
    if "returns_col" in params:
        kwargs["returns_col"] = "close"
    if "shift" in params:
        kwargs["shift"] = -1
    if "horizon" in params:
        kwargs["horizon"] = 1

    return method(*args, **kwargs)


def compare(before: pd.DataFrame, after: Any):
    if not isinstance(after, pd.DataFrame):
        return [], [], 0, ["NON_DATAFRAME_OUTPUT"]

    before_cols = set(before.columns)
    after_cols = set(after.columns)
    added = sorted(after_cols - before_cols)
    removed = sorted(before_cols - after_cols)
    modified = 0
    for col in before_cols & after_cols:
        try:
            if not before[col].equals(after[col]):
                modified += 1
        except Exception:
            modified += 1

    warnings = []
    if len(before) != len(after):
        warnings.append("ROW_COUNT_CHANGED")
    if any(c.startswith("target_") for c in added):
        warnings.append("TARGET_COLUMN_ADDED")
    for col in added:
        try:
            if after[col].isna().mean() > 0.5:
                warnings.append(f"HIGH_NAN_RATIO:{col}")
        except Exception:
            pass
    return added, removed, modified, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="src")
    ap.add_argument("--out", default="diagnostic_reports")
    ap.add_argument("--timeout", type=int, default=15, help="Seconds per component")
    ap.add_argument("--max-components", type=int, default=150)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if str(Path(".").resolve()) not in sys.path:
        sys.path.insert(0, str(Path(".").resolve()))

    df = make_sample_df()
    results: list[HarnessResult] = []

    # Only test high-value categories to avoid hanging on heavy models
    IMPORTANT_KEYWORDS = ["Enricher", "Calculator", "Analyzer", "Detector",
                          "Selector", "Validator", "Guard", "Checker"]
    SKIP_MODULES = {"neural", "tensorflow", "torch", "colab", "huggingface",
                    "roberta", "transformers", "heavy"}

    candidates = []
    for mod_name, class_name in discover_classes(root):
        if any(skip in mod_name.lower() for skip in SKIP_MODULES):
            continue
        if any(kw in class_name for kw in IMPORTANT_KEYWORDS):
            candidates.append((mod_name, class_name))
        if len(candidates) >= args.max_components:
            break

    print(f"Testing {len(candidates)} components (timeout={args.timeout}s each)...")

    import subprocess as _sub

    worker = str(Path("diagnostics/_harness_worker.py").resolve())

    for mod_name, class_name in candidates:
        comp = f"{mod_name}.{class_name}"
        try:
            proc = _sub.run(
                [sys.executable, worker, mod_name, class_name],
                capture_output=True, text=True, timeout=args.timeout
            )
            last_line = [l for l in proc.stdout.strip().splitlines() if l.strip()]
            if last_line:
                try:
                    r = json.loads(last_line[-1])
                    results.append(HarnessResult(
                        r.get("comp", comp), r.get("mod", mod_name), r.get("cls", class_name),
                        r.get("method", ""), r.get("status", "UNKNOWN"),
                        df.shape[0], r.get("after_len", -1),
                        r.get("added", ""), r.get("removed", ""),
                        r.get("modified", 0), r.get("warnings", ""), r.get("error", "")
                    ))
                except Exception as e:
                    results.append(HarnessResult(comp, mod_name, class_name, "", "PARSE_ERROR",
                        df.shape[0], -1, "", "", 0, "", str(e)[:100]))
            else:
                err = (proc.stderr or "no output")[-200:]
                results.append(HarnessResult(comp, mod_name, class_name, "", "IMPORT_FAILED",
                    df.shape[0], -1, "", "", 0, "", err))
        except _sub.TimeoutExpired:
            results.append(HarnessResult(comp, mod_name, class_name, "", "TIMEOUT",
                df.shape[0], -1, "", "", 0, f"timeout>{args.timeout}s", ""))
        except Exception as exc:
            results.append(HarnessResult(comp, mod_name, class_name, "", "HARNESS_ERROR",
                df.shape[0], -1, "", "", 0, "", repr(exc)[:100]))

    with (out / "component_harness_results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(asdict(results[0]).keys()) if results else list(HarnessResult.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    summary: dict[str, int] = {}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1
    (out / "component_harness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
if __name__ == "__main__":
    main()
