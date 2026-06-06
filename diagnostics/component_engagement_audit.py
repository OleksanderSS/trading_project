"""
Component Engagement Audit v4.

Purpose:
- discover enrichers, calculators, analyzers, algorithms, context maps, selectors, validators, collectors, factories, models
- check static engagement: imported, referenced by strings/config/registry/factory, likely entrypoint reachability
- estimate output behavior from source: added columns, target leakage risk, calculate/enrich/analyze method availability
- classify each component:
    ACTIVE_LIKELY
    REGISTERED_NOT_IMPORTED
    IMPORTED_NOT_REGISTERED
    ORPHAN_POTENTIALLY_VALUABLE
    ORPHAN_LOW_SIGNAL
    ACTIVE_RISKY
    NEEDS_RUNTIME_CONFIRMATION

Run:
    python diagnostics/component_engagement_audit.py --root src --configs configs . --out diagnostic_reports
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import defaultdict, Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CATEGORY_BY_PATH = [
    ("enricher", ["features/enrichers", "feature_engineering", "features/builders"]),
    ("calculator", ["calculators", "metrics"]),
    ("analyzer", ["analyzers", "analysis"]),
    ("algorithm", ["algorithms", "optimization"]),
    ("context_map", ["context", "context_map", "maps"]),
    ("validator", ["validators", "validation", "guards"]),
    ("collector", ["collectors", "data_sources"]),
    ("selector", ["selector", "selection"]),
    ("detector", ["detector", "detectors"]),
    ("model", ["models"]),
    ("factory_registry", ["factory", "factories", "registry"]),
    ("risk", ["risk", "risk_management"]),
    ("pipeline_stage", ["pipeline/stages", "orchestrator", "stage_"]),
    ("trading", ["trading", "backtesting"]),
]

CATEGORY_BY_NAME = [
    ("enricher", ["Enricher", "FeatureBuilder"]),
    ("calculator", ["Calculator", "Metric", "Metrics"]),
    ("analyzer", ["Analyzer", "Analysis"]),
    ("algorithm", ["Algorithm", "Optimizer", "Allocator", "Sizer"]),
    ("context_map", ["Context", "Map"]),
    ("validator", ["Validator", "Guard", "Checker"]),
    ("collector", ["Collector", "DataSource", "Fetcher"]),
    ("selector", ["Selector"]),
    ("detector", ["Detector"]),
    ("model", ["Model", "Estimator", "Predictor"]),
    ("factory_registry", ["Factory", "Registry"]),
    ("pipeline_stage", ["Stage", "Orchestrator", "Pipeline"]),
]

METHOD_NAMES = {
    "enricher": ["enrich", "transform", "build_features", "add_features"],
    "calculator": ["calculate", "compute", "fit_transform"],
    "analyzer": ["analyze", "run", "evaluate"],
    "algorithm": ["run", "optimize", "allocate", "size", "fit"],
    "context_map": ["build", "map", "transform", "get_context", "analyze"],
    "validator": ["validate", "check"],
    "collector": ["collect", "fetch", "load"],
    "selector": ["select", "fit", "transform"],
    "detector": ["detect", "fit_predict"],
    "model": ["fit", "predict"],
    "risk": ["calculate", "evaluate", "check"],
    "pipeline_stage": ["run", "execute", "process"],
}

HIGH_VALUE_CATEGORIES = {
    "enricher", "calculator", "analyzer", "algorithm", "context_map",
    "validator", "selector", "detector", "risk", "pipeline_stage"
}

RISK_PATTERNS = [
    ("P0", "TARGET_COLUMN_IN_NON_TARGET_COMPONENT", re.compile(r"target_[A-Za-z0-9_]*|f['\"]target_"), "Non-target component may create/use target columns."),
    ("P0", "FUTURE_SHIFT", re.compile(r"\.shift\(\s*-\s*\w*|\bshift\s*=\s*-\d+"), "Future shift requires ticker grouping and horizon handling."),
    ("P0", "BFILL", re.compile(r"\.bfill\(|method\s*=\s*['\"]bfill['\"]"), "Backfill can leak future data."),
    ("P1", "FILLNA_ZERO", re.compile(r"\.fillna\(\s*0\s*\)"), "fillna(0) can hide missing targets/returns/risk."),
    ("P1", "SYNTHETIC_PRIMARY", re.compile(r"combined_metric\s*=|synthetic_metric"), "Synthetic/stress metric may affect primary score."),
    ("P1", "HEAVY_IMPORT", re.compile(r"^import tensorflow|^from tensorflow|^import torch|^from torch|^import transformers|^from transformers", re.M), "Heavy import may be on lightweight path."),
    ("P2", "SILENT_EMPTY_RETURN", re.compile(r"return\s+(None|\{\}|\[\])"), "Silent empty return can hide failure."),
]


@dataclass
class ComponentRecord:
    component: str
    category: str
    file: str
    class_or_function: str
    kind: str
    public_methods: str
    expected_methods_found: str
    imported_by_count: int
    referenced_by_count: int
    reachable_from_entrypoint: bool
    registered_or_config_referenced: bool
    added_columns_static: str
    risk_count: int
    risk_rules: str
    has_test_reference: bool
    status: str
    recommended_action: str


@dataclass
class Edge:
    source: str
    target: str
    kind: str
    line: int


def iter_py(root: Path) -> list[Path]:
    return [
        p for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and not p.name.endswith(".pyc")
    ]


def module_name(root: Path, p: Path) -> str:
    rel = p.relative_to(root).with_suffix("")
    return ".".join(("src", *rel.parts)) if root.name == "src" else ".".join(rel.parts)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_ast(path: Path):
    text = read_text(path)
    try:
        return text, ast.parse(text)
    except SyntaxError:
        return text, None


def guess_category(file_rel: str, symbol: str) -> str:
    low = file_rel.replace("\\", "/").lower()
    for cat, parts in CATEGORY_BY_PATH:
        if any(p.lower() in low for p in parts):
            return cat
    for cat, parts in CATEGORY_BY_NAME:
        if any(p in symbol for p in parts):
            return cat
    return "utility_or_unknown"


def extract_import_edges(tree: ast.AST, source_rel: str) -> list[Edge]:
    edges = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(Edge(source_rel, alias.name, "import", node.lineno))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                edges.append(Edge(source_rel, f"{mod}.{alias.name}" if mod else alias.name, "from", node.lineno))
    return edges


def class_methods(node: ast.ClassDef) -> list[str]:
    return [
        n.name for n in node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def public_functions(tree: ast.AST) -> list[str]:
    return [
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")
    ]


def extract_added_columns(text: str) -> list[str]:
    patterns = [
        re.compile(r"df\s*\[\s*['\"]([^'\"]+)['\"]\s*\]\s*="),
        re.compile(r"df_enriched\s*\[\s*['\"]([^'\"]+)['\"]\s*\]\s*="),
        re.compile(r"features\s*\[\s*['\"]([^'\"]+)['\"]\s*\]\s*="),
        re.compile(r"\.assign\(([^)]*)\)"),
    ]
    cols = []
    for rx in patterns[:3]:
        cols.extend(rx.findall(text))
    for m in patterns[3].finditer(text):
        body = m.group(1)
        cols.extend(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", body))
    return sorted(set(cols))


def scan_risks(text: str, file_rel: str, category: str) -> list[tuple[str, str, str]]:
    findings = []
    for severity, rule, rx, why in RISK_PATTERNS:
        if rule == "TARGET_COLUMN_IN_NON_TARGET_COMPONENT" and (file_rel.startswith("targets/") or category == "target_calculator"):
            continue
        if rule == "HEAVY_IMPORT" and "models/neural" in file_rel.replace("\\", "/"):
            # Heavy imports inside neural model modules are expected; the problem is factory/top-level paths.
            continue
        if rx.search(text):
            findings.append((severity, rule, why))
    return findings


def load_config_texts(paths: list[Path]) -> dict[str, str]:
    out = {}
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            out[str(path)] = read_text(path)
            continue
        for p in path.rglob("*"):
            # ✅ FIX: skip non-config dirs to avoid scanning entire project
            skip_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules",
                        "data", "logs", "cache", ".mypy_cache", ".ruff_cache",
                        "backups", "trained_models", "mlruns", "diagnostic_reports"}
            if any(part in skip_dirs for part in p.parts):
                continue
            if p.suffix.lower() in {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}:
                out[str(p)] = read_text(p)
    return out


def build_reachability(files: list[str], edges: list[Edge], module_by_file: dict[str, str]) -> set[str]:
    """Fast reachability: mark files directly imported by entrypoints (1-hop only for speed)."""
    entry_tokens = ["cli/", "main/", "pipeline/", "orchestrator", "run_", "stage_"]
    entrypoints = set(f for f in files if any(t in f.lower() for t in entry_tokens))

    # Build simple source→target dict
    imports: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        imports[e.source].add(e.target)

    # Only 2-hop BFS for speed (full BFS over 600+ files × 6000+ edges is too slow)
    reachable = set(entrypoints)
    frontier = set(entrypoints)
    for _ in range(2):
        next_frontier: set[str] = set()
        for src in frontier:
            for tgt in imports.get(src, set()):
                # Find matching file
                for f, mod in module_by_file.items():
                    stem = Path(f).stem
                    if (tgt == mod or tgt.endswith("." + stem) or tgt == stem) and f not in reachable:
                        reachable.add(f)
                        next_frontier.add(f)
        frontier = next_frontier
        if not frontier:
            break
    return reachable


def status_and_action(category: str, imported_by: int, referenced_by: int, reachable: bool, registered: bool, risk_count: int, has_test: bool, added_cols: list[str]) -> tuple[str, str]:
    engaged = imported_by > 0 or referenced_by > 0 or reachable or registered

    if engaged and risk_count > 0:
        return "ACTIVE_RISKY", "Review/fix risks before trusting component output; add contract/golden tests."
    if engaged and added_cols and not has_test:
        return "ACTIVE_OUTPUT_UNTESTED", "Component appears to produce output columns; add lineage + correctness tests."
    if engaged and has_test:
        return "ACTIVE_WITH_TEST_REFERENCE", "Keep; verify runtime output reaches downstream model/evaluation."
    if engaged:
        return "ACTIVE_NEEDS_RUNTIME_CONFIRMATION", "Use runtime/lineage tracker to confirm execution and output usage."

    if category in HIGH_VALUE_CATEGORIES:
        return "UNUSED_POTENTIALLY_VALUABLE", "Do not delete; inspect logic and consider integration experiment/ablation."
    return "ORPHAN_LOW_SIGNAL", "Likely low-priority orphan; quarantine only after config/runtime confirmation."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="src")
    ap.add_argument("--configs", nargs="*", default=["configs", "."])
    ap.add_argument("--tests", default="tests")
    ap.add_argument("--out", default="diagnostic_reports")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    py_files = iter_py(root)
    file_rels = [str(p.relative_to(root)).replace("\\", "/") for p in py_files]
    module_by_file = {str(p.relative_to(root)).replace("\\", "/"): module_name(root, p) for p in py_files}

    all_edges = []
    file_text = {}
    file_tree = {}
    file_symbols = defaultdict(list)
    # ✅ PERF FIX: build per-file token sets instead of one giant string
    file_tokens: dict[str, set[str]] = {}
    # ✅ PERF FIX: use regex instead of full ast.walk for symbol/edge extraction
    CLASS_RE = re.compile(r'^class\s+(\w+)', re.M)
    FUNC_RE = re.compile(r'^def\s+(\w+)', re.M)
    IMPORT_RE = re.compile(r'^(?:from\s+([\w.]+)\s+import\s+([\w,\s*]+)|import\s+([\w.,\s]+))', re.M)

    for p in py_files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        text = read_text(p)
        file_text[rel] = text
        file_tree[rel] = None  # skip full AST
        file_tokens[rel] = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', text))

        # Extract classes with their methods (regex-based, much faster than ast.walk)
        classes_in_file = CLASS_RE.findall(text)
        for cls_name in classes_in_file:
            # Find methods belonging to this class (indented def)
            method_pattern = re.compile(
                rf'class\s+{re.escape(cls_name)}[^:]*:(.*?)(?=\nclass\s|\Z)',
                re.DOTALL
            )
            m = method_pattern.search(text)
            methods = []
            if m:
                methods = re.findall(r'^\s+def\s+(\w+)', m.group(1), re.M)
            file_symbols[rel].append(("class", cls_name, methods))

        # Top-level functions
        for func_name in FUNC_RE.findall(text):
            if not func_name.startswith("_"):
                file_symbols[rel].append(("function", func_name, []))

        # Extract import edges (regex-based)
        for from_mod, names, plain in IMPORT_RE.findall(text):
            if from_mod:
                for name in re.split(r',\s*', names.strip()):
                    name = name.strip()
                    if name and name != '*':
                        all_edges.append(Edge(rel, f"{from_mod}.{name}", "from", 0))
            elif plain:
                for mod in re.split(r',\s*', plain.strip()):
                    mod = mod.strip()
                    if mod:
                        all_edges.append(Edge(rel, mod, "import", 0))

    imported_by = defaultdict(set)
    # ✅ PERF FIX: build stem→file index first, then O(1) lookup per edge
    stem_to_file: dict[str, str] = {Path(f).stem: f for f in module_by_file}
    mod_to_file: dict[str, str] = {mod: f for f, mod in module_by_file.items()}
    for e in all_edges:
        tgt_stem = e.target.split(".")[-1]
        target_file = stem_to_file.get(tgt_stem) or mod_to_file.get(e.target)
        if target_file:
            imported_by[target_file].add(e.source)

    # ✅ PERF FIX: build global token→files index instead of scanning all_project_text per symbol
    token_to_files: dict[str, set[str]] = defaultdict(set)
    for f, tokens in file_tokens.items():
        for tok in tokens:
            token_to_files[tok].add(f)

    config_texts = load_config_texts([Path("src/config")])  # ✅ FIX: scan only config dir, not entire project
    config_text_combined = "\n".join(config_texts.values())
    config_tokens = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', config_text_combined))

    test_text = ""
    test_root = Path(args.tests)
    if test_root.exists():
        test_text = "\n".join(read_text(p) for p in test_root.rglob("*.py") if "__pycache__" not in p.parts)
    test_tokens = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', test_text))

    reachable = build_reachability(file_rels, all_edges, module_by_file)

    records = []
    for rel in file_rels:
        symbols = file_symbols.get(rel) or [("module", Path(rel).stem, [])]
        text = file_text[rel]
        added_cols = extract_added_columns(text)

        # ✅ PERF FIX: skip files with no interesting symbols to avoid processing every utility file
        cat_check = guess_category(rel, Path(rel).stem)
        has_interesting = (
            cat_check in HIGH_VALUE_CATEGORIES
            or any(guess_category(rel, s) in HIGH_VALUE_CATEGORIES for _, s, _ in symbols if _ == "class")
        )
        if not has_interesting and len(symbols) > 10:
            continue

        for kind, symbol, methods in symbols:
            category = guess_category(rel, symbol)
            expected = METHOD_NAMES.get(category, [])
            found_expected = sorted(set(methods) & set(expected)) if kind == "class" else ([symbol] if symbol in expected else [])

            risks = scan_risks(text, rel, category)
            risk_rules = sorted(set(r[1] for r in risks))

            referenced_by_count = 0
            registered = False
            search_tokens = {symbol, Path(rel).stem, module_by_file[rel].split(".")[-1]}
            for token in search_tokens:
                if not token:
                    continue
                # Count files referencing this token (excluding itself)
                refs = token_to_files.get(token, set())
                referenced_by_count += len(refs - {rel})
                if token in config_tokens:
                    registered = True

            has_test = any(tok in test_tokens for tok in search_tokens if tok)

            status, action = status_and_action(
                category=category,
                imported_by=len(imported_by.get(rel, set())),
                referenced_by=referenced_by_count,
                reachable=rel in reachable,
                registered=registered,
                risk_count=len(risks),
                has_test=has_test,
                added_cols=added_cols,
            )

            records.append(ComponentRecord(
                component=f"{module_by_file[rel]}.{symbol}" if kind != "module" else module_by_file[rel],
                category=category,
                file=rel,
                class_or_function=symbol,
                kind=kind,
                public_methods=";".join(methods),
                expected_methods_found=";".join(found_expected),
                imported_by_count=len(imported_by.get(rel, set())),
                referenced_by_count=referenced_by_count,
                reachable_from_entrypoint=rel in reachable,
                registered_or_config_referenced=registered,
                added_columns_static=";".join(added_cols),
                risk_count=len(risks),
                risk_rules=";".join(risk_rules),
                has_test_reference=has_test,
                status=status,
                recommended_action=action,
            ))

    with (out / "component_engagement.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(asdict(records[0]).keys()) if records else [field.name for field in ComponentRecord.__dataclass_fields__.values()]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    with (out / "component_import_edges.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(asdict(all_edges[0]).keys()) if all_edges else ["source", "target", "kind", "line"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in all_edges:
            writer.writerow(asdict(e))

    counts = Counter(r.status for r in records)
    cats = Counter(r.category for r in records)
    summary = {
        "components": len(records),
        "python_files": len(py_files),
        "status_counts": dict(counts),
        "category_counts": dict(cats),
        "reachable_files": len(reachable),
        "import_edges": len(all_edges),
    }
    (out / "component_engagement_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Component Engagement Audit", "", "## Status counts"]
    for k, v in counts.most_common():
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("## Category counts")
    for k, v in cats.most_common():
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("## How to use")
    md.extend([
        "1. Review `ACTIVE_RISKY` first.",
        "2. Review `ACTIVE_OUTPUT_UNTESTED` for enrichers/calculators/analyzers.",
        "3. Review `UNUSED_POTENTIALLY_VALUABLE` before deleting anything.",
        "4. Add runtime lineage tracking to prove output reaches model/evaluation.",
    ])
    (out / "component_engagement_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
