#!/usr/bin/env python3
"""
Deep static audit runner for trading/ML Python codebases.

Goal:
- aggregate checks that usually get duplicated across separate audit agents;
- emit normalized findings with deterministic fingerprints;
- keep checks offline and dependency-light;
- focus on high-risk trading/ML correctness patterns.

Usage:
    python audit_tools/deep_static_audit.py --root src --out audit_reports
    python audit_tools/deep_static_audit.py --root . --out audit_reports --format all

Outputs:
    findings.json
    findings.csv
    findings.md
    summary.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

PY_EXT = ".py"
EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".venv", "venv", "env", "node_modules", "dist", "build", ".tox", ".trunk"
}

SEVERITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "INFO": 4}


@dataclass(frozen=True)
class Finding:
    severity: str
    category: str
    rule_id: str
    file: str
    line: int
    symbol: str
    snippet: str
    problem: str
    why: str
    fix: str
    test: str
    confidence: str = "medium"

    @property
    def fingerprint(self) -> str:
        # Dedupe equivalent reports from different detector blocks while keeping line locality.
        # Snippet is normalized enough that whitespace-only changes do not create duplicates.
        normalized = " ".join(self.snippet.split())[:240]
        key = f"{self.category}|{self.rule_id}|{self.file}|{self.line}|{normalized}"
        return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def to_row(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint
        return d


class SourceFile:
    def __init__(self, path: Path, root: Path):
        self.path = path
        self.root = root
        self.rel = path.relative_to(root).as_posix()
        try:
            self.text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            self.text = path.read_text(encoding="latin-1", errors="ignore")
        self.lines = self.text.splitlines()
        self.ignores: dict[int, set[str]] = self._parse_ignores()
        try:
            self.tree = ast.parse(self.text, filename=self.rel)
        except SyntaxError:
            self.tree = None

    def _parse_ignores(self) -> dict[int, set[str]]:
        ignores = defaultdict(set)
        for i, line in enumerate(self.lines, start=1):
            if "# audit-ignore" in line:
                # Format: # audit-ignore: RULE_ID1, RULE_ID2 or just # audit-ignore
                parts = line.split("# audit-ignore", 1)[1].strip()
                if parts.startswith(":"):
                    rule_ids = [r.strip() for r in parts[1:].split(",") if r.strip()]
                    for r in rule_ids:
                        ignores[i].add(r)
                else:
                    ignores[i].add("*")  # Ignore all on this line
        return ignores

    def is_ignored(self, lineno: int, rule_id: str) -> bool:
        line_ignores = self.ignores.get(lineno, set())
        return "*" in line_ignores or rule_id in line_ignores

    def line(self, lineno: int) -> str:
        if lineno <= 0 or lineno > len(self.lines):
            return ""
        return self.lines[lineno - 1].strip()

    def window(self, lineno: int, before: int = 2, after: int = 2) -> str:
        start = max(1, lineno - before)
        end = min(len(self.lines), lineno + after)
        return "\n".join(f"{i}: {self.lines[i-1]}" for i in range(start, end + 1))


def iter_python_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*.py"):
        parts = set(p.parts)
        if parts.intersection(EXCLUDE_DIRS):
            continue
        yield p


def is_test_file(rel: str) -> bool:
    lower = rel.lower()
    return "/tests/" in lower or lower.startswith("tests/") or lower.endswith("_test.py") or lower.startswith("test_") or "/test_" in lower


def category_from_path(rel: str) -> set[str]:
    l = rel.lower()
    cats = set()
    for key in ["target", "feature", "pipeline", "calibration", "training", "risk", "metric", "model", "factory", "collector", "data", "security", "config", "validation", "test", "backtest"]:
        if key in l:
            cats.add(key)
    return cats


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return call_name(node.func)
    if isinstance(node, ast.Subscript):
        return call_name(node.value)
    return ""


def literal_value(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def has_keyword(call: ast.Call, name: str, value: Optional[object] = None) -> bool:
    for kw in call.keywords:
        if kw.arg == name:
            if value is None:
                return True
            return literal_value(kw.value) == value
    return False


def has_groupby_ancestor(node: ast.AST) -> bool:
    # Filled by parent links in scan_file().
    cur = getattr(node, "parent", None)
    while cur is not None:
        if isinstance(cur, ast.Call) and "groupby" in call_name(cur.func):
            return True
        cur = getattr(cur, "parent", None)
    return False


def function_context(node: ast.AST) -> str:
    cur = getattr(node, "parent", None)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return cur.name
        cur = getattr(cur, "parent", None)
    return ""


def make_finding(sf: SourceFile, node_or_line, severity: str, category: str, rule_id: str,
                 problem: str, why: str, fix: str, test: str, confidence: str = "medium",
                 snippet: Optional[str] = None) -> Finding:
    lineno = node_or_line if isinstance(node_or_line, int) else getattr(node_or_line, "lineno", 1)
    symbol = "" if isinstance(node_or_line, int) else function_context(node_or_line)
    return Finding(
        severity=severity,
        category=category,
        rule_id=rule_id,
        file=sf.rel,
        line=lineno,
        symbol=symbol,
        snippet=snippet if snippet is not None else sf.line(lineno),
        problem=problem,
        why=why,
        fix=fix,
        test=test,
        confidence=confidence,
    )


class StaticAuditRunner:
    # shift(-h) is expected in these target/label constructors; still worth surfacing,
    # but not as P0 by default.
    NEGATIVE_SHIFT_EXPECTED_FILES = {
        "pipeline/guards/temporal_target_guard.py",
        "pipeline/stages/stage_0_data_generation.py",
        "data/synthetic/data_generator.py",
    }

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.findings: list[Finding] = []
        self.imports_by_file: dict[str, set[str]] = {}
        self.model_lists: list[tuple[str, int, str, list[str]]] = []
        self.source_lists: list[tuple[str, int, str, list[str]]] = []

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def run(self) -> list[Finding]:
        files = [SourceFile(p, self.root) for p in iter_python_files(self.root)]
        for sf in files:
            self.scan_file(sf)
        self.scan_cross_file_consistency()
        
        # Filter findings based on inline ignores
        final_findings = []
        # Index SourceFiles by rel path for quick lookup
        sf_map = {sf.rel: sf for sf in files}
        for f in self.findings:
            sf = sf_map.get(f.file)
            if sf and sf.is_ignored(f.line, f.rule_id):
                continue
            final_findings.append(f)
            
        return self.deduplicate(final_findings)

    @staticmethod
    def deduplicate(findings: Iterable[Finding]) -> list[Finding]:
        by_fp: dict[str, Finding] = {}
        for f in findings:
            fp = f.fingerprint
            old = by_fp.get(fp)
            if old is None or SEVERITY_ORDER.get(f.severity, 99) < SEVERITY_ORDER.get(old.severity, 99):
                by_fp[fp] = f
        return sorted(by_fp.values(), key=lambda x: (SEVERITY_ORDER.get(x.severity, 99), x.category, x.file, x.line, x.rule_id))

    def scan_file(self, sf: SourceFile) -> None:
        if sf.tree is not None:
            for parent in ast.walk(sf.tree):
                for child in ast.iter_child_nodes(parent):
                    setattr(child, "parent", parent)
            self.scan_ast(sf)
        self.scan_regex(sf)
        self.scan_long_modules(sf)

    def scan_ast(self, sf: SourceFile) -> None:
        cats = category_from_path(sf.rel)
        imported = set()

        for node in ast.walk(sf.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
                    self.scan_import(sf, node, alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                imported.add(mod.split(".")[0])
                self.scan_import(sf, node, mod)

            if isinstance(node, ast.Call):
                name = call_name(node.func)
                lower_name = name.lower()
                self.scan_call(sf, node, name, lower_name, cats)

            if isinstance(node, ast.ExceptHandler):
                self.scan_except_handler(sf, node)

            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                self.scan_assignment(sf, node)

        self.imports_by_file[sf.rel] = imported

    def scan_import(self, sf: SourceFile, node: ast.AST, module_name: str) -> None:
        heavy = {"tensorflow", "torch", "transformers", "spacy", "pandas_ta", "talib", "yfinance"}
        root_mod = module_name.split(".")[0]
        rel = sf.rel.lower()
        if root_mod in heavy and not is_test_file(sf.rel) and self._is_module_level_import(node):
            # Importing heavy libs inside neural model file is less severe than importing them in factory/config/cli paths.
            severity = "P1" if any(k in rel for k in ["factory", "config", "cli", "pipeline", "main", "__init__"]) else "P2"
            self.add(make_finding(
                sf, node, severity, "heavy_imports", "HEAVY_TOP_LEVEL_IMPORT",
                f"Top-level import of heavy optional dependency '{module_name}'.",
                "Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.",
                "Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.",
                "Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.",
                confidence="high",
            ))

    @staticmethod
    def _is_module_level_import(node: ast.AST) -> bool:
        parent = getattr(node, "parent", None)
        if isinstance(parent, ast.Module):
            return True
        while isinstance(parent, (ast.Try, ast.If, ast.With)):
            parent = getattr(parent, "parent", None)
            if isinstance(parent, ast.Module):
                return True
        return False

    def scan_call(self, sf: SourceFile, node: ast.Call, name: str, lower_name: str, cats: set[str]) -> None:
        # Temporal correctness
        if lower_name.endswith(".shift") or lower_name == "shift":
            if node.args:
                val = literal_value(node.args[0])
                if isinstance(val, (int, float)) and val < 0:
                    severity = "P0" if ({"target", "feature", "pipeline", "calibration"} & cats) else "P1"
                    if sf.rel in self.NEGATIVE_SHIFT_EXPECTED_FILES:
                        severity = "P1"
                    confidence = "high" if not has_groupby_ancestor(node) else "medium"
                    self.add(make_finding(
                        sf, node, severity, "temporal", "NEGATIVE_SHIFT_LOOKAHEAD",
                        "Negative shift detected. It may be valid for target generation, but dangerous elsewhere.",
                        "shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.",
                        "Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.",
                        "Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.",
                        confidence=confidence,
                    ))

        if lower_name.endswith(".pct_change") or lower_name == "pct_change":
            if not has_keyword(node, "fill_method"):
                self.add(make_finding(
                    sf, node, "P1", "missing_policy", "PCT_CHANGE_IMPLICIT_FILL_METHOD",
                    "pct_change() called without explicit fill_method.",
                    "Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.",
                    "Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.",
                    "Add a test where a missing price gap does not become a zero or forward-filled return.",
                    confidence="high",
                ))

        if lower_name.endswith(".rolling") or lower_name == "rolling":
            if has_keyword(node, "center", True):
                self.add(make_finding(
                    sf, node, "P0", "temporal", "CENTERED_ROLLING_WINDOW",
                    "rolling(center=True) detected.",
                    "Centered windows use future observations for features at time t.",
                    "Use backward-looking rolling windows only; shift outputs if needed to represent availability time.",
                    "Add a causality test proving feature[t] does not change when rows after t are modified.",
                    confidence="high",
                ))

        if "merge_asof" in lower_name:
            # Missing direction means pandas default backward, but explicit is preferable in audit-critical code.
            if has_keyword(node, "direction", "forward") or has_keyword(node, "direction", "nearest"):
                self.add(make_finding(
                    sf, node, "P0", "temporal", "MERGE_ASOF_FORWARD_OR_NEAREST",
                    "merge_asof uses forward/nearest direction.",
                    "Forward/nearest joins can attach information not available at the left timestamp.",
                    "Use direction='backward' and a tolerance aligned with data availability.",
                    "Test that a macro/news value published after timestamp t is not joined to row t.",
                    confidence="high",
                ))
            elif not has_keyword(node, "direction"):
                self.add(make_finding(
                    sf, node, "P2", "temporal", "MERGE_ASOF_DIRECTION_NOT_EXPLICIT",
                    "merge_asof direction is not explicit.",
                    "Even if pandas defaults to backward, auditability is weak for temporal joins.",
                    "Set direction='backward' and document tolerance/availability policy.",
                    "Add a join fixture with one future macro/news row and assert it is not joined.",
                    confidence="medium",
                ))

        if lower_name.endswith("train_test_split") or lower_name == "train_test_split":
            is_shuffled = True
            # Default for sklearn is True, but check explicit kwarg
            for kw in node.keywords:
                if kw.arg == "shuffle":
                    val = literal_value(kw.value)
                    if val is False:
                        is_shuffled = False
            
            if is_shuffled:
                severity = "P0" if not is_test_file(sf.rel) else "P2"
                problem = "Random train_test_split detected in time-series/trading code path."
                why = "Random splits leak future regimes into train/validation and invalidate backtest-like evaluation."
            else:
                severity = "P2"
                problem = "train_test_split(shuffle=False) detected."
                why = "While chronological, this split lacks a 'purge gap' >= target horizon, which may lead to leakage if observations overlap."
            
            self.add(make_finding(
                sf, node, severity, "splits", "RANDOM_TRAIN_TEST_SPLIT",
                problem, why,
                "Use chronological or purged time split with gap >= max target horizon.",
                "Assert train max timestamp < validation min timestamp and purge gap >= target horizon.",
                confidence="high",
            ))

        # Missing policy
        if lower_name.endswith(".bfill") or lower_name == "bfill":
            self.add(make_finding(
                sf, node, "P0" if ({"feature", "target", "pipeline", "training"} & cats) else "P1", "missing_policy", "BFILL_IN_CAUSAL_DATA",
                "bfill() detected in likely causal time-series path.",
                "Backward fill moves future-known values into earlier timestamps.",
                "For train/eval causal data, remove bfill; use forward-fill only when availability policy permits, or leave NaN with indicator columns.",
                "Add a fixture where first known future value must not appear in earlier rows.",
                confidence="high",
            ))

        if lower_name.endswith(".fillna") or lower_name == "fillna":
            zero_fill = False
            if node.args and literal_value(node.args[0]) == 0:
                zero_fill = True
            for kw in node.keywords:
                if kw.arg in {"value", None} and literal_value(kw.value) == 0:
                    zero_fill = True
            if zero_fill:
                severity = "P0" if ({"target", "risk", "metric"} & cats) else "P1"
                self.add(make_finding(
                    sf, node, severity, "missing_policy", "FILLNA_ZERO_SUSPICIOUS",
                    "fillna(0) detected.",
                    "Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.",
                    "Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.",
                    "Add tests that target tails and missing price gaps are not converted to zeros.",
                    confidence="high",
                ))

        # Financial math
        if any(k in lower_name for k in ["std", "var", "cov", "corr"]):
            # broad context-only check handled with regex too; avoid too many INFOs here.
            pass

        # Security/path/artifact
        if lower_name.endswith(("pickle.load", "joblib.load", "torch.load")) or lower_name in {"pickle.load", "joblib.load", "torch.load"}:
            self.add(make_finding(
                sf, node, "P1", "security", "UNSAFE_MODEL_OR_PICKLE_LOAD",
                "Pickle/joblib/torch model load detected.",
                "These formats can execute code or load unsafe artifacts if the path is not trusted and validated.",
                "Only load from validated artifact directories; store metadata/hash and reject untrusted paths.",
                "Add tests that traversal/untrusted artifact paths are rejected before load.",
                confidence="medium",
            ))

        if lower_name.endswith(("read_csv", "read_json", "read_parquet", "open")) or lower_name in {"open", "read_csv", "read_json", "read_parquet"}:
            rel = sf.rel.lower()
            if any(k in rel for k in ["data_source", "collector", "file", "config", "security", "loader"]):
                self.add(make_finding(
                    sf, node, "P2", "security", "FILE_READ_NEEDS_PATH_VALIDATION",
                    "File read detected in config/data loading path.",
                    "User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.",
                    "Route all config paths through a single PathSecurityValidator before reading.",
                    "Test that '../secret.env' and absolute paths outside allowed base are rejected.",
                    confidence="medium",
                ))

    def scan_except_handler(self, sf: SourceFile, node: ast.ExceptHandler) -> None:
        catches_exception = False
        if node.type is None:
            catches_exception = True
        elif isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
            catches_exception = True
        elif isinstance(node.type, ast.Tuple):
            catches_exception = any(isinstance(e, ast.Name) and e.id in {"Exception", "BaseException"} for e in node.type.elts)
        if not catches_exception:
            return

        has_logger_error = False
        returns_silent = []
        raises = False
        returns_sample = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = call_name(child.func).lower()
                if any(name.endswith(x) for x in ["logger.error", "logger.exception", "logging.error", "logging.exception"]):
                    has_logger_error = True
                if "sample" in name or "synthetic" in name or "demo" in name or "mock" in name:
                    returns_sample = True
            if isinstance(child, ast.Raise):
                raises = True
            if isinstance(child, ast.Return):
                val = child.value
                if val is None:
                    returns_silent.append("None")
                elif isinstance(val, ast.Constant) and val.value is None:
                    returns_silent.append("None")
                elif isinstance(val, (ast.Dict, ast.List, ast.Tuple, ast.Set)) and len(getattr(val, "elts", getattr(val, "keys", [])) or []) == 0:
                    returns_silent.append(type(val).__name__)
                elif isinstance(val, ast.Call) and any(k in call_name(val.func).lower() for k in ["sample", "synthetic", "demo", "mock"]):
                    returns_sample = True

        if returns_sample:
            self.add(make_finding(
                sf, node, "P0", "synthetic_gates", "EXCEPTION_FALLS_BACK_TO_SAMPLE_DATA",
                "Exception handler appears to return sample/synthetic/demo data.",
                "A failed real collector can silently inject fake data into train/eval.",
                "Make sample fallback opt-in and mark data_kind/is_synthetic/eligible_for_training=False.",
                "Simulate collector failure and assert it raises or returns failed status unless allow_sample_fallback=True.",
                confidence="medium",
            ))
        if returns_silent:
            self.add(make_finding(
                sf, node, "P1", "error_policy", "BROAD_EXCEPTION_SILENT_RETURN",
                f"Broad exception returns silent fallback: {', '.join(sorted(set(returns_silent)))}.",
                "Pipeline may continue with None/{}/[] as if the stage succeeded.",
                "Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.",
                "Test that fatal failures in target generation/split/training do not continue silently.",
                confidence="high",
            ))
        if has_logger_error and raises:
            self.add(make_finding(
                sf, node, "P2", "error_policy", "LOGGER_ERROR_THEN_RAISE",
                "Exception is logged and re-raised in the same handler.",
                "If upper layers also log, this creates duplicate error reports and noisy traces.",
                "Log only at boundary layers, or add context and re-raise without error-level logging.",
                "Add a test/logger capture for one error event per failing operation.",
                confidence="medium",
            ))

    def scan_assignment(self, sf: SourceFile, node: ast.AST) -> None:
        targets = []
        value = None
        lineno = getattr(node, "lineno", 1)
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is None:
            return

        target_names = [call_name(t).lower() for t in targets]
        names_joined = " ".join(target_names)
        val = literal_value(value)

        if isinstance(val, list) and val and all(isinstance(x, str) for x in val):
            lname = names_joined
            if any(k in lname for k in ["model", "models", "alias", "aliases"]):
                self.model_lists.append((sf.rel, lineno, names_joined, val))
                if len(val) >= 3:
                    self.add(make_finding(
                        sf, node, "P2", "config_factory", "HARDCODED_MODEL_LIST",
                        "Hardcoded model list detected.",
                        "Duplicated model lists across factory/arena/pipeline drift over time.",
                        "Move models/aliases/capabilities into one registry/config and reference it everywhere.",
                        "Test that factory, CLI, arena, and prediction stage resolve the same registry entries.",
                        confidence="medium",
                    ))
            if any(k in lname for k in ["source", "sources", "collector", "collectors"]):
                self.source_lists.append((sf.rel, lineno, names_joined, val))

        if isinstance(val, dict) and val:
            keys = list(val.keys()) if all(isinstance(k, str) for k in val.keys()) else []
            lname = names_joined
            if keys and any(k in lname for k in ["model", "alias", "factory", "registry", "map"]):
                self.model_lists.append((sf.rel, lineno, names_joined, keys))
                if len(keys) >= 3:
                    self.add(make_finding(
                        sf, node, "P2", "config_factory", "HARDCODED_MODEL_MAP_OR_ALIASES",
                        "Hardcoded model map/alias registry detected.",
                        "Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.",
                        "Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.",
                        "Snapshot-test that all model names and aliases resolve from one source of truth.",
                        confidence="medium",
                    ))

    def scan_regex(self, sf: SourceFile) -> None:
        text = sf.text
        cats = category_from_path(sf.rel)

        # chained target fill after future shift within a short window
        future_shift_re = re.compile(r"shift\s*\(\s*-\s*\d+[^\)]*\)", re.I)
        for m in future_shift_re.finditer(text):
            line = text[:m.start()].count("\n") + 1
            window = "\n".join(sf.lines[max(0, line-1): min(len(sf.lines), line+8)])
            if re.search(r"\.(ffill|bfill|fillna)\s*\(", window):
                self.add(make_finding(
                    sf, line, "P0", "temporal", "FUTURE_SHIFT_THEN_FILL",
                    "Future shift is followed by fill operation in nearby lines.",
                    "Tail rows after future target shift have no real label; filling them fabricates labels.",
                    "Drop/mark tail horizon rows; never ffill/bfill/fillna target columns.",
                    "Assert the last horizon rows for each ticker are NaN or absent after target generation.",
                    confidence="high",
                    snippet=window,
                ))

        # autoencoder primary/fallback/routing
        for i, line in enumerate(sf.lines, start=1):
            low = line.lower()
            if "autoencoder" in low and any(k in low for k in ["predict", "primary", "fallback", "models", "alias", "return"]):
                self.add(make_finding(
                    sf, i, "P1", "model_routing", "AUTOENCODER_ROUTING_REVIEW",
                    "Autoencoder appears in model routing/prediction context.",
                    "Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.",
                    "Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.",
                    "Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.",
                    confidence="medium",
                ))

            # Sharpe/std zero nearby
            if "sharpe" in low or "sortino" in low:
                window = "\n".join(sf.lines[max(0, i-18): min(len(sf.lines), i+14)]).lower()
                calculation_window = "\n".join(sf.lines[max(0, i-2): min(len(sf.lines), i+3)]).lower()
                std_denominator = re.search(
                    r"/\s*\(?\s*(?:[\w\.]+\.std\s*\(|np\.std\s*\(|\w*std\w*\b|volatility\b|tracking_error\b|port_vol(?:atility)?\b|portfolio_volatility\b)",
                    calculation_window,
                )
                std_guard = re.search(
                    r"(?:np\.isfinite\s*\([^\)]*(?:std|volatility|tracking_error|port_vol)|"
                    r"not\s+np\.isfinite\s*\([^\)]*(?:std|volatility|tracking_error|port_vol)|"
                    r"(?:\w*std\w*|volatility|tracking_error|port_vol(?:atility)?|portfolio_volatility)\s*(?:==|<=|<|!=|>)\s*(?:0(?:\.0)?|1e-\d+)|"
                    r"np\.isclose\s*\([^\)]*(?:std|volatility|tracking_error|port_vol))",
                    window,
                )
                if std_denominator and not std_guard:
                    self.add(make_finding(
                        sf, i, "P1", "financial_math", "SHARPE_SORTINO_STD_ZERO_REVIEW",
                        "Sharpe/Sortino calculation near std usage without obvious zero-std guard.",
                        "Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.",
                        "Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.",
                        "Test constant returns and single-observation returns.",
                        confidence="low",
                    ))

            if "max_drawdown" in low or "drawdown" in low:
                if re.search(r"np\.min|min\(", low) or "current_drawdown" in low:
                    self.add(make_finding(
                        sf, i, "P2", "financial_math", "DRAWDOWN_SIGN_CONVENTION_REVIEW",
                        "Drawdown calculation found; sign convention needs review.",
                        "Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.",
                        "Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.",
                        "Test monotonic loss series and assert documented sign convention.",
                        confidence="low",
                    ))

            if "var" in low and any(k in low for k in ["percentile", "quantile", "return {'var': 0", 'return {"var": 0']):
                self.add(make_finding(
                    sf, i, "P1", "financial_math", "VAR_SIGN_OR_EMPTY_DATA_REVIEW",
                    "VaR percentile/zero-return pattern found.",
                    "VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.",
                    "Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.",
                    "Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.",
                    confidence="medium",
                ))

            if "datetime.now" in low or "pd.timestamp.now" in low or "date.today" in low:
                self.add(make_finding(
                    sf, i, "P3" if not is_test_file(sf.rel) else "P2", "determinism", "NON_INJECTED_CLOCK",
                    "Direct current-time call detected.",
                    "Runtime and tests become nondeterministic; relative dates can drift.",
                    "Inject a clock/reference_now parameter or central time provider.",
                    "Freeze clock in tests and assert stable outputs.",
                    confidence="medium",
                ))

            if any(token in low for token in ["requests.", "yfinance", "newsapi", "huggingface", "openai", "download"]):
                if is_test_file(sf.rel):
                    self.add(make_finding(
                        sf, i, "P1", "determinism", "NETWORK_IN_TEST_PATH",
                        "Potential network/API usage in test-like path.",
                        "Offline deterministic tests should not depend on network, APIs, or live market data.",
                        "Mock external APIs and use versioned fixtures.",
                        "Run tests with network blocked and assert no request is made.",
                        confidence="medium",
                    ))

            if any(k in low for k in ["feature", "enrich", "indicator"]) and any(k in low for k in ["df[", "df.", "assign("]):
                # This is intentionally low severity: prompts lineage review instead of declaring bug.
                if any(k in sf.rel.lower() for k in ["features", "enrich", "indicator"]):
                    window = "\n".join(sf.lines[max(0, i-2): min(len(sf.lines), i+4)]).lower()
                    if not any(k in window for k in ["lineage", "availability", "source", "data_kind", "metadata"]):
                        self.add(make_finding(
                            sf, i, "P3", "data_lineage", "FEATURE_WITHOUT_LOCAL_LINEAGE_HINT",
                            "Feature/enricher code updates data without nearby lineage/availability metadata.",
                            "Trading features need source, ticker/timeframe granularity, calculation window, and availability time.",
                            "Add feature manifest entries or emit lineage metadata from each enricher.",
                            "Test that every emitted feature has source, window, granularity, availability_time, causal flag.",
                            confidence="low",
                        ))

        # env loading and secrets
        if re.search(r"load_dotenv|dotenv_values|\.env", text):
            for i, line in enumerate(sf.lines, start=1):
                if ".env" in line or "load_dotenv" in line or "dotenv_values" in line:
                    self.add(make_finding(
                        sf, i, "P2", "security", "ENV_LOADING_REVIEW",
                        ".env loading/search path detected.",
                        "Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.",
                        "Make search paths explicit per environment and keep os.environ priority unless override=True.",
                        "Test that parent/home .env is not loaded in production/test mode.",
                        confidence="medium",
                    ))

        placeholder_re = re.compile(r"your[_-]?api[_-]?key|your[_-]?token|changeme|placeholder|dummy[_-]?key", re.I)
        sensitive_re = re.compile(r"api[_-]?key|token|secret|password|credential|bearer", re.I)
        placeholder_handler_re = re.compile(
            r"def\s+_?\w*placeholders?|_resolve_placeholders|_has_placeholders|"
            r"placeholders?\s*=|for\s+placeholder\b|resolved_placeholder|"
            r"contains .*placeholder|template placeholder|security protocol breach",
            re.I,
        )
        if placeholder_re.search(text):
            for i, line in enumerate(sf.lines, start=1):
                clean_line = line.strip()
                if clean_line.startswith(("#", '"""', "'''", "*", "'''", '"""')):
                    continue
                if not placeholder_re.search(line):
                    continue
                if placeholder_handler_re.search(line):
                    continue
                strong_fake_secret = re.search(r"your[_-]?api[_-]?key|your[_-]?token|changeme|dummy[_-]?key", line, re.I)
                placeholder_secret_value = (
                    "placeholder" in line.lower()
                    and sensitive_re.search(line)
                    and re.search(r"[:=]", line)
                )
                if strong_fake_secret or placeholder_secret_value:
                    self.add(make_finding(
                        sf, i, "P1", "security", "PLACEHOLDER_SECRET_REVIEW",
                        "Placeholder-looking secret/default detected.",
                        "Placeholders can be mistaken for valid credentials or leak into production config.",
                        "Validate secrets at startup and reject known placeholder patterns.",
                        "Test that placeholder secrets fail validation.",
                        confidence="medium",
                    ))

    def scan_long_modules(self, sf: SourceFile) -> None:
        loc = sum(1 for line in sf.lines if line.strip() and not line.strip().startswith("#"))
        if loc >= 700:
            self.add(make_finding(
                sf, 1, "P3", "architecture", "LONG_MODULE_REVIEW",
                f"Long module detected: {loc} non-comment LOC.",
                "Large modules tend to mix responsibilities, but mechanical splitting before tests is risky.",
                "Add characterization tests first, then split by data collection/validation/features/targets/split/training/evaluation/reporting.",
                "Characterization test should compare key outputs before and after refactor.",
                confidence="high",
                snippet=sf.lines[0].strip() if sf.lines else "",
            ))
        if sf.tree is not None:
            for node in ast.walk(sf.tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    if len(methods) >= 25:
                        self.add(make_finding(
                            sf, node, "P3", "architecture", "GOD_CLASS_REVIEW",
                            f"Class '{node.name}' has {len(methods)} methods.",
                            "God classes hide responsibilities and make fatal/non-fatal error policy hard to enforce.",
                            "Before splitting, add characterization tests; then extract cohesive services by responsibility.",
                            "Test public behavior of the class before extraction.",
                            confidence="high",
                        ))

    def scan_cross_file_consistency(self) -> None:
        # Model lists/maps duplicated in several files with overlapping entries.
        if len(self.model_lists) >= 2:
            normalized_sets = []
            for file, line, name, items in self.model_lists:
                norm = {str(x).lower().replace("_", "") for x in items}
                normalized_sets.append((file, line, name, norm, items))
            for idx, (file, line, name, norm, items) in enumerate(normalized_sets):
                overlaps = []
                for j, (file2, line2, name2, norm2, items2) in enumerate(normalized_sets):
                    if idx == j:
                        continue
                    inter = norm & norm2
                    if len(inter) >= 2:
                        overlaps.append(f"{file2}:{line2}")
                if overlaps:
                    # Fake SourceFile unavailable here; create directly.
                    snippet = f"{name} = {items[:10]}{'...' if len(items) > 10 else ''}"
                    f = Finding(
                        severity="P2",
                        category="config_factory",
                        rule_id="DUPLICATED_MODEL_REGISTRY_ENTRIES",
                        file=file,
                        line=line,
                        symbol="",
                        snippet=snippet,
                        problem="Model/alias registry entries overlap with other files.",
                        why="Duplicated registries drift and cause selector/factory/prediction inconsistencies.",
                        fix="Move all model names, aliases, class paths, role, heavy flag, and can_be_primary to one registry.",
                        test="Snapshot-test that factory, CLI, arena, and prediction load the same registry.",
                        confidence="medium",
                    )
                    self.add(f)
                    break


def write_outputs(findings: list[Finding], out: Path, formats: set[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [f.to_row() for f in findings]
    summary = {
        "total_findings": len(rows),
        "by_severity": dict(Counter(r["severity"] for r in rows)),
        "by_category": dict(Counter(r["category"] for r in rows)),
        "by_rule": dict(Counter(r["rule_id"] for r in rows)),
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if "json" in formats or "all" in formats:
        (out / "findings.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    if "csv" in formats or "all" in formats:
        fieldnames = list(rows[0].keys()) if rows else ["fingerprint", "severity", "category", "rule_id", "file", "line", "problem"]
        with (out / "findings.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if "md" in formats or "all" in formats:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            grouped[r["severity"]].append(r)
        md = []
        md.append("# Deep Static Audit Findings\n")
        md.append("## Summary\n")
        md.append(f"Total findings: **{len(rows)}**\n")
        md.append("### By severity\n")
        for sev in sorted(summary["by_severity"], key=lambda s: SEVERITY_ORDER.get(s, 99)):
            md.append(f"- {sev}: {summary['by_severity'][sev]}")
        md.append("\n### By category\n")
        for cat, count in sorted(summary["by_category"].items(), key=lambda x: (-x[1], x[0])):
            md.append(f"- {cat}: {count}")
        md.append("\n---\n")
        for sev in ["P0", "P1", "P2", "P3", "INFO"]:
            items = grouped.get(sev, [])
            if not items:
                continue
            md.append(f"\n## {sev}\n")
            for r in items:
                md.append(f"### {r['category']} / {r['rule_id']} — `{r['file']}:{r['line']}`")
                md.append(f"**Problem:** {r['problem']}")
                md.append(f"**Why:** {r['why']}")
                md.append(f"**Fix:** {r['fix']}")
                md.append(f"**Test:** {r['test']}")
                md.append(f"**Confidence:** {r['confidence']}  ")
                md.append("```python")
                md.append(r["snippet"][:1200])
                md.append("```\n")
        (out / "findings.md").write_text("\n".join(md), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deep static audit runner for trading/ML codebases")
    parser.add_argument("--root", required=True, help="Root directory to scan, usually src or project root")
    parser.add_argument("--out", default="audit_reports", help="Output directory")
    parser.add_argument("--format", choices=["json", "csv", "md", "all"], default="all")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        return 2

    runner = StaticAuditRunner(root)
    findings = runner.run()
    formats = {args.format}
    write_outputs(findings, Path(args.out).resolve(), formats)

    counts = Counter(f.severity for f in findings)
    print(f"Scanned: {root}")
    print(f"Findings: {len(findings)}")
    for sev in ["P0", "P1", "P2", "P3", "INFO"]:
        if counts.get(sev):
            print(f"  {sev}: {counts[sev]}")
    print(f"Reports written to: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
