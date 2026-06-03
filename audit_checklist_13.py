#!/usr/bin/env python3
"""
Audit Checklist (13 points)

Goal: fast, offline, static-ish checks that map findings to the 13 audit areas
we use for prioritization (correctness + stability/speed).

This is not a proof. It is a triage tool.

Usage:
  python audit_checklist_13.py --root src
  python audit_checklist_13.py --root src --json --output audit_13.json
  python audit_checklist_13.py --root src --max-issues 200
  python audit_checklist_13.py --root src --only Temporal,NaN,Errors
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}

SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}


CHECKLIST_AREAS = [
    "Temporal",      # lookahead/shift(-)/target alignment
    "NaN",           # fill/drop policy, pct_change defaults
    "Synthetic",     # gates and metadata for synthetic/sample data
    "Selection",     # model/feature selection fallbacks, dead routing
    "Metrics",       # sharpe/std=0, drawdown, annualization, aggregation
    "Async",         # await/async misuse, timeouts, blocking calls
    "Security",      # path traversal, secrets, unsafe eval/exec
    "Imports",       # heavy imports on import path
    "DeadCode",      # unused/abandoned modules (heuristics only)
    "Errors",        # silent exceptions, bare except, pass
    "Config",        # conflicting keys, hardcoded lists vs config
    "Tests",         # network usage, nondeterminism
    "Complexity",    # god modules / very long files (heuristic)
]


@dataclass
class Finding:
    area: str
    severity: str
    file: str
    line: int
    message: str
    code: str = ""
    hint: str = ""


def iter_py_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        yield path


def _severity_max(a: str, b: str) -> str:
    return a if SEVERITY_ORDER[a] <= SEVERITY_ORDER[b] else b


def scan_text(path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    rel = str(path).replace("\\", "/")

    def add(area: str, severity: str, line: int, message: str, code: str = "", hint: str = ""):
        findings.append(Finding(area=area, severity=severity, file=rel, line=line, message=message, code=code, hint=hint))

    # NaN: pct_change without explicit fill_method
    for m in re.finditer(r"\.pct_change\(([^)]*)\)", text):
        line = text[: m.start()].count("\n") + 1
        line_text = text.splitlines()[line - 1] if line - 1 < len(text.splitlines()) else ""
        if line_text.strip().startswith("#") or "# audit-ignore" in line_text:
            continue
        args = m.group(1)
        if "fill_method" not in args:
            add(
                "NaN",
                "HIGH",
                line,
                "pct_change() without explicit fill_method can forward-fill by default (pandas).",
                code=line_text.strip(),
                hint="Prefer pct_change(fill_method=None) and decide fill/drop explicitly.",
            )

    # Temporal: suspicious shift(-N) (context-dependent)
    for m in re.finditer(r"\.shift\(\s*-\s*\d+\s*\)", text):
        line = text[: m.start()].count("\n") + 1
        line_text = text.splitlines()[line - 1] if line - 1 < len(text.splitlines()) else ""
        if line_text.strip().startswith("#") or "# audit-ignore" in line_text:
            continue
        add(
            "Temporal",
            "MEDIUM",
            line,
            "shift(-N) indicates future access; OK for target construction, risky if used in features.",
            code=line_text.strip(),
            hint="Verify it is only used for targets/labels and never fed into model features.",
        )

    # Errors: bare except / except Exception with pass
    for m in re.finditer(r"^\s*except\s*:\s*$", text, flags=re.MULTILINE):
        line = text[: m.start()].count("\n") + 1
        if "# audit-ignore" in text.splitlines()[line - 1]:
            continue
        add(
            "Errors",
            "HIGH",
            line,
            "Bare except: catches BaseException; often hides bugs.",
            code=text.splitlines()[line - 1].strip(),
            hint="Catch specific exceptions; log with context; avoid swallowing errors.",
        )

    for m in re.finditer(r"^\s*except\s+Exception\s*:\s*$", text, flags=re.MULTILINE):
        line = text[: m.start()].count("\n") + 1
        if "# audit-ignore" in text.splitlines()[line - 1]:
            continue
        add(
            "Errors",
            "MEDIUM",
            line,
            "except Exception: without binding/logging can hide root cause.",
            code=text.splitlines()[line - 1].strip(),
            hint="Bind as 'e' and log; consider re-raise vs fallback.",
        )

    # Security: eval/exec
    # Avoid false positives:
    # - `.eval(` is a common ML API (torch modules), not Python's builtin eval.
    # - `def eval(` is just a method name (still potentially confusing, but not eval()).
    for m in re.finditer(r"(?<!\.)\b(eval|exec)\s*\(", text):
        line = text[: m.start()].count("\n") + 1
        line_text = text.splitlines()[line - 1] if line - 1 < len(text.splitlines()) else ""
        stripped = line_text.lstrip()
        # Ignore mentions in comments/docstrings.
        if stripped.startswith("#") or stripped.startswith(("'''", '"""')):
            continue
        if "without using eval" in line_text.lower():
            continue
        if re.match(r"^\s*def\s+(eval|exec)\s*\(", line_text):
            continue
        if "# audit-ignore" in line_text:
            continue
        add(
            "Security",
            "CRITICAL",
            line,
            f"Use of {m.group(1)}() is high risk.",
            code=line_text.strip(),
            hint="Avoid eval/exec; use safe parsers or controlled dispatch tables.",
        )

    # Imports: heavyweight libraries at module import time (heuristic)
    # Only match third-party imports, not local modules that happen to share names.
    heavy = ("torch", "tensorflow", "transformers", "spacy", "pandas_ta")
    for i, line_text in enumerate(text.splitlines(), start=1):
        if line_text.startswith("import ") or line_text.startswith("from "):
            for lib in heavy:
                if not re.search(rf"^\s*(import|from)\s+{re.escape(lib)}\b", line_text):
                    continue
                # Ignore local absolute imports like "from src...." even if they contain the token.
                if re.search(r"^\s*from\s+src\b", line_text):
                    continue
                    add(
                        "Imports",
                        "MEDIUM",
                        i,
                        f"Heavy import '{lib}' at module import time can slow startup/tests.",
                        code=line_text.strip(),
                        hint="Prefer lazy imports inside functions or optional-dependency guards.",
                    )

    # Complexity: very long files (heuristic)
    line_count = text.count("\n") + 1
    if line_count >= 1200:
        add(
            "Complexity",
            "LOW",
            1,
            f"Very large module ({line_count} lines). Consider splitting after behavior is test-locked.",
            hint="Refactor only with tests; split by responsibility boundaries.",
        )

    return findings


def scan_ast(path: Path, tree: ast.AST, text: str) -> list[Finding]:
    findings: list[Finding] = []
    rel = str(path).replace("\\", "/")

    def add(area: str, severity: str, node: ast.AST, message: str, code: str = "", hint: str = ""):
        line = getattr(node, "lineno", 1) or 1
        findings.append(Finding(area=area, severity=severity, file=rel, line=line, message=message, code=code, hint=hint))

    # Errors: "except: pass" / empty handlers
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                if h.type is None:
                    add("Errors", "HIGH", h, "Bare except handler in try/except.", hint="Avoid catching BaseException.")
                if len(h.body) == 1 and isinstance(h.body[0], ast.Pass):
                    add("Errors", "HIGH", h, "Exception handler contains only 'pass' (silent swallow).", hint="Log or re-raise; avoid silent failure.")

    # Tests: requests/http usage in tests (heuristic)
    if "/tests/" in rel or rel.startswith("tests/"):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name in ("requests", "httpx"):
                        add("Tests", "MEDIUM", node, f"Test imports '{n.name}' (risk of network).", hint="Mock network; avoid live calls in unit tests.")
            if isinstance(node, ast.ImportFrom) and node.module in ("requests", "httpx"):
                add("Tests", "MEDIUM", node, f"Test imports from '{node.module}' (risk of network).", hint="Mock network; avoid live calls in unit tests.")

    return findings


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Audit Checklist (13 points)")
    ap.add_argument("--root", default="src")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--output", default="")
    ap.add_argument("--max-issues", type=int, default=200)
    ap.add_argument("--only", default="", help="Comma-separated areas, e.g. Temporal,NaN,Errors")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        return 2

    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else set()
    findings: list[Finding] = []
    file_count = 0

    for path in iter_py_files(root):
        file_count += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            findings.append(Finding("Errors", "LOW", str(path), 1, f"Failed to read file: {e}"))
            continue

        file_findings = scan_text(path, text)

        try:
            tree = ast.parse(text)
            file_findings.extend(scan_ast(path, tree, text))
        except SyntaxError:
            findings.append(Finding("Errors", "HIGH", str(path).replace("\\", "/"), 1, "SyntaxError: file does not parse."))

        if only:
            file_findings = [f for f in file_findings if f.area in only]

        findings.extend(file_findings)
        if len(findings) >= args.max_issues:
            break

    # Sort for stable output: severity then file/line
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 99), f.file, f.line, f.area))

    payload: dict[str, Any] = {
        "root": str(root).replace("\\", "/"),
        "file_count_scanned": file_count,
        "issues": [asdict(f) for f in findings[: args.max_issues]],
    }

    if args.json or args.output:
        out = json.dumps(payload, indent=2, ensure_ascii=True)
        if args.output:
            Path(args.output).write_text(out, encoding="utf-8")
        else:
            print(out)
        return 0 if not findings else 1

    # Text output
    print(f"Audit(13): root={payload['root']} files={file_count} issues={len(payload['issues'])}")
    for f in payload["issues"]:
        print(f"[{f['severity']}] [{f['area']}] {f['file']}:{f['line']} {f['message']}")
        if f.get("code"):
            print(f"  > {f['code']}")
        if f.get("hint"):
            print(f"  - {f['hint']}")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
