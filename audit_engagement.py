#!/usr/bin/env python3
"""
--------------------------------------------------------------------------------
|              DEAN ENGAGEMENT & COVERAGE AUDITOR  v1.0                        |
|   Перевіряє user engagement, explainability, monitoring, test coverage       |
--------------------------------------------------------------------------------

Запуск:
    python audit_engagement.py --root src
    python audit_engagement.py --root src --json --output engagement_report.json
    python audit_engagement.py --root src --category ENG,EXP,MON,TEST,DOC

Категорії:
  [ENG]   User Engagement: user feedback loops, interactive components, config options
  [EXP]   Explainability: model explanation methods, feature importance, decision logging
  [MON]   Monitoring: alerting systems, performance metrics tracking, anomaly detection
  [TEST]  Test Coverage: integration tests, E2E tests, performance tests
  [DOC]   Documentation: user docs, API docs, architecture docs
"""

import ast
import collections
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEVERITY = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
SEVERITY_EMOJI = {
    "CRITICAL": "[CRITICAL]", "HIGH": "[HIGH]", "MEDIUM": "[MEDIUM]",
    "LOW": "[LOW]", "INFO": "[INFO]",
}
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env",
    "node_modules", ".mypy_cache", ".pytest_cache",
    "dist", "build",
}


@dataclass
class Issue:
    category: str
    severity: str
    file:     str
    line:     int
    message:  str
    code:     str = ""
    fix_hint: str = ""

    def __str__(self) -> str:
        emoji = SEVERITY_EMOJI.get(self.severity, "[INFO]")
        parts = [f"{emoji} [{self.severity}] [{self.category}] {self.file}:{self.line}"]
        parts.append(f"   {self.message}")
        if self.code:
            parts.append(f"   > {self.code.strip()[:120]}")
        if self.fix_hint:
            parts.append(f"   - {self.fix_hint}")
        return "\n".join(parts)


@dataclass
class EngagementResult:
    issues:     list[Issue] = field(default_factory=list)
    file_count: int = 0
    line_count: int = 0
    stats:      dict[str, Any] = field(default_factory=dict)

    def add(self, *args, **kwargs) -> None:
        self.issues.append(Issue(*args, **kwargs))

    def summary(self) -> dict[str, int]:
        return dict(collections.Counter(i.severity for i in self.issues))


def iter_py_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        yield path


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def read_file(path: Path) -> tuple[str, list[str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text, text.splitlines()
    except Exception:
        return "", []


def parse_ast(text: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def get_source_line(lines: list[str], lineno: int) -> str:
    return lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""


# -------------------------------------------------------------------------------
#  [ENG]  User Engagement
# -------------------------------------------------------------------------------

class EngagementChecker:
    """Перевіряє наявність user interaction points та feedback loops."""

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: EngagementResult) -> None:
        fname = rel(path, root)

        # Перевіряємо наявність user input methods
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # input() calls
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name) and child.func.id == "input":
                            result.add(
                                "ENG", "LOW", fname, child.lineno,
                                f"input()  {node.name}()   interactive CLI",
                                get_source_line(lines, child.lineno),
                                fix_hint="input()   CLI tools,  config files  API"
                            )

        # Перевіряємо наявність configuration classes
        has_config = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if "config" in node.name.lower() or "settings" in node.name.lower():
                    has_config = True
                    break

        # Перевіряємо наявність user-facing API endpoints
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    decorator_name = None
                    if isinstance(decorator, ast.Name):
                        decorator_name = decorator.id
                    elif isinstance(decorator, ast.Attribute):
                        decorator_name = decorator.attr
                    if decorator_name in ["route", "app.route", "api", "endpoint"]:
                        result.add(
                            "ENG", "INFO", fname, node.lineno,
                            f"API endpoint {node.name}()   user interaction",
                            get_source_line(lines, node.lineno),
                        )

        # Перевіряємо наявність feedback/logging mechanisms
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # User feedback collection
            if re.search(r'feedback|user_input|user_feedback|survey', line, re.IGNORECASE):
                result.add(
                    "ENG", "INFO", fname, i,
                    "User feedback mechanism detected",
                    stripped,
                )

            # Interactive plotting
            if re.search(r'plotly|bokeh|streamlit|gradio|dash', line, re.IGNORECASE):
                result.add(
                    "ENG", "INFO", fname, i,
                    "Interactive visualization library detected",
                    stripped,
                )


# -------------------------------------------------------------------------------
#  [EXP]  Explainability
# -------------------------------------------------------------------------------

class ExplainabilityChecker:
    """Перевіряє ML model explainability coverage."""

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: EngagementResult) -> None:
        fname = rel(path, root)

        # Перевіряємо наявність explainability libraries
        explainability_libs = {
            "shap", "lime", "eli5", "interpret", "alibi",
            "feature_importance", "permutation_importance",
            "partial_dependence", "pdp", "ice"
        }

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for lib in explainability_libs:
                if lib in line.lower():
                    result.add(
                        "EXP", "INFO", fname, i,
                        f"Explainability library/method detected: {lib}",
                        stripped,
                    )

        # Перевіряємо наявність feature importance tracking
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if "importance" in node.name.lower() or "explain" in node.name.lower():
                    result.add(
                        "EXP", "INFO", fname, node.lineno,
                        f"Explainability method: {node.name}()",
                        get_source_line(lines, node.lineno),
                    )

        # Перевіряємо наявність decision logging
        for i, line in enumerate(lines, 1):
            if re.search(r'decision.*log|log.*decision|record.*decision', line, re.IGNORECASE):
                result.add(
                    "EXP", "INFO", fname, i,
                    "Decision logging mechanism detected",
                    line.strip(),
                )

        # Перевіряємо наявність model interpretation methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if any("interpret" in method.name.lower() or "explain" in method.name.lower()
                       for method in node.body if isinstance(method, ast.FunctionDef)):
                    result.add(
                        "EXP", "INFO", fname, node.lineno,
                        f"Model interpretation class: {node.name}",
                        get_source_line(lines, node.lineno),
                    )


# -------------------------------------------------------------------------------
#  [MON]  Monitoring
# -------------------------------------------------------------------------------

class MonitoringChecker:
    """Перевіряє alerting та monitoring systems."""

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: EngagementResult) -> None:
        fname = rel(path, root)

        monitoring_libs = {
            "prometheus", "grafana", "sentry", "datadog", "newrelic",
            "alert", "monitoring", "telemetry", "metrics", "observability"
        }

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for lib in monitoring_libs:
                if lib in line.lower():
                    result.add(
                        "MON", "INFO", fname, i,
                        f"Monitoring/Alerting library detected: {lib}",
                        stripped,
                    )

        # Перевіряємо наявність alert conditions
        for i, line in enumerate(lines, 1):
            if re.search(r'alert|threshold|trigger|notify', line, re.IGNORECASE):
                if "if " in line or "when " in line:
                    result.add(
                        "MON", "INFO", fname, i,
                        "Alert condition detected",
                        line.strip(),
                    )

        # Перевіряємо наявність performance metrics tracking
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if any(metric in node.name.lower()
                       for metric in ["metric", "performance", "latency", "throughput"]):
                    result.add(
                        "MON", "INFO", fname, node.lineno,
                        f"Performance tracking method: {node.name}()",
                        get_source_line(lines, node.lineno),
                    )

        # Перевіряємо наявність anomaly detection
        for i, line in enumerate(lines, 1):
            if re.search(r'anomaly|outlier|detection|isolation', line, re.IGNORECASE):
                result.add(
                    "MON", "INFO", fname, i,
                    "Anomaly detection mechanism detected",
                    line.strip(),
                )


# -------------------------------------------------------------------------------
#  [TEST]  Test Coverage
# -------------------------------------------------------------------------------

class TestCoverageChecker:
    """Перевіряє наявність integration та E2E тестів."""

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: EngagementResult) -> None:
        fname = rel(path, root)

        # Перевіряємо чи це тестовий файл
        is_test_file = "test" in fname.lower() or fname.startswith("tests/")

        if is_test_file:
            # Перевіряємо тип тесту
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue

                # Integration test markers
                if re.search(r'integration|e2e|end.to.end', line, re.IGNORECASE):
                    result.add(
                        "TEST", "INFO", fname, i,
                        "Integration/E2E test detected",
                        stripped,
                    )

                # Performance test markers
                if re.search(r'performance|benchmark|load.*test', line, re.IGNORECASE):
                    result.add(
                        "TEST", "INFO", fname, i,
                        "Performance test detected",
                        stripped,
                    )

        # Перевіряємо наявність test fixtures
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if any("@pytest.fixture" in get_source_line(lines, d.lineno)
                       for d in node.decorator_list):
                    result.add(
                        "TEST", "INFO", fname, node.lineno,
                        f"Test fixture: {node.name}()",
                        get_source_line(lines, node.lineno),
                    )


# -------------------------------------------------------------------------------
#  [DOC]  Documentation
# -------------------------------------------------------------------------------

class DocumentationChecker:
    """Перевіряє наявність user docs, API docs, architecture docs."""

    def run(self, path: Path, root: Path, text: str,
            lines: list[str], tree: ast.Module, result: EngagementResult) -> None:
        fname = rel(path, root)

        # Перевіряємо docstring coverage
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                if not ast.get_docstring(node):
                    # Skip private methods
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("_"):
                        continue
                    result.add(
                        "DOC", "LOW", fname, node.lineno,
                        f"Missing docstring: {node.name if isinstance(node, ast.ClassDef) else node.name + '()'}",
                        fix_hint=f'"""Description of {node.name if isinstance(node, ast.ClassDef) else node.name}."""'
                    )

        # Перевіряємо наявність type hints
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("_") or node.name == "__init__":
                    continue
                if node.returns is None and len(node.body) > 3:
                    result.add(
                        "DOC", "LOW", fname, node.lineno,
                        f"Missing return type annotation: {node.name}()",
                        fix_hint=f"def {node.name}(...) -> ReturnType:"
                    )

        # Перевіряємо наявність README та docs
        if fname.lower() in ["readme.md", "readme.txt", "docs.md", "architecture.md"]:
            result.add(
                "DOC", "INFO", fname, 1,
                "Documentation file detected",
                fname,
            )


# -------------------------------------------------------------------------------
#  Main Auditor
# -------------------------------------------------------------------------------

class EngagementAuditor:

    def __init__(self, root: Path) -> None:
        self.root    = root
        self.result  = EngagementResult()
        self.checkers = [
            EngagementChecker(),
            ExplainabilityChecker(),
            MonitoringChecker(),
            TestCoverageChecker(),
            DocumentationChecker(),
        ]

    def audit(self) -> EngagementResult:
        print(f" Engagement & Coverage Audit: {self.root}")
        files = list(iter_py_files(self.root))
        self.result.file_count = len(files)

        for path in files:
            text, lines = read_file(path)
            self.result.line_count += len(lines)
            if not text:
                continue

            tree = parse_ast(text)
            if tree is None:
                continue

            for checker in self.checkers:
                try:
                    checker.run(path, self.root, text, lines, tree, self.result)
                except Exception as exc:
                    print(f"   Checker {checker.__class__.__name__} failed on {path}: {exc}")

        self.result.stats = {
            "files":  self.result.file_count,
            "lines":  self.result.line_count,
            "issues": len(self.result.issues),
            "by_severity": self.result.summary(),
            "by_category": dict(
                collections.Counter(i.category for i in self.result.issues)
            ),
        }
        return self.result


def print_report(result: EngagementResult, show_fix: bool = True, max_issues: int = 1000) -> None:
    sev_order = list(SEVERITY.keys())
    issues_sorted = sorted(
        result.issues,
        key=lambda i: (SEVERITY.get(i.severity, 99), i.category, i.file, i.line),
    )

    current_sev = None
    shown = 0
    for issue in issues_sorted:
        if shown >= max_issues:
            print(f"\n...   {len(issues_sorted) - shown} issues")
            break
        if issue.severity != current_sev:
            current_sev = issue.severity
            print(f"\n{'-' * 70}")
            print(f"  {issue.severity}")
            print(f"{'-' * 70}")
        print(f"\n[{issue.category}] {issue.file}:{issue.line}")
        print(f"   {issue.message}")
        if issue.code:
            try:
                print(f"   > {issue.code[:110]}")
            except UnicodeEncodeError:
                print(f"   > [code contains unicode characters]")
        if show_fix and issue.fix_hint:
            print(f"   - {issue.fix_hint}")
        shown += 1

    s = result.stats
    print(f"\n{'-' * 70}")
    print(f"   ENGAGEMENT & COVERAGE AUDIT  ")
    print(f"{'-' * 70}")
    print(f"   Files: {s['files']}   Lines: {s['lines']:,}   Issues: {s['issues']}")
    print()
    for sev in sev_order:
        cnt = s['by_severity'].get(sev, 0)
        if cnt:
            print(f"  {sev:<12}: {cnt}")
    print()
    print("   Categories:")
    for cat, cnt in sorted(s['by_category'].items(), key=lambda x: -x[1]):
        desc = {
            "ENG":  "User Engagement",
            "EXP":  "Explainability",
            "MON":  "Monitoring",
            "TEST": "Test Coverage",
            "DOC":  "Documentation",
        }.get(cat, "")
        print(f"    [{cat}] {cnt:3}  {desc}")
    print(f"{'-' * 70}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="DEAN Engagement & Coverage Auditor")
    parser.add_argument("--root",       default=".",    help=" ")
    parser.add_argument("--json",       action="store_true")
    parser.add_argument("--output",     default="",     help="  ")
    parser.add_argument("--severity",   default="LOW",  help="CRITICAL/HIGH/MEDIUM/LOW/INFO")
    parser.add_argument("--category",   default="",     help="ENG,EXP,MON,TEST,DOC")
    parser.add_argument("--max-issues", default=1000,   type=int)
    args = parser.parse_args()

    # Validate and resolve path (audit tool needs to scan user-provided paths)
    try:
        root = Path(args.root).resolve()
        # Ensure path is within current directory or absolute
        if not root.exists():
            print(f"ERROR:  : {root}")
            sys.exit(1)
    except (OSError, RuntimeError) as e:
        print(f"ERROR: Invalid path: {e}")
        sys.exit(1)

    auditor = EngagementAuditor(root)
    result  = auditor.audit()

    min_sev = SEVERITY.get(args.severity.upper(), 3)
    result.issues = [
        i for i in result.issues
        if SEVERITY.get(i.severity, 99) <= min_sev
        and (not args.category or i.category in args.category.upper().split(","))
    ]

    if args.json:
        output = json.dumps(
            {
                "stats":  result.stats,
                "issues": [
                    {"category": i.category, "severity": i.severity,
                     "file": i.file, "line": i.line,
                     "message": i.message, "fix_hint": i.fix_hint}
                    for i in result.issues
                ],
            },
            ensure_ascii=False, indent=2,
        )
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"DONE: {args.output}")
        else:
            print(output)
    else:
        if args.output:
            import contextlib
            with open(args.output, "w", encoding="utf-8") as f:
                with contextlib.redirect_stdout(f):
                    print_report(result, True, args.max_issues)
            print(f"DONE: {args.output}")
        else:
            print_report(result, True, args.max_issues)

    critical = result.stats.get("by_severity", {}).get("CRITICAL", 0)
    high     = result.stats.get("by_severity", {}).get("HIGH", 0)
    sys.exit(2 if critical > 0 else (1 if high > 0 else 0))


if __name__ == "__main__":
    main()
