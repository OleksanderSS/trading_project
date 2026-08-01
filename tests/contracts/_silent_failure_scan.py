"""AST scan for failure paths that leave no trace a caller can act on.

Three shapes, each of which has already produced a real defect here:

LOGGED_THEN_EMPTY
    A logger.error/critical call followed, in the same block, by a return of
    an empty literal. The stage logged "Enriched data not found. Skipping
    Modeling Stage." and returned {}, and the pipeline orchestrator reported
    a success (commit 4a8e804e). The log says failure; the return value says
    success; the caller only sees the return value.

SWALLOWED
    An except handler whose body neither logs nor re-raises. The failure is
    erased entirely.

NARROW_TUPLE
    except (ValueError, TypeError, AttributeError, KeyError,
    ZeroDivisionError). Reads exhaustive, is not: it omits OSError,
    RuntimeError, IndexError and every library exception. Written across the
    codebase in bulk by scripts/auto_refactor_exceptions.py, a regex that
    replaced every `except Exception as e:` under src/. Three defects traced
    to it so far -- CatBoostError, sqlite3.IntegrityError, yaml.YAMLError,
    all of which inherit straight from Exception. json.JSONDecodeError only
    survived because it happens to be a ValueError.

Import this from the contract test; it is not a test module itself.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"

# Not audited: archived code kept for reference, and the scripts that
# performed one-off migrations.
EXCLUDED_PARTS = {"archive", "__pycache__", "dead_pipeline_code"}

NARROW_TUPLE = frozenset(
    {"ValueError", "TypeError", "AttributeError", "KeyError", "ZeroDivisionError"}
)

_LOUD_LEVELS = {"error", "critical", "exception"}
_LOG_ATTRS = {"logger", "log", "_logger"}


@dataclass(frozen=True)
class Finding:
    kind: str
    module: str
    line: int
    context: str

    def __str__(self) -> str:
        return f"{self.module}:{self.line}  [{self.kind}]  {self.context}"


def _is_empty_literal(node: ast.expr | None) -> bool:
    """True for the values that read as 'nothing to report'."""
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        return node.value in (None, False, 0, "") and not isinstance(node.value, float)
    if isinstance(node, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
        return not node.elts if hasattr(node, "elts") else not node.keys
    return False


def _is_loud_log(node: ast.stmt) -> str | None:
    """Return the log level if this statement is a logger.error/critical call."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return None
    func = node.value.func
    if not isinstance(func, ast.Attribute) or func.attr not in _LOUD_LEVELS:
        return None
    target = func.value
    name = (
        target.id if isinstance(target, ast.Name)
        else target.attr if isinstance(target, ast.Attribute)
        else ""
    )
    return func.attr if name in _LOG_ATTRS else None


def _mentions_logging(body: list[ast.stmt]) -> bool:
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Attribute) and node.attr in (
            _LOUD_LEVELS | {"warning", "info", "debug"}
        ):
            return True
        # handle_error / handle_stage_error / _handle_stage_error count as
        # reporting: they route into ErrorHandler, which logs.
        if isinstance(node, ast.Attribute) and "handle" in node.attr and "error" in node.attr:
            return True
    return False


_DEPENDENCY_ERRORS = {"ImportError", "ModuleNotFoundError"}


def _is_optional_dependency_guard(node: ast.ExceptHandler) -> bool:
    """`except ImportError: HAVE_X = False` is a deliberate feature switch,
    not a swallowed failure -- the absence is recorded in a flag the code
    then branches on."""
    caught = node.type
    if isinstance(caught, ast.Name):
        names = {caught.id}
    elif isinstance(caught, ast.Tuple):
        names = {e.id for e in caught.elts if isinstance(e, ast.Name)}
    else:
        return False
    return bool(names) and names <= _DEPENDENCY_ERRORS


def _reraises(body: list[ast.stmt]) -> bool:
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Raise):
            return True
    return False


class _Scanner(ast.NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.findings: list[Finding] = []

    # -- shape 1: a loud log immediately followed by an empty return --------
    def _scan_block(self, body: list[ast.stmt]) -> None:
        for earlier, later in zip(body, body[1:]):
            level = _is_loud_log(earlier)
            if level and isinstance(later, ast.Return) and _is_empty_literal(later.value):
                self.findings.append(
                    Finding(
                        "LOGGED_THEN_EMPTY",
                        self.module,
                        later.lineno,
                        f"logger.{level}(...) then `return "
                        f"{ast.unparse(later.value) if later.value else 'None'}`",
                    )
                )

    def generic_visit(self, node: ast.AST) -> None:
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list):
                self._scan_block(block)
        super().generic_visit(node)

    # -- shapes 2 and 3: except handlers ------------------------------------
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if (
            not _is_optional_dependency_guard(node)
            and not _mentions_logging(node.body)
            and not _reraises(node.body)
        ):
            self.findings.append(
                Finding("SWALLOWED", self.module, node.lineno,
                        "except handler neither logs nor re-raises")
            )

        caught = node.type
        if isinstance(caught, ast.Tuple):
            names = {e.id for e in caught.elts if isinstance(e, ast.Name)}
            if names == NARROW_TUPLE:
                self.findings.append(
                    Finding("NARROW_TUPLE", self.module, node.lineno,
                            "except (ValueError, TypeError, AttributeError, "
                            "KeyError, ZeroDivisionError)")
                )
        self.generic_visit(node)


def _python_files(root: Path) -> list[Path]:
    return [
        path for path in sorted(root.rglob("*.py"))
        if not EXCLUDED_PARTS & set(path.parts)
    ]


def scan(root: Path | None = None) -> list[Finding]:
    """Scan every source file and return findings sorted by module."""
    findings: list[Finding] = []
    for path in _python_files(root or SOURCE_ROOT):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        scanner = _Scanner(path.relative_to(PROJECT_ROOT).as_posix())
        scanner.visit(tree)
        findings.extend(scanner.findings)
    return findings


def by_kind(findings: list[Finding]) -> dict[str, list[Finding]]:
    grouped: dict[str, list[Finding]] = {}
    for finding in findings:
        grouped.setdefault(finding.kind, []).append(finding)
    return grouped


if __name__ == "__main__":
    grouped = by_kind(scan())
    for kind in sorted(grouped):
        entries = grouped[kind]
        print(f"\n=== {kind}: {len(entries)} ===")
        for entry in entries[:40]:
            print(f"  {entry}")
        if len(entries) > 40:
            print(f"  ... and {len(entries) - 40} more")
