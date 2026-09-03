"""AST scan for failure paths that leave no trace a caller can act on.

Three shapes, each of which has already produced a real defect here:

LOGGED_THEN_EMPTY
    A logger.error/critical call followed, in the same block, by a return of
    an empty literal. The stage logged "Enriched data not found. Skipping
    Modeling Stage." and returned {}, and the pipeline orchestrator reported
    a success (commit 4a8e804e). The log says failure; the return value says
    success; the caller only sees the return value.

SWALLOWED
    An except handler whose body neither logs nor re-raises. Counted by
    SEVERITY, because the single number said four different things at once
    and so meant none of them. Measured 2026-09-01 across 112 handlers:

        SWALLOWED_ERASED       59  the handler does nothing and execution
                                   continues as though the call had worked
        SWALLOWED_EMPTY        29  returns None / {} / [] / False
        SWALLOWED_FABRICATED   17  returns a made-up VALUE, which is the worst
                                   of the four: an invented number cannot be
                                   told from a measured one. The dashboard
                                   returns cpu 45.2%, memory 8.5 GB and
                                   `status: 'healthy'` when reading the real
                                   metrics fails.

    The fourth group is not counted at all: a handler that returns an explicit
    marker -- `{'error': ...}`, `{'passive_status': 'unreadable'}`, `np.nan`,
    `"n/a"` -- has given its caller a state distinct from success, which is
    exactly what the rule asks for. Counting those as violations taught the
    opposite of the rule.

QUIET_THEN_EMPTY
    An except handler that reports only below error level -- debug, info or
    warning -- and then returns an empty value. LOGGED_THEN_EMPTY does not
    see these, because it requires error/critical, and that gap has a name:
    REGISTER #189. The walk-forward stability rung filtered rows by a ticker
    called `__POOLED__`, matched none of 159,149, raised, wrote the reason to
    `logger.debug` and returned None. The caller reads None as "cannot
    measure, skipping", so the only rung that asks whether an edge holds over
    TIME was off for every pooled context, and both champions of 2026-08-31
    were promoted without it. Nothing was hidden: the failure was recorded at
    a level nobody reads and returned as a value that means success.

    The rule this enforces is the one the dead rungs cost us: a check must
    have a distinct "could not measure" state, and that state may never be
    the same value as "passed".

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


_QUIET_LEVELS = {"debug", "info", "warning"}


def _quiet_log_level(body: list[ast.stmt]) -> str | None:
    """The quiet level this handler reports at, if it reports only quietly."""
    quiet: str | None = None
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr in _LOUD_LEVELS:
            return None
        if "handle" in node.attr and "error" in node.attr:
            return None
        if node.attr in _QUIET_LEVELS and quiet is None:
            quiet = node.attr
    return quiet


def _returns_empty(body: list[ast.stmt]) -> ast.Return | None:
    """The first `return <empty>` in this handler, if there is one."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Return) and _is_empty_literal(node.value):
            return node
    return None


#: Words that mark a returned value as "this is not a measurement".
_MARKER_WORDS = (
    "error", "fail", "unavailable", "unknown", "missing", "invalid",
    "unreadable", "n/a", "nan",
)
# `None` is deliberately NOT here. It reads like a marker and is the opposite
# of one: it is the single most common way a failure is made to look like an
# ordinary empty result. Including it moved 26 handlers out of the count in
# one stroke, which is how a scanner is talked into agreeing with the code.


def _is_explicit_marker(node: ast.expr | None) -> bool:
    """True when the returned value tells the caller it is not a result."""
    if node is None:
        return False
    text = ast.unparse(node).lower()
    return any(word in text for word in _MARKER_WORDS)


def _swallow_severity(body: list[ast.stmt]) -> str | None:
    """How badly this handler hides the failure, or None if it does not."""
    returns = [
        node for node in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(node, ast.Return)
    ]
    if not returns:
        return "SWALLOWED_ERASED"
    if all(_is_explicit_marker(node.value) for node in returns):
        return None
    if all(_is_empty_literal(node.value) for node in returns):
        return "SWALLOWED_EMPTY"
    return "SWALLOWED_FABRICATED"


def _reraises(body: list[ast.stmt]) -> bool:
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Raise):
            return True
    return False


#: A function whose name asks a QUESTION is not exempt. `is_ready()` returning
#: False after an exception answers "not ready" when the truth is "could not
#: determine", and those are different facts about the world. A function whose
#: name describes an ACTION -- save, upload, send -- has False as its honest
#: failure value, because the action really did not happen.
_QUESTION_PREFIXES = ("is_", "has_", "can_", "should_", "was_", "are_", "does_")


def _is_predicate(node: ast.AST) -> bool:
    """True when every `return` in this function yields True or False.

    `logger.error(...)` followed by `return False` is the WHOLE contract of a
    function like `FileManager.save_yaml() -> bool`: the log says what went
    wrong and the boolean says it went wrong, and a caller that ignores the
    boolean would ignore any other signal too. That is not the shape this scan
    was written for. The shape is `return {}` from a function that was asked
    for DATA -- ModelingStage returning an empty dict, which the orchestrator
    reported as a successful stage (commit 4a8e804e), because there the empty
    value is indistinguishable from a real, if unremarkable, result.

    Without this distinction the count mixed the two and could only be lowered
    by rewriting correct code, which is how a ratchet turns into a ritual.
    """
    returns = [
        child for child in ast.walk(node)
        if isinstance(child, ast.Return)
        and _enclosing_function(node, child) is node
    ]
    if not returns:
        return False
    name = getattr(node, "name", "")
    if name.startswith(_QUESTION_PREFIXES):
        return False
    return all(
        isinstance(r.value, ast.Constant) and isinstance(r.value.value, bool)
        for r in returns
    )


def _enclosing_function(func: ast.AST, target: ast.Return) -> ast.AST | None:
    """The nearest function that owns `target`, so nested defs do not lie."""
    found: ast.AST | None = None

    def walk(node: ast.AST, current: ast.AST | None) -> None:
        nonlocal found
        for child in ast.iter_child_nodes(node):
            owner = child if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) else current
            if child is target:
                found = owner
                return
            walk(child, owner)

    walk(func, func)
    return found


class _Scanner(ast.NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.findings: list[Finding] = []
        self._predicate_depth = 0

    # -- shape 1: a loud log immediately followed by an empty return --------
    def _scan_block(self, body: list[ast.stmt]) -> None:
        for earlier, later in zip(body, body[1:]):
            level = _is_loud_log(earlier)
            returns_false = (
                isinstance(later, ast.Return)
                and isinstance(later.value, ast.Constant)
                and later.value.value is False
            )
            if returns_false and self._predicate_depth:
                continue
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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        predicate = _is_predicate(node)
        self._predicate_depth += int(predicate)
        self.generic_visit(node)
        self._predicate_depth -= int(predicate)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    # -- shapes 2 and 3: except handlers ------------------------------------
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if (
            not _is_optional_dependency_guard(node)
            and not _mentions_logging(node.body)
            and not _reraises(node.body)
        ):
            severity = _swallow_severity(node.body)
            if severity is not None:
                self.findings.append(
                    Finding(severity, self.module, node.lineno,
                            "except handler neither logs nor re-raises")
                )

        if not _reraises(node.body):
            level = _quiet_log_level(node.body)
            returned = _returns_empty(node.body) if level else None
            if level and returned is not None:
                self.findings.append(
                    Finding(
                        "QUIET_THEN_EMPTY", self.module, returned.lineno,
                        f"logger.{level}(...) then `return "
                        f"{ast.unparse(returned.value) if returned.value else 'None'}`"
                        " -- the caller cannot tell this from success",
                    )
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
