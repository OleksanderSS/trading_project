"""Find statements that can never run: code after return/raise/continue/break.

This exists because of a real regression. `ModelingStage.__init__` ended with

    self.models_dir = ...
    self.diary_path = ...
    self._init_infrastructure()

and a later commit inserted a `_resolve_test_size` method above them. The
three lines ended up after that method's `return`, inside its `except`
block, and became unreachable. Nothing failed at import, nothing failed in
the unit tests, and the class simply never gained two attributes.

The cost was invisible until a full run: every one of 22 tickers raised
`'ModelingStage' object has no attribute 'diary_path'` at the point the
champion was about to be written, the per-ticker handler swallowed it, and
Stage 4 finished having logged ZERO champions out of 44 training runs.

Python does not warn about this. Neither pyflakes nor py_compile treats it
as an error, because it is syntactically fine. An AST walk does.

Runnable standalone:  python tests/contracts/_unreachable_scan.py
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

EXCLUDED_PARTS = ("archive", "draft", "__pycache__")

#: Statements that end control flow outright.
_TERMINAL = (ast.Return, ast.Raise, ast.Continue, ast.Break)


def _terminates(node: ast.stmt) -> bool:
    """True when nothing after `node` in the same block can execute.

    A bare Return check is not enough, and the regression that prompted this
    file proves it: the dead code sat after a try/except whose body AND
    whose handler both returned. The `Try` node is not itself a Return, so a
    naive scan walked straight past it -- the first version of this scanner
    did exactly that and reported the sample clean.
    """
    if isinstance(node, _TERMINAL):
        return True

    if isinstance(node, ast.If):
        # Only when BOTH branches leave; a missing else always falls through.
        return bool(node.orelse) and _body_terminates(node.body) and _body_terminates(node.orelse)

    if isinstance(node, ast.Try):
        if node.finalbody and _body_terminates(node.finalbody):
            return True
        # Control reaches past the try if the body completes normally (or its
        # else does), or if any handler completes normally.
        body_exits = _body_terminates(node.orelse or node.body)
        handlers_exit = bool(node.handlers) and all(
            _body_terminates(handler.body) for handler in node.handlers
        )
        return body_exits and handlers_exit

    if isinstance(node, (ast.With, ast.AsyncWith)):
        return _body_terminates(node.body)

    return False


def _body_terminates(body: list[ast.stmt]) -> bool:
    return bool(body) and any(_terminates(statement) for statement in body)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    context: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}  after {self.context}  {self.text}"


def _describe(node: ast.stmt) -> str:
    return {
        ast.Return: "return", ast.Raise: "raise",
        ast.Continue: "continue", ast.Break: "break",
        ast.If: "an if whose branches all exit",
        ast.Try: "a try whose body and handlers all exit",
        ast.With: "a with whose body exits",
        ast.AsyncWith: "a with whose body exits",
    }.get(type(node), "an exiting statement")


def _scan_body(body: list[ast.stmt], path: str, lines: list[str],
               findings: list[Finding]) -> None:
    for index, statement in enumerate(body[:-1]):
        if _terminates(statement):
            dead = body[index + 1]
            # A docstring-only expression or `...` after a return is a
            # stylistic placeholder, not a lost statement.
            if isinstance(dead, ast.Expr) and isinstance(
                dead.value, (ast.Constant,)
            ):
                continue
            text = lines[dead.lineno - 1].strip() if dead.lineno <= len(lines) else ""
            findings.append(
                Finding(path, dead.lineno, _describe(statement), text[:110])
            )
            break


def scan(root: Path | None = None) -> list[Finding]:
    findings: list[Finding] = []
    base = root or SRC_ROOT
    for file_path in sorted(base.rglob("*.py")):
        if any(part in EXCLUDED_PARTS for part in file_path.parts):
            continue
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        lines = source.splitlines()
        relative = file_path.relative_to(base.parent).as_posix()
        for node in ast.walk(tree):
            for field in ("body", "orelse", "finalbody"):
                body = getattr(node, field, None)
                if isinstance(body, list) and body:
                    _scan_body(body, relative, lines, findings)
            for handler in getattr(node, "handlers", []) or []:
                if handler.body:
                    _scan_body(handler.body, relative, lines, findings)
    return findings


if __name__ == "__main__":
    results = scan()
    for finding in results:
        print(finding)
    print(f"\n{len(results)} unreachable statement(s)")
