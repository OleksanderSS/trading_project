"""
Data source contracts.

Checks for API/secrets/import-time side effects by source scan.
"""

from __future__ import annotations

import ast
from pathlib import Path


# Attribute names that mean "talk to the network" when called.
NETWORK_ATTRS = {"get", "post", "put", "patch", "delete", "request", "urlopen",
                 "ClientSession", "urlretrieve"}
# Modules whose attribute access is network even for names not in NETWORK_ATTRS.
NETWORK_MODULES = {"requests", "httpx", "aiohttp", "urllib", "socket"}
# ``get``/``post`` are also dict and config method names, so a bare ``.get(``
# means nothing on its own. Only calls on one of these receivers count.
NETWORK_RECEIVERS = NETWORK_MODULES | {"session", "client", "http", "conn",
                                       "connection", "api"}


def _receiver_name(node: ast.AST) -> str:
    """Left-most name of an attribute chain: ``a.b.c`` -> ``a``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id.lower() if isinstance(node, ast.Name) else ""


def _is_network_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        receiver = _receiver_name(func.value)
        if receiver in NETWORK_MODULES:
            return True
        return func.attr in NETWORK_ATTRS and receiver in NETWORK_RECEIVERS
    if isinstance(func, ast.Name):
        return func.id in {"urlopen", "urlretrieve"}
    return False


def _module_level_nodes(tree: ast.Module):
    """Yield nodes that run at import time.

    Function and class *bodies* are skipped — they only run when called — but
    decorators and default arguments are kept, since those do run on import.
    """
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for sub in stmt.decorator_list:
                yield from ast.walk(sub)
            continue
        yield from ast.walk(stmt)


def test_no_network_calls_at_import_time_by_source_scan():
    """
    An import-time network call turns "import this module" into "reach out to a
    third party", which makes tests non-hermetic and imports fail offline.

    Checked over the AST rather than by string matching: the old scan flagged
    every ``metrics.get('sharpe_ratio')`` sitting inside a multi-line f-string,
    which is a dict lookup and not a request.
    """
    offenders = []
    for p in sorted(Path("src").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError as exc:
            offenders.append(f"{p}: does not parse: {exc}")
            continue
        for node in _module_level_nodes(tree):
            if isinstance(node, ast.Call) and _is_network_call(node):
                offenders.append(f"{p}:{node.lineno}: {ast.unparse(node)[:120]}")

    assert not offenders, (
        "Possible import-time network calls found. Move calls inside functions/classes. "
        + str(offenders[:20])
    )


def test_no_hardcoded_common_secret_names_by_source_scan():
    offenders = []
    secret_tokens = ["API_KEY =", "SECRET_KEY =", "PASSWORD =", "TOKEN =", "PRIVATE_KEY ="]
    for p in Path("src").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for token in secret_tokens:
            if token in text:
                offenders.append(str(p))
    assert not offenders, f"Possible hardcoded secrets. Review: {offenders[:20]}"
