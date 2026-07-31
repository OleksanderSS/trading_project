"""
Data source contracts.

Checks for API/secrets/import-time side effects by source scan.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: Calls that genuinely perform network I/O, as dotted suffixes.
_NETWORK_CALLS = {
    ("requests", "get"), ("requests", "post"), ("requests", "put"),
    ("requests", "delete"), ("requests", "head"), ("requests", "request"),
    ("httpx", "get"), ("httpx", "post"), ("httpx", "put"),
    ("httpx", "delete"), ("httpx", "head"), ("httpx", "request"),
    ("aiohttp", "ClientSession"),
    ("urlopen", None), ("urlretrieve", None),
}


def _call_names(node: ast.Call) -> tuple[str | None, str | None]:
    """(owner, attribute) for `owner.attribute(...)`, else (None, name)."""
    func = node.func
    if isinstance(func, ast.Attribute):
        owner = func.value
        owner_name = owner.id if isinstance(owner, ast.Name) else None
        return owner_name, func.attr
    if isinstance(func, ast.Name):
        return None, func.id
    return None, None


def _module_level_network_calls(tree: ast.AST) -> list[str]:
    """Network calls executed at import time, i.e. outside any def/class."""
    hits: list[str] = []
    for node in tree.body:  # top level only -- never descends into def/class
        for sub in ast.walk(node):
            if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                break
            if not isinstance(sub, ast.Call):
                continue
            owner, attr = _call_names(sub)
            if (owner, attr) in _NETWORK_CALLS or (attr, None) in _NETWORK_CALLS:
                hits.append(f"line {sub.lineno}: {owner + '.' if owner else ''}{attr}(...)")
    return hits


def test_no_network_calls_at_import_time_by_source_scan():
    """No HTTP at import time.

    Parsed with `ast` rather than scanned as text. The previous version looked
    for the substrings ".get(" and ".post(" on unindented lines, which matched
    ordinary `dict.get()` calls sitting inside multi-line HTML/f-string
    templates -- those start at column 0, so the "unindented means module
    level" heuristic reported them as import-time network I/O. Every finding
    it produced was of that kind (error_handler.py's HTML template,
    report_generator.py's summary block).
    """
    offenders = []
    for p in Path("src").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for hit in _module_level_network_calls(tree):
            offenders.append(f"{p}:{hit}")

    assert not offenders, (
        "Possible import-time network calls found. Move calls inside "
        "functions/classes. " + str(offenders[:20])
    )


def test_the_import_time_network_scan_still_catches_a_real_violation():
    """Guard the guard, since the previous check only ever produced noise."""
    violating = ast.parse("import requests\nDATA = requests.get('https://x')\n")
    assert _module_level_network_calls(violating)

    inside_function = ast.parse(
        "import requests\ndef load():\n    return requests.get('https://x')\n"
    )
    assert not _module_level_network_calls(inside_function)

    dict_get_in_template = ast.parse(
        'TEMPLATE = """\n<b>Time:</b> {info.get(\'timestamp\')}\n"""\n'
    )
    assert not _module_level_network_calls(dict_get_in_template)


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
