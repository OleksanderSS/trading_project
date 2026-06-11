"""
Data source contracts.

Checks for API/secrets/import-time side effects by source scan.
"""

from __future__ import annotations

from pathlib import Path


def test_no_network_calls_at_import_time_by_source_scan():
    offenders = []
    network_tokens = [".get(", ".post(", "requests.get", "requests.post", "httpx.get", "httpx.post", "aiohttp.ClientSession"]
    for p in Path("src").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if any(tok in stripped for tok in network_tokens):
                # This is heuristic. Review findings manually.
                if not line.startswith(" ") and not line.startswith("\t"):
                    offenders.append(f"{p}:{i}:{stripped}")

    assert not offenders, "Possible import-time network calls found. Move calls inside functions/classes. " + str(offenders[:20])


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
