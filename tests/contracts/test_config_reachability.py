"""
Config reachability contracts.

These are lightweight checks around dynamic config/registry references.
"""

from __future__ import annotations

from pathlib import Path
import re


def test_no_obvious_missing_class_paths_in_config_files():
    config_roots = [Path("configs"), Path(".")]
    config_files = []
    for root in config_roots:
        if not root.exists():
            continue
        if root.is_file():
            config_files.append(root)
        else:
            for p in root.rglob("*"):
                if ".git" in p.parts or ".venv" in p.parts or "venv" in p.parts:
                    continue
                if p.suffix.lower() in {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}:
                    config_files.append(p)

    src_text = ""
    if Path("src").exists():
        src_text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in Path("src").rglob("*.py") if "__pycache__" not in p.parts)

    class_path_re = re.compile(r"(?:class_path|component|factory|model_class)\s*[:=]\s*['\"](?P<value>src\.[A-Za-z0-9_.]+)['\"]")
    missing = []

    for cf in config_files:
        text = cf.read_text(encoding="utf-8", errors="ignore")
        for m in class_path_re.finditer(text):
            value = m.group("value")
            simple = value.split(".")[-1]
            if simple not in src_text:
                missing.append((str(cf), value))

    assert not missing, f"Config class paths may be missing from src: {missing[:20]}"
