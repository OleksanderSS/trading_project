"""
Config reachability contracts.

These are lightweight checks around dynamic config/registry references.
"""

from __future__ import annotations

import os
from pathlib import Path
import re


# Directories to prune from the repo-root walk below -- these hold generated
# data/artifacts, not config (data/ and models/ alone were >65k files). Path.rglob()
# cannot skip a subtree once it descends into it; it only filters the yielded
# results afterward, so the expensive directory listing/stat I/O for these
# huge trees already happened by the time the filter runs. That turned an
# innocent-looking `Path(".").rglob("*")` into a multi-hour scan (confirmed
# via wmic: the process was genuinely at 90%+ CPU the whole time, not
# deadlocked -- just walking everything). os.walk(topdown=True) can prune a
# directory before descending into it (mutate dirnames in place), which is
# the only way to actually avoid the I/O, not just the result.
_EXCLUDED_TOP_LEVEL_DIRS = {
    ".git", ".venv", "venv", "data", "reports", "logs", "outputs", "mlruns",
    "models", "archive", "audit", "node_modules", ".trunk",
}
_CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}


def _walk_config_files(root: Path) -> list[Path]:
    config_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_TOP_LEVEL_DIRS]
        for name in filenames:
            if Path(name).suffix.lower() in _CONFIG_SUFFIXES:
                config_files.append(Path(dirpath) / name)
    return config_files


def test_no_obvious_missing_class_paths_in_config_files():
    config_files: list[Path] = []
    if Path("configs").exists():
        config_files.extend(_walk_config_files(Path("configs")))
    if Path(".").exists():
        config_files.extend(_walk_config_files(Path(".")))

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
