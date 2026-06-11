"""
Config reachability checker.

Purpose:
- find config files
- extract likely component/model/enricher/stage names
- check whether referenced class/module names appear in src
- report config entries that may point to missing or renamed code

Run:
    python diagnostics/config_reachability_checker.py --root src --configs configs . --out diagnostic_reports
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


NAME_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Enricher|Calculator|Validator|Collector|Model|Factory|Analyzer|Detector|Selector|Guard|Stage|Agent)\b")
KEY_VALUE_RE = re.compile(r"(?P<key>model|model_name|enricher|calculator|validator|collector|stage|class|class_path|component|name)\s*[:=]\s*['\"]?(?P<value>[A-Za-z0-9_.\-]+)")


@dataclass
class ConfigReference:
    config_file: str
    key: str
    value: str
    status: str
    matched_files: str
    recommendation: str


def iter_source_texts(root: Path):
    entries = []
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        entries.append((str(p.relative_to(root)).replace("\\", "/"), text))
    return entries


def iter_config_files(paths: list[Path]):
    files = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".py"}:
            files.append(path)
        elif path.is_dir():
            for p in path.rglob("*"):
                if "__pycache__" in p.parts or ".git" in p.parts or ".venv" in p.parts or "venv" in p.parts:
                    continue
                if p.suffix.lower() in {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}:
                    files.append(p)
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="src")
    ap.add_argument("--configs", nargs="*", default=["configs", "."])
    ap.add_argument("--out", default="diagnostic_reports")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    source_texts = iter_source_texts(root)
    config_files = iter_config_files([Path(p) for p in args.configs if Path(p).exists()])

    refs: list[ConfigReference] = []
    seen = set()

    for cf in config_files:
        text = cf.read_text(encoding="utf-8", errors="ignore")
        candidates = []

        for m in NAME_RE.finditer(text):
            candidates.append(("class_like_name", m.group(0)))

        for m in KEY_VALUE_RE.finditer(text):
            candidates.append((m.group("key"), m.group("value")))

        for key, value in candidates:
            if len(value) < 3:
                continue
            ident = (str(cf), key, value)
            if ident in seen:
                continue
            seen.add(ident)

            simple = value.split(".")[-1].replace("-", "_")
            matched = []
            for file, src_text in source_texts:
                if value in src_text or simple in src_text or simple.lower() in file.lower():
                    matched.append(file)

            if matched:
                status = "FOUND_IN_SOURCE"
                recommendation = "Verify this reference is used through a registry/factory/pipeline path."
            else:
                status = "POSSIBLY_MISSING"
                recommendation = "Check if config references renamed/deleted component or if component is loaded dynamically."

            refs.append(ConfigReference(
                config_file=str(cf),
                key=key,
                value=value,
                status=status,
                matched_files=";".join(matched[:20]),
                recommendation=recommendation,
            ))

    with (out / "config_reachability.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(asdict(refs[0]).keys()) if refs else ["config_file", "key", "value", "status", "matched_files", "recommendation"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in refs:
            writer.writerow(asdict(r))

    summary = {
        "config_files_scanned": len(config_files),
        "references_found": len(refs),
        "possibly_missing": sum(1 for r in refs if r.status == "POSSIBLY_MISSING"),
    }
    (out / "config_reachability_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Config Reachability Summary", ""]
    for k, v in summary.items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("Review `config_reachability.csv`. Missing references are not automatically bugs; dynamic loading may require manual confirmation.")
    (out / "config_reachability_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
