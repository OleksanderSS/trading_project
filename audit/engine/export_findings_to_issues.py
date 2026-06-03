#!/usr/bin/env python3
"""Export audit findings into GitHub issue markdown files.

This does not call GitHub; it creates reviewable `.md` files that can be pasted
or used with `gh issue create --title ... --body-file ...`.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s.strip()).strip("-").lower()
    return s[:120] or "finding"


def load_findings(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("findings JSON must be a list")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--findings", default="audit_reports/findings.json")
    ap.add_argument("--out", default="audit_reports/issues")
    ap.add_argument("--severity", default="P0,P1")
    ap.add_argument("--max-per-rule", type=int, default=10)
    args = ap.parse_args()

    wanted = {s.strip().upper() for s in args.severity.split(",") if s.strip()}
    findings = [f for f in load_findings(Path(args.findings)) if f.get("severity") in wanted]
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for f in findings:
        groups[(f.get("severity", ""), f.get("category", ""), f.get("rule_id", ""))].append(f)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index_lines = ["# Audit issue export", ""]
    count = 0
    for (sev, category, rule_id), items in sorted(groups.items()):
        count += 1
        title = f"[{sev}] {category}: {rule_id}"
        filename = f"{count:03d}-{slugify(title)}.md"
        path = out / filename
        shown = items[: args.max_per_rule]
        body = [f"# {title}", "", f"Total findings in group: {len(items)}", ""]
        body += ["## Why this matters", "", shown[0].get("why", ""), ""]
        body += ["## Suggested fix", "", shown[0].get("fix", ""), ""]
        body += ["## Suggested test", "", shown[0].get("test", ""), ""]
        body += ["## Examples", ""]
        for f in shown:
            body += [
                f"### `{f.get('file')}:{f.get('line')}`",
                "",
                f"Problem: {f.get('problem')}",
                "",
                "```python",
                str(f.get("snippet", "")),
                "```",
                "",
                f"Fingerprint: `{f.get('fingerprint')}`",
                "",
            ]
        if len(items) > len(shown):
            body.append(f"_...and {len(items) - len(shown)} more findings in this group._")
        path.write_text("\n".join(body), encoding="utf-8")
        index_lines.append(f"- `{filename}` — {title} ({len(items)})")

    (out / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"Wrote {count} issue files to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
