#!/usr/bin/env python3
"""Create a compact triage report from deep_static_audit findings."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SEVERITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "INFO": 4}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--findings", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-per-rule", type=int, default=5)
    p.add_argument("--severity", default="P0,P1")
    args = p.parse_args()

    findings = json.loads(Path(args.findings).read_text(encoding="utf-8"))
    allowed = {s.strip().upper() for s in args.severity.split(",") if s.strip()}
    selected = [f for f in findings if f.get("severity") in allowed]

    by_rule = defaultdict(list)
    for f in selected:
        by_rule[(f.get("severity"), f.get("category"), f.get("rule_id"))].append(f)

    lines = ["# Audit Triage Report\n"]
    lines.append(f"Selected severities: {', '.join(sorted(allowed, key=lambda s: SEVERITY_ORDER.get(s, 99)))}")
    lines.append(f"Selected findings: **{len(selected)}**\n")

    lines.append("## Counts\n")
    lines.append("### By severity")
    sev_counts = Counter(f.get("severity") for f in selected)
    for sev in sorted(sev_counts, key=lambda s: SEVERITY_ORDER.get(str(s), 99)):
        lines.append(f"- {sev}: {sev_counts[sev]}")
    lines.append("\n### By rule")
    for key, items in sorted(by_rule.items(), key=lambda kv: (SEVERITY_ORDER.get(str(kv[0][0]), 99), -len(kv[1]), kv[0])):
        sev, cat, rule = key
        lines.append(f"- {sev} {cat}/{rule}: {len(items)}")

    lines.append("\n---\n")
    for key, items in sorted(by_rule.items(), key=lambda kv: (SEVERITY_ORDER.get(str(kv[0][0]), 99), -len(kv[1]), kv[0])):
        sev, cat, rule = key
        lines.append(f"## {sev} {cat} / {rule} — {len(items)} finding(s)\n")
        first = items[0]
        lines.append(f"**Problem:** {first.get('problem', '')}")
        lines.append(f"**Why:** {first.get('why', '')}")
        lines.append(f"**Fix:** {first.get('fix', '')}")
        lines.append(f"**Test:** {first.get('test', '')}\n")
        lines.append("### Examples")
        for f in items[: args.max_per_rule]:
            lines.append(f"- `{f.get('file')}:{f.get('line')}` fingerprint `{f.get('fingerprint')}` confidence `{f.get('confidence')}`")
        if len(items) > args.max_per_rule:
            lines.append(f"- ... {len(items) - args.max_per_rule} more")
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
