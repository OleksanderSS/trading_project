"""
Static module engagement diagnostic.

Run:
    python diagnostics/module_diagnostic.py --root src --out diagnostic_reports
"""

from __future__ import annotations

import argparse, ast, csv, json, re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

CATEGORY_PATTERNS = {
    "enricher": ["Enricher"], "calculator": ["Calculator", "Metric"],
    "target_calculator": ["Target", "RegressionCalculator", "ClassificationCalculator"],
    "validator": ["Validator", "Guard", "Checker"], "collector": ["Collector", "DataSource"],
    "model": ["Model", "Estimator"], "factory": ["Factory", "Registry"],
    "analyzer": ["Analyzer"], "detector": ["Detector"], "selector": ["Selector"],
    "orchestrator": ["Orchestrator", "Pipeline", "Stage"], "risk": ["Risk", "VaR", "Drawdown", "Sharpe"],
}
HIGH_RISK_PATTERNS = [
    ("P0","TEMPORAL_NEGATIVE_SHIFT",re.compile(r"\.shift\(\s*-\s*\w*|\bshift\s*=\s*-\d+"),"Future shift. Check ticker grouping."),
    ("P0","TARGET_IN_FEATURE_MODULE",re.compile(r"target_[A-Za-z0-9_]*|f['\"]target_"),"Feature-like module may create target columns."),
    ("P0","BFILL_CAUSAL_RISK",re.compile(r"\.bfill\(|method\s*=\s*['\"]bfill['\"]"),"Backfill can leak future data."),
    ("P1","FILLNA_ZERO_RISK",re.compile(r"\.fillna\(\s*0\s*\)"),"fillna(0) can hide missing data."),
    ("P0","TRAIN_TEST_SPLIT_TIMESERIES",re.compile(r"\btrain_test_split\s*\("),"Random split is dangerous for time-series."),
    ("P1","SYNTHETIC_PRIMARY_SCORE",re.compile(r"combined_metric\s*=|0\.7\s*\*\s*real_metric|0\.3\s*\*\s*synthetic_metric"),"Synthetic score may affect primary selection."),
    ("P1","TOP_LEVEL_TENSORFLOW_IMPORT",re.compile(r"^import tensorflow|^from tensorflow", re.M),"Heavy TensorFlow import."),
    ("P2","BROAD_EXCEPT",re.compile(r"except Exception"),"Broad exception. Classify fatal/degraded paths."),
]

@dataclass
class ModuleRecord:
    path: str; module: str; classes: str; functions: str; category_guess: str; imports_count: int; imported_by_count: int; risk_count: int

@dataclass
class ImportRecord:
    source: str; imported: str; kind: str; line: int

@dataclass
class RiskFinding:
    severity: str; rule_id: str; file: str; line: int; snippet: str; why: str

def iter_py(root): return [p for p in Path(root).rglob("*.py") if "__pycache__" not in p.parts]
def modname(root, p):
    rel = p.relative_to(root).with_suffix("")
    return ".".join(("src", *rel.parts)) if Path(root).name == "src" else ".".join(rel.parts)

def parse(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    try: return text, ast.parse(text)
    except SyntaxError: return text, None

def guess(path, classes, funcs):
    hay = " ".join([path, *classes, *funcs]).lower()
    scores = {c: sum(1 for p in pats if p.lower() in hay) for c,pats in CATEGORY_PATTERNS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "utility_or_unknown"

def extract_imports(tree, source):
    out=[]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names: out.append(ImportRecord(source,a.name,"import",node.lineno))
        elif isinstance(node, ast.ImportFrom):
            mod=node.module or ""
            for a in node.names: out.append(ImportRecord(source, f"{mod}.{a.name}" if mod else a.name, "from", node.lineno))
    return out

def write_csv(path, rows):
    rows=list(rows); Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows: Path(path).write_text("", encoding="utf-8"); return
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="src"); ap.add_argument("--out",default="diagnostic_reports"); args=ap.parse_args()
    root=Path(args.root); out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data={}; imports=[]; risks=[]
    for p in iter_py(root):
        rel=str(p.relative_to(root)).replace("\\","/"); text, tree=parse(p); classes=[]; funcs=[]; imps=[]
        if tree:
            classes=[n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            funcs=[n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef, ast.AsyncFunctionDef))]
            imps=extract_imports(tree, rel); imports.extend(imps)
        fr=[]
        for sev,rid,rx,why in HIGH_RISK_PATTERNS:
            if rid=="TARGET_IN_FEATURE_MODULE" and rel.startswith("targets/"): continue
            for m in rx.finditer(text):
                line=text[:m.start()].count("\n")+1; snippet=text.splitlines()[line-1].strip() if text.splitlines() else ""
                if "audit-ignore" in snippet: continue
                fr.append(RiskFinding(sev,rid,rel,line,snippet,why))
        risks.extend(fr)
        data[rel]={"module":modname(root,p),"classes":classes,"functions":funcs,"imports":imps,"risks":fr,"category":guess(rel,classes,funcs)}
    imported_by=defaultdict(set)
    for imp in imports:
        for rel,d in data.items():
            if imp.imported == d["module"] or imp.imported.startswith(d["module"]+".") or d["module"].endswith(imp.imported):
                imported_by[rel].add(imp.source)
    records=[ModuleRecord(rel,d["module"],";".join(d["classes"]),";".join(d["functions"][:50]),d["category"],len(d["imports"]),len(imported_by.get(rel,set())),len(d["risks"])) for rel,d in data.items()]
    orphans=[r.path for r in records if r.imported_by_count==0 and not r.path.endswith("__init__.py")]
    write_csv(out/"module_inventory.csv", [asdict(r) for r in records])
    write_csv(out/"static_imports.csv", [asdict(i) for i in imports])
    write_csv(out/"risk_findings.csv", [asdict(r) for r in risks])
    (out/"orphan_modules.txt").write_text("\n".join(orphans), encoding="utf-8")
    summary={"python_files":len(data),"classes":sum(len(d["classes"]) for d in data.values()),"functions":sum(len(d["functions"]) for d in data.values()),"risk_findings":len(risks),"orphans_static":len(orphans)}
    (out/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out/"diagnostic_summary.md").write_text("# Module Diagnostic Summary\n\n"+"\n".join(f"- **{k}**: {v}" for k,v in summary.items()), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
if __name__=="__main__": main()
