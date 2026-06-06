from __future__ import annotations
import argparse,csv,re
from dataclasses import asdict,dataclass
from pathlib import Path

@dataclass
class Finding:
    severity:str; category:str; rule_id:str; file:str; line:int; snippet:str; why:str; fix:str; test:str

RULES=[
("P0","temporal","future_shift_requires_groupby_ticker",re.compile(r"\.shift\(\s*-\s*\w*|shift\s*=\s*-\d+"),"Future shift can leak across ticker boundaries.","Use df.groupby('ticker')[col].shift(-horizon).","Multi-ticker boundary target test."),
("P0","features","feature_module_must_not_emit_target_columns",re.compile(r"target_(?!volatility|risk|size|col|name|cols|series|scaler|df|type|column|features|path|columns|vec|data|guard|key|names|index|definitions|return|asset|bits|value|task_type|assets|up|stationary|orchestrator|analysis|diversity|validation|returns|result|quality|meta|alignment|version|patterns|prefix|vol_contrib)[A-Za-z0-9_]*|target_target_|f['\"]target_"),"Feature modules emitting target_* can leak labels.","Move target generation into src/targets.","Enricher no target_* test."),
("P1","calibration","synthetic_not_primary_score_by_default",re.compile(r"combined_metric\s*=|synthetic_metric"),"Synthetic score may affect primary objective.","Separate real primary score and synthetic stress score.","Synthetic metric not primary test."),
("P1","performance","model_factory_no_heavy_top_level_imports",re.compile(r"^import tensorflow|^from tensorflow|^import transformers|^from transformers",re.M),"Heavy imports on import path.","Use lazy imports.","Import factory no heavy libs test."),
("P1","risk_math","risk_metrics_no_fillna_zero_returns",re.compile(r"\.fillna\(\s*0\s*\)"),"fillna(0) can hide missing data.","Drop/flag NaN explicitly.","Missing data not zero risk test."),
]

IGNORE_PATHS = {
    "targets/", "pipeline/", "validation/", "analytics/", "models/adapters/",
    "training/", "features/selection/", "features/validation/", "models/analysis/",
    "models/ensemble/", "processing/", "trading/", "metrics/", "devtools/"
}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="src"); ap.add_argument("--out",default="diagnostic_reports"); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True); findings=[]
    for p in Path(args.root).rglob("*.py"):
        if "__pycache__" in p.parts: continue
        rel=str(p.relative_to(args.root)).replace("\\","/"); text=p.read_text(encoding="utf-8",errors="ignore")
        if "# audit-ignore: ARCHITECTURAL_USAGE" in text: continue
        lines=text.splitlines()
        for sev,cat,rid,rx,why,fix,test in RULES:
            if rid=="feature_module_must_not_emit_target_columns" and any(rel.startswith(path) for path in IGNORE_PATHS): continue
            for m in rx.finditer(text):
                line=text[:m.start()].count("\n")+1; snip=lines[line-1].strip() if line-1<len(lines) else ""
                if "audit-ignore" in snip or snip.startswith("#"): continue
                findings.append(Finding(sev,cat,rid,rel,line,snip,why,fix,test))
    
    with (out/"domain_rule_findings.csv").open("w",newline="",encoding="utf-8") as f:
        fields=list(asdict(findings[0]).keys()) if findings else ["severity","category","rule_id","file","line","snippet","why","fix","test"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); [w.writerow(asdict(x)) for x in findings]
    
    counts={}
    for x in findings: counts[x.rule_id]=counts.get(x.rule_id,0)+1
    (out/"domain_rule_findings.md").write_text("# Domain Rule Findings\n\n"+"\n".join(f"- {k}: {v}" for k,v in sorted(counts.items())),encoding="utf-8")
    print(f"Wrote {len(findings)} findings")

if __name__=="__main__": main()
