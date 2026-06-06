from __future__ import annotations
import argparse,json,re
from collections import defaultdict
from pathlib import Path
MODEL_NAME_RE=re.compile(r"['\"]([A-Za-z0-9_\-]*(?:LightGBM|XGBoost|RandomForest|LSTM|GRU|CNN|Transformer|Autoencoder|CatBoost|TabNet|MLP|SVM|Ridge|Lasso|ElasticNet)[A-Za-z0-9_\-]*)['\"]",re.I)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="src"); ap.add_argument("--out",default="diagnostic_reports"); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); refs=defaultdict(list); reg=[]; lazy=[]; fac=[]
    for p in Path(args.root).rglob("*.py"):
        if "__pycache__" in p.parts: continue
        rel=str(p.relative_to(args.root)).replace("\\","/"); text=p.read_text(encoding="utf-8",errors="ignore")
        if "factory" in rel.lower(): fac.append(rel)
        if "model_registry" in text or "ModelRegistry" in text: reg.append(rel)
        if "lazy_loader" in text or "LazyLoader" in text or "lazy_import" in text: lazy.append(rel)
        for m in MODEL_NAME_RE.finditer(text): refs[m.group(1).lower()].append(rel)
    dups={k:sorted(set(v)) for k,v in refs.items() if len(set(v))>1}
    report={"factory_files":sorted(fac),"registry_mentions":sorted(set(reg)),"lazy_loader_mentions":sorted(set(lazy)),"duplicated_model_name_references":dups}
    (out/"registry_consistency_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"registry_consistency_report.md").write_text("# Registry Consistency Report\n\n"+f"- Factory files: {len(fac)}\n- Registry mentions: {len(set(reg))}\n- Lazy mentions: {len(set(lazy))}\n- Duplicate model refs: {len(dups)}",encoding="utf-8")
    print(json.dumps({k:len(v) if isinstance(v,list) else len(v) for k,v in report.items()},indent=2))
if __name__=="__main__": main()
