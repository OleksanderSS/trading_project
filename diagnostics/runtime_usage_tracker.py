from __future__ import annotations
import atexit, functools, json, os, time
from collections import Counter, defaultdict
from pathlib import Path
class RuntimeUsageTracker:
    counter=Counter(); metadata=defaultdict(list); events=[]; enabled=os.environ.get("DIAGNOSTIC_RUNTIME_TRACKING","1")!="0"; output=Path(os.environ.get("DIAGNOSTIC_RUNTIME_REPORT","diagnostic_reports/runtime_usage_report.json"))
    @classmethod
    def record(cls,name,metadata=None):
        if not cls.enabled: return
        metadata=metadata or {}; cls.counter[name]+=1; cls.events.append({"name":name,"timestamp":time.time(),"metadata":metadata})
        if metadata: cls.metadata[name].append(metadata)
    @classmethod
    def report(cls):
        return {"total_calls":sum(cls.counter.values()),"unique_components_called":len(cls.counter),"called_components":dict(cls.counter.most_common()),"metadata_samples":{k:v[:5] for k,v in cls.metadata.items()},"events_tail":cls.events[-100:]}
    @classmethod
    def save(cls,path=None):
        out=Path(path) if path else cls.output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(cls.report(),indent=2,ensure_ascii=False),encoding="utf-8")
def track_call(name=None, metadata_fn=None):
    def deco(func):
        comp=name or f"{func.__module__}.{func.__qualname__}"
        @functools.wraps(func)
        def wrap(*args,**kwargs):
            md={}
            if metadata_fn:
                try: md=metadata_fn(*args,**kwargs) or {}
                except Exception: md={"metadata_error":True}
            RuntimeUsageTracker.record(comp,md); return func(*args,**kwargs)
        return wrap
    return deco
def patch_method(cls, method_name, component_name=None):
    if not hasattr(cls,method_name): return False
    orig=getattr(cls,method_name)
    if getattr(orig,"_diagnostic_tracked",False): return True
    wrapped=track_call(component_name or f"{cls.__module__}.{cls.__name__}.{method_name}")(orig); wrapped._diagnostic_tracked=True; setattr(cls,method_name,wrapped); return True
atexit.register(lambda: RuntimeUsageTracker.save())
