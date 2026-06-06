"""Worker called as subprocess by component_harness_runner.py"""
import sys, json
sys.path.insert(0, '.')

import pandas as pd

METHOD_CANDIDATES = ["enrich", "calculate", "compute", "analyze", "detect",
                     "select", "validate", "transform", "run"]

def make_sample_df():
    return pd.DataFrame({
        "ticker": ["A"] * 8 + ["B"] * 8,
        "timestamp": list(pd.date_range("2024-01-01", periods=8, freq="D")) * 2,
        "open":   [100,101,102,103,104,105,106,107, 200,199,198,197,196,195,194,193],
        "high":   [102,103,104,105,106,107,108,109, 202,201,200,199,198,197,196,195],
        "low":    [99, 100,101,102,103,104,105,106, 198,197,196,195,194,193,192,191],
        "close":  [101,102,103,104,105,106,107,108, 199,198,197,196,195,194,193,192],
        "volume": [1000,1100,1200,1300,1400,1500,1600,1700,
                   2000,2100,2200,2300,2400,2500,2600,2700],
    }).sort_values(["ticker","timestamp"]).reset_index(drop=True)

def instantiate(cls):
    for fn in [lambda: cls(), lambda: cls(config={}), lambda: cls({})]:
        try: return fn()
        except Exception as exc: last = exc
    raise last

def call_method(obj, method_name, df):
    import inspect
    method = getattr(obj, method_name)
    sig = inspect.signature(method)
    params = sig.parameters
    if len(params) == 0:
        return method()
    first = next(iter(params.values()))
    args, kwargs = [], {}
    if first.kind in (first.POSITIONAL_ONLY, first.POSITIONAL_OR_KEYWORD):
        args.append(df.copy())
    elif "df" in params:
        kwargs["df"] = df.copy()
    elif "data" in params:
        kwargs["data"] = df.copy()
    for k, v in [("base_col","close"),("target_col","close"),
                  ("returns_col","close"),("shift",-1),("horizon",1)]:
        if k in params: kwargs[k] = v
    return method(*args, **kwargs)

def compare(before, after):
    if not isinstance(after, pd.DataFrame):
        return [], [], 0, ["NON_DATAFRAME_OUTPUT"]
    before_cols, after_cols = set(before.columns), set(after.columns)
    added = sorted(after_cols - before_cols)
    removed = sorted(before_cols - after_cols)
    modified = sum(1 for c in before_cols & after_cols
                   if not before[c].equals(after[c]))
    warnings = []
    if len(before) != len(after): warnings.append("ROW_COUNT_CHANGED")
    if any(c.startswith("target_") for c in added): warnings.append("TARGET_COLUMN_ADDED")
    for col in added:
        try:
            if after[col].isna().mean() > 0.5: warnings.append(f"HIGH_NAN:{col}")
        except Exception: pass
    return added, removed, modified, warnings

def run(mod_name, class_name):
    df = make_sample_df()
    r = {"comp": f"{mod_name}.{class_name}", "mod": mod_name,
         "cls": class_name, "method": "", "status": "INIT", "error": "",
         "added": "", "removed": "", "modified": 0, "warnings": "", "after_len": -1}
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, class_name)
    except Exception as e:
        r.update({"status": "IMPORT_FAILED", "error": repr(e)[:150]})
        return r
    try:
        obj = instantiate(cls)
    except Exception as e:
        r.update({"status": "INSTANTIATE_SKIPPED", "error": repr(e)[:150]})
        return r
    method_name = next((m for m in METHOD_CANDIDATES if hasattr(obj, m)), "")
    if not method_name:
        r.update({"status": "NO_KNOWN_METHOD"})
        return r
    r["method"] = method_name
    try:
        output = call_method(obj, method_name, df)
        added, removed, modified, warnings = compare(df, output)
        r.update({"status": "EXECUTED", "added": ";".join(added),
                  "removed": ";".join(removed), "modified": modified,
                  "warnings": ";".join(warnings),
                  "after_len": len(output) if hasattr(output, "__len__") else -1})
    except Exception as e:
        r.update({"status": "EXECUTION_FAILED", "error": repr(e)[:150]})
    return r

if __name__ == "__main__":
    mod_name = sys.argv[1]
    class_name = sys.argv[2]
    result = run(mod_name, class_name)
    print(json.dumps(result))
