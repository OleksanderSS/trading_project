import json
import os
import numpy as np
import joblib
from pathlib import Path

# Check what scaler files exist
batch_dir = Path('data/colab/accumulated/main_database')
print("=== Files in batch directory ===")
for f in sorted(batch_dir.iterdir()):
    size = f.stat().st_size if f.is_file() else 0
    print(f"  {f.name} ({size:,} bytes)")

# Check scaler files
scaler_files = list(batch_dir.glob('scaler_*.pkl'))
print(f"\n=== Scaler files: {len(scaler_files)} ===")
for sf in scaler_files:
    try:
        scaler = joblib.load(sf)
        print(f"\n  {sf.name}:")
        print(f"    Type: {type(scaler).__name__}")
        if hasattr(scaler, 'scale_'):
            print(f"    scale_: {scaler.scale_}")
            print(f"    mean_: {scaler.mean_}")
            print(f"    n_features: {scaler.scale_.shape[0]}")
        elif hasattr(scaler, 'center_'):
            print(f"    center_: {scaler.center_}")
        else:
            print(f"    Attributes: {[a for a in dir(scaler) if not a.startswith('_')]}")
    except Exception as e:
        print(f"  {sf.name}: ERROR - {e}")

# Check models metadata
meta_path = batch_dir / 'models_metadata.json'
if meta_path.exists():
    meta = json.load(open(meta_path))
    print(f"\n=== Models metadata: {len(meta)} entries ===")
    for cid, m in list(meta.items())[:5]:
        print(f"\n  {cid}:")
        print(f"    ticker: {m.get('ticker')}")
        print(f"    target: {m.get('target')}")
        print(f"    model_type: {m.get('model_type')}")
        print(f"    model_path: {m.get('model_path')}")
        print(f"    selected_features count: {len(m.get('selected_features', []))}")

# Check runtime_params
rp = batch_dir / 'runtime_params.json'
if rp.exists():
    params = json.load(open(rp))
    print(f"\n=== Runtime params ===")
    for k, v in params.items():
        print(f"  {k}: {v}")
