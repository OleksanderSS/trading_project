#!/usr/bin/env python
import yaml
import sys

try:
    with open('src/config/targets.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"✅ YAML valid. Keys: {list(cfg.keys())}")
    targets = cfg.get('targets', {})
    print(f"✅ Targets count: {len(targets)}")
    print(f"✅ First 5 targets: {list(targets.keys())[:5]}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
