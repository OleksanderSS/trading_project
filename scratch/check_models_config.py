import sys
from pathlib import Path
import json

project_path = Path("D:/trading_project")
sys.path.insert(0, str(project_path))
sys.path.insert(0, str(project_path / "src"))

from src.config.unified_config_manager import UnifiedConfigManager

manager = UnifiedConfigManager()
models_config = manager.get_config('models')

print("=== MODELS CONFIG KEYS ===")
print(list(models_config.keys()))

print("\n=== PER_MODEL SECTION ===")
if 'per_model' in models_config:
    print(json.dumps(models_config['per_model'], indent=2))
else:
    print("NOT FOUND IN ROOT")

print("\n=== TRYING TO READ YAML DIRECTLY ===")
import yaml
with open(project_path / "src" / "config" / "models.yaml", "r") as f:
    raw_yaml = yaml.safe_load(f)
    print("Raw YAML root keys:", list(raw_yaml.keys()))
    if 'per_model' in raw_yaml:
        print("Raw YAML per_model:", json.dumps(raw_yaml['per_model'], indent=2))
