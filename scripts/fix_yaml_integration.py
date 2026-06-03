#!/usr/bin/env python3
"""
Simple fix to make feature selectors read from YAML configuration.

No duplication - just fix existing selectors to use config.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Fix existing feature selectors to use YAML configuration."""
    print("🔧 Fixing YAML integration for feature selectors...")
    
    # Files to fix
    files_to_fix = [
        'src/features/feature_selector.py',
        'src/features/colab_context_integration.py',
        'src/features/improved_colab_selector.py'
    ]
    
    for file_path in files_to_fix:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"⚠️ File not found: {file_path}")
            continue
        
        print(f"📝 Fixing {file_path}...")
        
        # Read file
        with open(full_path, 'r') as f:
            content = f.read()
        
        # Simple fix - replace hardcoded values with YAML-based function
        if '_get_model_max_features' in content:
            # Replace the function with YAML integration
            new_function = '''    def _get_model_max_features(self, model_type: str) -> int:
        """Get max features from YAML configuration."""
        from src.config.unified_config_manager import get_current_config
        
        try:
            config_manager = get_current_config()
            models_config = config_manager.get('models', {})
            per_model = models_config.get('per_model', {})
            
            if model_type.lower() in per_model:
                max_features = per_model[model_type.lower()].get('max_features')
                if max_features is not None:
                    return int(max_features)
            
            # Fallback to defaults matching models.yaml
            defaults = {
                'mlp': 100, 'lstm': 110, 'gru': 105, 'cnn': 95,
                'transformer': 115, 'tabnet': 90, 'autoencoder': 100,
                'random_forest': 40, 'catboost': 45, 'lightgbm': 50,
                'xgboost': 48, 'linear': 35, 'svm': 42, 'knn': 38
            }
            return defaults.get(model_type.lower(), 100)
        except Exception as e:
            print(f"Error reading YAML config: {e}")
            return 100  # Safe fallback'''
            
            # Find and replace the function
            import re
            pattern = r'def _get_model_max_features\(self, model_type: str\) -> int:.*?return max_features_map\.get\(model_type\.lower\(\), \d+\)'
            
            if re.search(pattern, content, re.DOTALL):
                content = re.sub(pattern, new_function, content, flags=re.DOTALL)
                print(f"  ✅ Fixed _get_model_max_features in {file_path}")
            else:
                print(f"  ⚠️ Could not find function to fix in {file_path}")
        
        # Write back
        with open(full_path, 'w') as f:
            f.write(content)
    
    print("\n✅ YAML integration fix completed!")
    print("Now feature selectors will read from models.yaml configuration")


if __name__ == "__main__":
    main()
