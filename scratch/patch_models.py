
import os

def patch_models_yaml():
    path = r'D:\trading_project\src\config\models.yaml'
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if 'light: ["catboost"' in line:
            new_lines.append('    light: ["catboost", "lightgbm", "xgboost", "random_forest", "linear", "svm", "knn", "mlp", "ensemble"]\n')
        elif 'heavy: ["mlp", "cnn"' in line:
            new_lines.append('    heavy: ["cnn", "lstm", "gru", "transformer", "tabnet", "autoencoder"]\n')
        else:
            new_lines.append(line)
            
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Patched {path}")

def patch_modeling_stage():
    path = r'D:\trading_project\src\pipeline\stages\stage_4_modeling.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_code = """    def _get_light_model_types(self):
        \"\"\"Get list of light model types to train.\"\"\"
        return ['catboost', 'lightgbm', 'xgboost', 'random_forest', 'linear', 'svm', 'knn']"""
        
    new_code = """    def _get_light_model_types(self):
        \"\"\"Get list of light model types to train from config.\"\"\"
        models_config = self.config_manager.get_config('models') or {}
        
        # Try to get from dual_model_manager or categories.light
        light_models = models_config.get('dual_model_manager', {}).get('light_models', [])
        if not light_models:
            light_models = models_config.get('categories', {}).get('light', [])
        
        # Final fallback to a sensible default if config is missing
        if not light_models:
            light_models = ['catboost', 'lightgbm', 'xgboost', 'random_forest', 'linear', 'svm', 'knn']
            
        return light_models"""
    
    if old_code in content:
        new_content = content.replace(old_code, new_code)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Patched {path}")
    else:
        print(f"Could not find old_code in {path}")

if __name__ == "__main__":
    patch_models_yaml()
    patch_modeling_stage()
