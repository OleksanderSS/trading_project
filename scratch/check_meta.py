import json
from pathlib import Path

results_file = Path("data/colab/accumulated/main_database/colab_results.json")
if results_file.exists():
    with open(results_file, "r") as f:
        data = json.load(f)
    print("Keys in colab_results:", list(data.keys()))
    if "models_metadata" in data:
        meta = data["models_metadata"]
        print("Number of models in metadata:", len(meta))
        tsm_keys = [k for k in meta.keys() if "TSM" in k]
        print("TSM models:", tsm_keys)
        for k in tsm_keys:
            if "lstm" in k:
                print(k, "selected_features length:", len(meta[k].get("selected_features", [])))
                print(k, "keys:", list(meta[k].keys()))
else:
    print("colab_results.json not found!")
