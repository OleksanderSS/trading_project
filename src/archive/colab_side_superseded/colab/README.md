# src/colab/ — Colab-Side Execution Module

This module runs **inside Google Colab**, not locally.

## Purpose
Provides utilities for the heavy training phase that runs in Colab:
- `config/` — loads training configuration from batch files prepared by `--mode prepare`
- `environment/` — sets up Colab environment (GPU, Drive mount, dependencies)
- `memory/` — monitors GPU/RAM usage during heavy model training
- `models/` — model factory and PyTorch model definitions for LSTM, GRU, Transformer, etc.
- `utils/` — batch size tuning, data signatures, retry logic, metrics

## Workflow
```
Local:  python run_hybrid_pipeline.py --mode prepare
          ↓ creates data/colab/accumulated/<batch_name>/
Colab:  upload batch folder → run notebook using src/colab/
          ↓ trains heavy models, saves colab_results_summary.json
Local:  python run_hybrid_pipeline.py --mode continue --batch-name <batch_name>
```

## Not imported by local pipeline
This module is intentionally not imported by `run_hybrid_pipeline.py` or any stage.
It is copied/uploaded to Colab and executed there.
