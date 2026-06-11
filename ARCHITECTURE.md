# ARCHITECTURE: Unified Intelligence Ecosystem (DEAN Principles)

## 1. Core Philosophy: Modularity and Self-Correction
The system is an autonomous financial intelligence platform moving beyond static prediction toward a **self-aware ecosystem**. It bootstraps its own evolution through the interaction of hot-swappable configuration dictionaries and high-performance data processing.
- **UnifiedConfigManager [src/config/unified_config_manager.py]:** The central nervous system governing all components.
- **PipelineOrchestrator [src/pipeline/pipeline_orchestrator.py]:** The executive hub managing the 8-stage data lifecycle (0-7).
- **DataManager [src/data/management/data_manager.py]:** The **Single Source of Truth** for DuckDB access.

## 2. Pipeline Architecture (Stages 0-7)
The system follows a strict sequential lifecycle, orchestrated by **src/pipeline/pipeline_orchestrator.py**, which ensures data integrity, memory management, and checkpointing:

- **Stage 0: Setup** (src/config/ & src/devtools/): Environment initialization, configuration validation via `UnifiedConfigManager`, and pre-flight system integrity checks via `SystemValidator`.
- **Stage 1: Collection** (src/data/collectors/ & src/data/management/): Dynamic discovery and execution of collectors (Market, Macro, News). Data is gathered and stored in DuckDB via `DataManager`. Supports **Synthetic Market Generation** for stress-testing.
- **Stage 2: Processing & Context** (src/processing/ & src/analytics/context/): Data sanitization (cleaning, sampling) and market regime identification to establish the "World State".
- **Stage 3: Features & Targets** (src/features/ & src/targets/): Generation of technical indicators, NLP-based sentiment, and modular enrichers. Performs ticker-aware labeling for regression and classification targets.
- **Stage 4: Modeling** (src/training/ & src/models/): A 3-level training hierarchy (Strategy, Orchestrator, Workers) utilizing the Unified Model Repository. Strictly enforces **Leakage Protection via Purging & Embargo** to safeguard the Train/Test boundary.
- **Stage 5: Prediction** (src/predictions/ & src/ensembling/): Forecast generation and refinement through `StackedEnsemble` consensus. Dynamically weights **Light Models** (LGBM, RF) and **Heavy Models** (Transformers, LSTM) based on context.
- **Stage 6: Trading** (src/trading/ & src/optimization/): Decision execution via **Risk Gates** (Kill-Switch, News Timeout, Exposure Limits). Signals are processed by the `ConsensusEngine` and capital is distributed via the `PortfolioOptimizer`.
- **Stage 7: Strategic Evaluation & Meta-Learning** (src/analytics/reporting/ & src/backtesting/): Performance analytics, **Trading Arena** battles, and **Explainable AI (XAI)** to identify feature contribution and the 'Why' behind every signal.

The **Experience Diary [src/meta_learning/experience_diary.py]** serves as the critical **Feedback Loop**, capturing the "memory" of the system. It enables **Dynamic Model Weighting** by linking XAI insights and performance metadata back to the selection logic in Stage 5.

## 3. Scientific Foundation: Causal Intelligence
The system avoids "curve-fitting" by implementing **Causal Inference** principles within the **CausalEngine [src/analytics/context/causal_engine.py]**:
- **Baseline Benchmarking:** Every ML model competes against a scientific baseline (VAR - Vector Auto Regression) to ensure it adds predictive value.
- **Granger Causality:** Automated testing to distinguish between random correlations and actual causal links between macro factors and price movements.
- **Counterfactuals ("What-If"):** Training models on simulated scenarios to ensure structural robustness in unseen market conditions.
- **Synthetic Resilience Testing:** Validating **Risk Gates** by generating synthetic flash crashes and extreme volatility scenarios.

## 4. Multi-Timeframe Logic (5m, 15m, 1h, 1d)
The system's foundation is a **Dynamic Configuration Dictionary** (located in `assets.yaml` and `unified_config.yaml`).
- **No Hardcoding:** Timeframes are defined by parameters.
- **Synchronization:** All 4 timeframes are aligned in Stage 2 to ensure multi-scale features and Consensus (Stage 6) operate on a consistent temporal view.

## 5. System Standards
- **Consolidation:** All analysis and reporting logic is unified under `src/analytics/`.
- **Feature Hub:** All data enrichment logic is unified under `src/features/enrichers/`.
- **Infrastructure:** All core utilities (Logging, Error Handling, Calendar) reside in `src/core/` and `src/utils/` to serve as the project's foundation.

---

## 6. Serverless NLP Microservice for News Sentiment

This section details an independent, serverless microservice designed for real-time sentiment analysis of news articles. It integrates with the broader system via Google Cloud Storage.

### Overview

The pipeline automates the process of sentiment analysis. It watches a specific GCS folder for new text files, processes them using a pre-trained financial NLP model, and writes the structured results to another GCS folder.

### Components

1.  **Google Cloud Storage (GCS):**
    *   **Input Bucket:** `gs://trading_multi_project/data/raw/`
    *   **Output Bucket:** `gs://trading_multi_project/data/processed/`
    *   Acts as the event source and the destination for processed data.

2.  **Google Cloud Function:**
    *   **Name:** `nlp-news-processor`
    *   **Trigger:** Event-driven, fires on `google.storage.object.finalize` when a new file is created in the input bucket.
    *   **Runtime:** Python 3.11
    *   **Core Logic (`cloud_function/main.py`):**
        *   Uses the `transformers` library to load the `ProsusAI/finbert` model.
        *   Parses the incoming text file.
        *   Performs sentiment analysis.
        *   Outputs a JSON object containing the original text and the sentiment scores.
    *   **Key Configuration:**
        *   **Memory:** `2048MB` - This is a critical requirement, as loading the FinBERT model consumes over 1GB of memory.
        *   **Timeout:** `540s` - A generous timeout to accommodate model download on cold starts.

### Workflow

1.  A user or an automated process uploads a text file (e.g., `my_article.txt`) to `gs://trading_multi_project/data/raw/`.
2.  The GCS event immediately triggers the `nlp-news-processor` Cloud Function.
3.  On its first invocation (cold start), the function downloads the FinBERT model from Hugging Face. This may take several seconds.
4.  The function reads the content of `my_article.txt`.
5.  The text is passed through the FinBERT sentiment analysis pipeline.
6.  The function constructs a JSON object with the results.
7.  The function saves this result to `gs://trading_multi_project/data/processed/my_article_sentiment.json`.

### Deployment

The function is deployed from the local `cloud_function` directory using the following gcloud command. Ensure you are in the project's root directory when running it.

```bash
gcloud functions deploy nlp-news-processor \
--project=seismic-vista-470410-i5 \
--region=europe-central2 \
--runtime=python311 \
--source=./cloud_function \
--entry-point=process_news_file \
--trigger-resource=trading_multi_project \
--trigger-event=google.storage.object.finalize \
--timeout=540s \
--memory=2048MB
```

---

## 7. Code Audit & Recent Fixes (2026-05-08)

This section documents the recent code audit, fixes applied, and structural improvements to the trading_project codebase.

### 7.1 Resolved TODOs & Dead Code Removal
| File | Issue | Fix Applied |
|------|-------|-------------|
| `src/training/unified_training_manager.py` | Pending TODO: `run_battle` missing `actual_targets` parameter | Updated `execute_unified_training` to pass `data_context.get('actual_targets')` to `self.arena.run_battle`, enabling automatic arena battles. |
| `src/pipeline/stages/stage_4_modeling.py` | Dead import: `ModelEnsembleComposer` (module not found) | Removed dead import/comments, retained `EnsembleModel` integration. Ensemble creation now uses existing `EnsembleModel` class with top-N model selection by `r2_score`. |
| `src/core/security/secure_secrets_manager.py` | Pending TODO: `mask_secret` implementation | Confirmed `mask_secret` is fully implemented and functional (masks secrets as `XXXX...XXXX`). |
| `src/trading/adaptive_parameter_manager.py` & `src/processing/data_filter.py` | Dead constants: `MarketRegime.DEAD`, `MarketRegime.VOLATILE` | Removed unused enum values and associated preset configurations to reduce clutter. |

### 7.2 Consolidated Deduplication Logic
Created a new utility module `src/processing/deduplication_utils.py` with a reusable function:
```python
def deduplicate_dataframe(df: pd.DataFrame, subset_cols: List[str]) -> tuple[pd.DataFrame, int]:
    """Drop duplicates on given columns, return cleaned DF + count of removed rows."""
    if not subset_cols:
        return df, 0
    duplicates = int(df.duplicated(subset=subset_cols).sum())
    if duplicates > 0:
        df = df.drop_duplicates(subset=subset_cols)
    return df, duplicates
```
This replaces duplicated deduplication logic in:
- `src/pipeline/stages/stage_1_collection.py` (`_remove_news_duplicates`)
- `src/pipeline/stages/stage_2_processing.py` (`_deduplicate_news_data`)
- `src/processing/data_filter.py` (`_deduplicate_news`)

### 7.3 Ensemble Creation Logic (Stage 4)
- Removed dead `ModelEnsembleComposer` references.
- Ensemble configuration is now generated via `_create_ensemble_from_top_models_async`:
  1. Selects top-N models by `r2_score` (or configurable metric).
  2. Builds `ensemble_config` dictionary with model names, metrics, equal weights.
  3. Persists config to `models_dir/ensemble_{ticker}_{target_name}.json` for use by `TradingModelArena`.

### 7.4 Next Steps & Recommendations
| Area | Action |
|------|--------|
| **Testing** | Add unit tests for: <br>• `run_battle` receiving `actual_targets` correctly.<br>• Ensemble config generation/loading.<br>• `deduplicate_dataframe` correctness across stages. |
| **Deduplication Integration** | Complete replacement of all remaining duplicate-handling logic with `deduplicate_dataframe` calls. |
| **Performance Monitoring** | Enhance memory logging in `pipeline_orchestrator.py` to detect leaks during large DataFrame operations. |
| **Documentation** | Maintain this audit section with future code changes. |