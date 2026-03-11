# 📦 Stages - The Core Pipeline Logic

This directory contains the core logic of the data processing pipeline, broken down into sequential "stages." Each stage is a self-contained module responsible for a specific part of the data transformation process, from initial collection to final prediction.

---

## 🏛️ Architecture: A Shift Towards Asynchronicity and Dynamic Configuration

The pipeline is evolving towards a more modern, flexible, and performant architecture. The refactored `Stage 1` serves as the template for this new design.

### Key Principles:

1.  **Configuration-Driven:** Stages should be driven by the central YAML configuration files located in `src/config`. This avoids hard-coded logic and allows for easy modification of the pipeline's behavior without changing code.
2.  **Asynchronous by Default:** For I/O-bound operations (like data collection, database access, or API calls), stages should leverage Python's `asyncio` to perform tasks concurrently. This dramatically improves performance.
3.  **Modularity:** Each stage should have a clear, well-defined responsibility and operate on the data produced by the previous stage.

---

## 🚀 Stage 1: The Asynchronous Data Collection Engine

`stage_1_collectors_layer.py` is the first and most critical stage. It has been completely refactored to be fully asynchronous and dynamic.

### How it Works:

1.  **No Hard-coded Logic:** The stage does **not** import or even know about specific collector classes.
2.  **Dynamic Instantiation:** It uses the `CollectorFactory` (`src/collectors/collector_factory.py`) to find and instantiate all data collectors that are currently enabled in `src/config/collectors.yaml`.
3.  **Concurrent Execution:** It runs the `.collect()` method on all instantiated collectors **concurrently** using `asyncio.gather()`. This means dozens of data sources can be fetched in parallel, maximizing I/O throughput.
4.  **Data Aggregation:** It receives the results from all collectors and aggregates them into a dictionary of pandas DataFrames, grouped by the `type` defined in the YAML config (e.g., `market_data`, `news`, `macro`).
5.  **Resilience:** The failure of a single collector does not stop the entire process. The stage logs the error and continues with the data it successfully collected.

This new design makes the data collection process incredibly efficient and easy to manage. Adding a new data source is now as simple as adding a new entry to a YAML file.

---

## ⏩ Other Stages

Subsequent stages are responsible for:

- **Stage 2: Enrichment:** Adding calculated fields, indicators, and other metadata.
- **Stage 3: Feature Engineering:** Transforming raw data into features suitable for machine learning models.
- **Stage 4: Unification & Modeling:** Combining all data sources into a unified dataset and training models.
- **Stage 5: Prediction:** Using trained models to make predictions.

These stages are the next candidates for refactoring to align with the new asynchronous and configuration-driven architecture.
