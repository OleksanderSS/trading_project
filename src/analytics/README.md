# Strategic Intelligence & Reporting Hub (Stages 2 & 7) - DEAN Trading System

The Analytics module is the "Strategic Intelligence" center of the DEAN (Dynamic Ensemble Analysis Network) architecture. It provides a unified interface for market context detection (**Stage 2**) and comprehensive strategy evaluation (**Stage 7**), ensuring every signal and regime change is quantified through a professional risk and performance lens.

## Overview

The Analytics module is designed to be plug-and-play. Instead of hard-coding metrics, the system uses a dynamic orchestration engine that loads analyzers based on configuration. This allows for rapid experimentation with new alpha factors, risk metrics, or contextual filters without modifying the core pipeline.

---

## Core Components

### 1. UnifiedAnalyticsEngine
The `UnifiedAnalyticsEngine` is the central orchestrator. It is responsible for:
*   **Dynamic Loading**: Importing analyzer classes at runtime using `importlib`.
*   **Data Mapping**: Routing specific data streams (price, news, macro) to the relevant analyzers.
*   **Parallel Execution**: Running multiple heavy analyses concurrently using a `ThreadPoolExecutor` to minimize latency.
*   **Result Aggregation**: Consolidating outputs into a unified dictionary for reports and downstream decision-making.

### 2. IAnalyzer Interface
All analysis components implement the `IAnalyzer` abstract base class found in `src/analytics/interfaces.py` to maintain strict modularity.

---

## Module Structure

The following sub-packages form the intelligence ecosystem:

| Sub-package | Role | Primary Stage |
| :--- | :--- | :--- |
| **`analyzers/`** | Specialized modules for deep data analysis (e.g., News Impact, Hedge Fund styles, KNN Similarity). | Stage 2 / 7 |
| **`calculators/`** | Core mathematical calculations for risk, volatility, Drawdowns, and Fama-French factors. | Stage 7 |
| **`context/`** | Market regime detection, macro analysis, and causal modeling for context-aware trading. | Stage 2 |
| **`detectors/`** | Real-time anomaly and critical event detection (e.g., price shocks, volume spikes). | Stage 2 |
| **`reporting/`** | Automated generation of reports, visualizations, and the "Arena" for model comparison. | Stage 7 |

---

## Registered Analyzers (Examples)

| Analyzer | Description | DEAN Principle |
| :--- | :--- | :--- |
| **CausalEngine** | Projects "Trigger Events" (e.g., Fed hikes) into future "Implied Feature" ripples. | *Causal Vectors* |
| **MarketPhaseAnalyzer**| Identifies the current market regime (Bull, Bear, Sideways). | *Context Awareness* |
| **NewsImpactAnalyzer** | Quantifies the immediate price reaction following high-impact news. | *Sentiment Analysis* |
| **CriticalSignalDetector**| Detects extreme market conditions requiring immediate risk adjustment. | *Anomaly Detection* |
| **AdaptiveThresholdsAnalyzer** | Calculates dynamic significance levels based on current volatility. | *Dynamic Thresholds* |


---

## Configuration

Analyzers are managed via `src/config/analysis.yaml`. To add a new analyzer, define it in the `engine.analyzers` list:

```yaml
engine:
  max_workers: 4
  analyzers:
    - name: "my_new_factor"
      module: "src.analytics.analyzers.my_custom_module"
      class: "MyCustomAnalyzer"
      params:
        window: 14
      data_mapping: ["price_data", "market_regime"]
```

---

## Pipeline Integration

The Analytics module provides the primary source of feedback for the system:

1.  **Stage 2 (Context)**: Detectors and Context Analyzers identify the market environment, providing a "fingerprint" for the models.
2.  **Stage 7 (Evaluation)**: The `EvaluationStage` utilizes the `UnifiedAnalyticsEngine` and the `reporting/` sub-package to generate a comprehensive performance profile.
3.  **Feedback Loop**: Results are used to update the `ExperienceDiary` (Meta-Learning), generate Markdown reports, and trigger automated notifications.

---

## DEAN Philosophy: Beyond Simple Metrics
Unlike traditional systems that only look at Sharpe Ratios, the DEAN Analytics module seeks to understand **why** a strategy is performing. By using the `CausalEngine` and `MarketPhaseAnalyzer`, the system distinguishes between alpha and random noise, allowing the `AdaptiveThresholds` to tighten or loosen risk limits in real-time.