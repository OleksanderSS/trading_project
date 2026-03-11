# Meta-Learning Module

This module forms the **Self-Improving Intelligence Hub (Stages 4, 6, 7)** of the trading system, enabling it to learn how to learn and adapt its behavior over time. Meta-learning sits 'above' regular machine learning models, monitoring their performance and the market environment to improve overall system outcomes.

## Components

- **`experience_diary.py`**: Acts as the system's persistent "memory." It provides storage for all model results, predictions, and their eventual outcomes, linked directly to the specific market context. This forms a database of 'lessons learned' that prevents the repetition of past mistakes.

- **`dual_learning_loops.py`**: Enables a dual-pathway learning architecture. It allows the system to simultaneously train on price data and its own prediction errors, balancing immediate adaptation with long-term structural understanding.

- **`realtime_context_awareness.py`**: Provides dynamic market fingerprinting for context-aware predictions. By understanding the immediate market regime, this module enables the system to switch model weights or strategies based on the detected context.

- **`optimization/bayesian_optimizer.py`**: A specialized tool for automated hyperparameter tuning. It is used to optimize the parameters of individual models as well as the meta-learning system's own decision-making logic.

## Integration

The Meta-Learning module is deeply integrated with the following project components:
- **`src/analytics/context/`**: For receiving detailed market regime data.
- **`src/models/model_selector/`**: For informing the selection of the best-performing models in specific contexts.

## Goal

The goal of this module is to ensure that the trading system does not remain static. By analyzing its own performance and the evolving market, the Meta-Learning module allows the agent to evolve, reducing model drift and improving signal confidence over time. This ensures that the most effective models and strategies are prioritized in any given market condition.