# Unified Financial Intelligence Ecosystem

This repository contains a comprehensive, multi-stage platform for financial market analysis, prediction, and trading. The system is designed based on DEAN Principles (Dynamic, Evolving, Autonomous, and Networked) to create a self-correcting and adaptive intelligence ecosystem.

## Core System

The main pipeline consists of a 7-stage process that handles everything from data collection and processing to feature engineering, model training, prediction, and simulated trading. It is designed for high-throughput batch processing and complex causal analysis.

**For a detailed explanation of the core system, its scientific foundation, and multi-stage pipeline, please see the [ARCHITECTURE.md](ARCHITECTURE.md) document.**

## Serverless NLP Microservice

As part of this ecosystem, the project also includes a real-time, event-driven microservice for news sentiment analysis.

-   **Technology**: Google Cloud Functions, Google Cloud Storage
-   **Function**: Automatically analyzes the sentiment of news articles uploaded to a GCS bucket using the FinBERT NLP model.
-   **Purpose**: Provides a scalable, serverless way to enrich the system's data with real-time sentiment scores, which can be consumed by the main pipeline.

**Full details, workflow, and deployment commands for this microservice are documented in Section 6 of the [ARCHITECTURE.md](ARCHITECTURE.md) document.**
