# Core Module (`src/core`)

This directory contains the **Project Infrastructure (Internal Organs)**. It provides the essential system-level services required for all other modules to function reliably. 

As the lowest level of the architecture, the **CORE** module is designed to be highly independent; it SHOULD NOT depend on other `src/` modules (with the exception of `src/utils/`).

## Sub-packages

- **`logging/`**: Unified logging system and cross-platform notifications (Telegram/Email) used for system-wide status reporting.
- **`error_handling/`**: Global error management, exception tracking, and centralized retry logic to ensure system resilience.
- **`cache/`**: High-performance object and query caching systems to reduce database load and API latency.
- **`clients/`**: Base HTTP client factories equipped with rate-limiting and connection pooling for data collection.
- **`security/`**: Secure management of API keys, credentials, and encrypted secrets.
- **`file_management/`**: Centralized utilities for safe file operations, directory structure management, and path resolution.
- **`system/`**: System versioning, batch processing primitives, and low-level lifecycle management.

## Architectural Role

The Core module acts as the foundation of the entire project. While higher-level stages (like Modeling or Trading) focus on business logic and financial intelligence, the Core ensures that the underlying infrastructure—memory, files, network, and security—is stable and performant.