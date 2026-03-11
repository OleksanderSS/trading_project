# Integrations Module (`src/integrations`)

This module serves as the **External Gateway** of the project, responsible for **Bridging the System with External Platforms (Cloud, CI/CD)**. It ensures that the local pipeline can seamlessly communicate with enterprise-grade infrastructure and automation tools.

### Core Components

1.  **`bigquery/`**: Facilitates enterprise-level data storage and retrieval using Google Cloud Platform. It allows the system to offload massive historical datasets and store final analytical results in a scalable environment.
2.  **`ci_cd/`**: Handles integration with version control pipelines (e.g., GitHub Actions, GitLab CI) to enable automated testing, code quality checks, and seamless deployment of model updates.

### Pipeline Integration

*   **Stage 1 (Data Collection)**: Cloud connectors are used to fetch raw data from or archive it to BigQuery for persistent, long-term storage.
*   **Stage 7 (Evaluation & Reporting)**: Final performance metrics and strategy reports are exported to external platforms for centralized monitoring and stakeholder access.

This module ensures that the project remains scalable, maintainable, and ready for production-grade deployment across hybrid environments.