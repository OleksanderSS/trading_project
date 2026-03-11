# Data Acquisition & Management Gateway (Stage 1)

This directory houses the data acquisition and management layer of the trading system, serving as the **Stage 1 (Collection)** foundation for the entire pipeline. It is designed for high-performance, asynchronous operations and reliable incremental updates.

## Core Components

### 1. `collectors/` (Data Acquisition)
Responsible for the dynamic collection of price, news, macro, and fundamental data from various API-based and file-based sources.
- **Market Data**: Polygon, Yahoo Finance, and custom CSV collectors.
- **Macro/Fundamental**: FRED, SEC Filings, Economic Calendars, and Insider Trading.
- **Alternative Data**: Google News, RSS feeds, HuggingFace datasets, and Google Trends.

### 2. `management/` (Storage & Orchestration)
Provides central database access, asset list management, and data versioning.
- **DataManager**: The 'Source of Truth' for all data flowing through the pipeline, leveraging **DuckDB** and **BigQuery** for persistent, high-performance analytical storage.
- **Deduplication**: Uses a robust hashing mechanism to identify and filter out duplicate records.
- **AssetManager**: Manages ticker lists and asset-specific configurations.

### 3. `clients/` (Low-Level API Clients)
Houses low-level API clients (e.g., `YFinanceClient`) that handle the technical details of communication with external providers, including rate limiting and authentication.

## Data Pipeline Flow

1.  **Request**: A collector (managed by `CollectorFactory`) initiates a data fetch via a specific client.
2.  **Deduplicate**: The `DataManager` compares the fetched records against existing records to ensure integrity.
3.  **Normalize**: Raw records are transformed into a unified schema for the downstream pipeline.
4.  **Persist**: Cleaned and unique records are saved into the database for analysis and training.

## Usage

This module is the starting point for every operational mode, including **Train**, **Backtest**, and **Live/Intelligent** trading. The system is configuration-driven, with collectors enabled and tuned via `src/config/collectors.yaml`.

```python
from src.data.management.data_manager import DataManager

# Loading historical data for multiple tickers
data_manager = DataManager()
historical_data = data_manager.load_data_for_tickers(['AAPL', 'TSLA', 'SPY'])
```