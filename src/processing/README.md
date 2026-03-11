# Data Processing Layer (Stage 2: Data Sanitization & Sampling)

This directory contains the logic for the **Data Processing** stage (Stage 2) of the trading system. Its primary role is to transform raw, noisy data from the `src/data` layer into a clean, "sanitized" format. This stage acts as a bridge between raw collection and sophisticated feature engineering.

### Key Components

*   **`cleaners.py`**: Handles data integrity by removing outliers, performing sophisticated NaN imputation, and correcting spikes or errors in historical price series.
*   **`sampling.py`**: Manages timeframe resampling logic (e.g., aggregating 5-minute bars into 1-hour or 1-day candles) and handles data windowing to prepare "skeletons" for the feature layer.
*   **`price_preprocessor.py`**: Standardizes market data. It calculates primary price derivatives such as log-returns and performs normalization/scaling on OHLCV (Open, High, Low, Close, Volume) data.
*   **`parallel_processor.py`**: A performance utility that utilizes multi-core processing to speed up cleaning and transformation tasks on large historical datasets.
*   **`data_filter.py`**: Filters out illiquid tickers or assets with insufficient historical depth before they reach the modeling stage.

### Architecture Update
To maintain strict separation of concerns, all NLP, semantic extraction, and news-specific logic (formerly `news_utils.py` and `ticker_extractor.py`) has been moved to **Stage 3** within `src/features/nlp/`. This ensures that Stage 2 remains focused on numerical sanitization and structural data preparation.

### Workflow
The processing layer acts as a professional filter: **Raw Data** (DuckDB/CSV) $\rightarrow$ **Sanitization** (Cleaning/Resampling) $\rightarrow$ **Refined Skeletal Data**. By isolating these tasks, we ensure that the Feature Engineering stage always receives consistent, high-quality numerical inputs.