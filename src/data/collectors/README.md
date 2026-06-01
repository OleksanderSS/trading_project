# News Collectors (`src/collectors/news`)

This directory contains collectors designed to gather unstructured text-based data, primarily news articles, from various online sources.

## Architecture

The news collectors follow the dependency-injected architecture defined in the root `collectors` directory. They leverage a specialized abstract base class, `BaseNewsCollector`, which adds a crucial pre-processing step to the collection pipeline.

### `BaseNewsCollector`

This abstract class extends `BaseCollector` and introduces a filtering mechanism. Its primary responsibilities are:

1.  **Loading Filter Terms**: It retrieves a master list of keywords and tickers from the `strategy.unified_filter` section of the configuration.
2.  **Compiling a Regex**: It compiles these terms into a single, efficient, case-insensitive regular expression.
3.  **Filtering Raw Data**: It overrides the `fetch_raw_data` method. After a child collector fetches data (by implementing `_fetch_from_source`), this base class filters the results, ensuring that only items matching the regex are passed on for normalization and storage. This significantly reduces noise and saves processing time.

### Concrete Implementations

Each specific news collector (e.g., `GoogleNewsCollector`, `NewsAPICollector`, `RSSCollector`) inherits from `BaseNewsCollector` and has a very focused responsibility:

-   **Implement `_fetch_from_source()`**: This is the only abstract method they must implement. Its role is to connect to the specific data source (e.g., a REST API, an RSS feed URL) and return a list of raw data records (as dictionaries).
-   **No Transformation Logic**: Collectors in this layer **do not** transform data. They fetch it in its raw format and rely on the injected `normalizer` function (configured in `data.normalization.yaml`) to map it to the unified project schema.

### `HuggingFaceCollector`

Note: The `HuggingFaceCollector` is also located in this directory but is an exception. It collects metadata about models from the Hugging Face Hub and inherits directly from `BaseCollector`, as it does not require the news-specific text filtering provided by `BaseNewsCollector`.

## How to Add a New News Collector

1.  **Create the File**: Create a new file, e.g., `my_news_source_collector.py`.
2.  **Inherit from `BaseNewsCollector`**: 
    ```python
    from .base_news_collector import BaseNewsCollector

    class MyNewsSourceCollector(BaseNewsCollector):
        collector_type = "my_news_source" # Unique type for the factory
    ```
3.  **Implement the Constructor**: The constructor must accept all standard dependencies and pass them to the parent class.
    ```python
    def __init__(self, collector_name, config, db_manager, http_client_factory, error_handler, normalizer):
        super().__init__(collector_name, config, db_manager, http_client_factory, error_handler, normalizer)
        # Add any custom initialization logic here, like loading specific config values.
        self.my_api_endpoint = self.config.get("my_api_endpoint")
    ```
4.  **Implement `_fetch_from_source()`**: Write the logic to fetch data from your source. Use `self._get_async_http_client()` for making HTTP requests.
    ```python
    async def _fetch_from_source(self) -> List[Dict[str, Any]]:
        async with self._get_async_http_client() as client:
            response = await client.get(self.my_api_endpoint)
            response.raise_for_status()
            return response.json()['articles'] # Return the list of raw records
    ```
5.  **Configure the Collector**: Add a new entry for your collector in `configs/collectors.yaml`.
    ```yaml
    my_awesome_news_feed:
      enabled: true
      module: collectors.news.my_news_source_collector
      class: MyNewsSourceCollector
      type: news # Group with other news data
      config:
        my_api_endpoint: "https://api.mynewssource.com/v1/latest"
        # Any other parameters your collector needs
    ```
6.  **Update Normalization Rules**: Add a mapping in `configs/data/normalization.yaml` to tell the system how to transform the raw data from your new source into the unified format.
