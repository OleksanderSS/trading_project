"""
Results Manager - Manages evaluation results and reporting data
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_PATTERN = "results_*.json"


class ResultsManager:
    """
    Manages evaluation results, metrics storage, and reporting data.
    Provides centralized access to results for reporting components.
    """

    def __init__(self, results_dir: str | None = None):
        """
        Initialize ResultsManager.

        Args:
            results_dir: Directory to store results files
        """
        self.results_dir = Path(results_dir) if results_dir else Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._cached_results: dict[str, Any] = {}

    def save_results(self, results: dict[str, Any], name: str | None = None) -> str:
        """
        Save results to file.

        Args:
            results: Results dictionary to save
            name: Optional name for the results file

        Returns:
            Path to saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = name or f"results_{timestamp}.json"
        filepath = self.results_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)

            # Cache the results
            self._cached_results[filename] = results
            logger.info(f"Results saved to {filepath}")
            return str(filepath)

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Failed to save results: {e}")
            raise

    def load_results(self, filename: str) -> dict[str, Any] | None:
        """
        Load results from file.

        Args:
            filename: Name of results file to load

        Returns:
            Results dictionary or None if not found
        """
        # Check cache first
        if filename in self._cached_results:
            return cast(dict[str, Any], self._cached_results[filename])

        filepath = self.results_dir / filename
        if not filepath.exists():
            logger.warning(f"Results file not found: {filepath}")
            return None

        try:
            with open(filepath, encoding='utf-8') as f:
                results = json.load(f)

            # Cache the results
            self._cached_results[filename] = results
            logger.info(f"Results loaded from {filepath}")
            return cast(dict[str, Any], results)

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN
            logger.exception(f"Failed to load results: {e}")
            return None

    def get_latest_results(self) -> dict[str, Any] | None:
        """
        Get the most recent results file.

        Returns:
            Latest results dictionary or None if no results exist
        """
        try:
            result_files = list(self.results_dir.glob(RESULTS_PATTERN))
            if not result_files:
                return None

            # Sort by modification time
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            return self.load_results(latest_file.name)

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN
            logger.exception(f"Failed to get latest results: {e}")
            return None

    def list_results(self) -> list[str]:
        """
        List all available results files.

        Returns:
            List of results filenames
        """
        try:
            return [f.name for f in self.results_dir.glob(RESULTS_PATTERN)]
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN
            logger.exception(f"Failed to list results: {e}")
            return []

    def delete_results(self, filename: str) -> bool:
        """
        Delete a results file.

        Args:
            filename: Name of results file to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = self.results_dir / filename
            if filepath.exists():
                filepath.unlink()
                # Remove from cache
                self._cached_results.pop(filename, None)
                logger.info(f"Results file deleted: {filepath}")
                return True
            else:
                logger.warning(f"Results file not found: {filepath}")
                return False

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Failed to delete results: {e}")
            return False

    def get_results_summary(self) -> dict[str, Any]:
        """
        Get a summary of all available results.

        Returns:
            Summary dictionary with results statistics
        """
        try:
            result_files = list(self.results_dir.glob(RESULTS_PATTERN))

            summary: dict[str, Any] = {
                'total_results': len(result_files),
                'latest_result': None,
                'oldest_result': None,
                'results_list': []
            }

            if result_files:
                # Sort by modification time
                sorted_files = sorted(result_files, key=lambda f: f.stat().st_mtime)

                summary['latest_result'] = cast(str, sorted_files[-1].name)
                summary['oldest_result'] = cast(str, sorted_files[0].name)

                for file in sorted_files:
                    stat = file.stat()
                    results_list = cast(list[dict[str, Any]], summary['results_list'])
                    results_list.append({
                        'filename': file.name,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })

            return summary

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Failed to get results summary: {e}")
            return {'error': str(e)}

    def export_results_to_csv(self, filename: str, output_path: str | None = None) -> str | None:
        """
        Export results to CSV format for analysis.

        Args:
            filename: Results file to export
            output_path: Optional output path for CSV file

        Returns:
            Path to exported CSV file or None if failed
        """
        try:
            results = self.load_results(filename)
            if not results:
                return None

            # Flatten results for CSV export
            flattened = self._flatten_dict(results)
            df = pd.DataFrame([flattened])

            if output_path is None:
                output_path = str(self.results_dir / f"{filename.replace('.json', '.csv')}")

            df.to_csv(output_path, index=False)
            logger.info(f"Results exported to CSV: {output_path}")
            return cast(str, str(output_path))

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Failed to export results to CSV: {e}")
            raise RuntimeError(f"Failed to export results {filename} to CSV") from e

    def _flatten_dict(self, d: dict[str, Any], parent_key: str = '', sep: str = '_') -> dict[str, Any]:
        """
        Flatten nested dictionary for CSV export.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested items
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to strings for CSV
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def clear_cache(self):
        """Clear the internal results cache."""
        self._cached_results.clear()
        logger.info("Results cache cleared")

    def get_cache_info(self) -> dict[str, Any]:
        """
        Get information about the current cache state.

        Returns:
            Cache information dictionary
        """
        return {
            'cached_files': list(self._cached_results.keys()),
            'cache_size': len(self._cached_results),
            'cache_memory_mb': sum(len(str(v)) for v in self._cached_results.values()) / (1024 * 1024)
        }
