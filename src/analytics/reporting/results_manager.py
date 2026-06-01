"""
Results Manager - Manages evaluation results and reporting data
"""

import json
import pandas as pd
from typing import Dict, Any, List, Optional, cast
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ResultsManager:
    """
    Manages evaluation results, metrics storage, and reporting data.
    Provides centralized access to results for reporting components.
    """
    
    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize ResultsManager.
        
        Args:
            results_dir: Directory to store results files
        """
        self.results_dir = Path(results_dir) if results_dir else Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._cached_results: Dict[str, Any] = {}
        
    def save_results(self, results: Dict[str, Any], name: Optional[str] = None) -> str:
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
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}", exc_info=True)
            raise
    
    def load_results(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load results from file.
        
        Args:
            filename: Name of results file to load
            
        Returns:
            Results dictionary or None if not found
        """
        # Check cache first
        if filename in self._cached_results:
            return cast(Dict[str, Any], self._cached_results[filename])
        
        filepath = self.results_dir / filename
        if not filepath.exists():
            logger.warning(f"Results file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Cache the results
            self._cached_results[filename] = results
            logger.info(f"Results loaded from {filepath}")
            return cast(Dict[str, Any], results)
            
        except Exception as e:
            logger.error(f"Failed to load results: {e}", exc_info=True)
            return None
    
    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent results file.
        
        Returns:
            Latest results dictionary or None if no results exist
        """
        try:
            result_files = list(self.results_dir.glob("results_*.json"))
            if not result_files:
                return None
            
            # Sort by modification time
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            return self.load_results(latest_file.name)
            
        except Exception as e:
            logger.error(f"Failed to get latest results: {e}", exc_info=True)
            return None
    
    def list_results(self) -> List[str]:
        """
        List all available results files.
        
        Returns:
            List of results filenames
        """
        try:
            return [f.name for f in self.results_dir.glob("results_*.json")]
        except Exception as e:
            logger.error(f"Failed to list results: {e}", exc_info=True)
            return []  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN
    
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
                
        except Exception as e:
            logger.error(f"Failed to delete results: {e}", exc_info=True)
            return False
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all available results.
        
        Returns:
            Summary dictionary with results statistics
        """
        try:
            result_files = list(self.results_dir.glob("results_*.json"))
            
            summary: Dict[str, Any] = {
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
                    results_list = cast(List[Dict[str, Any]], summary['results_list'])
                    results_list.append({
                        'filename': file.name,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get results summary: {e}", exc_info=True)
            return {'error': str(e)}
    
    def export_results_to_csv(self, filename: str, output_path: Optional[str] = None) -> Optional[str]:
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
            
        except Exception as e:
            logger.error(f"Failed to export results to CSV: {e}", exc_info=True)
            return None
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten nested dictionary for CSV export.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested items
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items: List[tuple[str, Any]] = []
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
    
    def get_cache_info(self) -> Dict[str, Any]:
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
