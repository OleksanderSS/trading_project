# auto_colab_sync.py - Automated Synchronization for Google Colab

import hashlib
import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("AutoColabSync")

class AutoColabSync:
    """
    Automated synchronization of project data with Colab using standardized naming and structure.
    """

    def __init__(self):
        self.project_root = Path.cwd()
        self.data_dir = self.project_root / "data"
        self.colab_dir = self.data_dir / "colab"
        self.backup_dir = self.data_dir / "backup_for_colab"

        self.colab_dir.mkdir(parents=True, exist_ok=True)

        # Naming and compression settings
        self.naming_config = {
            'base_name': 'trading_pipeline_data',
            'version_format': '%Y%m%d_%H%M%S',
            'compression': 'tar.gz',
            'max_size_mb': 100  # Optimized size for Colab uploads
        }

        logger.info("AutoColabSync initialized using project root: %s", self.project_root)

    def generate_filename(self, timestamp: datetime | None = None) -> str:
        """Generates a standardized filename for the data package."""
        if timestamp is None:
            timestamp = datetime.now()

        base_name = self.naming_config['base_name']
        version = timestamp.strftime(self.naming_config['version_format'])
        compression = self.naming_config['compression']

        return f"{base_name}_{version}.{compression}"

    def calculate_data_hash(self, data_files: list[Path]) -> str:
        """Calculates a MD5 hash of the data files to detect changes."""
        hash_md5 = hashlib.md5()

        for file_path in sorted(data_files):
            if file_path.exists():
                file_stat = file_path.stat()
                hash_md5.update(f"{file_path.name}:{file_stat.st_mtime}:{file_stat.st_size}".encode())

        return hash_md5.hexdigest()

    def create_automated_package(self) -> Path:
        """Creates an automated package containing the latest project state for Colab."""
        logger.info("Creating automated Colab package...")

        timestamp = datetime.now()
        filename = self.generate_filename(timestamp)
        package_file = self.colab_dir / filename

        # Temporary staging directory
        temp_dir = self.colab_dir / "temp_package"
        temp_dir.mkdir(parents=True, exist_ok=True)

        package_data = {
            'metadata': {
                'created_at': timestamp.isoformat(),
                'version': '3.0.0',
                'description': 'Automated trading pipeline data for Colab (DuckDB)',
                'filename': filename,
                'total_size': 0,
                'total_files': 0,
                'data_hash': None
            },
            'data_structure': {
                'database': {},
                'processed_data': {},
                'configs': {}
            }
        }

        total_files = 0
        total_size = 0
        staged_files = []

        # 1. Add Main Database (DuckDB)
        db_path = self.data_dir / "main.duckdb"
        if db_path.exists():
            dest_path = temp_dir / "main.duckdb"
            shutil.copy2(db_path, dest_path)
            package_data['data_structure']['database']['main.duckdb'] = {
                'type': 'duckdb',
                'size': db_path.stat().st_size,
                'modified': datetime.fromtimestamp(db_path.stat().st_mtime).isoformat()
            }
            staged_files.append(dest_path)
            total_files += 1
            total_size += db_path.stat().st_size
            logger.info("Added main database: main.duckdb")

        # 2. Add Processed Parquet Files
        processed_dir = self.data_dir / "processed"
        if processed_dir.exists():
            for p_file in processed_dir.glob("**/*.parquet"):
                rel_path = p_file.relative_to(processed_dir)
                dest_path = temp_dir / "processed" / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p_file, dest_path)

                package_data['data_structure']['processed_data'][str(rel_path)] = {
                    'size': p_file.stat().st_size,
                    'modified': datetime.fromtimestamp(p_file.stat().st_mtime).isoformat()
                }
                staged_files.append(dest_path)
                total_files += 1
                total_size += p_file.stat().st_size
            logger.info("Processed directory synced.")

        # 3. Finalize Metadata
        package_data['metadata']['total_files'] = total_files
        package_data['metadata']['total_size'] = total_size
        package_data['metadata']['data_hash'] = self.calculate_data_hash(staged_files)

        with open(temp_dir / "package_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(package_data, f, indent=2, default=str)

        # Create Archive
        with tarfile.open(package_file, 'w:gz') as tar:
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    tar.add(file_path, arcname=file_path.relative_to(temp_dir))

        shutil.rmtree(temp_dir)
        logger.info("Automated package created: %s (%d files, %.1f MB)",
                    filename, total_files, total_size / (1024*1024))

        return package_file

    def cleanup_old_packages(self, keep_count: int = 5):
        """Removes old sync packages to save space."""
        logger.info("Cleaning up old packages (keeping last %d)...", keep_count)
        package_files = list(self.colab_dir.glob("trading_pipeline_data_*.tar.gz"))
        package_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for old_package in package_files[keep_count:]:
            try:
                old_package.unlink()
                logger.info("Deleted old package: %s", old_package.name)
            except Exception as e:
                logger.error("Error deleting %s: %s", old_package.name, e)

def main():
    """CLI entry point for Colab synchronization."""
    print("Automated Colab Sync System")
    print("=" * 50)

    try:
        sync_system = AutoColabSync()
        print("1. Create Automated Package")
        print("2. Cleanup Old Packages")
        print("0. Exit")

        choice = input("\nSelect action: ").strip()

        if choice == "1":
            path = sync_system.create_automated_package()
            print(f"[OK] Package created: {path}")
        elif choice == "2":
            count = sync_system.cleanup_old_packages()
            print("[OK] Cleanup complete.")
        elif choice == "0":
            print("Goodbye.")
    except Exception as e:
        logger.error("Sync failed: %s", e, exc_info=True)

if __name__ == "__main__":
    main()
