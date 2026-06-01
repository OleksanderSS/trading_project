"""
System Version Management - Management of system operations,
configurations and models
"""
import logging

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("VersionManager")


@dataclass
class ConfigVersion:
    """Configuration version"""
    version: str
    date: str
    description: str
    changes: List[str]
    migration_required: bool = False
    migration_script: Optional[str] = None


class ConfigVersionManager:
    """
    System version manager with support for models and pipeline runs
    """
    def __init__(self, history_path: Optional[str] = None):
        self.history_path = Path(
            history_path or "src/config/version_history.json"
        )
        self.versions: List[ConfigVersion] = []
        self.current_version = "1.3.0"
        self._load_versions()

    def _load_versions(self):
        """Loads version history from a JSON file"""
        if not self.history_path.exists():
            logger.warning(
                f"Version history file not found at {self.history_path}. "
                "Initializing with defaults."
            )
            self.versions = self._get_default_versions()
            self._save_versions()
            return

        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.versions = [
                    ConfigVersion(**v) for v in data.get("versions", [])
                ]
                self.current_version = data.get(
                    "current_version", self.current_version
                )
            logger.info(
                f"Loaded {len(self.versions)} versions "
                f"from {self.history_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load version history: {e}")
            self.versions = self._get_default_versions()

    def _save_versions(self):
        """Saves version history to a JSON file"""
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "current_version": self.current_version,
                "versions": [asdict(v) for v in self.versions]
            }
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save version history: {e}")

    def _get_default_versions(self) -> List[ConfigVersion]:
        """Initial versions if the file is missing"""
        return [
            ConfigVersion(
                version="1.0.0",
                date="2024-01-01",
                description="Initial version",
                changes=[
                    "Basic collector configuration",
                    "Thresholds system"
                ]
            ),
            ConfigVersion(
                version="1.3.0",
                date="2024-04-08",
                description="Unified system",
                changes=[
                    "Unified configuration",
                    "Enhanced error handling"
                ],
                migration_required=True,
                migration_script="migrate_to_v1_3_0"
            )
        ]

    def get_current_version(self) -> str:
        """Get the current system version"""
        return self.current_version

    def get_model_version(self, ticker: str, model_type: str) -> str:
        """
        Generates a version for the model based on date and type.
        Format: {model_type}_{ticker}_{YYYYMMDD_HHMM}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        version = f"{model_type.lower()}_{ticker.upper()}_{timestamp}"
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Generated model version: {version}")
        return version

    def tag_pipeline_run(self) -> str:
        """Generates a unique ID for the current pipeline run"""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Pipeline run tagged: {run_id}")
        return run_id

    def get_version_info(
        self, version: Optional[str] = None
    ) -> Optional[ConfigVersion]:
        """Get version information"""
        target_version = version or self.current_version
        for v in self.versions:
            if v.version == target_version:
                return v
        return None

    def get_all_versions(self) -> List[ConfigVersion]:
        """Get all versions"""
        return self.versions.copy()

    def get_migration_path(
        self, from_version: str, to_version: Optional[str] = None
    ) -> List[str]:
        """Get the migration path"""
        target_version = to_version or self.current_version
        from_idx = next(
            (i for i, v in enumerate(self.versions)
             if v.version == from_version),
            None
        )
        to_idx = next(
            (i for i, v in enumerate(self.versions)
             if v.version == target_version),
            None
        )

        if from_idx is None or to_idx is None or from_idx >= to_idx:
            return []

        return [
            self.versions[i].version
            for i in range(from_idx + 1, to_idx + 1)
        ]

    def needs_migration(self, from_version: str) -> bool:
        """Check if migration is required"""
        return len(self.get_migration_path(from_version)) > 0

    def add_version(
        self,
        version: str,
        description: str,
        changes: List[str],
        migration_required: bool = False,
        migration_script: Optional[str] = None
    ):
        """Add a new version to the history"""
        new_version = ConfigVersion(
            version=version,
            date=datetime.now().strftime("%Y-%m-%d"),
            description=description,
            changes=changes,
            migration_required=migration_required,
            migration_script=migration_script
        )
        self.versions.append(new_version)
        self.current_version = version
        self._save_versions()
        logger.info(f"Added new configuration version: {version}")


# Global instance of the version manager
_version_manager = None


def get_version_manager() -> ConfigVersionManager:
    """Get the global instance of the version manager"""
    global _version_manager
    if _version_manager is None:
        _version_manager = ConfigVersionManager()
    return _version_manager


def get_current_version() -> str:
    return get_version_manager().get_current_version()


def get_model_version(ticker: str, model_type: str) -> str:
    return get_version_manager().get_model_version(ticker, model_type)


def tag_pipeline_run() -> str:
    return get_version_manager().tag_pipeline_run()


def needs_migration(from_version: str) -> bool:
    return get_version_manager().needs_migration(from_version)


def get_migration_path(
    from_version: str, to_version: Optional[str] = None
) -> List[str]:
    return get_version_manager().get_migration_path(from_version, to_version)


def get_version_info(
    version: Optional[str] = None
) -> Optional[ConfigVersion]:
    return get_version_manager().get_version_info(version)
