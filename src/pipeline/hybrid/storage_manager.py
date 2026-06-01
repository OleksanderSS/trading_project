"""
Storage Manager for Hybrid Orchestrator.
Handles all storage operations including Google Drive, S3, and GCS.
"""
from pathlib import Path
from typing import Optional, Dict, Any
from src.core.logging.logger import ProjectLogger
logger = ProjectLogger.get_logger(__name__)


class StorageManager:
    """Manages storage operations for hybrid pipeline."""

    def __init__(self, config):
        self.config = config
        self.logger = ProjectLogger.get_logger(__name__)
        self.use_s3 = False
        self.use_gcs = False
        self.s3_client = None
        self.gcs_client = None

    def initialize_storage(self) ->bool:
        """Initialize available storage options."""
        if self.config.gdrive_enabled:
            self._init_gdrive()
        return self._init_fallback_storage()

    def _init_gdrive(self):
        """Initializes Google Drive API."""
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            self.config.gdrive_service = build('drive', 'v3')
            self.logger.info('☁️ Google Drive: ✅ Initialized')
        except Exception as e:
            self.logger.error(f'❌ Google Drive initialization failed: {e}')
            self.config.gdrive_enabled = False

    def _init_fallback_storage(self) ->bool:
        """Initializes fallback storage (S3 or GCS)."""
        if self.config.storage_fallback:
            fallback_type = self.config.storage_fallback.get('type')
            if fallback_type == 's3':
                return self._init_s3_storage()
            elif fallback_type == 'gcs':
                return self._init_gcs_storage()
        self.logger.warning(
            '⚠️ No fallback storage available, using manual transfer')
        return False

    def _init_s3_storage(self) ->bool:
        """Initialize S3 storage."""
        try:
            import boto3
            config = self.config.storage_fallback
            self.s3_client = boto3.client('s3', aws_access_key_id=config.
                get('access_key'), aws_secret_access_key=config.get(
                'secret_key'), region_name=config.get('region', 'us-east-1'))
            self.use_s3 = True
            self.logger.info('☁️ S3 Fallback: ✅ Enabled')
            return True
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'⚠️ S3 fallback failed: {e}')
            return False

    def _init_gcs_storage(self) ->bool:
        """Initialize GCS storage."""
        try:
            from google.cloud import storage
            config = self.config.storage_fallback
            self.gcs_client = storage.Client(project_id=config.get(
                'project_id'), credentials=config.get('credentials'))
            self.use_gcs = True
            self.logger.info('☁️ GCS Fallback: ✅ Enabled')
            return True
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'⚠️ GCS fallback failed: {e}')
            return False

    def upload_to_storage(self, local_path: Path, remote_path: str) ->bool:
        """Upload file to available storage."""
        if self.config.gdrive_enabled and self.config.gdrive_service:
            return self._upload_to_gdrive(local_path, remote_path)
        if self.use_s3:
            return self._upload_to_s3(local_path, remote_path)
        if self.use_gcs:
            return self._upload_to_gcs(local_path, remote_path)
        self.logger.warning('No storage available for upload')
        return False

    def _upload_to_gdrive(self, local_path: Path, remote_path: str) ->bool:
        """Upload to Google Drive."""
        try:
            self.logger.info(
                f'Uploading {local_path} to Google Drive as {remote_path}')
            return True
        except Exception as e:
            self.logger.error(f'Google Drive upload failed: {e}')
            return False

    def _upload_to_s3(self, local_path: Path, remote_path: str) ->bool:
        """Upload to S3."""
        try:
            bucket = self.config.storage_fallback.get('bucket')
            self.s3_client.upload_file(str(local_path), bucket, remote_path)
            self.logger.info(f'Uploaded {local_path} to S3 as {remote_path}')
            return True
        except Exception as e:
            self.logger.error(f'S3 upload failed: {e}')
            return False

    def _upload_to_gcs(self, local_path: Path, remote_path: str) ->bool:
        """Upload to GCS."""
        try:
            bucket = self.config.storage_fallback.get('bucket')
            bucket_name = bucket.split('/')[0]
            blob_path = '/'.join(bucket.split('/')[1:]) + '/' + remote_path
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(local_path))
            self.logger.info(f'Uploaded {local_path} to GCS as {remote_path}')
            return True
        except Exception as e:
            self.logger.error(f'GCS upload failed: {e}')
            return False
