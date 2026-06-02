from google.cloud import storage
from pathlib import Path
from typing import Optional
import time
from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
logger = ProjectLogger.get_logger('GCSManager')


class GCSManager:
    """A manager to handle all interactions with Google Cloud Storage."""

    def __init__(self, config: Optional[dict]=None):
        self.client = None
        self.bucket = None
        self.bucket_name = None
        try:
            if config is None:
                config = get_current_config()
            self.cloud_storage_config = config.get('cloud_storage')
            if not self.cloud_storage_config:
                logger.warning(
                    'Cloud Storage configuration is missing. GCS disabled.')
                return
            self.bucket_name = self.cloud_storage_config.get('bucket_name')
            if not self.bucket_name:
                logger.warning(
                    'GCS bucket name is not configured. GCS disabled.')
                return
            self.client = storage.Client()
            self.bucket = self.client.get_bucket(self.bucket_name)
            logger.info(
                f"Successfully connected to GCS bucket: '{self.bucket_name}'")
        except Exception as e:
            logger.error(
                f'Failed to initialize GCS: {e}. Continuing without cloud storage.',
                exc_info=True
                )
            self.client = None
            self.bucket = None

    def upload_file(self, source_file_path: str, destination_blob_name: str
        ) ->bool:
        """Uploads a file to the GCS bucket."""
        if not self.client or not self.bucket:
            logger.warning('GCS not initialized. Cannot upload file.')
            return False
        try:
            logger.info(
                f"Uploading '{source_file_path}' to GCS path '{destination_blob_name}'..."
                )
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_path)
            logger.info('Upload successful.')
            return True
        except Exception as e:
            logger.error(
                f"Failed to upload file '{source_file_path}' to '{destination_blob_name}': {e}"
                , exc_info=True)
            return False

    def upload_blob_from_memory(self, file_obj, destination_blob_name: str
        ) ->bool:
        """Uploads a file-like object to the GCS bucket."""
        if not self.client or not self.bucket:
            logger.warning('GCS not initialized. Cannot upload blob.')
            return False
        try:
            logger.info(
                f"Uploading from memory to GCS path '{destination_blob_name}'..."
                )
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_file(file_obj)
            logger.info('Upload successful.')
            return True
        except Exception as e:
            logger.error(
                f"Failed to upload from memory to '{destination_blob_name}': {e}"
                , exc_info=True)
            return False

    def download_file(self, source_blob_name: str, destination_file_path: str
        ) ->bool:
        """Downloads a file from the GCS bucket."""
        try:
            logger.info(
                f"Downloading GCS path '{source_blob_name}' to '{destination_file_path}'..."
                )
            blob = self.bucket.blob(source_blob_name)
            Path(destination_file_path).parent.mkdir(parents=True, exist_ok
                =True)
            blob.download_to_filename(destination_file_path)
            logger.info('Download successful.')
            return True
        except Exception as e:
            logger.error(
                f"Failed to download file '{source_blob_name}' to '{destination_file_path}': {e}"
                , exc_info=True)
            return False

    def list_files(self, prefix: Optional[str]=None) ->list[str]:
        """Lists all the files in the bucket with an optional prefix."""
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Failed to list files with prefix '{prefix}': {e}",
                exc_info=True)
            raise RuntimeError(f"Failed to list GCS files with prefix '{prefix}'") from e

    def file_exists(self, blob_name: str) ->bool:
        """Checks if a file exists in the GCS bucket."""
        if not self.client or not self.bucket:
            logger.warning('GCS not initialized. Cannot check file existence.')
            return False
        try:
            blob = self.bucket.blob(blob_name)
            return blob.exists()
        except Exception as e:
            logger.error(
                f"Error checking for existence of blob '{blob_name}': {e}",
                exc_info=True)
            return False

    def wait_for_blob(self, blob_name: str, timeout: int=300) ->Optional[
        storage.Blob]:
        """Waits for a blob to exist in GCS, with a timeout."""
        if not self.client or not self.bucket:
            logger.warning('GCS not initialized. Cannot wait for blob.')
            return None
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.file_exists(blob_name):
                logger.info(f"Blob '{blob_name}' found.")
                return self.bucket.blob(blob_name)
            logger.info(
                f"Waiting for blob '{blob_name}'... polling again in 10 seconds."
                )
            time.sleep(10)
        logger.warning(
            f"Timed out waiting for blob '{blob_name}' after {timeout} seconds."
            )
        return None
