from importlib.util import find_spec
from pathlib import Path

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("Hybrid.StorageHelpers")


def init_gdrive(stage) -> bool:
    """Attempt to initialize Google Drive API service."""
    if not _gdrive_available():
        stage.logger.warning("Google Drive API libraries not installed. Skipping GDrive init.")
        return False
    try:
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        sa_path = stage.config.system_config.get("google_drive", {}).get("service_account_path")
        if sa_path and Path(sa_path).exists():
            creds = service_account.Credentials.from_service_account_file(
                sa_path, scopes=["https://www.googleapis.com/auth/drive"]
            )
            stage.config.gdrive_service = build("drive", "v3", credentials=creds)
            stage.logger.info("✅ Google Drive API initialized via service account")
            return True
        creds_path = stage.config.system_config.get("google_drive", {}).get("credentials_path", "credentials.json")
        if Path(creds_path).exists():
            creds = Credentials.from_authorized_user_file(creds_path)
            stage.config.gdrive_service = build("drive", "v3", credentials=creds)
            stage.logger.info("✅ Google Drive API initialized via OAuth credentials")
            return True
        stage.logger.warning(
            "⚠️ No GDrive credentials found (service_account_path or credentials_path). Falling back to manual file transfer."
        )
        return False
    except Exception as e:
        stage.logger.error(f"❌ Google Drive initialization error: {e}")
        return False


def _gdrive_available() -> bool:
    return all(
        find_spec(module_name) is not None
        for module_name in (
            "google.oauth2.credentials",
            "googleapiclient.discovery",
            "googleapiclient.http",
        )
    )


def init_fallback_storage(stage) -> bool:
    """Initializes fallback storage (S3 or GCS)."""
    if stage.use_s3:
        return init_s3_storage(stage)
    if stage.use_gcs:
        return init_gcs_storage(stage)
    stage.logger.warning("⚠️ No fallback storage available, using manual transfer")
    return False


def init_s3_storage(stage) -> bool:
    """Initialize S3 storage."""
    try:
        import boto3

        s3_config = stage.storage_fallback.get("s3", {})
        stage.s3_client = boto3.client(
            "s3",
            aws_access_key_id=s3_config.get("access_key"),
            aws_secret_access_key=s3_config.get("secret_key"),
            region_name=s3_config.get("region", "us-east-1"),
        )
        stage.s3_bucket = s3_config.get("bucket")
        stage.logger.info(f"✅ S3 fallback initialized: {stage.s3_bucket}")
        return True
    except Exception as e:
        stage.logger.error(f"Виникла помилка: {e}", exc_info=True)
        stage.logger.warning(f"⚠️ S3 fallback failed: {e}")
        return False


def init_gcs_storage(stage) -> bool:
    """Initialize GCS storage."""
    try:
        from google.cloud import storage

        gcs_config = stage.storage_fallback.get("gcs", {})
        stage.gcs_client = storage.Client(project=gcs_config.get("project_id"))
        stage.gcs_bucket = stage.gcs_client.bucket(gcs_config.get("bucket"))
        stage.logger.info(f"✅ GCS fallback initialized: {gcs_config.get('bucket')}")
        return True
    except Exception as e:
        stage.logger.error(f"Виникла помилка: {e}", exc_info=True)
        stage.logger.warning(f"⚠️ GCS fallback failed: {e}")
        return False
