"""Cache synchronization with S3-compatible storage (MinIO)."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)

# Metadata file for tracking sync state
SYNC_METADATA_FILE = ".sync-metadata.json"


class CacheSyncConfig:
    """Configuration for cache synchronization."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = True,
    ):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.secure = secure


class CacheSync:
    """Handles cache synchronization with MinIO/S3."""

    def __init__(self, config: CacheSyncConfig, cache_path: Path):
        """Initialize cache sync client.

        Args:
            config: MinIO configuration
            cache_path: Local cache directory path
        """
        self.config = config
        self.cache_path = cache_path
        self.metadata_path = cache_path / SYNC_METADATA_FILE

        # Initialize MinIO client
        self.client = Minio(
            config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure,
        )

        # Ensure bucket exists
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.config.bucket):
                self.client.make_bucket(self.config.bucket)
                logger.info(f"Created bucket: {self.config.bucket}")
        except S3Error as e:
            logger.error(f"Error checking/creating bucket: {e}")
            raise

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            MD5 hash as hex string
        """
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _load_metadata(self) -> Dict:
        """Load sync metadata from local file.

        Returns:
            Metadata dictionary
        """
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {"files": {}, "last_sync": None}

    def _save_metadata(self, metadata: Dict) -> None:
        """Save sync metadata to local file.

        Args:
            metadata: Metadata dictionary to save
        """
        self.cache_path.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _get_local_files(self, test_id: Optional[str] = None) -> List[Path]:
        """Get list of local cache files.

        Args:
            test_id: Optional test ID to filter files

        Returns:
            List of file paths relative to cache_path
        """
        files = []
        search_path = self.cache_path / test_id if test_id else self.cache_path

        if not search_path.exists():
            return files

        for file_path in search_path.rglob("*"):
            if file_path.is_file() and file_path.name != SYNC_METADATA_FILE:
                files.append(file_path.relative_to(self.cache_path))

        return files

    def _get_remote_files(self, test_id: Optional[str] = None) -> Dict[str, dict]:
        """Get list of remote files with metadata.

        Args:
            test_id: Optional test ID to filter files

        Returns:
            Dictionary mapping file paths to metadata
        """
        prefix = f"{test_id}/" if test_id else ""
        remote_files = {}

        try:
            objects = self.client.list_objects(
                self.config.bucket, prefix=prefix, recursive=True
            )
            for obj in objects:
                if obj.object_name != SYNC_METADATA_FILE:
                    remote_files[obj.object_name] = {
                        "size": obj.size,
                        "last_modified": obj.last_modified,
                        "etag": obj.etag,
                    }
        except S3Error as e:
            logger.error(f"Error listing remote files: {e}")
            raise

        return remote_files

    def upload(self, test_id: Optional[str] = None, force: bool = False) -> int:
        """Upload local cache files to remote storage.

        Args:
            test_id: Optional test ID to filter files
            force: Force upload even if file hasn't changed

        Returns:
            Number of files uploaded
        """
        local_files = self._get_local_files(test_id)
        metadata = self._load_metadata()
        uploaded_count = 0

        logger.info(f"Uploading {len(local_files)} files to {self.config.bucket}...")

        for rel_path in local_files:
            local_file = self.cache_path / rel_path
            object_name = str(rel_path).replace("\\", "/")  # Ensure forward slashes

            # Check if file needs upload
            file_hash = self._get_file_hash(local_file)
            cached_info = metadata["files"].get(object_name, {})

            if not force and cached_info.get("hash") == file_hash:
                logger.debug(f"Skipping unchanged file: {object_name}")
                continue

            try:
                self.client.fput_object(
                    self.config.bucket,
                    object_name,
                    str(local_file),
                )
                logger.info(f"Uploaded: {object_name}")

                # Update metadata
                metadata["files"][object_name] = {
                    "hash": file_hash,
                    "uploaded_at": datetime.utcnow().isoformat(),
                }
                uploaded_count += 1

            except S3Error as e:
                logger.error(f"Failed to upload {object_name}: {e}")
                raise

        metadata["last_sync"] = datetime.utcnow().isoformat()
        self._save_metadata(metadata)

        return uploaded_count

    def download(self, test_id: Optional[str] = None, force: bool = False) -> int:
        """Download remote cache files to local storage.

        Args:
            test_id: Optional test ID to filter files
            force: Force download even if file exists locally

        Returns:
            Number of files downloaded
        """
        remote_files = self._get_remote_files(test_id)
        metadata = self._load_metadata()
        downloaded_count = 0

        logger.info(
            f"Downloading {len(remote_files)} files from {self.config.bucket}..."
        )

        for object_name, remote_info in remote_files.items():
            local_file = self.cache_path / object_name
            local_file.parent.mkdir(parents=True, exist_ok=True)

            # Check if download is needed
            if not force and local_file.exists():
                file_hash = self._get_file_hash(local_file)
                cached_info = metadata["files"].get(object_name, {})

                # Skip if local file matches cached hash
                if cached_info.get("hash") == file_hash:
                    logger.debug(f"Skipping unchanged file: {object_name}")
                    continue

            try:
                self.client.fget_object(
                    self.config.bucket,
                    object_name,
                    str(local_file),
                )
                logger.info(f"Downloaded: {object_name}")

                # Update metadata
                file_hash = self._get_file_hash(local_file)
                metadata["files"][object_name] = {
                    "hash": file_hash,
                    "downloaded_at": datetime.utcnow().isoformat(),
                }
                downloaded_count += 1

            except S3Error as e:
                logger.error(f"Failed to download {object_name}: {e}")
                raise

        metadata["last_sync"] = datetime.utcnow().isoformat()
        self._save_metadata(metadata)

        return downloaded_count

    def sync(
        self, test_id: Optional[str] = None, prefer_remote: bool = True
    ) -> Dict[str, int]:
        """Bidirectional sync between local and remote storage.

        Args:
            test_id: Optional test ID to filter files
            prefer_remote: If True, prefer remote files in conflicts

        Returns:
            Dictionary with upload and download counts
        """
        logger.info("Starting bidirectional sync...")

        local_files = set(
            str(f).replace("\\", "/") for f in self._get_local_files(test_id)
        )
        remote_files = self._get_remote_files(test_id)
        remote_file_names = set(remote_files.keys())

        metadata = self._load_metadata()
        uploaded = 0
        downloaded = 0

        # Files only in remote - download them
        remote_only = remote_file_names - local_files
        for object_name in remote_only:
            local_file = self.cache_path / object_name
            local_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                self.client.fget_object(
                    self.config.bucket,
                    object_name,
                    str(local_file),
                )
                logger.info(f"Downloaded (new): {object_name}")

                file_hash = self._get_file_hash(local_file)
                metadata["files"][object_name] = {
                    "hash": file_hash,
                    "downloaded_at": datetime.utcnow().isoformat(),
                }
                downloaded += 1

            except S3Error as e:
                logger.error(f"Failed to download {object_name}: {e}")

        # Files only in local - upload them
        local_only = local_files - remote_file_names
        for rel_path_str in local_only:
            local_file = self.cache_path / rel_path_str

            try:
                self.client.fput_object(
                    self.config.bucket,
                    rel_path_str,
                    str(local_file),
                )
                logger.info(f"Uploaded (new): {rel_path_str}")

                file_hash = self._get_file_hash(local_file)
                metadata["files"][rel_path_str] = {
                    "hash": file_hash,
                    "uploaded_at": datetime.utcnow().isoformat(),
                }
                uploaded += 1

            except S3Error as e:
                logger.error(f"Failed to upload {rel_path_str}: {e}")

        # Files in both - check for changes and resolve conflicts
        common_files = local_files & remote_file_names
        for object_name in common_files:
            local_file = self.cache_path / object_name
            local_hash = self._get_file_hash(local_file)
            cached_info = metadata["files"].get(object_name, {})

            # Check if either side has changed
            local_changed = cached_info.get("hash") != local_hash
            # For remote, we can check etag or last_modified
            # Simplified: assume remote changed if local didn't change but etag differs

            if local_changed:
                if prefer_remote:
                    # Download remote version
                    try:
                        self.client.fget_object(
                            self.config.bucket,
                            object_name,
                            str(local_file),
                        )
                        logger.info(
                            f"Downloaded (conflict, prefer remote): {object_name}"
                        )

                        file_hash = self._get_file_hash(local_file)
                        metadata["files"][object_name] = {
                            "hash": file_hash,
                            "downloaded_at": datetime.utcnow().isoformat(),
                        }
                        downloaded += 1

                    except S3Error as e:
                        logger.error(f"Failed to download {object_name}: {e}")
                else:
                    # Upload local version
                    try:
                        self.client.fput_object(
                            self.config.bucket,
                            object_name,
                            str(local_file),
                        )
                        logger.info(f"Uploaded (conflict, prefer local): {object_name}")

                        metadata["files"][object_name] = {
                            "hash": local_hash,
                            "uploaded_at": datetime.utcnow().isoformat(),
                        }
                        uploaded += 1

                    except S3Error as e:
                        logger.error(f"Failed to upload {object_name}: {e}")

        metadata["last_sync"] = datetime.utcnow().isoformat()
        self._save_metadata(metadata)

        return {"uploaded": uploaded, "downloaded": downloaded}
