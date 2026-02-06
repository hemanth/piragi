"""Change detection for automatic updates."""

import hashlib
import logging
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class ChangeDetector:
    """Detects changes in files and URLs for automatic updates."""

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """
        Compute SHA256 hash of content.

        Args:
            content: Content to hash

        Returns:
            Hex digest of SHA256 hash
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def is_url(source: str) -> bool:
        """Check if source is a URL."""
        parsed = urlparse(source)
        return parsed.scheme in ("http", "https")

    @staticmethod
    def check_file_changed(
        source: str, stored_mtime: Optional[float], stored_hash: str
    ) -> bool:
        """
        Check if a file has changed using mtime and content hash.

        Args:
            source: File path
            stored_mtime: Previously stored modification time
            stored_hash: Previously stored content hash

        Returns:
            True if file changed, False otherwise
        """
        if not os.path.exists(source):
            return False

        # Quick check: modification time
        current_mtime = os.path.getmtime(source)
        if stored_mtime and current_mtime == stored_mtime:
            # File hasn't been touched, definitely not changed
            return False

        # Modification time changed, check actual content
        try:
            with open(source, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            current_hash = ChangeDetector.compute_content_hash(content)
            return current_hash != stored_hash
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to read {source}: {e}")
            return True

    @staticmethod
    def check_url_changed(
        source: str,
        stored_etag: Optional[str],
        stored_last_modified: Optional[str],
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """
        Check if a URL has changed using HTTP headers.
        Uses conditional requests for minimal latency.

        Args:
            source: URL
            stored_etag: Previously stored ETag
            stored_last_modified: Previously stored Last-Modified
            timeout: Request timeout in seconds

        Returns:
            Dict with 'changed' bool and optional 'etag', 'last_modified'
        """
        try:
            headers = {}

            # Add conditional request headers
            if stored_etag:
                headers["If-None-Match"] = stored_etag
            if stored_last_modified:
                headers["If-Modified-Since"] = stored_last_modified

            # Send HEAD request first (faster, no body download)
            response = requests.head(source, headers=headers, timeout=timeout, allow_redirects=True)

            # 304 Not Modified - content hasn't changed
            if response.status_code == 304:
                return {"changed": False}

            # If HEAD not supported, try GET with same conditional headers
            if response.status_code == 405:  # Method Not Allowed
                response = requests.get(
                    source, headers=headers, timeout=timeout, stream=True, allow_redirects=True
                )
                # Close connection immediately without downloading body
                response.close()

            # 200 OK - content might have changed
            if response.status_code == 200:
                new_etag = response.headers.get("ETag")
                new_last_modified = response.headers.get("Last-Modified")

                # If server provides ETag or Last-Modified, use them
                if new_etag and new_etag == stored_etag:
                    return {"changed": False}
                if new_last_modified and new_last_modified == stored_last_modified:
                    return {"changed": False}

                # Headers changed or not available, assume content changed
                return {
                    "changed": True,
                    "etag": new_etag,
                    "last_modified": new_last_modified,
                }

            # Other status codes - assume changed to be safe
            return {"changed": True}

        except Exception as e:
            # Network error - can't verify, assume not changed
            # This prevents errors from forcing unnecessary updates
            return {"changed": False, "error": str(e)}

    @staticmethod
    def get_file_metadata(source: str, content: str) -> Dict[str, Any]:
        """
        Get metadata for a file source.

        Args:
            source: File path
            content: File content

        Returns:
            Metadata dict with mtime and content_hash
        """
        mtime = os.path.getmtime(source) if os.path.exists(source) else None
        content_hash = ChangeDetector.compute_content_hash(content)

        return {
            "source": source,
            "last_checked": time.time(),
            "content_hash": content_hash,
            "mtime": mtime,
            "etag": None,
            "last_modified": None,
            "check_interval": 300.0,  # 5 minutes default
        }

    @staticmethod
    def get_url_metadata(
        source: str, content: str, timeout: int = 10
    ) -> Dict[str, Any]:
        """
        Get metadata for a URL source.

        Args:
            source: URL
            content: URL content
            timeout: Request timeout

        Returns:
            Metadata dict with etag, last_modified, and content_hash
        """
        content_hash = ChangeDetector.compute_content_hash(content)

        # Fetch HTTP headers
        try:
            response = requests.head(source, timeout=timeout, allow_redirects=True)
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
        except Exception:
            etag = None
            last_modified = None

        return {
            "source": source,
            "last_checked": time.time(),
            "content_hash": content_hash,
            "mtime": None,
            "etag": etag,
            "last_modified": last_modified,
            "check_interval": 300.0,  # 5 minutes default for URLs
        }

    @staticmethod
    def should_check_now(last_checked: float, check_interval: float) -> bool:
        """
        Determine if enough time has passed to check for updates.

        Args:
            last_checked: Unix timestamp of last check
            check_interval: Seconds between checks

        Returns:
            True if should check now
        """
        return (time.time() - last_checked) >= check_interval
