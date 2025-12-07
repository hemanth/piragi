"""Document loading using markitdown."""

import asyncio
import glob
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse, urljoin

from markitdown import MarkItDown

from .types import Document

# Remote filesystem schemes supported via fsspec
REMOTE_SCHEMES = {"s3", "gs", "gcs", "az", "abfs", "abfss", "hdfs", "webhdfs", "sftp", "ftp"}

# Lazy import fsspec - only when needed
_fsspec = None

# Lazy import crawl4ai - only when needed
_crawl4ai = None


def _get_fsspec():
    """Lazy load fsspec, raising helpful error if not installed."""
    global _fsspec
    if _fsspec is None:
        try:
            import fsspec
            _fsspec = fsspec
        except ImportError:
            raise ImportError(
                "fsspec is required for remote filesystem support. "
                "Install it with: pip install piragi[remote] or pip install fsspec"
            )
    return _fsspec


def _get_crawl4ai():
    """Lazy load crawl4ai, raising helpful error if not installed."""
    global _crawl4ai
    if _crawl4ai is None:
        try:
            import crawl4ai
            _crawl4ai = crawl4ai
        except ImportError:
            raise ImportError(
                "crawl4ai is required for recursive URL crawling. "
                "Install it with: pip install piragi[crawler] or pip install crawl4ai"
            )
    return _crawl4ai


class DocumentLoader:
    """Load documents from various sources using markitdown."""

    def __init__(self) -> None:
        """Initialize the document loader."""
        self.converter = MarkItDown()

    def load(self, source: Union[str, List[str]]) -> List[Document]:
        """
        Load documents from file paths, URLs, or glob patterns.

        Args:
            source: Single path/URL, list of paths/URLs, or glob pattern

        Returns:
            List of loaded documents
        """
        if isinstance(source, str):
            sources = [source]
        else:
            sources = source

        documents = []
        for src in sources:
            documents.extend(self._load_single(src))

        return documents

    def _load_single(self, source: str) -> List[Document]:
        """Load from a single source (file, URL, glob pattern, or remote URI)."""
        # Check if it's a remote filesystem URI (s3://, gs://, az://, etc.)
        if self._is_remote_uri(source):
            return self._load_remote(source)

        # Check if it's a crawl URL (ends with /**)
        if self._is_crawl_url(source):
            return self._crawl_url(source)

        # Check if it's a URL (http/https)
        if self._is_url(source):
            return [self._load_url(source)]

        # Check if it's a glob pattern
        if any(char in source for char in ["*", "?", "[", "]"]):
            return self._load_glob(source)

        # Single file
        if os.path.isfile(source):
            return [self._load_file(source)]

        # Directory - load all files
        if os.path.isdir(source):
            return self._load_directory(source)

        raise ValueError(f"Invalid source: {source}")

    def _is_remote_uri(self, source: str) -> bool:
        """Check if source is a remote filesystem URI (s3://, gs://, az://, etc.)."""
        try:
            parsed = urlparse(source)
            return parsed.scheme in REMOTE_SCHEMES
        except Exception:
            return False

    def _is_url(self, source: str) -> bool:
        """Check if source is an HTTP/HTTPS URL."""
        try:
            result = urlparse(source)
            return result.scheme in ("http", "https") and bool(result.netloc)
        except Exception:
            return False

    def _is_crawl_url(self, source: str) -> bool:
        """Check if source is a crawl URL (http(s) URL ending with /**)."""
        if not source.endswith("/**"):
            return False
        # Remove /** and check if it's a valid URL
        base_url = source[:-3]
        return self._is_url(base_url)

    def _load_file(self, file_path: str) -> Document:
        """Load a single file."""
        try:
            result = self.converter.convert(file_path)
            content = result.text_content

            # Extract metadata
            metadata = {
                "filename": os.path.basename(file_path),
                "file_type": Path(file_path).suffix.lstrip("."),
                "file_path": os.path.abspath(file_path),
            }

            return Document(content=content, source=file_path, metadata=metadata)

        except Exception as e:
            raise RuntimeError(f"Failed to load file {file_path}: {e}")

    def _load_url(self, url: str) -> Document:
        """Load content from a URL."""
        try:
            result = self.converter.convert(url)
            content = result.text_content

            metadata = {
                "filename": url.split("/")[-1] or "index",
                "file_type": "url",
                "file_path": url,
            }

            return Document(content=content, source=url, metadata=metadata)

        except Exception as e:
            raise RuntimeError(f"Failed to load URL {url}: {e}")

    def _load_glob(self, pattern: str) -> List[Document]:
        """Load files matching a glob pattern."""
        files = glob.glob(pattern, recursive=True)
        files = [f for f in files if os.path.isfile(f)]

        if not files:
            raise ValueError(f"No files found matching pattern: {pattern}")

        return [self._load_file(f) for f in files]

    def _load_directory(self, directory: str) -> List[Document]:
        """Load all files from a directory recursively."""
        pattern = os.path.join(directory, "**", "*")
        files = glob.glob(pattern, recursive=True)
        files = [f for f in files if os.path.isfile(f)]

        if not files:
            raise ValueError(f"No files found in directory: {directory}")

        documents = []
        for f in files:
            try:
                documents.append(self._load_file(f))
            except Exception:
                # Skip files that can't be processed
                continue

        return documents

    def _load_remote(self, uri: str) -> List[Document]:
        """
        Load files from remote filesystems (S3, GCS, Azure, etc.) using fsspec.

        Supports glob patterns in the URI path, e.g.:
            s3://bucket/docs/**/*.pdf
            gs://bucket/reports/*.md
            az://container/files/*.txt
        """
        # Parse the URI to get scheme
        parsed = urlparse(uri)
        scheme = parsed.scheme

        # Get fsspec (lazy import)
        fsspec = _get_fsspec()

        # Get the filesystem
        try:
            fs = fsspec.filesystem(scheme)
        except ImportError as e:
            # Provide helpful error message for missing dependencies
            pkg_map = {
                "s3": "s3fs",
                "gs": "gcsfs",
                "gcs": "gcsfs",
                "az": "adlfs",
                "abfs": "adlfs",
                "abfss": "adlfs",
            }
            pkg = pkg_map.get(scheme, scheme)
            raise ImportError(
                f"To use {scheme}:// URIs, install the required package: "
                f"pip install piragi[{scheme}] or pip install {pkg}"
            ) from e

        # Remove scheme for fsspec path
        remote_path = uri.split("://", 1)[1]

        # Check if it's a glob pattern
        is_glob = any(char in remote_path for char in ["*", "?", "[", "]"])

        if is_glob:
            # Use fsspec glob to find matching files
            files = fs.glob(remote_path)
            if not files:
                raise ValueError(f"No files found matching pattern: {uri}")
        else:
            # Single file or directory
            if fs.isdir(remote_path):
                # Load all files from remote directory
                files = [f for f in fs.find(remote_path) if not fs.isdir(f)]
            else:
                files = [remote_path]

        if not files:
            raise ValueError(f"No files found at: {uri}")

        documents = []
        for remote_file in files:
            try:
                doc = self._load_remote_file(fs, scheme, remote_file)
                documents.append(doc)
            except Exception:
                # Skip files that can't be processed
                continue

        if not documents:
            raise ValueError(f"No files could be loaded from: {uri}")

        return documents

    def _load_remote_file(self, fs, scheme: str, remote_path: str) -> Document:
        """Load a single file from a remote filesystem."""
        # Download to temp file for markitdown processing
        filename = os.path.basename(remote_path)
        suffix = Path(filename).suffix

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            # Download the file
            fs.get(remote_path, tmp_path)

        try:
            # Convert using markitdown
            result = self.converter.convert(tmp_path)
            content = result.text_content

            # Build the full URI for source tracking
            source_uri = f"{scheme}://{remote_path}"

            metadata = {
                "filename": filename,
                "file_type": suffix.lstrip(".") if suffix else "unknown",
                "file_path": source_uri,
                "remote_scheme": scheme,
            }

            return Document(content=content, source=source_uri, metadata=metadata)

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    def _crawl_url(
        self,
        source: str,
        max_depth: int = 3,
        max_pages: int = 100,
    ) -> List[Document]:
        """
        Recursively crawl a website and load all pages.

        Uses crawl4ai for efficient async crawling with JS rendering support.

        Args:
            source: URL ending with /** (e.g., https://docs.example.com/**)
            max_depth: Maximum depth to crawl (default: 3)
            max_pages: Maximum number of pages to crawl (default: 100)

        Returns:
            List of Document objects for each crawled page
        """
        # Get crawl4ai (lazy import)
        crawl4ai = _get_crawl4ai()
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

        # Remove /** suffix to get base URL
        base_url = source[:-3]
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc

        # Track visited URLs and documents
        visited: set = set()
        documents: List[Document] = []
        to_visit: List[tuple] = [(base_url, 0)]  # (url, depth)

        async def crawl_page(crawler: AsyncWebCrawler, url: str) -> Optional[tuple]:
            """Crawl a single page and return (content, links)."""
            try:
                config = CrawlerRunConfig()
                result = await crawler.arun(url=url, config=config)
                if result.success:
                    return result.markdown, result.links.get("internal", [])
                return None
            except Exception:
                return None

        async def run_crawler():
            """Run the async crawler."""
            nonlocal documents, visited, to_visit

            browser_config = BrowserConfig(headless=True)
            async with AsyncWebCrawler(config=browser_config) as crawler:
                while to_visit and len(documents) < max_pages:
                    url, depth = to_visit.pop(0)

                    # Skip if already visited or exceeds max depth
                    if url in visited or depth > max_depth:
                        continue

                    visited.add(url)

                    # Crawl the page
                    result = await crawl_page(crawler, url)
                    if result is None:
                        continue

                    content, links = result

                    # Skip empty content
                    if not content or len(content.strip()) < 50:
                        continue

                    # Create document
                    metadata = {
                        "filename": url.split("/")[-1] or "index",
                        "file_type": "url",
                        "file_path": url,
                        "crawl_depth": depth,
                        "crawl_source": base_url,
                    }
                    documents.append(Document(
                        content=content,
                        source=url,
                        metadata=metadata,
                    ))

                    # Add internal links to queue (same domain only)
                    if depth < max_depth:
                        for link in links:
                            link_url = link.get("href", "") if isinstance(link, dict) else str(link)
                            if not link_url:
                                continue

                            # Resolve relative URLs
                            full_url = urljoin(url, link_url)
                            parsed = urlparse(full_url)

                            # Only follow same-domain links
                            if parsed.netloc == base_domain and full_url not in visited:
                                # Remove fragments and normalize
                                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                                if parsed.query:
                                    clean_url += f"?{parsed.query}"

                                if clean_url not in visited:
                                    to_visit.append((clean_url, depth + 1))

        # Run the async crawler
        try:
            asyncio.get_event_loop().run_until_complete(run_crawler())
        except RuntimeError:
            # No event loop running, create a new one
            asyncio.run(run_crawler())

        if not documents:
            raise ValueError(f"No pages could be crawled from: {source}")

        return documents
