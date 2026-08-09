"""Async background updater for automatic document refresh."""

import asyncio
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union
from queue import Queue, Empty
import logging

from .change_detection import ChangeDetector

logger = logging.getLogger(__name__)


class UpdateTask:
    """Represents an update task."""

    def __init__(self, source: str, priority: int = 0):
        self.source = source
        self.priority = priority
        self.created_at = time.time()


class AsyncUpdater:
    """
    Background updater that checks and refreshes sources asynchronously.

    This prevents queries from being blocked while checking for updates.
    Updates happen in a background thread/task.
    """

    def __init__(
        self,
        refresh_callback: Callable[[Union[str, List[str]]], None],
        check_interval: float = 300.0,
        max_workers: int = 2,
    ):
        """
        Initialize the async updater.

        Args:
            refresh_callback: Function to call when refresh is needed (e.g., kb.refresh)
            check_interval: Default interval between checks in seconds
            max_workers: Number of background workers
        """
        self.refresh_callback = refresh_callback
        self.check_interval = check_interval
        self.max_workers = max_workers

        # Source metadata tracking
        self.sources_metadata: Dict[str, Dict[str, Any]] = {}

        # Update queue and control
        self.update_queue: Queue = Queue()
        self.running = False
        self._stop_event = threading.Event()
        self.workers: List[threading.Thread] = []
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "checks_performed": 0,
            "updates_performed": 0,
            "last_check_time": None,
        }

    def register_source(
        self,
        source: str,
        content: str,
        check_interval: Optional[float] = None,
    ) -> None:
        """
        Register a source for automatic update checking.

        Args:
            source: File path or URL
            content: Current content
            check_interval: Custom check interval (uses default if None)
        """
        with self._lock:
            if ChangeDetector.is_url(source):
                metadata = ChangeDetector.get_url_metadata(source, content)
            else:
                metadata = ChangeDetector.get_file_metadata(source, content)

            if check_interval is not None:
                metadata["check_interval"] = check_interval

            self.sources_metadata[source] = metadata
            logger.info(f"Registered source for auto-update: {source}")

    def unregister_source(self, source: str) -> None:
        """Remove a source from auto-update tracking."""
        with self._lock:
            if source in self.sources_metadata:
                del self.sources_metadata[source]
                logger.info(f"Unregistered source: {source}")

    def start(self) -> None:
        """Start background update workers."""
        if self.running:
            logger.warning("Updater already running")
            return

        self.running = True

        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncUpdater-Worker-{i}",
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)

        # Start scheduler thread
        scheduler = threading.Thread(
            target=self._scheduler_loop,
            name="AsyncUpdater-Scheduler",
            daemon=True,
        )
        scheduler.start()
        self.workers.append(scheduler)

        logger.info(f"Started {self.max_workers} update workers + scheduler")

    def stop(self) -> None:
        """Stop background workers."""
        self.running = False
        self._stop_event.set()
        logger.info("Stopping async updater...")

        # Wait for workers to finish current tasks
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5.0)

        self.workers.clear()
        logger.info("Async updater stopped")

    def _scheduler_loop(self) -> None:
        """
        Main scheduler loop that checks which sources need updating.
        Runs in background thread.
        """
        while self.running:
            try:
                current_time = time.time()

                with self._lock:
                    sources_to_check = []

                    for source, metadata in self.sources_metadata.items():
                        # Check if enough time has passed
                        if ChangeDetector.should_check_now(
                            metadata["last_checked"], metadata["check_interval"]
                        ):
                            sources_to_check.append(source)

                # Queue checks
                for source in sources_to_check:
                    self.update_queue.put(UpdateTask(source))

                self.stats["last_check_time"] = current_time

                # Sleep until next scheduled check (check every 10s for any due sources)
                self._stop_event.wait(timeout=10)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                self._stop_event.wait(timeout=10)

    def _worker_loop(self) -> None:
        """
        Worker loop that processes update tasks.
        Runs in background thread.
        """
        while self.running:
            try:
                # Get task from queue (timeout to allow checking self.running)
                try:
                    task = self.update_queue.get(timeout=1.0)
                except Empty:
                    continue

                # Process the update task
                self._check_and_refresh(task.source)

                self.update_queue.task_done()

            except Exception as e:
                logger.error(f"Worker error: {e}")

    def _check_and_refresh(self, source: str) -> None:
        """
        Check if source changed and refresh if needed.

        Args:
            source: Source to check
        """
        try:
            with self._lock:
                if source not in self.sources_metadata:
                    return

                metadata = self.sources_metadata[source]

            self.stats["checks_performed"] += 1

            # Check for changes
            changed = False
            new_metadata = {}

            if ChangeDetector.is_url(source):
                result = ChangeDetector.check_url_changed(
                    source,
                    metadata.get("etag"),
                    metadata.get("last_modified"),
                )
                changed = result.get("changed", False)
                if "etag" in result:
                    new_metadata["etag"] = result["etag"]
                if "last_modified" in result:
                    new_metadata["last_modified"] = result["last_modified"]
            else:
                changed = ChangeDetector.check_file_changed(
                    source, metadata.get("mtime"), metadata["content_hash"]
                )

            # Update last_checked timestamp
            with self._lock:
                self.sources_metadata[source]["last_checked"] = time.time()
                if new_metadata:
                    self.sources_metadata[source].update(new_metadata)

            # Refresh if changed
            if changed:
                logger.info(f"Change detected in {source}, refreshing...")
                self.refresh_callback(source)
                self.stats["updates_performed"] += 1

                # Update metadata after refresh
                # Note: In production, you'd want to read the new content and hash
                # For now, we just update the timestamp

        except Exception as e:
            logger.error(f"Error checking source {source}: {e}")

    def queue_update(self, source: str, priority: int = 0) -> None:
        """
        Manually queue a source for update check.

        Args:
            source: Source to check
            priority: Priority (higher = more urgent)
        """
        self.update_queue.put(UpdateTask(source, priority))
        logger.debug(f"Queued update check for {source}")

    def get_stats(self) -> Dict[str, Any]:
        """Get updater statistics."""
        with self._lock:
            return {
                **self.stats,
                "registered_sources": len(self.sources_metadata),
                "queue_size": self.update_queue.qsize(),
                "running": self.running,
            }


class AsyncUpdaterMixin:
    """
    Mixin to add async update capabilities to Ragi.

    Usage:
        class AutoUpdateRagi(AsyncUpdaterMixin, Ragi):
            pass

        kb = AutoUpdateRagi("./docs", auto_update=True)
    """

    def __init__(self, *args, auto_update: bool = False, **kwargs):
        """
        Initialize with optional auto-update.

        Args:
            auto_update: Enable background auto-updates
            **kwargs: Additional config including:
                - auto_update_interval: Check interval in seconds
                - auto_update_workers: Number of background workers
        """
        super().__init__(*args, **kwargs)

        self.auto_update_enabled = auto_update
        self.updater: Optional[AsyncUpdater] = None

        if auto_update:
            # Extract auto-update config
            config = kwargs.get("config", {})
            auto_config = config.get("auto_update", {})

            interval = auto_config.get("interval", 300.0)
            workers = auto_config.get("workers", 2)

            # Initialize updater
            self.updater = AsyncUpdater(
                refresh_callback=self.refresh,
                check_interval=interval,
                max_workers=workers,
            )

            # Start background workers
            self.updater.start()

    def add(self, sources, **kwargs):
        """Override add to register sources with auto-updater."""
        result = super().add(sources, **kwargs)

        # Register sources with updater if auto-update enabled
        if self.auto_update_enabled and self.updater:
            # Note: Would need access to loaded documents to register properly
            # This is a simplified version
            source_list = sources if isinstance(sources, list) else [sources]
            for source in source_list:
                # Register with default metadata
                # In production, you'd track the actual content
                self.updater.register_source(source, "", check_interval=None)

        return result

    def get_update_stats(self) -> Dict[str, Any]:
        """Get auto-update statistics."""
        if self.updater:
            return self.updater.get_stats()
        return {"auto_update_enabled": False}

    def stop_auto_update(self) -> None:
        """Stop auto-update background workers."""
        if self.updater:
            self.updater.stop()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "updater") and self.updater:
            self.updater.stop()
