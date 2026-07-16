from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PySide6.QtCore import QThread
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from mdexplore_app.file_coordination import advisory_file_lock
from pdfexplore.app import PdfExploreWindow
from pdfexplore.workers import (
    BACKGROUND_LEASE_BUSY,
    PdfTextCacheGcWorker,
    PdfTextPrefetchWorker,
)


class PdfTextCacheGcWorkerTests(unittest.TestCase):
    def test_worker_reports_only_entries_with_definitely_missing_sources(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdfexplore-cache-gc-worker-") as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            existing_source = root / "existing.pdf"
            existing_source.write_bytes(b"pdf")
            missing_source = root / "missing.pdf"

            existing_cache = cache_dir / "existing.txt.gz"
            existing_cache.write_bytes(b"cached")
            existing_metadata = cache_dir / "existing.txt.gz.meta.json"
            existing_metadata.write_text(
                json.dumps({"source_path": str(existing_source)}),
                encoding="utf-8",
            )

            missing_cache = cache_dir / "missing.txt.gz"
            missing_cache.write_bytes(b"cached")
            missing_metadata = cache_dir / "missing.txt.gz.meta.json"
            missing_metadata.write_text(
                json.dumps({"source_path": str(missing_source)}),
                encoding="utf-8",
            )

            orphan_metadata = cache_dir / "orphan.txt.gz.meta.json"
            orphan_metadata.write_text(
                json.dumps({"source_path": str(existing_source)}),
                encoding="utf-8",
            )

            emitted: dict[str, object] = {}
            worker = PdfTextCacheGcWorker(
                7,
                [str(existing_source), str(missing_source)],
                cache_dir,
                batch_size=20,
            )
            worker.signals.finished.connect(
                lambda request_id, payload, error: emitted.update(
                    request_id=request_id,
                    payload=payload,
                    error=error,
                )
            )
            worker.run()

            self.assertEqual(emitted.get("request_id"), 7)
            self.assertEqual(emitted.get("error"), "")
            payload = emitted.get("payload")
            self.assertIsInstance(payload, dict)
            assert isinstance(payload, dict)
            self.assertEqual(
                payload.get("missing_memory_path_keys"),
                [str(missing_source)],
            )
            self.assertEqual(
                [entry["source_path"] for entry in payload["missing_disk_entries"]],
                [str(missing_source)],
            )
            self.assertEqual(
                payload.get("stale_metadata_paths"),
                [str(orphan_metadata)],
            )

    def test_worker_stops_before_disk_glob_when_cancelled_during_memory_scan(
        self,
    ) -> None:
        abort_state = {"value": False}
        worker = PdfTextCacheGcWorker(
            8,
            ["/missing/one.pdf", "/missing/two.pdf"],
            Path("/unused/cache"),
            should_abort=lambda: abort_state["value"],
        )

        def _check_one(_path_key: str) -> bool:
            abort_state["value"] = True
            return False

        with (
            patch.object(worker, "_source_is_missing", side_effect=_check_one),
            patch("pdfexplore.workers.os.scandir") as disk_scan,
        ):
            worker.run()

        disk_scan.assert_not_called()

    def test_partial_memory_batch_advances_cursor_only_through_checked_entry(
        self,
    ) -> None:
        abort_state = {"value": False}
        emitted: dict[str, object] = {}
        path_keys = ["/documents/a.pdf", "/documents/b.pdf", "/documents/c.pdf"]
        worker = PdfTextCacheGcWorker(
            81,
            path_keys,
            Path("/unused/cache"),
            batch_size=len(path_keys),
            should_abort=lambda: abort_state["value"],
        )
        worker.signals.finished.connect(
            lambda _request_id, payload, error: emitted.update(
                payload=payload,
                error=error,
            )
        )

        def _check_first(path_key: str) -> bool:
            self.assertEqual(path_key, path_keys[0])
            abort_state["value"] = True
            return False

        with (
            patch.object(worker, "_source_is_missing", side_effect=_check_first),
            patch("pdfexplore.workers.os.scandir") as disk_scan,
        ):
            worker.run()

        disk_scan.assert_not_called()
        self.assertEqual(emitted.get("error"), "")
        payload = emitted.get("payload")
        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertEqual(payload.get("memory_cursor"), path_keys[0])

    def test_partial_disk_batch_advances_cursor_only_through_checked_entry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="pdfexplore-cache-gc-cursor-") as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            metadata_names: list[str] = []
            source_paths: list[str] = []
            for label in ("a", "b", "c"):
                cache_path = cache_dir / f"{label}.txt.gz"
                cache_path.write_bytes(b"cached")
                source_path = str(root / f"{label}.pdf")
                metadata_path = cache_dir / f"{label}.txt.gz.meta.json"
                metadata_path.write_text(
                    json.dumps({"source_path": source_path}),
                    encoding="utf-8",
                )
                metadata_names.append(metadata_path.name)
                source_paths.append(source_path)

            abort_state = {"value": False}
            checked_sources: list[str] = []
            emitted: dict[str, object] = {}
            worker = PdfTextCacheGcWorker(
                82,
                [],
                cache_dir,
                batch_size=len(metadata_names),
                should_abort=lambda: abort_state["value"],
            )
            worker.signals.finished.connect(
                lambda _request_id, payload, error: emitted.update(
                    payload=payload,
                    error=error,
                )
            )

            def _check_first(source_path: str) -> bool:
                checked_sources.append(source_path)
                abort_state["value"] = True
                return False

            with patch.object(
                worker,
                "_source_is_missing",
                side_effect=_check_first,
            ):
                worker.run()

            self.assertEqual(checked_sources, [source_paths[0]])
            self.assertEqual(emitted.get("error"), "")
            payload = emitted.get("payload")
            self.assertIsInstance(payload, dict)
            assert isinstance(payload, dict)
            self.assertEqual(payload.get("disk_cursor"), metadata_names[0])

    def test_worker_defers_metadata_read_io_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdfexplore-cache-gc-metadata-io-") as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            source = root / "existing.pdf"
            source.write_bytes(b"pdf")
            cache_path = cache_dir / "entry.txt.gz"
            cache_path.write_bytes(b"cached")
            metadata_path = cache_dir / "entry.txt.gz.meta.json"
            metadata_path.write_text(
                json.dumps({"source_path": str(source)}),
                encoding="utf-8",
            )
            original_read_text = Path.read_text
            emitted: dict[str, object] = {}

            def _read_text(path: Path, *args, **kwargs) -> str:
                if path == metadata_path:
                    raise PermissionError("metadata read denied")
                return original_read_text(path, *args, **kwargs)

            worker = PdfTextCacheGcWorker(12, [], cache_dir)
            worker.signals.finished.connect(
                lambda _request_id, payload, error: emitted.update(
                    payload=payload,
                    error=error,
                )
            )
            with patch.object(Path, "read_text", new=_read_text):
                worker.run()

            self.assertEqual(emitted.get("error"), "")
            payload = emitted.get("payload")
            self.assertIsInstance(payload, dict)
            assert isinstance(payload, dict)
            self.assertEqual(payload.get("missing_disk_entries"), [])
            self.assertEqual(payload.get("stale_metadata_paths"), [])

    def test_worker_reports_busy_lease_distinctly(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdfexplore-cache-gc-busy-") as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            lease_path = Path(tmpdir) / "background.lock"
            emitted: dict[str, object] = {}
            worker = PdfTextCacheGcWorker(
                9,
                [],
                cache_dir,
                lease_lock_path=lease_path,
            )
            worker.signals.finished.connect(
                lambda request_id, payload, error: emitted.update(
                    request_id=request_id,
                    payload=payload,
                    error=error,
                )
            )

            with advisory_file_lock(lease_path, blocking=True) as acquired:
                self.assertTrue(acquired)
                worker.run()

            self.assertEqual(emitted.get("request_id"), 9)
            self.assertEqual(emitted.get("payload"), {})
            self.assertEqual(emitted.get("error"), BACKGROUND_LEASE_BUSY)

    def test_finished_signal_runs_after_successful_gc_lease_release(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdfexplore-cache-gc-release-") as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            lease_path = Path(tmpdir) / "background.lock"
            lease_available_in_callback: list[bool] = []
            worker = PdfTextCacheGcWorker(
                10,
                [],
                cache_dir,
                lease_lock_path=lease_path,
            )

            def _on_finished(
                _request_id: int,
                _payload: object,
                _error: str,
            ) -> None:
                with advisory_file_lock(
                    lease_path,
                    exclusive=True,
                    blocking=False,
                ) as acquired:
                    lease_available_in_callback.append(acquired)

            worker.signals.finished.connect(_on_finished)
            worker.run()

            self.assertEqual(lease_available_in_callback, [True])

    def test_gc_worker_reports_busy_when_lease_parent_is_not_a_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="pdfexplore-cache-gc-lock-error-") as tmpdir:
            blocked_parent = Path(tmpdir) / "not-a-directory"
            blocked_parent.write_text("blocked", encoding="utf-8")
            emitted: dict[str, object] = {}
            worker = PdfTextCacheGcWorker(
                11,
                [],
                Path(tmpdir) / "cache",
                lease_lock_path=blocked_parent / "background.lock",
            )
            worker.signals.finished.connect(
                lambda request_id, payload, error: emitted.update(
                    request_id=request_id,
                    payload=payload,
                    error=error,
                )
            )

            worker.run()

            self.assertEqual(emitted.get("request_id"), 11)
            self.assertEqual(emitted.get("payload"), {})
            self.assertEqual(emitted.get("error"), BACKGROUND_LEASE_BUSY)


class PdfTextCacheGcWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="pdfexplore-cache-gc-window-")
        self.root = Path(self._tempdir.name)
        self.window = PdfExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.root / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        self.window._pdf_text_disk_cache_dir = self.root / "text-cache"
        self.window._last_observed_global_activity_wall_time = (
            time.time()
            - max(
                self.window.PREFETCH_IDLE_SECONDS,
                self.window.PDF_TEXT_CACHE_GC_IDLE_SECONDS,
            )
            - 2.0
        )

    def tearDown(self) -> None:
        self.window._active_pdf_text_cache_gc_worker = None
        self.window.close()
        QApplication.processEvents()
        self._tempdir.cleanup()

    def test_gc_removes_memory_disk_metadata_and_badge_for_deleted_pdf(self) -> None:
        source = self.root / "deleted.pdf"
        source.write_bytes(b"pdf")
        stat = source.stat()
        path_key = self.window._path_key(source)
        self.window._store_cached_pdf_text(
            path_key,
            stat.st_mtime_ns,
            stat.st_size,
            "extracted text",
        )
        self.window._store_pdf_text_to_disk_cache(
            path_key,
            stat.st_mtime_ns,
            stat.st_size,
            "extracted text",
        )
        self.window._record_cached_pdf_path_key(path_key)

        cache_path = self.window._pdf_text_disk_cache_file_path(
            path_key,
            stat.st_mtime_ns,
            stat.st_size,
        )
        metadata_path = self.window._pdf_text_disk_cache_metadata_path(cache_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata.get("source_path"), path_key)

        source.unlink()
        removed_memory, removed_disk = self.window._apply_pdf_text_cache_gc_payload(
            {
                "missing_memory_path_keys": [path_key],
                "missing_disk_entries": [
                    {
                        "source_path": path_key,
                        "cache_path": str(cache_path),
                        "metadata_path": str(metadata_path),
                    }
                ],
                "stale_metadata_paths": [],
            }
        )

        self.assertEqual((removed_memory, removed_disk), (1, 1))
        self.assertEqual(self.window._pdf_text_cache_total_chars, 0)
        self.assertNotIn(path_key, self.window._pdf_text_cache)
        self.assertNotIn(path_key, self.window._cached_pdf_path_keys)
        self.assertFalse(cache_path.exists())
        self.assertFalse(metadata_path.exists())

    def test_quota_trim_is_deferred_to_abortable_idle_gc(self) -> None:
        first = self.root / "first.pdf"
        second = self.root / "second.pdf"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        self.window.PDF_TEXT_DISK_CACHE_TRIM_INTERVAL = 1
        self.window.PDF_TEXT_DISK_CACHE_MAX_FILES = 1
        self.window.PDF_TEXT_DISK_CACHE_MAX_BYTES = 1024 * 1024

        first_stat = first.stat()
        first_key = self.window._path_key(first)
        self.window._store_pdf_text_to_disk_cache(
            first_key,
            first_stat.st_mtime_ns,
            first_stat.st_size,
            "first text",
        )
        time.sleep(0.01)
        second_stat = second.stat()
        second_key = self.window._path_key(second)
        self.window._store_pdf_text_to_disk_cache(
            second_key,
            second_stat.st_mtime_ns,
            second_stat.st_size,
            "second text",
        )

        self.assertEqual(
            len(list(self.window._pdf_text_disk_cache_dir.glob("*.txt.gz"))),
            2,
        )
        self.window._apply_pdf_text_cache_gc_payload(
            {
                "missing_memory_path_keys": [],
                "missing_disk_entries": [],
                "stale_metadata_paths": [],
            }
        )

        self.assertEqual(
            len(list(self.window._pdf_text_disk_cache_dir.glob("*.txt.gz"))),
            1,
        )
        self.assertEqual(
            self.window._pdf_text_disk_cache_trim_completed_revision,
            self.window._pdf_text_disk_cache_trim_requested_revision,
        )

    def test_quota_trim_defers_on_cache_entry_stat_error(self) -> None:
        source = self.root / "stat-error.pdf"
        source.write_bytes(b"pdf")
        source_stat = source.stat()
        path_key = self.window._path_key(source)
        self.window._store_pdf_text_to_disk_cache(
            path_key,
            source_stat.st_mtime_ns,
            source_stat.st_size,
            "cached text",
        )
        cache_path = self.window._pdf_text_disk_cache_file_path(
            path_key,
            source_stat.st_mtime_ns,
            source_stat.st_size,
        )
        self.window.PDF_TEXT_DISK_CACHE_MAX_FILES = 0
        original_stat = Path.stat

        def _stat(path: Path, *args, **kwargs):
            if path == cache_path:
                raise PermissionError("cache stat denied")
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, "stat", new=_stat):
            completed = self.window._trim_pdf_text_disk_cache()

        self.assertFalse(completed)
        self.assertTrue(cache_path.exists())

    def test_shared_trim_dirty_marker_triggers_quota_trim_in_another_window(
        self,
    ) -> None:
        other = PdfExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.root / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        other._pdf_text_disk_cache_dir = self.window._pdf_text_disk_cache_dir
        try:
            self.window.PDF_TEXT_DISK_CACHE_TRIM_INTERVAL = 1000
            other.PDF_TEXT_DISK_CACHE_MAX_FILES = 1
            other.PDF_TEXT_DISK_CACHE_MAX_BYTES = 1024 * 1024
            other._pdf_text_disk_cache_trim_completed_revision = (
                other._pdf_text_disk_cache_trim_requested_revision
            )

            for label in ("first", "second"):
                source = self.root / f"{label}.pdf"
                source.write_bytes(label.encode("utf-8"))
                source_stat = source.stat()
                self.assertTrue(
                    self.window._store_pdf_text_to_disk_cache(
                        self.window._path_key(source),
                        source_stat.st_mtime_ns,
                        source_stat.st_size,
                        f"{label} text",
                    )
                )

            dirty_path = other._pdf_text_disk_cache_trim_dirty_path()
            self.assertTrue(dirty_path.is_file())
            self.assertEqual(other._pdf_text_disk_cache_store_count, 0)
            self.assertEqual(
                other._pdf_text_disk_cache_trim_completed_revision,
                other._pdf_text_disk_cache_trim_requested_revision,
            )
            self.assertEqual(
                len(list(other._pdf_text_disk_cache_dir.glob("*.txt.gz"))),
                2,
            )

            other._apply_pdf_text_cache_gc_payload(
                {
                    "missing_memory_path_keys": [],
                    "missing_disk_entries": [],
                    "stale_metadata_paths": [],
                }
            )

            self.assertEqual(
                len(list(other._pdf_text_disk_cache_dir.glob("*.txt.gz"))),
                1,
            )
            self.assertFalse(dirty_path.exists())
        finally:
            other.close()
            QApplication.processEvents()

    def test_cancelled_disk_store_removes_its_temporary_file(self) -> None:
        source = self.root / "cancel-store.pdf"
        source.write_bytes(b"pdf")
        stat = source.stat()
        path_key = self.window._path_key(source)
        checks = 0

        def _should_abort() -> bool:
            nonlocal checks
            checks += 1
            return checks >= 2

        stored = self.window._store_pdf_text_to_disk_cache(
            path_key,
            stat.st_mtime_ns,
            stat.st_size,
            "x" * (2 * 1024 * 1024),
            should_abort=_should_abort,
        )

        self.assertFalse(stored)
        self.assertFalse(
            self.window._pdf_text_disk_cache_file_path(
                path_key,
                stat.st_mtime_ns,
                stat.st_size,
            ).exists()
        )
        self.assertFalse(
            any(self.window._pdf_text_disk_cache_dir.glob(".*.tmp.*"))
        )

    def test_gzip_read_does_not_hold_shared_process_lock(self) -> None:
        source = self.root / "read-lock.pdf"
        source.write_bytes(b"pdf")
        stat = source.stat()
        path_key = self.window._path_key(source)
        cache_path = self.window._pdf_text_disk_cache_file_path(
            path_key,
            stat.st_mtime_ns,
            stat.st_size,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"placeholder")
        entered = threading.Event()
        release = threading.Event()
        result: list[str | None] = []

        class _BlockingGzipReader:
            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def read(self, _size: int = -1) -> str:
                entered.set()
                release.wait(5.0)
                if getattr(self, "_returned", False):
                    return ""
                self._returned = True
                return "cached text"

        with patch("pdfexplore.app.gzip.open", return_value=_BlockingGzipReader()):
            thread = threading.Thread(
                target=lambda: result.append(
                    self.window._load_pdf_text_from_disk_cache(
                        path_key,
                        stat.st_mtime_ns,
                        stat.st_size,
                    )
                )
            )
            thread.start()
            self.assertTrue(entered.wait(3.0))
            with advisory_file_lock(
                self.window._pdf_text_disk_cache_process_lock_path(),
                exclusive=True,
                blocking=False,
            ) as acquired:
                self.assertTrue(acquired)
            release.set()
            thread.join(5.0)

        self.assertEqual(result, ["cached text"])

    def test_cancelled_chunked_gzip_read_preserves_valid_cache_entry(self) -> None:
        source = self.root / "cancel-read.pdf"
        source.write_bytes(b"pdf")
        source_stat = source.stat()
        path_key = self.window._path_key(source)
        cache_path = self.window._pdf_text_disk_cache_file_path(
            path_key,
            source_stat.st_mtime_ns,
            source_stat.st_size,
        )
        metadata_path = self.window._pdf_text_disk_cache_metadata_path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"valid compressed placeholder")
        metadata_path.write_text(
            json.dumps({"source_path": path_key}),
            encoding="utf-8",
        )
        state = {"reads": 0}

        class _ChunkedReader:
            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def read(self, size: int = -1) -> str:
                self.assert_read_size = size
                state["reads"] += 1
                return "first chunk"

        with patch("pdfexplore.app.gzip.open", return_value=_ChunkedReader()):
            loaded = self.window._load_pdf_text_from_disk_cache(
                path_key,
                source_stat.st_mtime_ns,
                source_stat.st_size,
                should_abort=lambda: state["reads"] >= 1,
            )

        self.assertIsNone(loaded)
        self.assertEqual(state["reads"], 1)
        self.assertTrue(cache_path.is_file())
        self.assertTrue(metadata_path.is_file())

    def test_read_cancellation_after_disk_load_does_not_publish_cache_state(
        self,
    ) -> None:
        source = self.root / "cancel-after-read.pdf"
        source.write_bytes(b"pdf")
        path_key = self.window._path_key(source)
        cancelled = {"value": False}

        def _load(*_args, **_kwargs) -> str:
            cancelled["value"] = True
            return "cached text"

        with patch.object(
            self.window,
            "_load_pdf_text_from_disk_cache",
            side_effect=_load,
        ):
            loaded = self.window._read_pdf_text(
                source,
                should_abort=lambda: cancelled["value"],
            )

        self.assertEqual(loaded, "")
        self.assertNotIn(path_key, self.window._pdf_text_cache)
        self.assertNotIn(path_key, self.window._cached_pdf_path_keys)

    def test_cache_producer_lock_avoids_duplicate_extraction(self) -> None:
        source = self.root / "shared-extraction.pdf"
        source.write_bytes(b"pdf")
        other = PdfExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.root / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        other._pdf_text_disk_cache_dir = self.window._pdf_text_disk_cache_dir
        entered = threading.Event()
        release = threading.Event()
        page = Mock()

        def _extract_once() -> str:
            entered.set()
            release.wait(5.0)
            return "shared cached text"

        page.extract_text.side_effect = _extract_once
        reader_factory = Mock(return_value=Mock(pages=[page]))
        results: list[str] = []
        try:
            with patch("pdfexplore.app.PdfReader", reader_factory):
                first_thread = threading.Thread(
                    target=lambda: results.append(
                        self.window._read_pdf_text(source)
                    )
                )
                second_thread = threading.Thread(
                    target=lambda: results.append(other._read_pdf_text(source))
                )
                first_thread.start()
                self.assertTrue(entered.wait(3.0))
                second_thread.start()
                time.sleep(0.05)
                self.assertEqual(reader_factory.call_count, 1)
                release.set()
                first_thread.join(5.0)
                second_thread.join(5.0)
        finally:
            other.close()
            QApplication.processEvents()

        self.assertEqual(reader_factory.call_count, 1)
        self.assertEqual(
            sorted(results),
            ["shared cached text", "shared cached text"],
        )

    def test_gc_preserves_stale_candidate_repaired_after_worker_scan(self) -> None:
        source = self.root / "repaired.pdf"
        source.write_bytes(b"pdf")
        cache_path = self.window._pdf_text_disk_cache_dir / "repaired.txt.gz"
        metadata_path = self.window._pdf_text_disk_cache_metadata_path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"fresh cache")
        metadata_path.write_text(
            json.dumps({"version": 1, "source_path": str(source.resolve())}),
            encoding="utf-8",
        )

        self.window._apply_pdf_text_cache_gc_payload(
            {
                "missing_memory_path_keys": [],
                "missing_disk_entries": [],
                "stale_metadata_paths": [str(metadata_path)],
            }
        )

        self.assertTrue(cache_path.is_file())
        self.assertTrue(metadata_path.is_file())

    def test_gc_does_not_delete_replacement_after_worker_scan(self) -> None:
        missing_source = self.root / "missing.pdf"
        cache_path = self.window._pdf_text_disk_cache_dir / "replace.txt.gz"
        metadata_path = self.window._pdf_text_disk_cache_metadata_path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"old cache")
        metadata_path.write_text(
            json.dumps({"source_path": str(missing_source.resolve())}),
            encoding="utf-8",
        )
        scanned_stat = cache_path.stat()
        scanned_identity = [
            scanned_stat.st_dev,
            scanned_stat.st_ino,
            scanned_stat.st_mtime_ns,
            scanned_stat.st_size,
        ]
        replacement = cache_path.with_name("replacement.tmp")
        replacement.write_bytes(b"new cache")
        replacement.replace(cache_path)

        removed_memory, removed_disk = self.window._apply_pdf_text_cache_gc_payload(
            {
                "missing_memory_path_keys": [],
                "missing_disk_entries": [
                    {
                        "source_path": str(missing_source.resolve()),
                        "cache_path": str(cache_path),
                        "metadata_path": str(metadata_path),
                        "cache_identity": scanned_identity,
                    }
                ],
                "stale_metadata_paths": [],
            }
        )

        self.assertEqual((removed_memory, removed_disk), (0, 0))
        self.assertEqual(cache_path.read_bytes(), b"new cache")
        self.assertTrue(metadata_path.is_file())

    def test_gc_preserves_cache_when_metadata_owner_changes_after_scan(self) -> None:
        missing_source = self.root / "missing-owner.pdf"
        replacement_source = self.root / "replacement-owner.pdf"
        replacement_source.write_bytes(b"pdf")
        replacement_stat = replacement_source.stat()
        replacement_path_key = self.window._path_key(replacement_source)
        cache_path = self.window._pdf_text_disk_cache_file_path(
            replacement_path_key,
            replacement_stat.st_mtime_ns,
            replacement_stat.st_size,
        )
        metadata_path = self.window._pdf_text_disk_cache_metadata_path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"cached text")
        metadata_path.write_text(
            json.dumps({"source_path": str(missing_source.resolve())}),
            encoding="utf-8",
        )
        scanned_stat = cache_path.stat()
        scanned_identity = [
            scanned_stat.st_dev,
            scanned_stat.st_ino,
            scanned_stat.st_mtime_ns,
            scanned_stat.st_size,
        ]

        metadata_path.write_text(
            json.dumps({"source_path": replacement_path_key}),
            encoding="utf-8",
        )
        removed_memory, removed_disk = self.window._apply_pdf_text_cache_gc_payload(
            {
                "missing_memory_path_keys": [],
                "missing_disk_entries": [
                    {
                        "source_path": str(missing_source.resolve()),
                        "cache_path": str(cache_path),
                        "metadata_path": str(metadata_path),
                        "cache_identity": scanned_identity,
                    }
                ],
                "stale_metadata_paths": [],
            }
        )

        self.assertEqual((removed_memory, removed_disk), (0, 0))
        self.assertEqual(cache_path.read_bytes(), b"cached text")
        self.assertEqual(
            json.loads(metadata_path.read_text(encoding="utf-8"))["source_path"],
            replacement_path_key,
        )

    def test_gc_commits_gzip_removal_when_metadata_unlink_fails(self) -> None:
        source = self.root / "deleted-with-stuck-metadata.pdf"
        source.write_bytes(b"pdf")
        source_stat = source.stat()
        path_key = self.window._path_key(source)
        self.window._store_cached_pdf_text(
            path_key,
            source_stat.st_mtime_ns,
            source_stat.st_size,
            "cached text",
        )
        self.window._record_cached_pdf_path_key(path_key)
        cache_path = self.window._pdf_text_disk_cache_file_path(
            path_key,
            source_stat.st_mtime_ns,
            source_stat.st_size,
        )
        metadata_path = self.window._pdf_text_disk_cache_metadata_path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"cached text")
        metadata_path.write_text(
            json.dumps({"source_path": path_key}),
            encoding="utf-8",
        )
        scanned_stat = cache_path.stat()
        scanned_identity = [
            scanned_stat.st_dev,
            scanned_stat.st_ino,
            scanned_stat.st_mtime_ns,
            scanned_stat.st_size,
        ]
        source.unlink()
        original_unlink = Path.unlink

        def _unlink(path: Path, *args, **kwargs) -> None:
            if path == metadata_path:
                raise PermissionError("metadata unlink denied")
            original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", new=_unlink):
            removed_memory, removed_disk = (
                self.window._apply_pdf_text_cache_gc_payload(
                    {
                        "missing_memory_path_keys": [],
                        "missing_disk_entries": [
                            {
                                "source_path": path_key,
                                "cache_path": str(cache_path),
                                "metadata_path": str(metadata_path),
                                "cache_identity": scanned_identity,
                            }
                        ],
                        "stale_metadata_paths": [],
                    }
                )
            )

        self.assertEqual((removed_memory, removed_disk), (1, 1))
        self.assertFalse(cache_path.exists())
        self.assertTrue(metadata_path.exists())
        self.assertEqual(self.window._pdf_text_cache_total_chars, 0)
        self.assertNotIn(path_key, self.window._pdf_text_cache)
        self.assertNotIn(path_key, self.window._cached_pdf_path_keys)

    def test_gc_stops_before_memory_mutation_when_activity_resumes(self) -> None:
        source = self.root / "stop-before-memory.pdf"
        source.write_bytes(b"pdf")
        stat = source.stat()
        path_key = self.window._path_key(source)
        self.window._store_cached_pdf_text(
            path_key,
            stat.st_mtime_ns,
            stat.st_size,
            "keep until next idle pass",
        )
        self.window._record_cached_pdf_path_key(path_key)
        source.unlink()
        abort_checks = 0

        def _should_abort() -> bool:
            nonlocal abort_checks
            abort_checks += 1
            # Candidate validation uses two checks; cancel immediately before
            # the final in-memory mutation phase.
            return abort_checks >= 3

        self.window._apply_pdf_text_cache_gc_payload(
            {
                "missing_memory_path_keys": [path_key],
                "missing_disk_entries": [],
                "stale_metadata_paths": [],
            },
            _should_abort,
        )

        self.assertIn(path_key, self.window._pdf_text_cache)
        self.assertIn(path_key, self.window._cached_pdf_path_keys)

    def test_text_extraction_cancellation_between_pages_does_not_cache_partial_text(
        self,
    ) -> None:
        source = self.root / "cancelled.pdf"
        source.write_bytes(b"pdf")
        first_page = Mock()
        second_page = Mock()
        second_page.extract_text.return_value = "second page"
        reader = Mock(pages=[first_page, second_page])
        abort_state = {"value": False}

        def _extract_first_page() -> str:
            abort_state["value"] = True
            return "partial first page"

        first_page.extract_text.side_effect = _extract_first_page

        def should_abort() -> bool:
            return abort_state["value"]

        with (
            patch("pdfexplore.app.PdfReader", return_value=reader),
            patch.object(self.window, "_lookup_cached_pdf_text", return_value=None),
            patch.object(
                self.window,
                "_load_pdf_text_from_disk_cache",
                return_value=None,
            ),
            patch.object(self.window, "_store_cached_pdf_text") as store_memory,
            patch.object(self.window, "_store_pdf_text_to_disk_cache") as store_disk,
            patch.object(self.window, "_record_cached_pdf_path_key") as record_badge,
        ):
            extracted = self.window._read_pdf_text(
                source,
                should_abort=should_abort,
            )

        self.assertEqual(extracted, "")
        first_page.extract_text.assert_called_once_with()
        second_page.extract_text.assert_not_called()
        store_memory.assert_not_called()
        store_disk.assert_not_called()
        record_badge.assert_not_called()

    def test_idle_prefetch_populates_cached_badge_without_gui_cache_probe(
        self,
    ) -> None:
        from reportlab.pdfgen import canvas

        source = self.root / "idle-prefetch.pdf"
        writer = canvas.Canvas(str(source))
        writer.drawString(72, 720, "idle cache text")
        writer.save()
        path_key = self.window._path_key(source)
        self.window._last_user_interaction_at = time.monotonic()
        with (
            patch.object(
                self.window,
                "_list_visible_pdf_files_in_tree",
                return_value=[source],
            ),
            patch.object(self.window._prefetch_pool, "start") as too_early,
        ):
            self.window._start_scope_prefetch()
        too_early.assert_not_called()
        self.window._scope_prefetch_timer.stop()

        self.window._last_user_interaction_at = (
            time.monotonic() - self.window.PREFETCH_IDLE_SECONDS - 1.0
        )
        self.window._prefetch_paused_until = 0.0
        self.window._next_scope_prefetch_at = 0.0

        with (
            patch.object(
                self.window,
                "_list_visible_pdf_files_in_tree",
                return_value=[source],
            ),
            patch.object(
                self.window,
                "_is_pdf_text_cached_for_path",
                side_effect=AssertionError("GUI-thread cache probe"),
            ) as cache_probe,
            patch.object(self.window._prefetch_pool, "start") as start_worker,
        ):
            self.window._start_scope_prefetch()
            cache_probe.assert_not_called()

        start_worker.assert_called_once()
        worker = start_worker.call_args.args[0]
        self.window._last_user_interaction_at = time.monotonic()
        self.assertTrue(worker.should_abort())
        self.window._last_user_interaction_at = (
            time.monotonic() - self.window.PREFETCH_IDLE_SECONDS - 1.0
        )
        worker.run()
        QApplication.processEvents()
        self.window._sync_cached_tree_badges()

        self.assertIn(path_key, self.window._cached_pdf_path_keys)
        self.assertIn(path_key, self.window.model._cached_text_paths)
        stat = source.stat()
        self.assertTrue(
            self.window._pdf_text_disk_cache_file_path(
                path_key,
                stat.st_mtime_ns,
                stat.st_size,
            ).is_file()
        )

    def test_stale_search_worker_completion_still_publishes_cache_badges(
        self,
    ) -> None:
        source = self.root / "stale-search-cache.pdf"
        source.write_bytes(b"pdf")
        path_key = self.window._path_key(source)
        self.window._record_cached_pdf_path_key(path_key)
        self.window._cached_badge_sync_timer.stop()
        self.window._search_request_id = 9

        self.window._on_search_finished(8, [], {}, [], "")

        self.assertTrue(self.window._cached_badge_sync_timer.isActive())

    def test_immediate_badge_probe_restores_persisted_entries_without_idle_wait(
        self,
    ) -> None:
        cached_sources = [self.root / "cached-one.pdf", self.root / "cached-two.pdf"]
        uncached_source = self.root / "uncached.pdf"
        for index, source in enumerate([*cached_sources, uncached_source]):
            source.write_bytes(f"pdf-{index}".encode("utf-8"))

        cached_keys: list[str] = []
        for source in cached_sources:
            source_stat = source.stat()
            path_key = self.window._path_key(source)
            cached_keys.append(path_key)
            self.assertTrue(
                self.window._store_pdf_text_to_disk_cache(
                    path_key,
                    source_stat.st_mtime_ns,
                    source_stat.st_size,
                    f"cached text for {source.name}",
                )
            )

        # Simulate a fresh process: memory and badge state are empty while the
        # compressed cache files remain on disk.
        with self.window._pdf_text_cache_lock:
            self.window._pdf_text_cache.clear()
            self.window._pdf_text_cache_total_chars = 0
        with self.window._cached_pdf_path_keys_lock:
            self.window._cached_pdf_path_keys.clear()
            self.window._cached_pdf_path_keys_revision += 1
        self.window._cached_pdf_path_keys_synced_revision = -1
        self.window._cache_badge_probe_timer.stop()

        started_workers: list[PdfTextPrefetchWorker] = []

        def _run_immediately(worker: PdfTextPrefetchWorker, *_args) -> None:
            started_workers.append(worker)
            worker.run()

        with (
            patch.object(
                self.window,
                "_list_visible_pdf_files_in_tree",
                return_value=[*cached_sources, uncached_source],
            ),
            patch.object(
                self.window._cache_badge_probe_pool,
                "start",
                side_effect=_run_immediately,
            ),
            patch.object(self.window, "_prefetch_pdf_text") as text_prefetch,
        ):
            self.window._start_cached_badge_probe()
            QApplication.processEvents()

        self.assertEqual(len(started_workers), 1)
        self.assertEqual(
            started_workers[0].paths,
            [*cached_sources, uncached_source],
        )
        text_prefetch.assert_not_called()
        for path_key in cached_keys:
            self.assertIn(path_key, self.window._cached_pdf_path_keys)
        self.assertNotIn(
            self.window._path_key(uncached_source),
            self.window._cached_pdf_path_keys,
        )

    def test_directory_load_requests_immediate_persisted_cache_badge_probe(self) -> None:
        with (
            patch.object(self.window, "_request_cached_badge_probe") as request_probe,
            patch.object(self.window, "_request_scope_prefetch") as request_prefetch,
        ):
            self.window._on_model_directory_loaded(str(self.root))

        request_probe.assert_called_once_with()
        request_prefetch.assert_called_once_with()

    def test_idle_prefetch_restores_existing_disk_cache_badge_on_worker_thread(
        self,
    ) -> None:
        from reportlab.pdfgen import canvas

        source = self.root / "already-cached.pdf"
        writer = canvas.Canvas(str(source))
        writer.drawString(72, 720, "existing cache text")
        writer.save()
        stat = source.stat()
        path_key = self.window._path_key(source)
        self.window._store_pdf_text_to_disk_cache(
            path_key,
            stat.st_mtime_ns,
            stat.st_size,
            "existing cache text",
        )
        self.assertNotIn(path_key, self.window._cached_pdf_path_keys)

        self.window._last_user_interaction_at = (
            time.monotonic() - self.window.PREFETCH_IDLE_SECONDS - 1.0
        )
        self.window._prefetch_paused_until = 0.0
        self.window._next_scope_prefetch_at = 0.0
        original_probe = self.window._is_pdf_text_cached_for_path
        probe_ran_on_gui_thread: list[bool] = []

        def _probe(
            path: Path,
            *,
            mark_cached_badge: bool = False,
            should_abort=None,
        ) -> bool:
            probe_ran_on_gui_thread.append(
                QThread.currentThread() == QApplication.instance().thread()
            )
            return original_probe(
                path,
                mark_cached_badge=mark_cached_badge,
                should_abort=should_abort,
            )

        with (
            patch.object(
                self.window,
                "_list_visible_pdf_files_in_tree",
                return_value=[source],
            ),
            patch.object(
                self.window,
                "_is_pdf_text_cached_for_path",
                side_effect=_probe,
            ),
            patch.object(
                self.window,
                "_global_idle_elapsed_seconds",
                return_value=self.window.PREFETCH_IDLE_SECONDS + 1.0,
            ),
        ):
            self.window._start_scope_prefetch()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                QApplication.processEvents()
                if (
                    path_key in self.window.model._cached_text_paths
                    and not self.window._active_prefetch_workers
                ):
                    break
                time.sleep(0.005)

        self.assertTrue(probe_ran_on_gui_thread)
        self.assertFalse(any(probe_ran_on_gui_thread))
        self.assertIn(path_key, self.window._cached_pdf_path_keys)
        self.assertIn(path_key, self.window.model._cached_text_paths)

    def test_cancel_removes_queued_prefetch_worker_from_active_set(self) -> None:
        worker = PdfTextPrefetchWorker(17, [self.root / "queued.pdf"], lambda _path: "")
        self.window._active_prefetch_workers.add(worker)

        with (
            patch.object(self.window._prefetch_pool, "tryTake", return_value=True),
            patch.object(self.window._prefetch_pool, "clear"),
        ):
            self.window._cancel_scope_prefetch()

        self.assertNotIn(worker, self.window._active_prefetch_workers)

    def test_repeated_activity_does_not_recancel_stale_running_worker(self) -> None:
        request_id = self.window._prefetch_request_id
        worker = PdfTextPrefetchWorker(request_id, [], lambda _path: "")
        self.window._active_prefetch_workers.add(worker)

        with (
            patch.object(self.window._prefetch_pool, "tryTake", return_value=False),
            patch.object(self.window._prefetch_pool, "clear"),
            patch.object(
                self.window,
                "_cancel_scope_prefetch",
                wraps=self.window._cancel_scope_prefetch,
            ) as cancel,
            patch.object(self.window, "_request_scope_prefetch"),
        ):
            self.window._mark_user_interaction()
            self.window._mark_user_interaction()

        self.assertEqual(cancel.call_count, 1)
        self.window._active_prefetch_workers.discard(worker)

    def test_deleted_current_pdf_does_not_starve_visible_prefetch_rotation(self) -> None:
        deleted_current = self.root / "deleted-current.pdf"
        visible = self.root / "visible.pdf"
        visible.write_bytes(b"pdf")
        self.window.current_file = deleted_current
        self.window._last_user_interaction_at = (
            time.monotonic() - self.window.PREFETCH_IDLE_SECONDS - 1.0
        )
        self.window._prefetch_paused_until = 0.0
        self.window._next_scope_prefetch_at = 0.0

        with (
            patch.object(
                self.window,
                "_list_visible_pdf_files_in_tree",
                return_value=[visible],
            ),
            patch.object(self.window._prefetch_pool, "start") as start,
        ):
            self.window._start_scope_prefetch()
            first_worker = start.call_args.args[0]
            self.assertEqual(first_worker.paths, [deleted_current])
            self.window._on_scope_prefetch_finished(first_worker.request_id, 0, 0, "")
            self.window._scope_prefetch_timer.stop()
            self.window._next_scope_prefetch_at = 0.0

            self.window._start_scope_prefetch()
            second_worker = start.call_args.args[0]

        self.assertEqual(second_worker.paths, [visible])

    def test_cached_badge_does_not_suppress_worker_validation(self) -> None:
        source = self.root / "badge-needs-validation.pdf"
        source.write_bytes(b"pdf")
        self.window._record_cached_pdf_path_key(self.window._path_key(source))
        self.window._last_user_interaction_at = (
            time.monotonic() - self.window.PREFETCH_IDLE_SECONDS - 1.0
        )
        self.window._prefetch_paused_until = 0.0
        self.window._next_scope_prefetch_at = 0.0

        with (
            patch.object(
                self.window,
                "_list_visible_pdf_files_in_tree",
                return_value=[source],
            ),
            patch.object(self.window._prefetch_pool, "start") as start,
        ):
            self.window._start_scope_prefetch()

        start.assert_called_once()
        self.assertEqual(start.call_args.args[0].paths, [source])

    def test_prefetch_completion_gives_overdue_gc_first_refusal(self) -> None:
        request_id = self.window._prefetch_request_id
        worker = PdfTextPrefetchWorker(request_id, [], lambda _path: "")
        self.window._active_prefetch_workers.add(worker)

        with (
            patch.object(
                self.window,
                "_maybe_start_pdf_text_cache_gc",
                return_value=True,
            ) as start_gc,
            patch.object(self.window, "_request_scope_prefetch") as request_prefetch,
        ):
            self.window._on_scope_prefetch_finished(request_id, 0, 0, "")

        start_gc.assert_called_once_with()
        request_prefetch.assert_not_called()
        self.assertNotIn(worker, self.window._active_prefetch_workers)

    def test_prefetch_lease_busy_retries_without_completion_cooldown(self) -> None:
        request_id = self.window._prefetch_request_id
        worker = PdfTextPrefetchWorker(request_id, [], lambda _path: "")
        self.window._active_prefetch_workers.add(worker)
        self.window._next_scope_prefetch_at = time.monotonic() + 60.0

        with (
            patch.object(self.window._scope_prefetch_timer, "start") as start,
            patch.object(self.window, "_request_cached_badge_sync") as badge_sync,
            patch.object(self.window, "_maybe_start_pdf_text_cache_gc") as start_gc,
        ):
            self.window._on_scope_prefetch_finished(
                request_id,
                0,
                0,
                BACKGROUND_LEASE_BUSY,
            )

        self.assertNotIn(worker, self.window._active_prefetch_workers)
        self.assertEqual(self.window._next_scope_prefetch_at, 0.0)
        start.assert_called_once()
        badge_sync.assert_not_called()
        start_gc.assert_not_called()

    def test_gc_starts_only_after_idle_threshold(self) -> None:
        self.window._last_pdf_text_cache_gc_at = 0.0
        with patch.object(self.window._prefetch_pool, "start") as start:
            self.window._last_user_interaction_at = time.monotonic()
            self.assertFalse(self.window._maybe_start_pdf_text_cache_gc())
            start.assert_not_called()

            self.window._prefetch_paused_until = 0.0
            self.window._last_user_interaction_at = (
                time.monotonic() - self.window.PDF_TEXT_CACHE_GC_IDLE_SECONDS - 1.0
            )
            self.assertTrue(self.window._maybe_start_pdf_text_cache_gc())
            start.assert_called_once()
            worker, priority = start.call_args.args
            self.assertIsInstance(worker, PdfTextCacheGcWorker)
            self.assertEqual(priority, -2)

    def test_gc_lease_busy_preserves_cadence_and_cursors_then_retries(self) -> None:
        previous_cadence = 0.0
        self.window._last_pdf_text_cache_gc_at = previous_cadence
        self.window._pdf_text_cache_gc_memory_cursor = "memory-before"
        self.window._pdf_text_cache_gc_disk_cursor = "disk-before"
        self.window._last_user_interaction_at = (
            time.monotonic() - self.window.PDF_TEXT_CACHE_GC_IDLE_SECONDS - 1.0
        )
        self.window._prefetch_paused_until = 0.0

        with patch.object(self.window._prefetch_pool, "start") as start:
            self.assertTrue(self.window._maybe_start_pdf_text_cache_gc())

        worker = start.call_args.args[0]
        self.assertGreater(self.window._last_pdf_text_cache_gc_at, previous_cadence)

        with (
            patch("pdfexplore.app.QTimer.singleShot") as single_shot,
            patch.object(self.window, "_request_cached_badge_sync") as badge_sync,
            patch.object(self.window, "_request_scope_prefetch") as prefetch,
        ):
            self.window._on_pdf_text_cache_gc_finished(
                worker.request_id,
                {
                    "memory_cursor": "must-not-commit",
                    "disk_cursor": "must-not-commit",
                },
                BACKGROUND_LEASE_BUSY,
            )

        self.assertIsNone(self.window._active_pdf_text_cache_gc_worker)
        self.assertEqual(self.window._last_pdf_text_cache_gc_at, previous_cadence)
        self.assertEqual(
            self.window._pdf_text_cache_gc_memory_cursor,
            "memory-before",
        )
        self.assertEqual(self.window._pdf_text_cache_gc_disk_cursor, "disk-before")
        badge_sync.assert_not_called()
        prefetch.assert_not_called()
        single_shot.assert_called_once()
        retry_ms, callback = single_shot.call_args.args
        self.assertEqual(
            retry_ms,
            max(100, int(self.window.SCOPE_PREFETCH_TIMER_INTERVAL_MS)),
        )
        self.assertEqual(callback, self.window._maybe_start_pdf_text_cache_gc)

    def test_user_activity_cancels_gc_before_payload_application(self) -> None:
        source = self.root / "cancel-gc.pdf"
        source.write_bytes(b"pdf")
        stat = source.stat()
        path_key = self.window._path_key(source)
        self.window._store_cached_pdf_text(
            path_key,
            stat.st_mtime_ns,
            stat.st_size,
            "cached",
        )
        source.unlink()
        self.window._last_pdf_text_cache_gc_at = 0.0
        self.window._last_user_interaction_at = (
            time.monotonic() - self.window.PDF_TEXT_CACHE_GC_IDLE_SECONDS - 1.0
        )
        self.window._prefetch_paused_until = 0.0

        with (
            patch.object(self.window._prefetch_pool, "start") as start,
            patch.object(self.window, "_apply_pdf_text_cache_gc_payload") as apply,
        ):
            self.assertTrue(self.window._maybe_start_pdf_text_cache_gc())
            worker = start.call_args.args[0]
            self.window._mark_user_interaction()
            worker.run()
            QApplication.processEvents()

        apply.assert_not_called()
        self.assertIn(path_key, self.window._pdf_text_cache)
        self.window._scope_prefetch_timer.stop()

    def test_gc_payload_application_runs_on_worker_thread(self) -> None:
        source = self.root / "worker-gc.pdf"
        source.write_bytes(b"pdf")
        stat = source.stat()
        path_key = self.window._path_key(source)
        self.window._store_cached_pdf_text(
            path_key,
            stat.st_mtime_ns,
            stat.st_size,
            "cached",
        )
        self.window._store_pdf_text_to_disk_cache(
            path_key,
            stat.st_mtime_ns,
            stat.st_size,
            "cached",
        )
        self.window._record_cached_pdf_path_key(path_key)
        source.unlink()
        self.window._scope_prefetch_timer.stop()
        self.window._last_pdf_text_cache_gc_at = 0.0
        self.window._last_user_interaction_at = (
            time.monotonic() - self.window.PDF_TEXT_CACHE_GC_IDLE_SECONDS - 1.0
        )
        self.window._prefetch_paused_until = 0.0
        original_apply = self.window._apply_pdf_text_cache_gc_payload
        apply_ran_on_gui_thread: list[bool] = []

        def _apply(payload: object, should_abort=None) -> tuple[int, int]:
            apply_ran_on_gui_thread.append(
                QThread.currentThread() == QApplication.instance().thread()
            )
            return original_apply(payload, should_abort)

        with patch.object(
            self.window,
            "_apply_pdf_text_cache_gc_payload",
            side_effect=_apply,
        ):
            self.assertTrue(self.window._maybe_start_pdf_text_cache_gc())
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                QApplication.processEvents()
                if self.window._active_pdf_text_cache_gc_worker is None:
                    break
                time.sleep(0.005)

        self.assertTrue(apply_ran_on_gui_thread)
        self.assertFalse(any(apply_ran_on_gui_thread))
        self.assertNotIn(path_key, self.window._pdf_text_cache)
        self.assertNotIn(path_key, self.window._cached_pdf_path_keys)

    def test_gc_timer_does_not_rescan_visible_tree_on_gui_thread(self) -> None:
        self.window._prefetch_paused_until = 0.0
        self.window._last_pdf_text_cache_gc_at = 0.0
        self.window._last_user_interaction_at = (
            time.monotonic() - self.window.PDF_TEXT_CACHE_GC_IDLE_SECONDS - 1.0
        )
        with (
            patch.object(self.window._prefetch_pool, "start") as start,
            patch.object(
                self.window,
                "_list_visible_pdf_files_in_tree",
            ) as visible_tree_scan,
        ):
            self.window._on_pdf_text_cache_gc_timer()

        start.assert_called_once()
        visible_tree_scan.assert_not_called()

    def test_disk_cache_probe_does_not_read_metadata_on_gui_thread(self) -> None:
        source = self.root / "cached.pdf"
        source.write_bytes(b"pdf")
        stat = source.stat()
        path_key = self.window._path_key(source)
        cache_path = self.window._pdf_text_disk_cache_file_path(
            path_key,
            stat.st_mtime_ns,
            stat.st_size,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"cached")

        with patch.object(
            self.window,
            "_ensure_pdf_text_disk_cache_metadata_locked",
        ) as ensure_metadata:
            self.assertTrue(self.window._is_pdf_text_cached_for_path(source))

        ensure_metadata.assert_not_called()


if __name__ == "__main__":
    unittest.main()
