from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from pdfexplore.app import PdfExploreWindow
from pdfexplore.workers import PdfTextCacheGcWorker


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

    def test_gc_starts_only_after_idle_threshold(self) -> None:
        self.window._last_pdf_text_cache_gc_at = 0.0
        with patch.object(self.window._prefetch_pool, "start") as start:
            self.window._last_user_interaction_at = time.monotonic()
            self.assertFalse(self.window._maybe_start_pdf_text_cache_gc())
            start.assert_not_called()

            self.window._prefetch_paused_until = 0.0
            self.window._last_user_interaction_at = (
                time.monotonic() - self.window.PREFETCH_IDLE_SECONDS - 1.0
            )
            self.assertTrue(self.window._maybe_start_pdf_text_cache_gc())
            start.assert_called_once()
            worker, priority = start.call_args.args
            self.assertIsInstance(worker, PdfTextCacheGcWorker)
            self.assertEqual(priority, -2)


if __name__ == "__main__":
    unittest.main()
