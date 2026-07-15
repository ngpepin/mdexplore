from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PySide6.QtCore import QThread
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from pdfexplore.app import PdfExploreWindow
from pdfexplore.workers import PdfTextCacheGcWorker, PdfTextPrefetchWorker


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
            patch("pdfexplore.workers.Path.glob") as disk_glob,
        ):
            worker.run()

        disk_glob.assert_not_called()


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
        first_page.extract_text.return_value = "partial first page"
        second_page = Mock()
        second_page.extract_text.return_value = "second page"
        reader = Mock(pages=[first_page, second_page])
        abort_checks = 0

        def should_abort() -> bool:
            nonlocal abort_checks
            abort_checks += 1
            # Entry check, page-one check, then cancel before page two.
            return abort_checks >= 3

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
        self.assertEqual(abort_checks, 3)
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

        def _probe(path: Path, *, mark_cached_badge: bool = False) -> bool:
            probe_ran_on_gui_thread.append(
                QThread.currentThread() == QApplication.instance().thread()
            )
            return original_probe(path, mark_cached_badge=mark_cached_badge)

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
