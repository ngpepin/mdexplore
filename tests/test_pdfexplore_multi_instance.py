from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import multiprocessing
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from mdexplore_app.file_coordination import load_files_payload, update_files_sidecar
import pdfexplore.app as pdfexplore_app_module
from pdfexplore.app import (
    HIGHLIGHTING_FILE_NAME,
    VIEWS_FILE_NAME,
    PdfExploreWindow,
)
from pdfexplore.workers import PdfTextPrefetchWorker


def _persisted_session(page: int) -> dict:
    return {
        "active_view_id": 1,
        "next_view_id": 3,
        "next_view_sequence": 3,
        "next_tab_color_index": 2,
        "tabs": [
            {"view_id": 1, "sequence": 1, "color_slot": 0, "state": {"page": page}},
            {"view_id": 2, "sequence": 2, "color_slot": 1, "state": {"page": page + 1}},
        ],
    }


def _highlight(identifier: str, start: int) -> dict:
    return {
        "id": identifier,
        "page": 1,
        "start": start,
        "end": start + 4,
        "kind": "normal",
        "text": "text",
    }


class _SpawnedPdfTextCacheHarness:
    """Run the production cache methods without constructing a Qt window."""

    PDF_TEXT_CACHE_MAX_ENTRIES = PdfExploreWindow.PDF_TEXT_CACHE_MAX_ENTRIES
    PDF_TEXT_CACHE_MAX_CHARS = PdfExploreWindow.PDF_TEXT_CACHE_MAX_CHARS
    PDF_TEXT_DISK_CACHE_TRIM_INTERVAL = (
        PdfExploreWindow.PDF_TEXT_DISK_CACHE_TRIM_INTERVAL
    )
    PDF_TEXT_DISK_CACHE_TOUCH_INTERVAL_SECONDS = (
        PdfExploreWindow.PDF_TEXT_DISK_CACHE_TOUCH_INTERVAL_SECONDS
    )

    _path_key = staticmethod(PdfExploreWindow._path_key)
    _lookup_cached_pdf_text = PdfExploreWindow._lookup_cached_pdf_text
    _store_cached_pdf_text = PdfExploreWindow._store_cached_pdf_text
    _record_cached_pdf_path_key = PdfExploreWindow._record_cached_pdf_path_key
    _pdf_text_disk_cache_file_name = staticmethod(
        PdfExploreWindow._pdf_text_disk_cache_file_name
    )
    _pdf_text_disk_cache_file_path = (
        PdfExploreWindow._pdf_text_disk_cache_file_path
    )
    _pdf_text_disk_cache_process_lock_path = (
        PdfExploreWindow._pdf_text_disk_cache_process_lock_path
    )
    _pdf_text_disk_cache_trim_dirty_path = (
        PdfExploreWindow._pdf_text_disk_cache_trim_dirty_path
    )
    _pdf_text_disk_cache_metadata_path = staticmethod(
        PdfExploreWindow._pdf_text_disk_cache_metadata_path
    )
    _ensure_pdf_text_disk_cache_metadata_locked = (
        PdfExploreWindow._ensure_pdf_text_disk_cache_metadata_locked
    )
    _load_pdf_text_from_disk_cache = PdfExploreWindow._load_pdf_text_from_disk_cache
    _store_pdf_text_to_disk_cache = PdfExploreWindow._store_pdf_text_to_disk_cache
    _read_pdf_text = PdfExploreWindow._read_pdf_text

    def __init__(self, cache_dir: Path) -> None:
        self._pdf_text_disk_cache_dir = Path(cache_dir)
        self._pdf_text_cache_lock = threading.Lock()
        self._pdf_text_cache: OrderedDict[str, tuple[int, int, str]] = OrderedDict()
        self._pdf_text_cache_total_chars = 0
        self._cached_pdf_path_keys_lock = threading.Lock()
        self._cached_pdf_path_keys: set[str] = set()
        self._cached_pdf_path_keys_revision = 0
        self._pdf_text_disk_cache_lock = threading.Lock()
        self._pdf_text_disk_cache_store_count = 0
        self._pdf_text_disk_cache_trim_requested_revision = 1


def _extract_shared_pdf_text_process(
    source_path_text: str,
    cache_dir_text: str,
    ready_queue,
    start_event,
    producer_attempt_queue,
    extraction_entered_event,
    extraction_release_event,
    extraction_count,
    result_queue,
) -> None:
    """Read one miss while exposing producer-lock contention to the parent."""
    process_id = os.getpid()
    cache = _SpawnedPdfTextCacheHarness(Path(cache_dir_text))
    original_advisory_file_lock = pdfexplore_app_module.advisory_file_lock

    @contextmanager
    def _recording_advisory_file_lock(path, *args, **kwargs):
        if Path(path).name.startswith(".pdfexplore-producer-"):
            producer_attempt_queue.put(process_id)
        with original_advisory_file_lock(path, *args, **kwargs) as acquired:
            yield acquired

    class _BlockingPage:
        def extract_text(self) -> str:
            with extraction_count.get_lock():
                extraction_count.value += 1
            extraction_entered_event.set()
            if not extraction_release_event.wait(10.0):
                raise RuntimeError("parent did not release extraction")
            return "shared spawned-process text"

    class _Reader:
        def __init__(self, _path: str) -> None:
            self.pages = [_BlockingPage()]

    ready_queue.put(process_id)
    if not start_event.wait(10.0):
        result_queue.put(("error", process_id, "start timeout"))
        return
    try:
        with (
            patch.object(
                pdfexplore_app_module,
                "advisory_file_lock",
                _recording_advisory_file_lock,
            ),
            patch.object(pdfexplore_app_module, "PdfReader", _Reader),
        ):
            text = cache._read_pdf_text(Path(source_path_text))
    except Exception as exc:
        result_queue.put(("error", process_id, repr(exc)))
        return
    result_queue.put(("ok", process_id, text))


class PdfExploreMultiInstanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="pdfexplore-multi-")
        self.root = Path(self._tempdir.name)
        self.window = PdfExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.root / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        self.window._pdf_text_disk_cache_dir = self.root / "cache"

    def tearDown(self) -> None:
        self.window.close()
        QApplication.processEvents()
        self._tempdir.cleanup()

    def _wait_for_heartbeat_idle(self, timeout: float = 2.0) -> bool:
        """Process queued completion signals until the heartbeat writer is idle."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            QApplication.processEvents()
            if (
                self.window._active_global_activity_touch_worker is None
                and not self.window._global_activity_touch_pending
            ):
                return True
            time.sleep(0.01)
        QApplication.processEvents()
        return self.window._active_global_activity_touch_worker is None

    def _wait_for_activity_probe_idle(self, timeout: float = 2.0) -> bool:
        """Process queued completion signals until the heartbeat reader is idle."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            QApplication.processEvents()
            if self.window._active_global_activity_probe_worker is None:
                return True
            time.sleep(0.01)
        QApplication.processEvents()
        return self.window._active_global_activity_probe_worker is None

    def test_view_sidecar_is_written_beside_nested_pdf(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        pdf_path = nested / "document.pdf"
        pdf_path.write_bytes(b"pdf")
        path_key = self.window._path_key(pdf_path)
        self.window._document_view_sessions[path_key] = _persisted_session(3)

        with (
            patch.object(self.window, "_save_document_view_session"),
            patch.object(self.window, "_rebuild_tree_marker_cache"),
        ):
            self.window._persist_document_view_session(
                path_key,
                capture_current=False,
            )

        nested_sidecar = nested / VIEWS_FILE_NAME
        self.assertTrue(nested_sidecar.is_file())
        self.assertFalse((self.root / VIEWS_FILE_NAME).exists())
        self.assertEqual(
            load_files_payload(nested_sidecar),
            {pdf_path.name: _persisted_session(3)},
        )

    def test_stale_view_cache_merge_preserves_external_pdf(self) -> None:
        local = self.window._directory_view_states(self.root)
        external_session = _persisted_session(8)
        update_files_sidecar(
            self.root / VIEWS_FILE_NAME,
            {"external.pdf": external_session},
        )
        local["local.pdf"] = _persisted_session(2)

        self.window._save_directory_view_states(
            self.root,
            changed_file_names={"local.pdf"},
        )

        committed = load_files_payload(self.root / VIEWS_FILE_NAME)
        self.assertEqual(set(committed), {"external.pdf", "local.pdf"})

    def test_refresh_discards_nonactive_live_view_session_cache(self) -> None:
        source = self.root / "external.pdf"
        source.write_bytes(b"pdf")
        path_key = self.window._path_key(source)
        self.window._document_view_sessions[path_key] = _persisted_session(2)
        update_files_sidecar(
            self.root / VIEWS_FILE_NAME,
            {"external.pdf": _persisted_session(8)},
        )

        self.window._refresh_directory_view()

        refreshed = self.window._view_session_for_path_key(path_key)
        self.assertIsNotNone(refreshed)
        self.assertEqual(refreshed["tabs"][0]["state"]["page"], 8)

    def test_refresh_observes_external_sidecar_removals(self) -> None:
        current = self.root / "current.pdf"
        other = self.root / "other.pdf"
        current.write_bytes(b"pdf")
        other.write_bytes(b"pdf")
        current_key = self.window._path_key(current)
        other_key = self.window._path_key(other)
        color_path = self.root / self.window.model.COLOR_FILE_NAME

        update_files_sidecar(color_path, {"current.pdf": "#abcdef"})
        update_files_sidecar(
            self.root / HIGHLIGHTING_FILE_NAME,
            {"current.pdf": [_highlight("remove-me", 10)]},
        )
        update_files_sidecar(
            self.root / VIEWS_FILE_NAME,
            {"other.pdf": _persisted_session(4)},
        )
        self.assertEqual(self.window.model.color_for_file(current), "#abcdef")
        self.window._current_text_highlights = (
            self.window._load_text_highlights_for_path_key(current_key)
        )
        self.window._document_view_sessions[other_key] = _persisted_session(4)

        update_files_sidecar(color_path, {"current.pdf": None})
        update_files_sidecar(
            self.root / HIGHLIGHTING_FILE_NAME,
            {"current.pdf": None},
        )
        update_files_sidecar(self.root / VIEWS_FILE_NAME, {"other.pdf": None})
        self.window.current_file = current
        self.window._apply_persistent_text_highlights = lambda: None

        self.window._refresh_directory_view()

        self.assertIsNone(self.window.model.color_for_file(current))
        self.assertEqual(self.window._current_text_highlights, [])
        self.assertIsNone(self.window._view_session_for_path_key(other_key))

    def test_stale_highlight_cache_merge_preserves_external_pdf(self) -> None:
        local = self.window._directory_text_highlights(self.root)
        update_files_sidecar(
            self.root / HIGHLIGHTING_FILE_NAME,
            {"external.pdf": [_highlight("external-id", 10)]},
        )
        local["local.pdf"] = [_highlight("local-id", 20)]

        self.window._save_directory_text_highlights(
            self.root,
            changed_file_names={"local.pdf"},
        )

        committed = load_files_payload(self.root / HIGHLIGHTING_FILE_NAME)
        self.assertEqual(set(committed), {"external.pdf", "local.pdf"})

    def test_stale_color_cache_merge_preserves_external_pdf(self) -> None:
        local_pdf = self.root / "local.pdf"
        local_pdf.write_bytes(b"pdf")
        self.window.model._load_directory_colors(self.root)
        color_path = self.root / self.window.model.COLOR_FILE_NAME
        update_files_sidecar(color_path, {"external.pdf": "#112233"})

        self.window.model.set_color_for_file(local_pdf, "#abcdef")

        self.assertEqual(
            load_files_payload(color_path),
            {"external.pdf": "#112233", "local.pdf": "#abcdef"},
        )

    def test_highlight_ids_are_unique_across_windows(self) -> None:
        other = PdfExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.root / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        try:
            identifiers = {
                self.window._new_text_highlight_id(),
                other._new_text_highlight_id(),
            }
            self.assertEqual(len(identifiers), 2)
            self.assertTrue(all(value.startswith("pdfhl-") for value in identifiers))
        finally:
            other.close()
            QApplication.processEvents()

    def test_stale_windows_merge_highlight_adds_for_the_same_pdf(self) -> None:
        source = self.root / "shared.pdf"
        source.write_bytes(b"pdf")
        path_key = self.window._path_key(source)
        other = PdfExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.root / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        self.window._apply_persistent_text_highlights = lambda: None
        other._apply_persistent_text_highlights = lambda: None
        try:
            self.assertEqual(
                self.window._load_text_highlights_for_path_key(path_key),
                [],
            )
            self.assertEqual(other._load_text_highlights_for_path_key(path_key), [])

            self.window._replace_persistent_preview_highlight_range(
                path_key, 1, 10, 14, "normal", "one"
            )
            other._replace_persistent_preview_highlight_range(
                path_key, 1, 30, 34, "important", "two"
            )

            committed = load_files_payload(
                self.root / HIGHLIGHTING_FILE_NAME
            )["shared.pdf"]
            self.assertEqual(
                [(entry["start"], entry["kind"]) for entry in committed],
                [(10, "normal"), (30, "important")],
            )
        finally:
            other.close()
            QApplication.processEvents()

    def test_stale_remove_preserves_a_concurrent_same_pdf_add(self) -> None:
        source = self.root / "shared.pdf"
        source.write_bytes(b"pdf")
        path_key = self.window._path_key(source)
        other = PdfExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.root / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        self.window._apply_persistent_text_highlights = lambda: None
        other._apply_persistent_text_highlights = lambda: None
        try:
            self.window._replace_persistent_preview_highlight_range(
                path_key, 1, 10, 14, "normal", "old"
            )
            stale = other._load_text_highlights_for_path_key(path_key)
            old_id = stale[0]["id"]
            self.window._replace_persistent_preview_highlight_range(
                path_key, 1, 30, 34, "normal", "new"
            )

            other.current_file = source
            other._current_text_highlights = stale
            other._remove_persistent_preview_highlight(
                {"clickedHighlightId": old_id}
            )

            committed = load_files_payload(
                self.root / HIGHLIGHTING_FILE_NAME
            )["shared.pdf"]
            self.assertEqual([entry["start"] for entry in committed], [30])
        finally:
            other.close()
            QApplication.processEvents()

    def test_stale_add_does_not_resurrect_a_removed_highlight(self) -> None:
        source = self.root / "shared.pdf"
        source.write_bytes(b"pdf")
        path_key = self.window._path_key(source)
        other = PdfExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.root / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        self.window._apply_persistent_text_highlights = lambda: None
        other._apply_persistent_text_highlights = lambda: None
        try:
            self.window._replace_persistent_preview_highlight_range(
                path_key, 1, 10, 14, "normal", "keep"
            )
            self.window._replace_persistent_preview_highlight_range(
                path_key, 1, 30, 34, "normal", "remove"
            )
            stale = other._load_text_highlights_for_path_key(path_key)
            removed_id = next(
                entry["id"] for entry in stale if entry["start"] == 30
            )

            self.window.current_file = source
            self.window._remove_persistent_preview_highlight(
                {"clickedHighlightId": removed_id}
            )
            other._current_text_highlights = stale
            other._replace_persistent_preview_highlight_range(
                path_key, 1, 50, 54, "important", "new"
            )

            committed = load_files_payload(
                self.root / HIGHLIGHTING_FILE_NAME
            )["shared.pdf"]
            self.assertEqual([entry["start"] for entry in committed], [10, 50])
        finally:
            other.close()
            QApplication.processEvents()

    def test_activity_in_one_instance_suppresses_idle_work_in_another(self) -> None:
        source = self.root / "document.pdf"
        source.write_bytes(b"pdf")
        self.assertTrue(self._wait_for_heartbeat_idle())
        self.window._last_user_interaction_at = (
            time.monotonic() - self.window.PREFETCH_IDLE_SECONDS - 2.0
        )
        self.window._last_global_activity_touch_at = 0.0
        self.window._touch_global_activity_stamp(force=True)
        stamp_path = self.window._global_activity_stamp_path()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not stamp_path.is_file():
            QApplication.processEvents()
            time.sleep(0.01)
        self.assertTrue(stamp_path.is_file())

        with (
            patch.object(
                self.window,
                "_list_visible_pdf_files_in_tree",
                return_value=[source],
            ),
            patch.object(self.window._prefetch_pool, "start") as start,
        ):
            self.window._start_scope_prefetch()

        start.assert_not_called()
        self.window._scope_prefetch_timer.stop()

    def test_spawned_processes_single_flight_shared_text_extraction(self) -> None:
        source = self.root / "single-flight.pdf"
        source.write_bytes(b"pdf")
        cache_dir = self.root / "spawned-text-cache"
        context = multiprocessing.get_context("spawn")
        ready_queue = context.Queue()
        start_event = context.Event()
        producer_attempt_queue = context.Queue()
        extraction_entered_event = context.Event()
        extraction_release_event = context.Event()
        extraction_count = context.Value("i", 0)
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_extract_shared_pdf_text_process,
                args=(
                    str(source),
                    str(cache_dir),
                    ready_queue,
                    start_event,
                    producer_attempt_queue,
                    extraction_entered_event,
                    extraction_release_event,
                    extraction_count,
                    result_queue,
                ),
            )
            for _index in range(2)
        ]

        try:
            for process in processes:
                process.start()
            ready_processes = {ready_queue.get(timeout=10.0) for _ in processes}
            self.assertEqual(len(ready_processes), 2)
            start_event.set()
            self.assertTrue(extraction_entered_event.wait(10.0))
            producer_attempts = {
                producer_attempt_queue.get(timeout=10.0) for _ in processes
            }
            self.assertEqual(producer_attempts, ready_processes)
        finally:
            start_event.set()
            extraction_release_event.set()
            for process in processes:
                process.join(15.0)
                if process.is_alive():
                    process.terminate()
                    process.join(5.0)

        for process in processes:
            self.assertEqual(process.exitcode, 0)
        results = [result_queue.get(timeout=5.0) for _ in processes]
        self.assertTrue(
            all(status == "ok" for status, _process_id, _text in results),
            results,
        )
        self.assertEqual(
            {text for _status, _process_id, text in results},
            {"shared spawned-process text"},
        )
        self.assertEqual(extraction_count.value, 1)
        self.assertEqual(len(list(cache_dir.glob("*.txt.gz"))), 1)
        self.assertEqual(len(list(cache_dir.glob("*.txt.gz.meta.json"))), 1)
        self.assertFalse(any(cache_dir.glob(".*.tmp.*")))

    def test_read_pdf_text_falls_back_when_cache_lock_path_is_unavailable(
        self,
    ) -> None:
        source = self.root / "fallback-extraction.pdf"
        source.write_bytes(b"pdf")
        blocked_parent = self.root / "cache-parent-is-a-file"
        blocked_parent.write_bytes(b"not a directory")
        self.window._pdf_text_disk_cache_dir = blocked_parent / "text-cache"

        class _Page:
            @staticmethod
            def extract_text() -> str:
                return "fallback extracted text"

        reader = type("_Reader", (), {"pages": [_Page()]})()
        with patch("pdfexplore.app.PdfReader", return_value=reader) as pdf_reader:
            extracted = self.window._read_pdf_text(source)

        self.assertEqual(extracted, "fallback extracted text")
        pdf_reader.assert_called_once_with(str(source))
        source_stat = source.stat()
        path_key = self.window._path_key(source)
        self.assertEqual(
            self.window._lookup_cached_pdf_text(
                path_key,
                source_stat.st_mtime_ns,
                source_stat.st_size,
            ),
            "fallback extracted text",
        )
        self.assertIn(path_key, self.window._cached_pdf_path_keys)

    def test_signature_change_while_waiting_reacquires_path_stable_stripe(
        self,
    ) -> None:
        source = self.root / "signature-change.pdf"
        source.write_bytes(b"initial signature")
        path_key = self.window._path_key(source)
        initial_stat = source.stat()
        initial_signature = (initial_stat.st_mtime_ns, initial_stat.st_size)

        entered_lock_paths: list[Path] = []
        active_lock_paths: list[Path] = []
        extraction_lock_paths: list[Path | None] = []

        @contextmanager
        def _producer_lock(lock_path: Path, **_kwargs):
            normalized = Path(lock_path)
            entered_lock_paths.append(normalized)
            active_lock_paths.append(normalized)
            try:
                if len(entered_lock_paths) == 1:
                    # Model the source changing while this caller waited for the
                    # path-stable producer stripe. The stale attempt must leave
                    # this context and reacquire the same stripe before reading.
                    source.write_bytes(b"changed signature payload for retry")
                yield True
            finally:
                self.assertEqual(active_lock_paths.pop(), normalized)

        class _Page:
            @staticmethod
            def extract_text() -> str:
                extraction_lock_paths.append(
                    active_lock_paths[-1] if active_lock_paths else None
                )
                return "text from changed signature"

        reader = type("_Reader", (), {"pages": [_Page()]})()
        with (
            patch.object(
                self.window,
                "_load_pdf_text_from_disk_cache",
                return_value=None,
            ),
            patch.object(
                self.window,
                "_store_pdf_text_to_disk_cache",
                return_value=True,
            ),
            patch.object(
                pdfexplore_app_module,
                "advisory_file_lock",
                side_effect=_producer_lock,
            ),
            patch("pdfexplore.app.PdfReader", return_value=reader) as pdf_reader,
        ):
            extracted = self.window._read_pdf_text(source)

        self.assertEqual(extracted, "text from changed signature")
        changed_stat = source.stat()
        self.assertNotEqual(
            (changed_stat.st_mtime_ns, changed_stat.st_size),
            initial_signature,
        )
        self.assertEqual(len(entered_lock_paths), 2)
        self.assertEqual(entered_lock_paths[0], entered_lock_paths[1])
        self.assertEqual(extraction_lock_paths, [entered_lock_paths[1]])
        pdf_reader.assert_called_once_with(str(source))

    def test_symlink_retarget_restarts_cache_ownership_before_extraction(
        self,
    ) -> None:
        target_a = self.root / "target-a.pdf"
        target_b = self.root / "target-b.pdf"
        target_a.write_bytes(b"target a")
        target_b.write_bytes(b"target b replacement")
        source = self.root / "linked.pdf"
        source.symlink_to(target_a)
        target_a_key = self.window._path_key(source)
        target_b_key = str(target_b.resolve())
        entered_lock_paths: list[Path] = []

        @contextmanager
        def _producer_lock(lock_path: Path, **_kwargs):
            entered_lock_paths.append(Path(lock_path))
            if len(entered_lock_paths) == 1:
                source.unlink()
                source.symlink_to(target_b)
            yield True

        class _Page:
            @staticmethod
            def extract_text() -> str:
                return "text owned by target b"

        reader = type("_Reader", (), {"pages": [_Page()]})()
        with (
            patch.object(
                self.window,
                "_load_pdf_text_from_disk_cache",
                return_value=None,
            ),
            patch.object(
                self.window,
                "_store_pdf_text_to_disk_cache",
                return_value=True,
            ),
            patch.object(
                pdfexplore_app_module,
                "advisory_file_lock",
                side_effect=_producer_lock,
            ),
            patch("pdfexplore.app.PdfReader", return_value=reader) as pdf_reader,
        ):
            extracted = self.window._read_pdf_text(source)

        self.assertEqual(extracted, "text owned by target b")
        self.assertEqual(len(entered_lock_paths), 2)
        pdf_reader.assert_called_once_with(str(target_b.resolve()))
        self.assertNotIn(target_a_key, self.window._pdf_text_cache)
        self.assertIn(target_b_key, self.window._pdf_text_cache)

    def test_symlink_retarget_revalidates_outer_memory_cache_hit(self) -> None:
        target_a = self.root / "outer-memory-a.pdf"
        target_b = self.root / "outer-memory-b.pdf"
        target_a.write_bytes(b"target a")
        target_b.write_bytes(b"target b")
        source = self.root / "outer-memory-link.pdf"
        source.symlink_to(target_a)
        target_a_key = str(target_a.resolve())
        target_b_key = str(target_b.resolve())
        lookup_keys: list[str] = []

        def _lookup(path_key: str, _mtime_ns: int, _size: int) -> str:
            lookup_keys.append(path_key)
            if path_key == target_a_key:
                source.unlink()
                source.symlink_to(target_b)
                return "stale target a text"
            return "current target b text"

        with (
            patch.object(
                self.window,
                "_lookup_cached_pdf_text",
                side_effect=_lookup,
            ),
            patch.object(
                self.window,
                "_load_pdf_text_from_disk_cache",
            ) as disk_load,
            patch("pdfexplore.app.PdfReader") as pdf_reader,
        ):
            extracted = self.window._read_pdf_text(source)

        self.assertEqual(extracted, "current target b text")
        self.assertEqual(lookup_keys, [target_a_key, target_b_key])
        disk_load.assert_not_called()
        pdf_reader.assert_not_called()

    def test_symlink_retarget_revalidates_inner_memory_cache_hit(self) -> None:
        target_a = self.root / "inner-memory-a.pdf"
        target_b = self.root / "inner-memory-b.pdf"
        target_a.write_bytes(b"target a")
        target_b.write_bytes(b"target b")
        source = self.root / "inner-memory-link.pdf"
        source.symlink_to(target_a)
        target_a_key = str(target_a.resolve())
        target_b_key = str(target_b.resolve())
        lookup_keys: list[str] = []

        def _lookup(path_key: str, _mtime_ns: int, _size: int) -> str | None:
            lookup_keys.append(path_key)
            if len(lookup_keys) == 1:
                return None
            if path_key == target_a_key:
                source.unlink()
                source.symlink_to(target_b)
                return "stale target a text"
            return "current target b text"

        @contextmanager
        def _producer_lock(_lock_path: Path, **_kwargs):
            yield True

        with (
            patch.object(
                self.window,
                "_lookup_cached_pdf_text",
                side_effect=_lookup,
            ),
            patch.object(
                self.window,
                "_load_pdf_text_from_disk_cache",
                return_value=None,
            ) as disk_load,
            patch.object(
                pdfexplore_app_module,
                "advisory_file_lock",
                side_effect=_producer_lock,
            ),
            patch("pdfexplore.app.PdfReader") as pdf_reader,
        ):
            extracted = self.window._read_pdf_text(source)

        self.assertEqual(extracted, "current target b text")
        self.assertEqual(
            lookup_keys,
            [target_a_key, target_a_key, target_b_key],
        )
        disk_load.assert_called_once()
        pdf_reader.assert_not_called()

    def test_heartbeat_filesystem_write_runs_off_gui_thread(self) -> None:
        self.assertTrue(self._wait_for_heartbeat_idle())
        self.window._last_global_activity_touch_at = 0.0
        main_thread_id = threading.get_ident()
        observed_thread_ids: list[int] = []
        original_mkdir = Path.mkdir
        original_touch = Path.touch

        def _recording_mkdir(path: Path, *args, **kwargs):
            observed_thread_ids.append(threading.get_ident())
            return original_mkdir(path, *args, **kwargs)

        def _recording_touch(path: Path, *args, **kwargs):
            observed_thread_ids.append(threading.get_ident())
            return original_touch(path, *args, **kwargs)

        with (
            patch.object(Path, "mkdir", new=_recording_mkdir),
            patch.object(Path, "touch", new=_recording_touch),
        ):
            self.window._touch_global_activity_stamp(force=True)
            self.assertTrue(self._wait_for_heartbeat_idle())

        self.assertTrue(observed_thread_ids)
        self.assertTrue(
            all(thread_id != main_thread_id for thread_id in observed_thread_ids)
        )

    def test_heartbeat_filesystem_read_runs_off_gui_thread(self) -> None:
        self.assertTrue(self._wait_for_activity_probe_idle())
        self.window._global_activity_probe_timer.stop()
        stamp_path = self.window._global_activity_stamp_path()
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_path.touch()
        main_thread_id = threading.get_ident()
        observed_thread_ids: list[int] = []
        original_stat = Path.stat

        def _recording_stat(path: Path, *args, **kwargs):
            if path == stamp_path:
                observed_thread_ids.append(threading.get_ident())
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, "stat", new=_recording_stat):
            self.window._start_global_activity_probe_worker()
            self.assertTrue(self._wait_for_activity_probe_idle())

        self.assertTrue(observed_thread_ids)
        self.assertTrue(
            all(thread_id != main_thread_id for thread_id in observed_thread_ids)
        )
        self.window._last_user_interaction_at = (
            time.monotonic() - self.window.PREFETCH_IDLE_SECONDS - 1.0
        )
        with patch.object(
            Path,
            "stat",
            side_effect=AssertionError("GUI idle check touched filesystem"),
        ):
            self.assertLess(
                self.window._global_idle_elapsed_seconds(),
                self.window.PREFETCH_IDLE_SECONDS,
            )

    def test_user_interaction_cancels_locally_before_heartbeat_dispatch(self) -> None:
        self.assertTrue(self._wait_for_heartbeat_idle())
        self.window._last_global_activity_touch_at = 0.0
        request_id = self.window._prefetch_request_id
        prefetch_worker = PdfTextPrefetchWorker(
            request_id,
            [self.root / "queued.pdf"],
            lambda _path: "",
        )
        self.window._active_prefetch_workers.add(prefetch_worker)
        observed: dict[str, float | int] = {}

        def _capture_dispatch(_worker) -> None:
            observed["request_id"] = self.window._prefetch_request_id
            observed["idle_elapsed"] = (
                time.monotonic() - self.window._last_user_interaction_at
            )

        with (
            patch.object(self.window._prefetch_pool, "tryTake", return_value=False),
            patch.object(self.window._prefetch_pool, "clear"),
            patch.object(
                self.window._global_activity_touch_pool,
                "start",
                side_effect=_capture_dispatch,
            ) as start_heartbeat,
            patch.object(Path, "mkdir") as mkdir,
            patch.object(Path, "touch") as touch,
        ):
            self.window._mark_user_interaction()

        start_heartbeat.assert_called_once()
        mkdir.assert_not_called()
        touch.assert_not_called()
        self.assertGreater(int(observed["request_id"]), request_id)
        self.assertLess(float(observed["idle_elapsed"]), 0.5)

        self.window._active_prefetch_workers.discard(prefetch_worker)
        heartbeat_worker = self.window._active_global_activity_touch_worker
        self.assertIsNotNone(heartbeat_worker)
        self.window._on_global_activity_touch_finished(heartbeat_worker.request_id)
        self.window._scope_prefetch_timer.stop()

    def test_heartbeat_requests_coalesce_while_writer_is_active(self) -> None:
        self.assertTrue(self._wait_for_heartbeat_idle())
        self.window._last_global_activity_touch_at = 0.0
        dispatched: list[object] = []

        with patch.object(
            self.window._global_activity_touch_pool,
            "start",
            side_effect=lambda worker: dispatched.append(worker),
        ):
            self.window._touch_global_activity_stamp(force=True)
            self.window._touch_global_activity_stamp()
            self.window._touch_global_activity_stamp()
            self.window._touch_global_activity_stamp(force=True)
            self.assertEqual(len(dispatched), 1)
            self.assertTrue(self.window._global_activity_touch_pending)

            first = dispatched[0]
            self.window._on_global_activity_touch_finished(first.request_id)
            self.assertEqual(len(dispatched), 2)
            self.assertFalse(self.window._global_activity_touch_pending)

            second = dispatched[1]
            self.window._on_global_activity_touch_finished(second.request_id)

        self.assertIsNone(self.window._active_global_activity_touch_worker)

    def test_shutdown_drops_pending_heartbeat_without_rescheduling(self) -> None:
        self.assertTrue(self._wait_for_heartbeat_idle())
        self.window._last_global_activity_touch_at = 0.0
        dispatched: list[object] = []

        with patch.object(
            self.window._global_activity_touch_pool,
            "start",
            side_effect=lambda worker: dispatched.append(worker),
        ):
            self.window._touch_global_activity_stamp(force=True)
            self.window._touch_global_activity_stamp(force=True)
            self.assertEqual(len(dispatched), 1)
            self.assertTrue(self.window._global_activity_touch_pending)

            first = dispatched[0]
            self.window._closing = True
            self.window._stop_global_activity_heartbeat()
            self.assertFalse(self.window._global_activity_touch_pending)
            self.assertFalse(self.window._global_activity_touch_timer.isActive())

            self.window._on_global_activity_touch_finished(first.request_id)
            self.assertEqual(len(dispatched), 1)

        self.assertIsNone(self.window._active_global_activity_touch_worker)

    def test_preview_profile_is_off_record_for_process_isolation(self) -> None:
        source = self.root / "profile.pdf"
        source.write_bytes(b"pdf")
        preview = self.window._create_preview_widget(source)
        self.assertTrue(preview.page().profile().isOffTheRecord())
        preview.deleteLater()


if __name__ == "__main__":
    unittest.main()
