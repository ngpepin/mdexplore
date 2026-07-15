from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from pdfexplore.app import PdfExploreWindow


def _create_pdf_with_text(path: Path, text: str) -> None:
    from reportlab.pdfgen import canvas

    writer = canvas.Canvas(str(path))
    writer.drawString(72, 720, text)
    writer.save()


class PdfExploreSearchScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="pdfexplore-search-scope-")
        self.root = Path(self._tempdir.name)
        self.window = PdfExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.root / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        self.window.show()
        QApplication.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        QApplication.processEvents()
        self._tempdir.cleanup()

    def _wait_for(self, predicate, timeout_seconds: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            QApplication.processEvents()
            if predicate():
                return
            time.sleep(0.01)
        self.fail("Timed out waiting for condition")

    def test_visible_tree_pdf_listing_respects_expanded_branches(self) -> None:
        root_pdf = self.root / "root.pdf"
        _create_pdf_with_text(root_pdf, "root text")
        visible_dir = self.root / "visible"
        visible_dir.mkdir()
        visible_pdf = visible_dir / "visible.pdf"
        _create_pdf_with_text(visible_pdf, "visible text")
        hidden_dir = self.root / "hidden"
        hidden_dir.mkdir()
        hidden_pdf = hidden_dir / "hidden.pdf"
        _create_pdf_with_text(hidden_pdf, "hidden text")

        self.window._refresh_directory_view()
        QApplication.processEvents()

        visible_index = self.window.model.index(str(visible_dir))
        self.assertTrue(visible_index.isValid())
        self.window.tree.expand(visible_index)
        deadline = time.monotonic() + 3.0
        while (
            self.window.model.rowCount(visible_index) <= 0
            and time.monotonic() < deadline
        ):
            QApplication.processEvents()
            time.sleep(0.01)

        files = self.window._list_visible_pdf_files_in_tree()
        file_keys = {self.window._path_key(path) for path in files}

        self.assertIn(self.window._path_key(root_pdf), file_keys)
        self.assertIn(self.window._path_key(visible_pdf), file_keys)
        self.assertNotIn(self.window._path_key(hidden_pdf), file_keys)

    def test_search_scans_visible_tree_pdfs_not_collapsed_branches(self) -> None:
        visible_dir = self.root / "visible"
        visible_dir.mkdir()
        visible_pdf = visible_dir / "visible.pdf"
        _create_pdf_with_text(visible_pdf, "needle visible")

        hidden_dir = self.root / "hidden"
        hidden_dir.mkdir()
        hidden_pdf = hidden_dir / "hidden.pdf"
        _create_pdf_with_text(hidden_pdf, "needle hidden")

        self.window._refresh_directory_view()
        QApplication.processEvents()

        visible_index = self.window.model.index(str(visible_dir))
        self.window.tree.expand(visible_index)
        QApplication.processEvents()

        self.window.match_input.setText("needle")
        self.window._run_match_search_now()
        self._wait_for(lambda: len(self.window.current_match_files) == 1)

        match_keys = {
            self.window._path_key(path) for path in self.window.current_match_files
        }
        self.assertIn(self.window._path_key(visible_pdf), match_keys)
        self.assertNotIn(self.window._path_key(hidden_pdf), match_keys)

    def test_search_startup_does_not_probe_each_candidate_cache_on_gui_thread(
        self,
    ) -> None:
        candidates = [self.root / f"candidate-{index}.pdf" for index in range(3)]
        self.window.match_input.setText("needle")

        with (
            patch.object(
                self.window,
                "_list_visible_pdf_files_in_tree",
                return_value=candidates,
            ),
            patch.object(
                self.window,
                "_is_pdf_text_cached_for_path",
                side_effect=AssertionError("synchronous cache probe"),
            ) as cache_probe,
            patch.object(self.window.thread_pool, "start") as start_worker,
        ):
            self.window._run_match_search_now()

        cache_probe.assert_not_called()
        start_worker.assert_called_once()
        self.window._active_search_workers.clear()

    def test_search_scan_publishes_partial_hits_before_all_workers_finish(self) -> None:
        first_pdf = self.root / "first.pdf"
        _create_pdf_with_text(first_pdf, "needle first")
        second_pdf = self.root / "second.pdf"
        _create_pdf_with_text(second_pdf, "needle second")

        self.window._refresh_directory_view()
        QApplication.processEvents()

        first_key = self.window._path_key(first_pdf)
        second_key = self.window._path_key(second_pdf)
        request_id = self.window._search_request_id

        self.window._search_scan_total_candidates = 2
        self.window._search_scan_scope = self.root
        self.window._search_scan_candidate_order = {first_key: 0, second_key: 1}
        self.window._search_scan_expected_workers = 2
        self.window._search_scan_completed_workers = 0
        self.window._search_scan_match_counts = {}
        self.window._search_scan_filename_match_paths = set()

        dummy_worker = type("_DummySearchWorker", (), {"request_id": request_id})()
        self.window._active_search_workers.add(dummy_worker)

        self.window._on_search_finished(
            request_id,
            [first_key],
            {first_key: 3},
            [first_key],
            "",
        )

        self.assertEqual(self.window._search_scan_completed_workers, 1)
        self.assertEqual(self.window._search_scan_expected_workers, 2)
        self.assertEqual(
            {self.window._path_key(path) for path in self.window.current_match_files},
            {first_key},
        )
        self.assertEqual(self.window.model._search_match_counts.get(first_key), 3)
        self.assertNotIn(second_key, self.window.model._search_match_counts)

    def test_window_title_updates_effective_scope_model_state(self) -> None:
        child = self.root / "child"
        child.mkdir()
        self.window._refresh_directory_view()
        QApplication.processEvents()

        child_index = self.window.model.index(str(child))
        self.window.tree.setCurrentIndex(child_index)
        QApplication.processEvents()
        self.window._update_window_title()

        self.assertEqual(
            self.window.model._effective_scope_root_key,
            self.window._path_key(child),
        )

    def test_tree_visibility_changes_rerun_active_search(self) -> None:
        child = self.root / "child"
        child.mkdir()
        self.window._refresh_directory_view()
        QApplication.processEvents()

        child_index = self.window.model.index(str(child))
        self.window.tree.setCurrentIndex(child_index)
        self.window.match_input.setText("needle")
        QApplication.processEvents()

        calls: list[str] = []

        def fake_run_match_search() -> None:
            calls.append("run")

        self.window._run_match_search = fake_run_match_search  # type: ignore[method-assign]

        self.window._on_tree_directory_expanded(child_index)
        self._wait_for(lambda: len(calls) >= 1)

        self.window._on_tree_directory_collapsed(child_index)
        self._wait_for(lambda: len(calls) >= 2)

        self.assertGreaterEqual(len(calls), 2)

    def test_rerun_active_search_cancels_inflight_scan_immediately(self) -> None:
        self.window.match_input.setText("needle")
        QApplication.processEvents()

        calls: list[str] = []

        def fake_cancel_pending_search_scan() -> None:
            calls.append("cancel")

        self.window._cancel_pending_search_scan = fake_cancel_pending_search_scan  # type: ignore[method-assign]
        self.window._rerun_active_search_for_scope()

        self.assertIn("cancel", calls)

    def test_tree_marker_rebuild_avoids_sync_root_walk(self) -> None:
        calls: list[str] = []

        def fake_start_tree_marker_scan() -> None:
            calls.append("start")

        self.window._start_tree_marker_scan = fake_start_tree_marker_scan  # type: ignore[method-assign]
        self.window._tree_marker_cache_root_key = None

        with patch("pdfexplore.app.os.walk", side_effect=AssertionError("sync walk")):
            self.window._rebuild_tree_marker_cache()

        self.assertEqual(calls, ["start"])


if __name__ == "__main__":
    unittest.main()
