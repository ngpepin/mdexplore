from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QSizePolicy

from pdfexplore.app import PdfExploreWindow


def _create_pdf_with_text(path: Path, text: str) -> None:
    from reportlab.pdfgen import canvas

    writer = canvas.Canvas(str(path))
    writer.drawString(72, 720, text)
    writer.save()


class PdfExploreWindowLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="pdfexplore-layout-")
        root = Path(self._tempdir.name)
        self.window = PdfExploreWindow(
            root=root,
            app_icon=QIcon(),
            config_path=root / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        self.window.show()
        QApplication.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        QApplication.processEvents()
        self._tempdir.cleanup()

    def test_long_path_label_does_not_raise_window_minimum_width(self) -> None:
        self.assertEqual(
            self.window.path_label.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Ignored,
        )
        baseline_width = self.window.centralWidget().minimumSizeHint().width()
        long_text = ("very/deep/pdf/path/" * 24) + "target-file.pdf"
        self.window.path_label.setText(long_text)
        self.window.path_label.setToolTip(long_text)
        QApplication.processEvents()
        expanded_width = self.window.centralWidget().minimumSizeHint().width()
        self.assertLessEqual(expanded_width, baseline_width + 80)

    def test_pdf_tree_uses_pdf_name_filter(self) -> None:
        self.assertEqual(self.window.model.nameFilters(), ["*.pdf"])

    def test_reuses_cached_preview_widget_for_previously_opened_pdf(self) -> None:
        first_pdf = Path(self._tempdir.name) / "first.pdf"
        second_pdf = Path(self._tempdir.name) / "second.pdf"
        _create_pdf_with_text(first_pdf, "first document")
        _create_pdf_with_text(second_pdf, "second document")

        self.window._open_path_in_active_view(first_pdf)
        QApplication.processEvents()
        first_view = self.window._current_preview_widget()

        self.window._add_document_view()
        self.window._open_path_in_active_view(second_pdf)
        QApplication.processEvents()
        second_view = self.window._current_preview_widget()

        self.window._open_path_in_active_view(first_pdf)
        QApplication.processEvents()
        reused_first_view = self.window._current_preview_widget()

        self.assertIsNotNone(first_view)
        self.assertIsNotNone(second_view)
        self.assertIs(first_view, reused_first_view)
        self.assertIsNot(first_view, second_view)
        self.assertIs(
            self.window._preview_widgets_by_path[self.window._path_key(first_pdf)],
            first_view,
        )

    def test_persisted_multi_view_session_restores_for_same_pdf(self) -> None:
        pdf_path = Path(self._tempdir.name) / "session.pdf"
        _create_pdf_with_text(pdf_path, "session document")
        path_key = self.window._path_key(pdf_path)

        self.window._open_path_in_active_view(pdf_path)
        self.window._add_document_view()
        QApplication.processEvents()
        self.assertEqual(self.window.view_tabs.count(), 2)

        self.window._persist_document_view_session(path_key, capture_current=False)
        self.window.close()
        QApplication.processEvents()

        reopened = PdfExploreWindow(
            root=Path(self._tempdir.name),
            app_icon=QIcon(),
            config_path=Path(self._tempdir.name) / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        reopened.show()
        reopened._open_path_in_active_view(pdf_path)
        QApplication.processEvents()

        self.assertEqual(reopened.view_tabs.count(), 2)
        self.assertTrue(reopened.view_tabs.isVisible())
        reopened.close()
        QApplication.processEvents()

    def test_single_default_view_tab_is_hidden_until_multi_view(self) -> None:
        pdf_path = Path(self._tempdir.name) / "single.pdf"
        _create_pdf_with_text(pdf_path, "single document")

        self.window._open_path_in_active_view(pdf_path)
        QApplication.processEvents()
        self.assertEqual(self.window.view_tabs.count(), 1)
        self.assertFalse(self.window.view_tabs.isVisible())

        self.window._add_document_view()
        QApplication.processEvents()
        self.assertEqual(self.window.view_tabs.count(), 2)
        self.assertTrue(self.window.view_tabs.isVisible())

    def test_capture_tab_state_uses_visible_preview_not_current_index(self) -> None:
        pdf_path = Path(self._tempdir.name) / "shared.pdf"
        _create_pdf_with_text(pdf_path, "shared document")
        path_key = self.window._path_key(pdf_path)

        first_index = self.window.view_tabs.addTab("first")
        self.window._set_tab_data(first_index, self.window._new_tab_data(pdf_path))
        second_index = self.window.view_tabs.addTab("second")
        self.window._set_tab_data(second_index, self.window._new_tab_data(pdf_path))
        self.window.view_tabs.setCurrentIndex(second_index)

        preview = self.window._preview_widget_for_path(pdf_path)
        self.window.preview_stack.setCurrentWidget(preview)
        self.window._viewer_bridge_ready_by_path[path_key] = True

        expected_state = {
            "page": 3,
            "pagesCount": 10,
            "scale": "page-width",
            "scrollTop": 120,
            "scrollRatio": 0.3,
        }

        def fake_run_viewer_js(_source, callback=None):
            if callback is not None:
                callback(expected_state)

        self.window._run_viewer_js = fake_run_viewer_js  # type: ignore[method-assign]

        self.window._capture_tab_state(first_index)
        data = self.window._tab_data(first_index) or {}

        self.assertEqual(data.get("state"), expected_state)
        self.assertAlmostEqual(float(data.get("progress", 0.0)), 0.3)

    def test_same_document_tab_restore_uses_tab_specific_state(self) -> None:
        pdf_path = Path(self._tempdir.name) / "shared.pdf"
        _create_pdf_with_text(pdf_path, "shared document")
        path_key = self.window._path_key(pdf_path)

        preview = self.window._preview_widget_for_path(pdf_path)
        preview.setUrl(self.window._viewer_url_for_pdf(pdf_path))
        self.window._viewer_bridge_ready_by_path[path_key] = True
        self.window.preview_stack.setCurrentWidget(preview)

        first_state = {
            "page": 2,
            "pagesCount": 10,
            "scale": "page-width",
            "scrollTop": 80,
            "scrollRatio": 0.2,
        }
        second_state = {
            "page": 7,
            "pagesCount": 10,
            "scale": "page-width",
            "scrollTop": 280,
            "scrollRatio": 0.7,
        }

        first_index = self.window.view_tabs.addTab("first")
        first_data = self.window._new_tab_data(pdf_path)
        first_data["state"] = dict(first_state)
        self.window._set_tab_data(first_index, first_data)

        second_index = self.window.view_tabs.addTab("second")
        second_data = self.window._new_tab_data(pdf_path)
        second_data["state"] = dict(second_state)
        self.window._set_tab_data(second_index, second_data)

        captured_sources: list[str] = []

        def fake_run_viewer_js(source, callback=None):
            captured_sources.append(source)
            if callback is not None:
                callback(None)

        self.window._run_viewer_js = fake_run_viewer_js  # type: ignore[method-assign]

        self.window._load_tab_index(second_index)

        restore_calls = [source for source in captured_sources if "restoreViewState" in source]
        self.assertTrue(restore_calls)
        self.assertIn('"page": 7', restore_calls[-1])
        self.assertIn('"scrollRatio": 0.7', restore_calls[-1])

    def test_refresh_clears_preview_when_current_pdf_is_removed(self) -> None:
        pdf_path = Path(self._tempdir.name) / "deleted.pdf"
        _create_pdf_with_text(pdf_path, "to be removed")
        self.window._open_path_in_active_view(pdf_path)
        QApplication.processEvents()

        pdf_path.unlink()
        self.window._refresh_directory_view()
        QApplication.processEvents()

        self.assertIsNone(self.window.current_file)
        self.assertEqual(self.window.view_tabs.count(), 0)
        self.assertIs(self.window.preview_stack.currentWidget(), self.window._empty_preview)

    def test_file_change_watch_triggers_reload_for_changed_signature(self) -> None:
        pdf_path = Path(self._tempdir.name) / "watched.pdf"
        _create_pdf_with_text(pdf_path, "watched")
        self.window._open_path_in_active_view(pdf_path)
        QApplication.processEvents()

        path_key = self.window._path_key(pdf_path)
        self.window._preview_signatures_by_path[path_key] = (0, 0)
        reasons: list[str] = []

        def fake_reload(reason: str) -> None:
            reasons.append(reason)

        self.window._reload_current_preview = fake_reload  # type: ignore[method-assign]
        self.window._on_file_change_watch_tick()

        self.assertEqual(reasons, ["file changed on disk"])

    def test_save_document_view_session_captures_latest_state(self) -> None:
        pdf_path = Path(self._tempdir.name) / "session-capture.pdf"
        _create_pdf_with_text(pdf_path, "session capture")
        self.window._open_path_in_active_view(pdf_path)
        QApplication.processEvents()

        path_key = self.window._path_key(pdf_path)
        self.window._viewer_bridge_ready_by_path[path_key] = True

        latest_state = {
            "page": 6,
            "pagesCount": 10,
            "scale": "page-width",
            "scrollTop": 240,
            "scrollRatio": 0.6,
        }

        def fake_run_viewer_js(_source, callback=None):
            if callback is not None:
                callback(latest_state)

        self.window._run_viewer_js = fake_run_viewer_js  # type: ignore[method-assign]
        self.window._save_document_view_session(path_key, capture_current=True)

        session = self.window._document_view_sessions.get(path_key) or {}
        tabs = session.get("tabs") if isinstance(session, dict) else None
        self.assertIsInstance(tabs, list)
        self.assertTrue(tabs)
        first_tab = tabs[0]
        self.assertEqual(first_tab.get("state", {}).get("page"), 6)


if __name__ == "__main__":
    unittest.main()
