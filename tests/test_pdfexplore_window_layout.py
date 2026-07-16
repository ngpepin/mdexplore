from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QPoint, QEvent, QRect, Qt
from PySide6.QtGui import QClipboard, QIcon, QImage, QKeyEvent, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOptionViewItem,
)

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

    def test_default_splitter_uses_configured_tree_fraction(
        self,
    ) -> None:
        tree_width, viewer_width = self.window.splitter.sizes()
        total_width = tree_width + viewer_width
        self.assertGreater(total_width, 0)
        self.assertAlmostEqual(
            tree_width / total_width,
            self.window.SPLITTER_TREE_DEFAULT_FRACTION,
            delta=0.01,
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

    def test_expand_and_collapse_buttons_sit_between_refresh_and_dark_mode(self) -> None:
        top_bar_widget = self.window.centralWidget().layout().itemAt(0).widget()
        top_bar = top_bar_widget.layout()
        button_labels = [
            top_bar.itemAt(index).widget().text()
            for index in range(top_bar.count())
            if hasattr(top_bar.itemAt(index).widget(), "text")
        ]
        refresh_index = button_labels.index("Refresh")
        self.assertEqual(
            button_labels[refresh_index : refresh_index + 5],
            ["Refresh", "Expand", "Collapse", "Dark", "Add View"],
        )

    def test_top_left_buttons_are_compactly_sized_to_their_labels(self) -> None:
        top_bar_widget = self.window.centralWidget().layout().itemAt(0).widget()
        top_bar = top_bar_widget.layout()
        buttons = {
            top_bar.itemAt(index).widget().text(): top_bar.itemAt(index).widget()
            for index in range(top_bar.count())
            if isinstance(top_bar.itemAt(index).widget(), QPushButton)
        }

        expected_padding = {
            "Recent": 12,
            "^": 10,
            "Refresh": 12,
            "Expand": 12,
            "Collapse": 12,
            "Dark": 12,
            "Add View": 12,
            "Edit": 12,
        }
        for label, padding in expected_padding.items():
            button = buttons[label]
            sizing_label = "Light" if label == "Dark" else label
            text_width = button.fontMetrics().horizontalAdvance(sizing_label)
            menu_indicator_width = 0
            if button.menu() is not None:
                menu_indicator_width = max(
                    8,
                    int(
                        button.style().pixelMetric(
                            QStyle.PixelMetric.PM_MenuButtonIndicator
                        )
                    ),
                )
            expected_width = max(20, text_width + menu_indicator_width + padding)
            self.assertEqual(button.width(), expected_width)
            self.assertEqual(
                button.sizePolicy().horizontalPolicy(),
                QSizePolicy.Policy.Fixed,
            )

    def _wait_for_tree_index(self, path: Path, timeout_seconds: float = 2.0):
        deadline = time.monotonic() + timeout_seconds
        index = self.window.model.index(str(path))
        while not index.isValid() and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.01)
            index = self.window.model.index(str(path))
        self.assertTrue(index.isValid(), f"Tree index did not load for {path}")
        return index

    def test_directory_context_menu_make_root_uses_selected_folder(self) -> None:
        root = Path(self._tempdir.name)
        target = root / "nested-root"
        target.mkdir()

        self.window._refresh_directory_view()
        QApplication.processEvents()
        index = self._wait_for_tree_index(target)
        self.window.tree.scrollTo(index)
        QApplication.processEvents()
        pos = self.window.tree.visualRect(index).center()

        class _FakeAction:
            def __init__(self, text: str) -> None:
                self.text = text

        class _FakeMenu:
            def __init__(self, *_args, **_kwargs) -> None:
                self.actions: list[_FakeAction] = []

            def addAction(self, text: str):
                action = _FakeAction(text)
                self.actions.append(action)
                return action

            def addSeparator(self) -> None:
                return None

            def exec(self, *_args, **_kwargs):
                return next(
                    action for action in self.actions if action.text == "Make Root"
                )

        with patch("pdfexplore.app.QMenu", _FakeMenu), patch.object(
            self.window, "_set_root_directory"
        ) as set_root_mock:
            self.window._show_tree_context_menu(pos)

        set_root_mock.assert_called_once_with(target)

    def test_folder_search_hit_counts_are_independent_of_selected_scope(self) -> None:
        root = Path(self._tempdir.name)
        selected_folder = root / "selected"
        other_folder = root / "other" / "nested"
        selected_folder.mkdir()
        other_folder.mkdir(parents=True)
        selected_file = selected_folder / "selected.pdf"
        selected_nested_folder = selected_folder / "nested"
        selected_nested_folder.mkdir()
        selected_nested_file = selected_nested_folder / "nested.pdf"
        other_file = other_folder / "other.pdf"
        _create_pdf_with_text(selected_file, "selected")
        _create_pdf_with_text(selected_nested_file, "nested")
        _create_pdf_with_text(other_file, "other")

        self.window.model.set_search_match_counts(
            {selected_file: 4, selected_nested_file: 9, other_file: 5}
        )
        self.window.model.set_effective_scope_directory(selected_folder)

        self.assertEqual(
            self.window.model.search_hit_count_for_directory(selected_folder), 2
        )
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(selected_nested_folder), 1
        )
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(other_folder), 1
        )
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(other_folder.parent), 1
        )

        folder_index = self.window.model.index(str(selected_folder))
        self.assertTrue(folder_index.isValid())
        self.window.model.set_reduce_paint_cost(True)
        image = QImage(420, 42, QImage.Format.Format_ARGB32)
        image.fill(0xFFFFFFFF)
        painter = QPainter(image)
        option = QStyleOptionViewItem()
        option.rect = QRect(0, 0, 420, 40)
        option.widget = self.window.tree
        self.window.tree.itemDelegate().paint(painter, option, folder_index)
        painter.end()

        orange_pixel_x_positions: list[int] = []
        for y in range(image.height()):
            for x in range(image.width()):
                color = image.pixelColor(x, y)
                if (
                    abs(color.red() - 243) <= 12
                    and abs(color.green() - 197) <= 12
                    and abs(color.blue() - 107) <= 12
                    and color.alpha() > 200
                ):
                    orange_pixel_x_positions.append(x)
        self.assertTrue(
            orange_pixel_x_positions,
            "PDF directories must use the shared mdexplore folder-pill painter.",
        )
        self.assertLess(
            max(orange_pixel_x_positions),
            option.rect.center().x(),
            "PDF folder pills should sit beside short labels, as in mdexplore.",
        )

    def test_folder_search_count_changes_invalidate_loaded_directory_rows(self) -> None:
        root = Path(self._tempdir.name)
        folder = root / "reports"
        nested_folder = folder / "quarterly"
        nested_folder.mkdir(parents=True)
        pdf_path = nested_folder / "results.pdf"
        _create_pdf_with_text(pdf_path, "results")

        self.window._refresh_directory_view()
        QApplication.processEvents()
        self._wait_for_tree_index(folder)
        self._wait_for_tree_index(nested_folder)

        changed_paths: list[Path] = []

        def _record_changed_path(top_left, _bottom_right, _roles) -> None:
            changed_paths.append(Path(self.window.model.filePath(top_left)))

        self.window.model.dataChanged.connect(_record_changed_path)
        self.window.model.set_search_match_counts({pdf_path: 3})

        self.assertIn(folder, changed_paths)
        self.assertIn(nested_folder, changed_paths)

        changed_paths.clear()
        self.window.model.clear_search_match_paths()

        self.assertIn(folder, changed_paths)
        self.assertIn(nested_folder, changed_paths)

    def test_folder_search_counts_follow_visible_pdf_symlink_parents(self) -> None:
        root = Path(self._tempdir.name)
        visible_folder = root / "topic"
        target_folder = root / "publisher"
        visible_folder.mkdir()
        target_folder.mkdir()
        target = target_folder / "book.pdf"
        _create_pdf_with_text(target, "needle")
        visible_link = visible_folder / "linked-book.pdf"
        try:
            visible_link.symlink_to(target)
        except OSError:
            self.skipTest("symlink creation is not supported")

        self.window.model.set_search_match_counts(
            {target: 4},
            directory_match_paths=[visible_link],
        )

        self.assertEqual(
            self.window.model.search_hit_count_for_directory(visible_folder),
            1,
        )
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(target_folder),
            0,
        )

    def test_expand_opens_only_pdf_bearing_branches_and_their_ancestors(self) -> None:
        root = Path(self._tempdir.name)
        pdf_parent = root / "reports"
        pdf_directory = pdf_parent / "quarterly"
        pdf_directory.mkdir(parents=True)
        _create_pdf_with_text(pdf_directory / "results.pdf", "results")

        direct_pdf_directory = root / "invoices"
        direct_pdf_directory.mkdir()
        _create_pdf_with_text(direct_pdf_directory / "invoice.pdf", "invoice")

        empty_directory = root / "empty" / "nested"
        empty_directory.mkdir(parents=True)

        self.window._refresh_directory_view()
        QApplication.processEvents()
        self.window.expand_btn.click()
        QApplication.processEvents()

        reports_index = self._wait_for_tree_index(pdf_parent)
        quarterly_index = self._wait_for_tree_index(pdf_directory)
        invoices_index = self._wait_for_tree_index(direct_pdf_directory)
        empty_index = self._wait_for_tree_index(root / "empty")

        self.assertTrue(self.window.tree.isExpanded(reports_index))
        self.assertTrue(self.window.tree.isExpanded(quarterly_index))
        self.assertTrue(self.window.tree.isExpanded(invoices_index))
        self.assertFalse(self.window.tree.isExpanded(empty_index))

    def test_collapse_closes_all_directory_branches(self) -> None:
        root = Path(self._tempdir.name)
        pdf_parent = root / "reports"
        pdf_directory = pdf_parent / "quarterly"
        pdf_directory.mkdir(parents=True)
        _create_pdf_with_text(pdf_directory / "results.pdf", "results")

        self.window._refresh_directory_view()
        QApplication.processEvents()
        self.window.expand_btn.click()
        QApplication.processEvents()

        reports_index = self._wait_for_tree_index(pdf_parent)
        quarterly_index = self._wait_for_tree_index(pdf_directory)
        self.assertTrue(self.window.tree.isExpanded(reports_index))
        self.assertTrue(self.window.tree.isExpanded(quarterly_index))

        self.window.collapse_btn.click()
        QApplication.processEvents()

        self.assertFalse(self.window.tree.isExpanded(reports_index))
        self.assertFalse(self.window.tree.isExpanded(quarterly_index))

    def test_dark_mode_button_is_disabled_without_a_preview(self) -> None:
        self.assertFalse(self.window.dark_mode_btn.isEnabled())

        pdf_path = Path(self._tempdir.name) / "dark-mode-state.pdf"
        _create_pdf_with_text(pdf_path, "dark mode state")
        self.window._open_path_in_active_view(pdf_path)
        QApplication.processEvents()

        self.assertTrue(self.window.dark_mode_btn.isEnabled())

        pdf_path.unlink()
        self.window._clear_preview_for_missing_file()
        QApplication.processEvents()

        self.assertFalse(self.window.dark_mode_btn.isEnabled())

    def test_dark_mode_button_toggles_viewer_colors_and_label(self) -> None:
        pdf_path = Path(self._tempdir.name) / "dark-mode-toggle.pdf"
        _create_pdf_with_text(pdf_path, "dark mode toggle")
        self.window._open_path_in_active_view(pdf_path)
        QApplication.processEvents()

        captured_sources: list[str] = []
        self.window._run_viewer_js = (  # type: ignore[method-assign]
            lambda source, callback=None: captured_sources.append(source)
        )

        initial_width = self.window.dark_mode_btn.width()

        self.window.dark_mode_btn.click()
        self.assertTrue(self.window._preview_dark_mode)
        self.assertEqual(self.window.dark_mode_btn.text(), "Light")
        self.assertEqual(self.window.dark_mode_btn.width(), initial_width)
        self.assertIn("setDarkMode(true)", captured_sources[-1])

        self.window.dark_mode_btn.click()
        self.assertFalse(self.window._preview_dark_mode)
        self.assertEqual(self.window.dark_mode_btn.text(), "Dark")
        self.assertEqual(self.window.dark_mode_btn.width(), initial_width)
        self.assertIn("setDarkMode(false)", captured_sources[-1])

    def test_toolbar_input_resets_background_idle_clock(self) -> None:
        pdf_path = Path(self._tempdir.name) / "toolbar-input.pdf"
        _create_pdf_with_text(pdf_path, "toolbar input")
        self.window._open_path_in_active_view(pdf_path)
        QApplication.processEvents()

        previous = time.monotonic() - self.window.PREFETCH_IDLE_SECONDS - 1.0
        self.window._last_user_interaction_at = previous

        QTest.mouseClick(
            self.window.dark_mode_btn,
            Qt.MouseButton.LeftButton,
        )
        QApplication.processEvents()

        self.assertGreater(self.window._last_user_interaction_at, previous)
        self.assertLess(
            time.monotonic() - self.window._last_user_interaction_at,
            1.0,
        )

    def test_near_query_groups_are_forwarded_to_pdf_viewer(self) -> None:
        captured_sources: list[str] = []
        self.window._run_viewer_js = (  # type: ignore[method-assign]
            lambda source, callback=None: captured_sources.append(source)
        )
        self.window._current_preview_path_key = lambda: "active.pdf"  # type: ignore[method-assign]
        self.window._viewer_bridge_ready_by_path["active.pdf"] = True
        self.window.match_input.setText("NEAR(alpha, 'Beta', gamma)")

        groups = self.window._current_near_term_groups()
        self.window._highlight_viewer_search_terms(
            self.window._current_search_terms(),
            scroll_to_first=True,
        )

        self.assertEqual(
            groups,
            [[("alpha", False), ("Beta", True), ("gamma", False)]],
        )
        self.assertTrue(captured_sources)
        self.assertIn("setSearchTerms", captured_sources[-1])
        self.assertIn('"text": "alpha"', captured_sources[-1])
        self.assertIn('"text": "Beta", "caseSensitive": true', captured_sources[-1])
        self.assertIn('"text": "gamma"', captured_sources[-1])
        self.assertTrue(captured_sources[-1].rstrip().endswith("true);"))

    def test_newly_opened_search_match_requests_first_hit_navigation(self) -> None:
        pdf_path = Path(self._tempdir.name) / "matched.pdf"
        _create_pdf_with_text(pdf_path, "needle")
        path_key = self.window._path_key(pdf_path)
        self.window.current_file = pdf_path
        self.window.current_match_files = [pdf_path]
        self.window.match_input.setText("needle")
        self.window._current_preview_path_key = lambda: path_key  # type: ignore[method-assign]
        self.window._viewer_bridge_ready_by_path[path_key] = True
        self.window._pending_search_scroll_to_first_path_keys.add(path_key)
        calls: list[bool] = []
        self.window._highlight_viewer_search_terms = (  # type: ignore[method-assign]
            lambda _terms, *, scroll_to_first=False: calls.append(scroll_to_first)
        )

        self.window._apply_active_search_to_viewer()

        self.assertEqual(calls, [True])
        self.assertNotIn(path_key, self.window._pending_search_scroll_to_first_path_keys)

    def test_tree_context_menu_copy_path_copies_shell_escaped_absolute_path(self) -> None:
        pdf_path = Path(self._tempdir.name) / "copy path [test].pdf"
        _create_pdf_with_text(pdf_path, "copy path")
        self.window._copy_tree_path_to_clipboard(pdf_path)
        clipboard_text = QApplication.clipboard().text(QClipboard.Mode.Clipboard)
        self.assertEqual(
            clipboard_text,
            str(pdf_path.resolve()).replace(" ", r"\ ").replace("[", r"\[").replace("]", r"\]"),
        )

    def test_new_document_defaults_to_page_fit_view(self) -> None:
        pdf_path = Path(self._tempdir.name) / "default-fit.pdf"
        _create_pdf_with_text(pdf_path, "default fit")

        self.window._open_path_in_active_view(pdf_path)
        QApplication.processEvents()

        tab_index = self.window.view_tabs.currentIndex()
        tab_data = self.window._tab_data(tab_index) or {}
        state = tab_data.get("state") if isinstance(tab_data, dict) else {}
        self.assertEqual((state or {}).get("scale"), "page-fit")

        viewer_url = self.window._viewer_url_for_pdf(pdf_path)
        self.assertEqual(viewer_url.fragment(), "zoom=page-fit")

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

    def test_preview_widget_cache_is_bounded_and_uses_lru_eviction(self) -> None:
        first_pdf = Path(self._tempdir.name) / "lru-first.pdf"
        second_pdf = Path(self._tempdir.name) / "lru-second.pdf"
        third_pdf = Path(self._tempdir.name) / "lru-third.pdf"
        _create_pdf_with_text(first_pdf, "first document")
        _create_pdf_with_text(second_pdf, "second document")
        _create_pdf_with_text(third_pdf, "third document")

        self.window._open_path_in_active_view(first_pdf)
        QApplication.processEvents()
        first_view = self.window._current_preview_widget()

        self.window._open_path_in_active_view(second_pdf)
        QApplication.processEvents()

        # Reopening the first document makes the second document the LRU entry.
        self.window._open_path_in_active_view(first_pdf)
        QApplication.processEvents()
        self.assertIs(self.window._current_preview_widget(), first_view)

        self.window._open_path_in_active_view(third_pdf)
        QApplication.processEvents()

        first_key = self.window._path_key(first_pdf)
        second_key = self.window._path_key(second_pdf)
        third_key = self.window._path_key(third_pdf)
        self.assertEqual(self.window.PREVIEW_WIDGET_CACHE_MAX_ENTRIES, 2)
        self.assertEqual(
            set(self.window._preview_widgets_by_path),
            {first_key, third_key},
        )
        self.assertNotIn(second_key, self.window._preview_widgets_by_path)
        self.assertIs(
            self.window._current_preview_widget(),
            self.window._preview_widgets_by_path[third_key],
        )
        self.assertLessEqual(
            len(self.window._preview_widgets_by_path),
            self.window.PREVIEW_WIDGET_CACHE_MAX_ENTRIES,
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
        self.assertFalse(self.window.edit_btn.isEnabled())
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

    def test_single_view_scroll_state_restores_when_returning_to_document(self) -> None:
        first_pdf = Path(self._tempdir.name) / "first.pdf"
        second_pdf = Path(self._tempdir.name) / "second.pdf"
        _create_pdf_with_text(first_pdf, "first")
        _create_pdf_with_text(second_pdf, "second")

        first_key = self.window._path_key(first_pdf)
        first_preview = self.window._preview_widget_for_path(first_pdf)
        first_preview.setUrl(self.window._viewer_url_for_pdf(first_pdf))
        self.window._viewer_bridge_ready_by_path[first_key] = True

        captured_sources: list[str] = []
        first_state = {
            "page": 4,
            "pagesCount": 10,
            "scale": "page-width",
            "scrollTop": 160,
            "scrollRatio": 0.4,
        }

        def fake_run_viewer_js(source, callback=None):
            captured_sources.append(source)
            if "getViewState" in source and callback is not None:
                callback(first_state)
                return
            if callback is not None:
                callback(None)

        self.window._run_viewer_js = fake_run_viewer_js  # type: ignore[method-assign]

        self.window._open_path_in_active_view(first_pdf)
        QApplication.processEvents()
        self.window._open_path_in_active_view(second_pdf)
        QApplication.processEvents()
        self.window._open_path_in_active_view(first_pdf)
        QApplication.processEvents()

        restore_calls = [
            source for source in captured_sources if "restoreViewState" in source
        ]
        self.assertTrue(restore_calls)
        self.assertIn('"page": 4', restore_calls[-1])
        self.assertIn('"scrollTop": 160', restore_calls[-1])
        self.assertIn('"scrollRatio": 0.4', restore_calls[-1])

    def test_show_preview_context_menu_uses_request_selected_text_hint(self) -> None:
        captured: dict[str, object] = {}
        selection_info = {
            "selectedText": "Selected from Qt",
            "hasSelection": True,
            "page": 2,
            "start": 4,
            "end": 12,
            "multiPageSelection": False,
        }

        class _FakeRequest:
            def selectedText(self) -> str:
                return "Selected from Qt"

        class _FakePreview:
            def lastContextMenuRequest(self):
                return _FakeRequest()

            def selectedText(self) -> str:
                return ""

        original_current_preview_widget = self.window._current_preview_widget
        original_show_with_cached_selection = (
            self.window._show_preview_context_menu_with_cached_selection
        )
        original_request_selection_info = (
            self.window._request_preview_context_menu_selection_info
        )
        try:
            self.window._current_preview_widget = lambda: _FakePreview()  # type: ignore[method-assign]
            self.window._request_preview_context_menu_selection_info = (  # type: ignore[method-assign]
                lambda click_x, click_y, selected_text_hint, callback: (
                    captured.update(
                        {
                            "request_click_x": click_x,
                            "request_click_y": click_y,
                            "request_selected_text_hint": selected_text_hint,
                        }
                    ),
                    callback(dict(selection_info)),
                )
            )
            self.window._show_preview_context_menu_with_cached_selection = (  # type: ignore[method-assign]
                lambda pos, info, selected_text_hint, **kwargs: captured.update(
                    {
                        "pos": pos,
                        "info": info,
                        "selected_text_hint": selected_text_hint,
                        "kwargs": kwargs,
                    }
                )
            )

            self.window._show_preview_context_menu(QPoint(11, 17))
        finally:
            self.window._current_preview_widget = original_current_preview_widget  # type: ignore[method-assign]
            self.window._show_preview_context_menu_with_cached_selection = (  # type: ignore[method-assign]
                original_show_with_cached_selection
            )
            self.window._request_preview_context_menu_selection_info = (  # type: ignore[method-assign]
                original_request_selection_info
            )

        self.assertEqual(captured.get("selected_text_hint"), "Selected from Qt")
        self.assertEqual(captured.get("info"), selection_info)
        self.assertEqual(captured.get("request_click_x"), 11)
        self.assertEqual(captured.get("request_click_y"), 17)
        self.assertEqual(
            captured.get("request_selected_text_hint"),
            "Selected from Qt",
        )
        kwargs = captured.get("kwargs")
        self.assertIsInstance(kwargs, dict)
        assert isinstance(kwargs, dict)
        self.assertEqual(kwargs.get("click_x"), 11)
        self.assertEqual(kwargs.get("click_y"), 17)

    def test_request_preview_context_menu_selection_info_reuses_selected_text_hint(self) -> None:
        captured: dict[str, object] = {}

        def fake_run_viewer_js(_source, callback=None):
            if callback is not None:
                callback(
                    {
                        "page": 2,
                        "start": 4,
                        "end": 12,
                        "selectedText": "",
                        "hasSelection": False,
                    }
                )

        self.window._run_viewer_js = fake_run_viewer_js  # type: ignore[method-assign]
        self.window._request_preview_context_menu_selection_info(
            11,
            17,
            "Selected from Qt",
            lambda info: captured.update(info),
        )

        self.assertEqual(captured.get("selectedText"), "Selected from Qt")
        self.assertEqual(captured.get("hasSelection"), True)

    def test_context_menu_highlight_keeps_pre_menu_range_when_late_probe_is_empty(
        self,
    ) -> None:
        captured: dict[str, object] = {}

        class _FakePreview:
            @staticmethod
            def mapToGlobal(pos: QPoint) -> QPoint:
                return pos

        class _FakeAction:
            def __init__(self, text: str) -> None:
                self.text = text

        class _FakeMenu:
            def __init__(self, *_args, **_kwargs) -> None:
                self.actions: list[_FakeAction] = []

            def addAction(self, text: str):
                action = _FakeAction(text)
                self.actions.append(action)
                return action

            @staticmethod
            def addSeparator() -> None:
                return None

            def exec(self, *_args, **_kwargs):
                return next(
                    action for action in self.actions if action.text == "Highlight"
                )

        selection_snapshot = {
            "selectedText": "Alpha",
            "hasSelection": True,
            "page": 1,
            "start": 0,
            "end": 5,
            "multiPageSelection": False,
            "threeUpActive": False,
            "clickedHighlightId": "",
        }
        late_probe = {
            "selectedText": "Alpha",
            "hasSelection": True,
            "page": None,
            "start": None,
            "end": None,
            "multiPageSelection": False,
            "threeUpActive": False,
            "clickedHighlightId": "",
        }

        with (
            patch.object(
                self.window,
                "_current_preview_widget",
                return_value=_FakePreview(),
            ),
            patch.object(
                self.window,
                "_request_preview_context_menu_selection_info",
                side_effect=lambda _x, _y, _hint, callback: callback(
                    dict(late_probe)
                ),
            ),
            patch.object(
                self.window,
                "_add_persistent_preview_highlight",
                side_effect=lambda info, **kwargs: captured.update(
                    {"info": info, **kwargs}
                ),
            ),
            patch("pdfexplore.app.QMenu", _FakeMenu),
        ):
            self.window._show_preview_context_menu_with_cached_selection(
                QPoint(11, 17),
                selection_snapshot,
                "Alpha",
                click_x=11,
                click_y=17,
            )

        merged = captured.get("info")
        self.assertIsInstance(merged, dict)
        assert isinstance(merged, dict)
        self.assertEqual(merged.get("page"), 1)
        self.assertEqual(merged.get("start"), 0)
        self.assertEqual(merged.get("end"), 5)
        self.assertEqual(captured.get("kind"), "normal")
        self.assertEqual(captured.get("selected_text_hint"), "Alpha")

    def test_preview_zoom_in_out_and_reset_adjust_current_view_only(self) -> None:
        pdf_path = Path(self._tempdir.name) / "zoom.pdf"
        _create_pdf_with_text(pdf_path, "zoom test")
        self.window._open_path_in_active_view(pdf_path)
        QApplication.processEvents()

        path_key = self.window._path_key(pdf_path)
        self.window._viewer_bridge_ready_by_path[path_key] = True
        captured_sources: list[str] = []
        zoom_state = {"currentScale": 1.0, "currentScaleValue": "page-width", "percent": 100}

        def fake_run_viewer_js(source, callback=None):
            captured_sources.append(source)
            if callback is not None:
                callback(None)

        def fake_request_zoom_state(callback):
            callback(dict(zoom_state))

        self.window._run_viewer_js = fake_run_viewer_js  # type: ignore[method-assign]
        self.window._request_preview_zoom_state = fake_request_zoom_state  # type: ignore[method-assign]

        self.window._zoom_preview_in()
        QApplication.processEvents()
        self.assertIn("setZoomScale", captured_sources[-1])
        self.assertIn("1.1", captured_sources[-1])

        zoom_state["currentScale"] = 1.1
        self.window._zoom_preview_out()
        QApplication.processEvents()
        self.assertIn("setZoomScale", captured_sources[-1])
        self.assertIn("1.0", captured_sources[-1])

        self.window._reset_preview_zoom()
        QApplication.processEvents()
        self.assertTrue(any("resetZoom" in source for source in captured_sources))

        self.assertTrue(self.window._preview_zoom_overlay.isVisible())
        self.assertEqual(self.window._preview_zoom_overlay.text(), "Fit Width")

    def test_toggle_preview_three_up_uses_bridge_and_updates_overlay(self) -> None:
        pdf_path = Path(self._tempdir.name) / "three-up.pdf"
        _create_pdf_with_text(pdf_path, "three up")
        self.window._open_path_in_active_view(pdf_path)
        QApplication.processEvents()

        path_key = self.window._path_key(pdf_path)
        self.window._viewer_bridge_ready_by_path[path_key] = True
        captured_expressions: list[str] = []

        def fake_run_viewer_js_json(expression, callback):
            captured_expressions.append(expression)
            callback({"threeUpActive": True, "onePageScale": 1.25, "percent": 125})

        self.window._run_viewer_js_json = fake_run_viewer_js_json  # type: ignore[method-assign]
        self.window._toggle_preview_three_up()

        self.assertTrue(captured_expressions)
        self.assertIn("toggleThreeUpMode", captured_expressions[-1])
        self.assertEqual(self.window._preview_zoom_overlay.text(), "3-Up")

    def test_set_preview_zoom_one_hundred_uses_bridge_and_updates_overlay(self) -> None:
        pdf_path = Path(self._tempdir.name) / "zoom-100.pdf"
        _create_pdf_with_text(pdf_path, "zoom 100")
        self.window._open_path_in_active_view(pdf_path)
        QApplication.processEvents()

        path_key = self.window._path_key(pdf_path)
        self.window._viewer_bridge_ready_by_path[path_key] = True
        captured_expressions: list[str] = []

        def fake_run_viewer_js_json(expression, callback):
            captured_expressions.append(expression)
            callback({"ok": True, "threeUpActive": False, "onePageScale": 1.0, "percent": 100})

        self.window._run_viewer_js_json = fake_run_viewer_js_json  # type: ignore[method-assign]
        self.window._set_preview_zoom_one_hundred()

        self.assertTrue(captured_expressions)
        self.assertIn("setOnePageZoom100", captured_expressions[-1])
        self.assertEqual(self.window._preview_zoom_overlay.text(), "100%")

    def test_edit_button_is_available(self) -> None:
        self.assertEqual(self.window.edit_btn.text(), "Edit")
        self.assertFalse(self.window.edit_btn.isEnabled())

    def test_edit_current_file_uses_okular_launcher_script(self) -> None:
        pdf_path = Path(self._tempdir.name) / "open-in-okular.pdf"
        _create_pdf_with_text(pdf_path, "open in okular")
        self.window._open_path_in_active_view(pdf_path)
        QApplication.processEvents()
        self.assertTrue(self.window.edit_btn.isEnabled())

        launcher = Path(self._tempdir.name) / "run-okular.sh"
        launcher.write_text("#!/bin/sh\n", encoding="utf-8")

        with patch("pdfexplore.app.OKULAR_EDIT_LAUNCHER", launcher), patch(
            "pdfexplore.app.subprocess.Popen"
        ) as popen_mock:
            self.window._edit_current_file()

        popen_mock.assert_called_once_with([str(launcher), str(pdf_path.resolve())])

    def test_context_menu_disables_highlight_actions_in_three_up(self) -> None:
        pdf_path = Path(self._tempdir.name) / "menu-three-up.pdf"
        _create_pdf_with_text(pdf_path, "menu three up")
        self.window._open_path_in_active_view(pdf_path)
        QApplication.processEvents()

        class _FakeAction:
            def __init__(self, text: str) -> None:
                self.text = text

        class _FakeMenu:
            last_labels: list[str] = []

            def __init__(self, *_args, **_kwargs) -> None:
                self.labels: list[str] = []
                _FakeMenu.last_labels = self.labels

            def addAction(self, text: str):
                self.labels.append(text)
                return _FakeAction(text)

            def addSeparator(self) -> None:
                self.labels.append("---")

            def exec(self, *_args, **_kwargs):
                return None

        info = {
            "selectedText": "Alpha",
            "hasSelection": True,
            "page": 1,
            "start": 0,
            "end": 5,
            "threeUpActive": True,
        }

        with patch("pdfexplore.app.QMenu", _FakeMenu):
            self.window._show_preview_context_menu_with_cached_selection(
                QPoint(4, 4),
                info,
                "Alpha",
                click_x=4,
                click_y=4,
            )

        labels = _FakeMenu.last_labels
        self.assertNotIn("Highlight", labels)
        self.assertNotIn("Highlight Important", labels)
        self.assertNotIn("Remove Highlight", labels)
        self.assertIn("Copy Selected Text", labels)

    def test_schedule_followup_view_restore_reapplies_restore_and_highlights(self) -> None:
        pdf_path = Path(self._tempdir.name) / "followup.pdf"
        _create_pdf_with_text(pdf_path, "followup")
        path_key = self.window._path_key(pdf_path)
        restore_state = {
            "page": 5,
            "pagesCount": 11,
            "scale": "page-width",
            "scrollTop": 220,
            "scrollRatio": 0.5,
        }
        captured_sources: list[str] = []
        apply_highlights_calls: list[str] = []
        apply_search_calls: list[str] = []

        self.window._current_preview_path_key = lambda: path_key  # type: ignore[method-assign]
        self.window._viewer_bridge_ready_by_path[path_key] = True
        self.window._run_viewer_js = lambda source, callback=None: captured_sources.append(source)  # type: ignore[method-assign]
        self.window._apply_persistent_text_highlights = lambda: apply_highlights_calls.append("h")  # type: ignore[method-assign]
        self.window._apply_active_search_to_viewer = lambda: apply_search_calls.append("s")  # type: ignore[method-assign]

        with patch("pdfexplore.app.QTimer.singleShot", new=lambda _delay, fn: fn()):
            self.window._schedule_followup_view_restore(path_key, restore_state)

        restore_calls = [
            source for source in captured_sources if "restoreViewState" in source
        ]
        self.assertEqual(len(restore_calls), 2)
        self.assertIn('"page": 5', restore_calls[-1])
        self.assertEqual(len(apply_highlights_calls), 2)
        self.assertEqual(len(apply_search_calls), 2)

    def test_event_filter_handles_ctrl_bar_toggle_shortcut(self) -> None:
        calls: list[str] = []
        self.window._toggle_preview_three_up = lambda: calls.append("toggle")  # type: ignore[method-assign]

        shifted_event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Backslash,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
            "|",
        )
        plain_event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Backslash,
            Qt.KeyboardModifier.ControlModifier,
            "\\",
        )
        override_event = QKeyEvent(
            QEvent.Type.ShortcutOverride,
            Qt.Key.Key_Backslash,
            Qt.KeyboardModifier.ControlModifier,
            "\\",
        )
        ctrl_nine_event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_9,
            Qt.KeyboardModifier.ControlModifier,
            "9",
        )

        handled_shifted = self.window.eventFilter(self.window, shifted_event)
        handled_plain = self.window.eventFilter(self.window, plain_event)
        handled_override = self.window.eventFilter(self.window, override_event)
        handled_ctrl_nine = self.window.eventFilter(self.window, ctrl_nine_event)

        self.assertTrue(handled_shifted)
        self.assertTrue(handled_plain)
        self.assertTrue(handled_override)
        self.assertTrue(handled_ctrl_nine)
        self.assertEqual(calls, ["toggle", "toggle", "toggle"])


if __name__ == "__main__":
    unittest.main()
