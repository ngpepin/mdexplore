from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QSizePolicy

import mdexplore
from mdexplore_app import search as search_query
from mdexplore_app.workers import SearchScanWorker


class WindowLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="mdexplore-layout-")
        root = Path(self._tempdir.name)
        self.window = mdexplore.MdExploreWindow(
            root=root,
            app_icon=QIcon(),
            config_path=root / ".mdexplore.cfg",
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
        long_text = ("very/deep/document/path/" * 24) + "target-file.md"
        self.window.path_label.setText(long_text)
        self.window.path_label.setToolTip(long_text)
        QApplication.processEvents()
        expanded_width = self.window.centralWidget().minimumSizeHint().width()

        self.assertLessEqual(
            expanded_width,
            baseline_width + 80,
            "Long document paths should not make the main window effectively unshrinkable",
        )

    def test_safe_path_helpers_swallow_permission_errors(self) -> None:
        class _DeniedPath:
            def is_dir(self):
                raise PermissionError("denied")

            def is_file(self):
                raise PermissionError("denied")

            def resolve(self):
                raise PermissionError("denied")

        denied = _DeniedPath()

        self.assertFalse(self.window._safe_is_dir(denied))
        self.assertFalse(self.window._safe_is_file(denied))
        self.assertIs(self.window._safe_resolve(denied), denied)

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

        with patch("mdexplore.QMenu", _FakeMenu), patch.object(
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
        selected_file = selected_folder / "selected.md"
        selected_nested_folder = selected_folder / "nested"
        selected_nested_folder.mkdir()
        selected_nested_file = selected_nested_folder / "nested.md"
        other_file = other_folder / "other.md"
        selected_file.write_text("selected", encoding="utf-8")
        selected_nested_file.write_text("nested", encoding="utf-8")
        other_file.write_text("other", encoding="utf-8")

        self.window.model.set_search_match_counts(
            {selected_file: 2, selected_nested_file: 8, other_file: 3}
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

    def test_search_worker_completion_removes_the_exact_worker(self) -> None:
        predicate = search_query.compile_match_predicate("alpha")
        hit_counter = search_query.compile_match_hit_counter("alpha")
        first = SearchScanWorker(7, [], predicate, hit_counter, [])
        second = SearchScanWorker(7, [], predicate, hit_counter, [])
        self.window._active_search_scan_workers = {first, second}
        self.window._search_scan_request_id = 7
        self.window._search_scan_expected_workers = 2
        self.window._search_scan_completed_workers = 0

        self.window._on_search_scan_finished(
            first.worker_token,
            7,
            [],
            {},
            [],
            "",
        )

        self.assertNotIn(first, self.window._active_search_scan_workers)
        self.assertIn(second, self.window._active_search_scan_workers)
        self.assertEqual(self.window._search_scan_completed_workers, 1)

    def test_expand_and_collapse_buttons_are_between_refresh_and_pdf(self) -> None:
        button_labels = [
            button.text()
            for button in self.window.centralWidget().findChildren(
                mdexplore.QPushButton
            )
        ]
        refresh_index = button_labels.index("Refresh")
        self.assertEqual(
            button_labels[refresh_index : refresh_index + 4],
            ["Refresh", "Expand", "Collapse", "PDF"],
        )

    def test_expand_opens_markdown_bearing_branches_and_ancestors(self) -> None:
        root = Path(self._tempdir.name)
        markdown_parent = root / "reports"
        markdown_directory = markdown_parent / "quarterly"
        markdown_directory.mkdir(parents=True)
        (markdown_directory / "results.md").write_text("# Results", encoding="utf-8")

        direct_markdown_directory = root / "notes"
        direct_markdown_directory.mkdir()
        (direct_markdown_directory / "todo.md").write_text("# Todo", encoding="utf-8")

        empty_directory = root / "empty" / "nested"
        empty_directory.mkdir(parents=True)

        self.window._refresh_directory_view()
        QApplication.processEvents()
        self.window.expand_btn.click()
        QApplication.processEvents()

        reports_index = self._wait_for_tree_index(markdown_parent)
        quarterly_index = self._wait_for_tree_index(markdown_directory)
        notes_index = self._wait_for_tree_index(direct_markdown_directory)
        empty_index = self._wait_for_tree_index(root / "empty")

        self.assertTrue(self.window.tree.isExpanded(reports_index))
        self.assertTrue(self.window.tree.isExpanded(quarterly_index))
        self.assertTrue(self.window.tree.isExpanded(notes_index))
        self.assertFalse(self.window.tree.isExpanded(empty_index))

    def test_collapse_closes_all_directory_branches(self) -> None:
        root = Path(self._tempdir.name)
        markdown_parent = root / "reports"
        markdown_directory = markdown_parent / "quarterly"
        markdown_directory.mkdir(parents=True)
        (markdown_directory / "results.md").write_text("# Results", encoding="utf-8")

        self.window._refresh_directory_view()
        QApplication.processEvents()
        self.window.expand_btn.click()
        QApplication.processEvents()

        reports_index = self._wait_for_tree_index(markdown_parent)
        quarterly_index = self._wait_for_tree_index(markdown_directory)
        self.assertTrue(self.window.tree.isExpanded(reports_index))
        self.assertTrue(self.window.tree.isExpanded(quarterly_index))

        self.window.collapse_btn.click()
        QApplication.processEvents()

        self.assertFalse(self.window.tree.isExpanded(reports_index))
        self.assertFalse(self.window.tree.isExpanded(quarterly_index))

    def test_status_bar_shows_mmdr_version_before_gpu(self) -> None:
        self.window.close()
        QApplication.processEvents()

        with patch.object(
            mdexplore.MarkdownRenderer,
            "mermaid_runtime_status_text",
            return_value="mmdr 0.2.2",
        ):
            self.window = mdexplore.MdExploreWindow(
                root=Path(self._tempdir.name),
                app_icon=QIcon(),
                config_path=Path(self._tempdir.name) / ".mdexplore.cfg",
                gpu_context_available=True,
            )

        self.window.show()
        QApplication.processEvents()

        status_widgets = [
            widget.text()
            for widget in self.window.statusBar().findChildren(mdexplore.QLabel)
            if widget.text() in {"mmdr 0.2.2", "GPU"}
        ]
        self.assertEqual(status_widgets, ["mmdr 0.2.2", "GPU"])
        self.assertEqual(
            self.window._mermaid_runtime_status_label.styleSheet(),
            self.window._gpu_status_label.styleSheet(),
        )


if __name__ == "__main__":
    unittest.main()
