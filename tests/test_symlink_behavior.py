from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QModelIndex
from PySide6.QtGui import QColor
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

import mdexplore
from mdexplore_app.tree import ColorizedMarkdownModel


class SymlinkIconModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_markdown_symlink_hides_view_and_highlight_icons(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mdexplore-symlink-icons-") as tmpdir:
            root = Path(tmpdir)
            target = root / "target.md"
            target.write_text("hello\n", encoding="utf-8")
            link = root / "link.md"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlink creation is not supported in this environment")

            model = ColorizedMarkdownModel()
            model.setRootPath(str(root))
            index = model.index(str(link))
            self.assertTrue(index.isValid())

            # Markers are tracked by resolved path key; symlink rows must still
            # suppress the marker/view icon decorations.
            model.set_multi_view_paths({target})
            model.set_persistent_highlight_paths({target})

            no_search_size = model.decoration_size_for_index(index)
            self.assertEqual(no_search_size.width(), model._ICON_SIZE)

            # Search-hit pills remain allowed for symlink rows.
            model.set_search_match_counts({link: 3})
            with_search_size = model.decoration_size_for_index(index)
            self.assertEqual(
                with_search_size.width(),
                model._SEARCH_SLOT_WIDTH + model._ICON_GAP + model._ICON_SIZE,
            )

    def test_markdown_symlink_uses_pastel_blue_grey_icon_tint(self) -> None:
        model = ColorizedMarkdownModel()
        icon_pixmap = model._symlink_icon.pixmap(model._ICON_SIZE, model._ICON_SIZE)
        image = icon_pixmap.toImage()
        expected = QColor(model.SYMLINK_ICON_COLOR)

        sampled = None
        for y in range(image.height()):
            for x in range(image.width()):
                color = image.pixelColor(x, y)
                if color.alpha() >= 200:
                    sampled = color
                    break
            if sampled is not None:
                break

        if sampled is None:
            for y in range(image.height()):
                for x in range(image.width()):
                    color = image.pixelColor(x, y)
                    if color.alpha() > 0:
                        sampled = color
                        break
                if sampled is not None:
                    break

        self.assertIsNotNone(sampled)
        self.assertIsNotNone(expected)
        self.assertLessEqual(abs(sampled.red() - expected.red()), 2)
        self.assertLessEqual(abs(sampled.green() - expected.green()), 2)
        self.assertLessEqual(abs(sampled.blue() - expected.blue()), 2)


class SymlinkNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="mdexplore-symlink-nav-")
        self.base = Path(self._tempdir.name)
        self.root = self.base / "docs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.window = mdexplore.MdExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.base / ".mdexplore.cfg",
            gpu_context_available=False,
        )

    def tearDown(self) -> None:
        self.window.close()
        QApplication.processEvents()
        self._tempdir.cleanup()

    def _build_symlink(self, target: Path, link: Path) -> None:
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("symlink creation is not supported in this environment")

    def test_symlink_target_within_session_top_scope_is_followed(self) -> None:
        content_dir = self.root / "content"
        links_dir = self.root / "links"
        content_dir.mkdir(parents=True, exist_ok=True)
        links_dir.mkdir(parents=True, exist_ok=True)

        target = content_dir / "target.md"
        target.write_text("# Target\n", encoding="utf-8")
        link = links_dir / "target-link.md"
        self._build_symlink(target, link)

        with patch.object(self.window, "_set_root_directory") as set_root_mock, patch.object(
            self.window, "_expand_directory_path_in_tree"
        ) as expand_mock, patch.object(
            self.window, "_select_tree_markdown_path_with_retry"
        ) as select_mock:
            followed = self.window._maybe_follow_markdown_symlink_selection(link)

        self.assertTrue(followed)
        set_root_mock.assert_not_called()
        expand_mock.assert_called_once_with(target.parent.resolve())
        select_mock.assert_called_once_with(target.resolve())

    def test_symlink_target_outside_session_top_scope_is_not_followed(self) -> None:
        links_dir = self.root / "links"
        links_dir.mkdir(parents=True, exist_ok=True)

        outside_dir = self.base / "outside"
        outside_dir.mkdir(parents=True, exist_ok=True)
        target = outside_dir / "outside.md"
        target.write_text("# Outside\n", encoding="utf-8")
        link = links_dir / "outside-link.md"
        self._build_symlink(target, link)

        self.window._session_top_visited_root = self.root.resolve()

        with patch.object(self.window, "_set_root_directory") as set_root_mock, patch.object(
            self.window, "_expand_directory_path_in_tree"
        ) as expand_mock, patch.object(
            self.window, "_select_tree_markdown_path_with_retry"
        ) as select_mock:
            followed = self.window._maybe_follow_markdown_symlink_selection(link)

        self.assertFalse(followed)
        set_root_mock.assert_not_called()
        expand_mock.assert_not_called()
        select_mock.assert_not_called()

    def test_symlink_target_can_reroot_when_within_session_top_scope(self) -> None:
        links_dir = self.root / "links"
        content_dir = self.root / "content"
        links_dir.mkdir(parents=True, exist_ok=True)
        content_dir.mkdir(parents=True, exist_ok=True)

        target = content_dir / "target.md"
        target.write_text("# Target\n", encoding="utf-8")
        link = links_dir / "target-link.md"
        self._build_symlink(target, link)

        # Simulate current root narrower than session top scope.
        self.window.root = links_dir.resolve()
        self.window._session_top_visited_root = self.root.resolve()

        with patch.object(self.window, "_set_root_directory") as set_root_mock, patch.object(
            self.window, "_expand_directory_path_in_tree"
        ) as expand_mock, patch.object(
            self.window, "_select_tree_markdown_path_with_retry"
        ) as select_mock:
            followed = self.window._maybe_follow_markdown_symlink_selection(link)

        self.assertTrue(followed)
        set_root_mock.assert_called_once_with(
            target.parent.resolve(),
            pending_preview_target=target.resolve(),
        )
        expand_mock.assert_called_once_with(target.parent.resolve())
        select_mock.assert_called_once_with(target.resolve())

    def test_set_root_directory_pending_target_shows_preparing_placeholder(self) -> None:
        pending_root = self.base / "pending-root"
        pending_root.mkdir(parents=True, exist_ok=True)
        target = pending_root / "pending.md"
        target.write_text("# Pending\n", encoding="utf-8")

        with patch.object(self.window, "_set_preview_html") as set_preview_html_mock:
            self.window._set_root_directory(
                pending_root,
                pending_preview_target=target,
            )

        self.assertTrue(set_preview_html_mock.called)
        placeholder_html = set_preview_html_mock.call_args_list[0].args[0]
        self.assertIn("Preparing markdown preview", placeholder_html)
        self.assertIn(target.name, placeholder_html)

    def test_select_tree_markdown_path_retry_forces_preview_when_signal_missed(self) -> None:
        target = self.root / "target.md"
        target.write_text("# Target\n", encoding="utf-8")
        self.window.model.setRootPath(str(self.root))
        self.window.tree.setRootIndex(self.window.model.index(str(self.root)))

        with patch.object(self.window.tree, "setCurrentIndex") as set_index_mock, patch.object(
            self.window.tree, "scrollTo"
        ) as scroll_mock, patch.object(self.window, "_load_preview") as load_preview_mock:
            self.window._select_tree_markdown_path_with_retry(target)

        set_index_mock.assert_called_once()
        scroll_mock.assert_called_once()
        load_preview_mock.assert_called_once_with(target.resolve())

    def test_select_tree_markdown_path_retry_shows_preparing_placeholder(self) -> None:
        target = self.root / "pending.md"
        target.write_text("# Pending\n", encoding="utf-8")

        with patch.object(self.window.model, "index", return_value=QModelIndex()), patch.object(
            self.window, "_set_preview_html"
        ) as set_preview_html_mock, patch.object(
            self.window, "_load_preview"
        ) as load_preview_mock:
            self.window._select_tree_markdown_path_with_retry(target, attempts_left=0)

        self.assertTrue(set_preview_html_mock.called)
        placeholder_html = set_preview_html_mock.call_args_list[0].args[0]
        self.assertIn("Preparing markdown preview", placeholder_html)
        self.assertIn(target.name, placeholder_html)
        load_preview_mock.assert_called_once_with(target.resolve())

    def test_load_preview_shows_rendering_placeholder_message(self) -> None:
        target = self.root / "rendering.md"
        target.write_text("# Rendering\n", encoding="utf-8")

        with patch.object(self.window, "_set_preview_html") as set_preview_html_mock, patch.object(
            self.window._render_pool, "start"
        ) as render_start_mock:
            self.window._load_preview(target)

        self.assertTrue(set_preview_html_mock.called)
        placeholder_html = set_preview_html_mock.call_args_list[0].args[0]
        self.assertIn("Rendering markdown preview", placeholder_html)
        self.assertIn(target.name, placeholder_html)
        render_start_mock.assert_called_once()

    def test_back_button_tracks_previous_document_once(self) -> None:
        first = self.root / "first.md"
        second = self.root / "second.md"
        first.write_text("# First\n", encoding="utf-8")
        second.write_text("# Second\n", encoding="utf-8")

        with patch.object(self.window._render_pool, "start"):
            self.window._load_preview(first)
            self.assertFalse(self.window.back_btn.isEnabled())
            self.window._load_preview(second)

        self.assertTrue(self.window.back_btn.isEnabled())
        with patch.object(
            self.window, "_select_tree_markdown_path_with_retry"
        ) as select_mock:
            self.window._go_back_document()
            self.window._go_back_document()

        select_mock.assert_called_once_with(first.resolve())
        self.assertFalse(self.window.back_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
