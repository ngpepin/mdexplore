from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

import mdexplore


class PreviewRelativeMarkdownLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(
            prefix="mdexplore-relative-md-links-"
        )
        self.base = Path(self._tempdir.name)
        self.root = self.base / "docs"
        self.root.mkdir(parents=True, exist_ok=True)

        self.index_path = self.root / "index.md"
        self.index_path.write_text("# Index\n", encoding="utf-8")

        self.section_dir = self.root / "01_Overview"
        self.section_dir.mkdir(parents=True, exist_ok=True)
        self.section_file = self.section_dir / "01_General_Overview.md"
        self.section_file.write_text("# Overview\n", encoding="utf-8")

        self.external_dir = self.base / "external"
        self.external_dir.mkdir(parents=True, exist_ok=True)
        self.external_file = self.external_dir / "external.md"
        self.external_file.write_text("# External\n", encoding="utf-8")

        self.window = mdexplore.MdExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.base / ".mdexplore.cfg",
            gpu_context_available=False,
        )
        self.window.current_file = self.index_path.resolve()

    def tearDown(self) -> None:
        self.window.close()
        QApplication.processEvents()
        self._tempdir.cleanup()

    def test_resolve_relative_markdown_link_target_accepts_relative_md(self) -> None:
        resolved = self.window._resolve_relative_markdown_link_target(
            "./01_Overview/01_General_Overview.md#overview"
        )
        self.assertEqual(resolved, self.section_file.resolve())

    def test_resolve_relative_markdown_link_target_rejects_absolute_paths(self) -> None:
        self.assertIsNone(
            self.window._resolve_relative_markdown_link_target(
                f"{self.section_file.resolve()}"
            )
        )
        self.assertIsNone(
            self.window._resolve_relative_markdown_link_target(
                f"file://{self.section_file.resolve()}"
            )
        )

    def test_preview_relative_link_inside_root_expands_and_selects_without_reroot(self) -> None:
        with patch.object(self.window, "_set_root_directory") as set_root_mock, patch.object(
            self.window, "_expand_directory_path_in_tree"
        ) as expand_mock, patch.object(
            self.window, "_select_tree_markdown_path_with_retry"
        ) as select_mock:
            self.window._on_preview_relative_markdown_link_requested(
                "./01_Overview/01_General_Overview.md"
            )

        set_root_mock.assert_not_called()
        expand_mock.assert_called_once_with(self.section_dir.resolve())
        select_mock.assert_called_once_with(self.section_file.resolve())

    def test_preview_relative_link_outside_root_reroots_then_opens(self) -> None:
        with patch.object(self.window, "_set_root_directory") as set_root_mock, patch.object(
            self.window, "_expand_directory_path_in_tree"
        ) as expand_mock, patch.object(
            self.window, "_select_tree_markdown_path_with_retry"
        ) as select_mock:
            self.window._on_preview_relative_markdown_link_requested(
                "../external/external.md"
            )

        set_root_mock.assert_called_once_with(
            self.external_dir.resolve(),
            pending_preview_target=self.external_file.resolve(),
        )
        expand_mock.assert_called_once_with(self.external_dir.resolve())
        select_mock.assert_called_once_with(self.external_file.resolve())


if __name__ == "__main__":
    unittest.main()
