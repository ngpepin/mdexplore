from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QSizePolicy

import mdexplore


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


if __name__ == "__main__":
    unittest.main()
