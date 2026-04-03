from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QSizePolicy

from pdfexplore.app import PdfExploreWindow


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


if __name__ == "__main__":
    unittest.main()
