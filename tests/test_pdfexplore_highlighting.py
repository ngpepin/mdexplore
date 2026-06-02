from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from pdfexplore.app import HIGHLIGHTING_FILE_NAME, PdfExploreWindow


class PdfExploreHighlightPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="pdfexplore-highlighting-")
        self.root = Path(self._tempdir.name)
        self.window = PdfExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.root / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        self.window.show()
        QApplication.processEvents()
        self.window._apply_persistent_text_highlights = lambda: None  # type: ignore[method-assign]

    def tearDown(self) -> None:
        self.window.close()
        QApplication.processEvents()
        self._tempdir.cleanup()

    def test_highlights_persist_to_pdfexplore_sidecar(self) -> None:
        pdf_path = self.root / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
        path_key = self.window._path_key(pdf_path)

        self.window._replace_persistent_preview_highlight_range(
            path_key,
            1,
            10,
            20,
            "normal",
            "alpha",
        )
        self.window._replace_persistent_preview_highlight_range(
            path_key,
            1,
            30,
            40,
            "important",
            "beta",
        )

        sidecar = self.root / HIGHLIGHTING_FILE_NAME
        self.assertTrue(sidecar.is_file())
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        entries = payload.get("files", {}).get("doc.pdf", [])
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["kind"], "normal")
        self.assertEqual(entries[1]["kind"], "important")

    def test_overlapping_highlight_replaces_range_and_preserves_non_overlap(self) -> None:
        pdf_path = self.root / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
        path_key = self.window._path_key(pdf_path)

        self.window._replace_persistent_preview_highlight_range(
            path_key,
            1,
            10,
            20,
            "normal",
            "alpha",
        )
        self.window._replace_persistent_preview_highlight_range(
            path_key,
            1,
            14,
            18,
            "important",
            "beta",
        )

        entries = self.window._load_text_highlights_for_path_key(path_key)
        self.assertEqual(
            [(entry["start"], entry["end"], entry["kind"]) for entry in entries],
            [
                (10, 14, "normal"),
                (14, 18, "important"),
                (18, 20, "normal"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
