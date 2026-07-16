from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from mdexplore_app.file_coordination import load_files_payload, update_files_sidecar
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

    def test_highlight_sidecar_is_written_beside_nested_pdf(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        pdf_path = nested / "doc.pdf"
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

        nested_sidecar = nested / HIGHLIGHTING_FILE_NAME
        self.assertTrue(nested_sidecar.is_file())
        self.assertFalse((self.root / HIGHLIGHTING_FILE_NAME).exists())
        self.assertEqual(
            load_files_payload(nested_sidecar)[pdf_path.name][0]["text"],
            "alpha",
        )

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

    def test_failed_highlight_add_reports_failure_and_restores_disk_state(self) -> None:
        pdf_path = self.root / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
        path_key = self.window._path_key(pdf_path)
        sidecar = self.root / HIGHLIGHTING_FILE_NAME
        existing = {
            "id": "persisted-highlight",
            "page": 1,
            "start": 10,
            "end": 20,
            "kind": "normal",
            "text": "existing",
        }
        update_files_sidecar(
            sidecar,
            {
                pdf_path.name: [existing],
                "other.pdf": [
                    {
                        **existing,
                        "id": "other-highlight",
                    }
                ],
            },
        )

        with patch(
            "mdexplore_app.file_coordination.atomic_write_text",
            side_effect=OSError("disk full"),
        ):
            self.window._replace_persistent_preview_highlight_range(
                path_key,
                1,
                30,
                40,
                "important",
                "new",
            )

        committed = load_files_payload(sidecar)[pdf_path.name]
        self.assertEqual([entry["id"] for entry in committed], ["persisted-highlight"])
        self.assertEqual(
            [entry["id"] for entry in self.window._current_text_highlights],
            ["persisted-highlight"],
        )
        self.assertEqual(
            self.window.statusBar().currentMessage(),
            "Highlight could not be saved",
        )

    def test_failed_highlight_remove_reports_failure_and_restores_disk_state(self) -> None:
        pdf_path = self.root / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
        path_key = self.window._path_key(pdf_path)
        sidecar = self.root / HIGHLIGHTING_FILE_NAME
        existing = {
            "id": "persisted-highlight",
            "page": 1,
            "start": 10,
            "end": 20,
            "kind": "normal",
            "text": "existing",
        }
        update_files_sidecar(
            sidecar,
            {
                pdf_path.name: [existing],
                "other.pdf": [
                    {
                        **existing,
                        "id": "other-highlight",
                    }
                ],
            },
        )
        self.window.current_file = pdf_path
        self.window._current_text_highlights = [dict(existing)]

        with patch(
            "mdexplore_app.file_coordination.atomic_write_text",
            side_effect=OSError("disk full"),
        ):
            self.window._remove_persistent_preview_highlight(
                {"clickedHighlightId": "persisted-highlight"}
            )

        committed = load_files_payload(sidecar)[pdf_path.name]
        self.assertEqual([entry["id"] for entry in committed], ["persisted-highlight"])
        self.assertEqual(
            [entry["id"] for entry in self.window._current_text_highlights],
            ["persisted-highlight"],
        )
        self.assertEqual(
            self.window.statusBar().currentMessage(),
            "Highlight removal could not be saved",
        )

    def test_failed_final_highlight_unlink_is_reported_and_restored(self) -> None:
        pdf_path = self.root / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
        sidecar = self.root / HIGHLIGHTING_FILE_NAME
        existing = {
            "id": "only-highlight",
            "page": 1,
            "start": 10,
            "end": 20,
            "kind": "normal",
            "text": "existing",
        }
        update_files_sidecar(sidecar, {pdf_path.name: [existing]})
        self.window.current_file = pdf_path
        self.window._current_text_highlights = [dict(existing)]

        with patch(
            "pathlib.Path.unlink",
            side_effect=OSError("read only"),
        ):
            self.window._remove_persistent_preview_highlight(
                {"clickedHighlightId": "only-highlight"}
            )

        committed = load_files_payload(sidecar)[pdf_path.name]
        self.assertEqual([entry["id"] for entry in committed], ["only-highlight"])
        self.assertEqual(
            [entry["id"] for entry in self.window._current_text_highlights],
            ["only-highlight"],
        )
        self.assertEqual(
            self.window.statusBar().currentMessage(),
            "Highlight removal could not be saved",
        )


if __name__ == "__main__":
    unittest.main()
