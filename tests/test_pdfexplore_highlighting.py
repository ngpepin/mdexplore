from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from mdexplore_app.file_coordination import load_files_payload, update_files_sidecar
from pdfexplore.app import HIGHLIGHTING_FILE_NAME, NOTES_FILE_NAME, PdfExploreWindow


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

    def test_violet_file_highlight_color_is_available(self) -> None:
        self.assertIn(("Violet", "#7D12FF"), PdfExploreWindow.HIGHLIGHT_COLORS)

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

    def test_notes_persist_per_directory_and_overlaps_remain_distinct(self) -> None:
        pdf_path = self.root / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
        path_key = self.window._path_key(pdf_path)
        first = {"id": "n1", "page": 1, "start": 10, "end": 30, "kind": "note", "text": "one"}
        second = {"id": "n2", "page": 1, "start": 20, "end": 40, "kind": "note", "text": "two"}
        self.window._persist_notes_for_path_key(path_key, [first, second])
        sidecar = self.root / NOTES_FILE_NAME
        self.assertTrue(sidecar.is_file())
        entries = load_files_payload(sidecar)["doc.pdf"]
        self.assertEqual([entry["id"] for entry in entries], ["n1", "n2"])
        self.assertEqual([(entry["start"], entry["end"]) for entry in entries], [(10, 30), (20, 40)])

    def test_note_payload_is_after_purple_highlights_and_deletion_restores_purple_only(self) -> None:
        calls = []
        self.window._run_viewer_js = lambda js, callback=None: calls.append(js)  # type: ignore[method-assign]
        self.window._current_text_highlights = [{"id": "h1", "page": 1, "start": 10, "end": 30, "kind": "normal", "text": "alpha"}]
        self.window._current_notes = [{"id": "n1", "page": 1, "start": 15, "end": 25, "kind": "note", "text": "memo"}]
        PdfExploreWindow._apply_persistent_text_highlights(self.window)
        self.assertIn('"kind": "normal"', calls[-1])
        self.assertIn('"kind": "note"', calls[-1])
        self.assertNotIn('"memo"', calls[-1])
        self.assertLess(calls[-1].index('"kind": "normal"'), calls[-1].index('"kind": "note"'))
        self.window._current_notes = []
        PdfExploreWindow._apply_persistent_text_highlights(self.window)
        self.assertIn('"kind": "normal"', calls[-1])
        self.assertNotIn('"kind": "note"', calls[-1])

    def test_note_tree_marker_uses_shared_green_note_path_set(self) -> None:
        pdf_path = self.root / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
        path_key = self.window._path_key(pdf_path)
        self.window._persist_notes_for_path_key(path_key, [{"id": "n1", "page": 1, "start": 1, "end": 5, "kind": "note", "text": "memo"}])
        self.window._refresh_tree_marker_cache_for_path(path_key)
        self.assertIn(path_key, self.window._tree_note_marker_paths)
        self.assertIn(path_key, self.window.model._noted_preview_paths)

    def test_note_live_selection_range_can_replace_stale_cached_partial_range(self) -> None:
        cached = {"page": 1, "start": 10, "end": 20, "selectedText": "partial", "hasSelection": True}
        live = {"page": 1, "start": 10, "end": 80, "selectedText": "complete multiline selection", "hasSelection": True}
        captured = []
        self.window.current_file = self.root / "doc.pdf"
        self.window.current_file.write_bytes(b"%PDF-1.4\n%stub\n")
        def fake_add_dialog(*args, **kwargs):
            self.window._last_note_dialog_size = (630, 410)
            return "ok", "memo"
        self.window._run_preview_note_dialog = fake_add_dialog  # type: ignore[method-assign]
        original_transform = self.window._transform_notes_for_path_key
        def capture_transform(path_key, transform):
            result = original_transform(path_key, transform)
            captured.extend(result[0])
            return result
        self.window._transform_notes_for_path_key = capture_transform  # type: ignore[method-assign]
        # The menu path passes the valid live range to _add_preview_note; assert
        # the note lifecycle preserves that full range rather than the stale one.
        self.window._add_preview_note(live, cached["selectedText"])
        self.assertEqual((captured[-1]["start"], captured[-1]["end"]), (10, 80))
        self.assertEqual(captured[-1]["dialog_size"], {"width": 630, "height": 410})

    def test_note_specific_dialog_size_round_trips_and_is_restored_on_edit(self) -> None:
        pdf_path = self.root / "specific-size.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
        path_key = self.window._path_key(pdf_path)
        note = {
            "id": "n-size",
            "page": 1,
            "start": 5,
            "end": 15,
            "kind": "note",
            "text": "original",
            "dialog_size": {"width": 650, "height": 430},
        }
        self.window._persist_notes_for_path_key(path_key, [note])
        self.window.current_file = pdf_path
        self.window._current_notes = self.window._load_notes_for_path_key(path_key)
        observed_sizes: list[tuple[int, int] | None] = []

        def fake_edit_dialog(*args, **kwargs):
            observed_sizes.append(kwargs.get("preferred_size"))
            self.window._last_note_dialog_size = (760, 520)
            return "ok", "edited"

        self.window._run_preview_note_dialog = fake_edit_dialog  # type: ignore[method-assign]
        self.window._edit_preview_note("n-size")
        reloaded = self.window._load_notes_for_path_key(path_key)
        self.assertEqual(observed_sizes, [(650, 430)])
        self.assertEqual(reloaded[0]["text"], "edited")
        self.assertEqual(reloaded[0]["dialog_size"], {"width": 760, "height": 520})

    def test_note_edit_and_delete_lifecycle_persists(self) -> None:
        pdf_path = self.root / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
        path_key = self.window._path_key(pdf_path)
        note = {"id": "n-edit", "page": 1, "start": 5, "end": 15, "kind": "note", "text": "original"}
        self.window._persist_notes_for_path_key(path_key, [note])
        self.window.current_file = pdf_path
        self.window._current_notes = self.window._load_notes_for_path_key(path_key)
        self.window._run_preview_note_dialog = lambda *args, **kwargs: ("ok", "edited text")  # type: ignore[method-assign]
        self.window._edit_preview_note("n-edit")
        edited = self.window._load_notes_for_path_key(path_key)
        self.assertEqual(len(edited), 1)
        self.assertEqual(edited[0]["text"], "edited text")

        self.window._current_notes = edited
        self.window._run_preview_note_dialog = lambda *args, **kwargs: ("delete", "edited text")  # type: ignore[method-assign]
        self.window._edit_preview_note("n-edit")
        self.assertEqual(self.window._load_notes_for_path_key(path_key), [])
        self.assertNotIn(path_key, self.window._tree_note_marker_paths)


if __name__ == "__main__":
    unittest.main()
