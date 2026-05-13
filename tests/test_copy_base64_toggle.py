from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

import mdexplore


_ONE_PIXEL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z6m8AAAAASUVORK5CYII="
)


class CopyBase64ToggleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="mdexplore-copy-base64-")
        self.root = Path(self._tempdir.name)
        self.window = mdexplore.MdExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.root / ".mdexplore.cfg",
            mermaid_backend=mdexplore.MERMAID_BACKEND_JS,
            gpu_context_available=False,
            debug_mode=False,
        )
        self.window.show()
        QApplication.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        QApplication.processEvents()
        self._tempdir.cleanup()

    def _write_fixture_image(self, path: Path) -> None:
        path.write_bytes(base64.b64decode(_ONE_PIXEL_PNG_BASE64))

    def test_toggle_button_updates_tooltip_and_state(self) -> None:
        self.assertFalse(self.window._copy_base64_images_enabled)
        self.assertEqual(
            self.window.copy_image_base64_btn.toolTip(),
            "Turn BASE64 image encoding on",
        )
        self.window._toggle_copy_base64_images_enabled()
        self.assertTrue(self.window._copy_base64_images_enabled)
        self.assertEqual(
            self.window.copy_image_base64_btn.toolTip(),
            "Turn BASE64 image encoding off",
        )

    def test_convert_inline_image_links_to_data_uri(self) -> None:
        source_md = self.root / "note.md"
        source_md.write_text("placeholder\n", encoding="utf-8")
        image_path = self.root / "pixel.png"
        self._write_fixture_image(image_path)

        markdown = "Before\n![logo](pixel.png)\nAfter\n"
        converted, count = self.window._convert_markdown_image_links_to_data_uri_for_copy(
            markdown,
            source_md,
        )

        self.assertEqual(count, 1)
        self.assertIn("data:image/png;base64,", converted)
        self.assertNotIn("![logo](pixel.png)", converted)

    def test_convert_reference_image_definition_to_data_uri(self) -> None:
        source_md = self.root / "ref.md"
        source_md.write_text("placeholder\n", encoding="utf-8")
        image_path = self.root / "ref-image.png"
        self._write_fixture_image(image_path)

        markdown = "![diagram][img-ref]\n\n[img-ref]: ref-image.png\n"
        converted, count = self.window._convert_markdown_image_links_to_data_uri_for_copy(
            markdown,
            source_md,
        )

        self.assertEqual(count, 1)
        self.assertIn("[img-ref]: data:image/png;base64,", converted)
        self.assertNotIn("[img-ref]: ref-image.png", converted)

    def test_convert_linked_image_markdown_to_data_uri(self) -> None:
        source_md = self.root / "linked.md"
        source_md.write_text("placeholder\n", encoding="utf-8")
        image_path = self.root / "linked-image.png"
        self._write_fixture_image(image_path)

        markdown = "[![linked](linked-image.png)](linked-image.png)\n"
        converted, count = self.window._convert_markdown_image_links_to_data_uri_for_copy(
            markdown,
            source_md,
        )

        self.assertEqual(count, 2)
        self.assertGreaterEqual(converted.count("data:image/png;base64,"), 2)
        self.assertNotIn("linked-image.png", converted)

    def test_clipboard_staging_uses_base64_rewritten_markdown(self) -> None:
        source_md = self.root / "clipboard.md"
        image_path = self.root / "clip-image.png"
        self._write_fixture_image(image_path)
        source_md.write_text("![clip](clip-image.png)\n", encoding="utf-8")

        self.window._copy_base64_images_enabled = True
        staged_paths, converted_links, failures = self.window._prepare_clipboard_copy_paths(
            [source_md]
        )

        self.assertEqual(failures, 0)
        self.assertEqual(converted_links, 1)
        self.assertEqual(len(staged_paths), 1)
        staged_text = staged_paths[0].read_text(encoding="utf-8")
        self.assertIn("data:image/png;base64,", staged_text)


if __name__ == "__main__":
    unittest.main()
