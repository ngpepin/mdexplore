from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mdexplore_app.highlight_labels import (
    effective_highlight_label,
    highlight_label_menu_text,
    label_assignment_directory,
    local_highlight_label,
    set_highlight_label,
)


class HighlightLabelPersistenceTests(unittest.TestCase):
    def test_label_is_inherited_and_child_override_wins(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mdexplore-labels-") as tmpdir:
            root = Path(tmpdir)
            child = root / "child"
            grandchild = child / "grandchild"
            grandchild.mkdir(parents=True)
            filename = ".mdexplore-labels.json"

            set_highlight_label(root, filename, "#FFFF00", "Needs review")
            self.assertEqual(
                effective_highlight_label(grandchild, filename, "#ffff00"),
                "Needs review",
            )

            set_highlight_label(child, filename, "#ffff00", "Approved")
            self.assertEqual(
                effective_highlight_label(grandchild, filename, "#FFFF00"),
                "Approved",
            )
            self.assertEqual(local_highlight_label(child, filename, "#ffff00"), "Approved")

    def test_blank_label_removes_only_local_override_and_empty_sidecar(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdfexplore-labels-") as tmpdir:
            root = Path(tmpdir)
            child = root / "child"
            child.mkdir()
            filename = ".pdfexplore-labels.json"

            set_highlight_label(root, filename, "#7D12FF", "Archive")
            set_highlight_label(child, filename, "#7d12ff", "Local")
            set_highlight_label(child, filename, "#7d12ff", "")

            self.assertFalse((child / filename).exists())
            self.assertEqual(
                effective_highlight_label(child, filename, "#7D12FF"), "Archive"
            )

            payload = json.loads((root / filename).read_text(encoding="utf-8"))
            self.assertEqual(payload["labels"]["#7d12ff"], "Archive")

    def test_parent_level_assignment_targets_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="highlight-parent-") as tmpdir:
            root = Path(tmpdir)
            child = root / "child"
            child.mkdir()

            self.assertEqual(label_assignment_directory(child, False), child.resolve())
            self.assertEqual(label_assignment_directory(child, True), root.resolve())

            filename = ".mdexplore-labels.json"
            set_highlight_label(child, filename, "#ffff00", "Local")
            target = label_assignment_directory(child, True)
            set_highlight_label(target, filename, "#ffff00", "Shared")
            set_highlight_label(child, filename, "#ffff00", "")
            self.assertIsNone(local_highlight_label(child, filename, "#ffff00"))
            self.assertEqual(
                effective_highlight_label(child, filename, "#ffff00"), "Shared"
            )

    def test_context_menu_text_uses_effective_label_and_truncates_to_25(self) -> None:
        with tempfile.TemporaryDirectory(prefix="highlight-menu-") as tmpdir:
            root = Path(tmpdir)
            child = root / "child"
            child.mkdir()
            filename = ".mdexplore-labels.json"

            self.assertEqual(
                highlight_label_menu_text(child, filename, "#ffff00", "Yellow"),
                "Yellow",
            )
            set_highlight_label(root, filename, "#ffff00", "A very long inherited review label")
            text = highlight_label_menu_text(child, filename, "#ffff00", "Yellow")
            self.assertEqual(len(text), 25)
            self.assertTrue(text.endswith("…"))
            self.assertTrue(text.startswith("A very long inherited"))


if __name__ == "__main__":
    unittest.main()
