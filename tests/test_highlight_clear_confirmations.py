import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QMessageBox

import mdexplore


class _FakeModel:
    def __init__(self) -> None:
        self.directory_clears: list[Path] = []
        self.recursive_clears: list[Path] = []

    def clear_directory_highlights(self, scope: Path) -> int:
        self.directory_clears.append(scope)
        return 3

    def clear_all_highlights(self, scope: Path) -> int:
        self.recursive_clears.append(scope)
        return 7


class _FakeStatusBar:
    def __init__(self) -> None:
        self.messages: list[tuple[str, int]] = []

    def showMessage(self, text: str, timeout: int) -> None:
        self.messages.append((text, timeout))


class HighlightClearConfirmationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = mdexplore.MdExploreWindow.__new__(mdexplore.MdExploreWindow)
        self.window.model = _FakeModel()
        self._status_bar = _FakeStatusBar()
        self.window.statusBar = lambda: self._status_bar
        self._tempdir = tempfile.TemporaryDirectory(prefix="mdexplore-clear-confirm-")
        self.scope = Path(self._tempdir.name)
        self.window._highlight_scope_directory = lambda: self.scope

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def test_clear_directory_highlights_requires_confirmation(self) -> None:
        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.No,
        ) as question:
            self.window._confirm_and_clear_directory_highlighting(self.scope)
        question.assert_called_once()
        self.assertEqual(self.window.model.directory_clears, [])

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ) as question:
            self.window._confirm_and_clear_directory_highlighting(self.scope)
        question.assert_called_once()
        self.assertEqual(self.window.model.directory_clears, [self.scope])
        self.assertEqual(len(self._status_bar.messages), 1)

    def test_clear_all_highlights_requires_confirmation(self) -> None:
        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.No,
        ) as question:
            self.window._confirm_and_clear_all_highlighting(self.scope)
        question.assert_called_once()
        self.assertEqual(self.window.model.recursive_clears, [])

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Yes,
        ) as question:
            self.window._confirm_and_clear_all_highlighting(self.scope)
        question.assert_called_once()
        self.assertEqual(self.window.model.recursive_clears, [self.scope])
        self.assertEqual(len(self._status_bar.messages), 1)


if __name__ == "__main__":
    unittest.main()
