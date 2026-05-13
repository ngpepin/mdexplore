"""Tree model and delegate helpers for the markdown file pane."""

from __future__ import annotations

from PySide6.QtGui import QIcon

from .file_tree import ColorizedExtensionModel, ExtensionTreeItemDelegate
from .icons import build_markdown_icon


class ColorizedMarkdownModel(ColorizedExtensionModel):
    """Filesystem model with per-directory persisted markdown highlights."""

    COLOR_FILE_NAME = ".mdexplore-colors.json"
    TARGET_EXTENSION = ".md"
    PRIMARY_ICON_NAME = "markdown.svg"
    PRIMARY_ICON_COLOR = "#bcc5d1"

    def _fallback_primary_icon(self) -> QIcon:
        return build_markdown_icon()


class MarkdownTreeItemDelegate(ExtensionTreeItemDelegate):
    """Paint filename-only highlight backgrounds for markdown rows."""
