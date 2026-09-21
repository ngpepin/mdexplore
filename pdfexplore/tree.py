"""PDF-specific tree model and delegate wrappers."""

from __future__ import annotations

from mdexplore_app.file_tree import (
    ColorizedExtensionModel,
    DirectorySortProxyModel,
    DirectorySortTreeView,
    ExtensionTreeItemDelegate,
)

from .settings import TREE_SETTINGS


def _tree_setting(name: str, fallback: str) -> str:
    value = TREE_SETTINGS.get(name, fallback)
    return str(value) if value is not None else fallback


class ColorizedPdfModel(ColorizedExtensionModel):
    """Filesystem model configured for PDF explorer semantics."""

    COLOR_FILE_NAME = _tree_setting("color_file_name", ".pdfexplore-colors.json")
    TARGET_EXTENSION = _tree_setting("target_extension", ".pdf")
    PRIMARY_ICON_NAME = _tree_setting("primary_icon_name", "pdf.svg")
    PRIMARY_ICON_COLOR = _tree_setting("primary_icon_color", "#e86060")


class PdfDirectorySortProxyModel(DirectorySortProxyModel):
    """Per-directory direct-child sorting for PDF files."""

    def __init__(self, source_model: ColorizedPdfModel, parent=None) -> None:
        super().__init__(source_model, ".pdfexplore-sort.json", parent)


class PdfTreeView(DirectorySortTreeView):
    """PDF tree with far-right per-directory sort controls."""


class PdfTreeItemDelegate(ExtensionTreeItemDelegate):
    """Use the shared mdexplore file and directory-row painting behavior."""
