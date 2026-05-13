"""Tree model helpers for the PDF explorer."""

from __future__ import annotations

from mdexplore_app.file_tree import ColorizedExtensionModel, ExtensionTreeItemDelegate


class ColorizedPdfModel(ColorizedExtensionModel):
    """Filesystem model with per-directory persisted PDF highlights."""

    COLOR_FILE_NAME = ".pdfexplore-colors.json"
    TARGET_EXTENSION = ".pdf"
    PRIMARY_ICON_NAME = "pdf.svg"
    PRIMARY_ICON_COLOR = "#e86060"


class PdfTreeItemDelegate(ExtensionTreeItemDelegate):
    """Paint filename-only highlight backgrounds for PDF rows."""
