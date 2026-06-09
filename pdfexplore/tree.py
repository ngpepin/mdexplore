"""PDF-specific tree model wrappers.

This module intentionally stays thin and delegates almost all rendering and
behavior to shared `mdexplore_app.file_tree` primitives.
"""

from __future__ import annotations

from mdexplore_app.file_tree import ColorizedExtensionModel, ExtensionTreeItemDelegate


class ColorizedPdfModel(ColorizedExtensionModel):
    """Filesystem model configured for PDF explorer semantics.

    The shared base class reads these constants to determine extension matching,
    icon shape/color, and sidecar filename for per-file color assignments.
    """

    COLOR_FILE_NAME = ".pdfexplore-colors.json"
    TARGET_EXTENSION = ".pdf"
    PRIMARY_ICON_NAME = "pdf.svg"
    PRIMARY_ICON_COLOR = "#e86060"


class PdfTreeItemDelegate(ExtensionTreeItemDelegate):
    """Delegate that paints PDF rows using shared extension-row behavior.

    The class exists primarily as an explicit PDF explorer type boundary so
    future PDF-only row behavior can be added without touching markdown explorer
    delegates.
    """
