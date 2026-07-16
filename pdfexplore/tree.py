"""PDF-specific tree model and delegate wrappers."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import QStyledItemDelegate

from mdexplore_app.file_tree import ColorizedExtensionModel, ExtensionTreeItemDelegate
from mdexplore_app.runtime import search_hit_count_font_family

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


class PdfTreeItemDelegate(ExtensionTreeItemDelegate):
    """Paint PDF rows, with a PDF-only final overlay for directory counts.

    Markdown's shared delegate supplies the count source, colors, and file-row
    behavior.  Directory counts are redrawn here after Qt has finished painting
    the native folder row, anchored inside the visible viewport.  This prevents
    platform styles or a clipped tree column from erasing or hiding the badge.
    """

    _PDF_DIR_PILL_BG = "#f3c56b"
    _PDF_DIR_PILL_FG = "#111111"
    _PDF_DIR_PILL_MARGIN = 8
    _PDF_DIR_PILL_PADDING_X = 7

    def paint(self, painter: QPainter, option, index) -> None:
        model = index.model()
        if not isinstance(model, ColorizedPdfModel):
            super().paint(painter, option, index)
            return

        info = model.fileInfo(index)
        if not info.isDir():
            super().paint(painter, option, index)
            return

        # Let Qt paint the complete native directory row first.  The count is
        # intentionally the final operation so a style cannot overpaint it.
        QStyledItemDelegate.paint(self, painter, option, index)

        hit_count = model.search_hit_count_for_directory(Path(info.filePath()))
        if hit_count <= 0:
            return
        self._paint_visible_directory_count(painter, option, hit_count)

    def _paint_visible_directory_count(
        self,
        painter: QPainter,
        option,
        hit_count: int,
    ) -> None:
        count_text = str(hit_count) if hit_count <= 99 else "++"

        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

            font = option.font
            font.setFamily(search_hit_count_font_family())
            base_size = font.pointSizeF() if font.pointSizeF() > 0 else 10.0
            font.setPointSizeF(max(7.0, base_size - 1.2))
            font.setBold(False)
            painter.setFont(font)
            metrics = QFontMetrics(font)

            row_rect = option.rect
            visible_right = row_rect.right()
            widget = getattr(option, "widget", None)
            viewport = getattr(widget, "viewport", None)
            if callable(viewport):
                viewport_widget = viewport()
                if viewport_widget is not None:
                    visible_right = min(
                        visible_right,
                        viewport_widget.rect().right(),
                    )

            pill_height = max(14, min(20, row_rect.height() - 4))
            pill_width = max(
                22,
                metrics.horizontalAdvance(count_text)
                + (2 * self._PDF_DIR_PILL_PADDING_X),
            )
            pill_x = max(
                row_rect.left() + 4,
                visible_right - pill_width - self._PDF_DIR_PILL_MARGIN,
            )
            pill_y = row_rect.top() + max(0, (row_rect.height() - pill_height) / 2)
            pill_rect = QRectF(pill_x, pill_y, pill_width, pill_height)
            radius = max(4.0, pill_height * 0.30)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(self._PDF_DIR_PILL_BG))
            painter.drawRoundedRect(pill_rect, radius, radius)
            painter.setPen(QColor(self._PDF_DIR_PILL_FG))
            painter.drawText(
                pill_rect,
                int(Qt.AlignmentFlag.AlignCenter),
                count_text,
            )
        finally:
            painter.restore()
