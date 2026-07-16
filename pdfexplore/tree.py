"""PDF-specific tree model wrappers.

This module intentionally stays thin and delegates almost all rendering and
behavior to shared `mdexplore_app.file_tree` primitives.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QStyledItemDelegate

from mdexplore_app.file_tree import ColorizedExtensionModel, ExtensionTreeItemDelegate
from mdexplore_app.runtime import search_hit_count_font_family

from .settings import TREE_SETTINGS


def _tree_setting(name: str, fallback: str) -> str:
    value = TREE_SETTINGS.get(name, fallback)
    return str(value) if value is not None else fallback


class ColorizedPdfModel(ColorizedExtensionModel):
    """Filesystem model configured for PDF explorer semantics.

    The shared base class reads these constants to determine extension matching,
    icon shape/color, and sidecar filename for per-file color assignments.
    """

    COLOR_FILE_NAME = _tree_setting("color_file_name", ".pdfexplore-colors.json")
    TARGET_EXTENSION = _tree_setting("target_extension", ".pdf")
    PRIMARY_ICON_NAME = _tree_setting("primary_icon_name", "pdf.svg")
    PRIMARY_ICON_COLOR = _tree_setting("primary_icon_color", "#e86060")
    DIRECTORY_SEARCH_PILL_BG = "#f3c56b"
    DIRECTORY_SEARCH_PILL_FG = "#111111"

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        """Return a native Qt decoration badge for matching directories.

        PDF folder counts intentionally use the model's DecorationRole instead
        of the shared custom row painter.  Native decorations are rendered by
        every Qt item-view style and therefore remain visible in environments
        where a custom directory paint pass may be skipped or overwritten.
        """
        if role == Qt.ItemDataRole.DecorationRole:
            info = self.fileInfo(index)
            if info.isDir():
                hit_count = self.search_hit_count_for_directory(
                    Path(info.filePath())
                )
                if hit_count > 0:
                    base_icon = super().data(index, role)
                    if not isinstance(base_icon, QIcon):
                        base_icon = QIcon()
                    return self._decorated_directory_icon(base_icon, hit_count)
        return super().data(index, role)

    def _decorated_directory_icon(
        self,
        base_icon: QIcon,
        search_hit_count: int,
    ) -> QIcon:
        """Combine an orange matching-file-count pill with the folder icon."""
        count_text = self._search_count_display_text(search_hit_count)
        if not count_text:
            return base_icon

        cache_key = (
            "pdf-directory",
            int(base_icon.cacheKey()) if not base_icon.isNull() else 0,
            count_text,
        )
        cached = self._decorated_icon_cache.get(cache_key)
        if cached is not None:
            return cached

        total_width = self.max_decoration_width()
        total_height = self._ICON_SIZE
        cluster_width = self._SEARCH_SLOT_WIDTH + self._ICON_GAP + self._ICON_SIZE
        cursor_x = max(0, total_width - cluster_width)
        canvas = QPixmap(total_width, total_height)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        pill_rect = QRect(
            cursor_x,
            1,
            self._SEARCH_SLOT_WIDTH,
            max(1, total_height - 2),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self.DIRECTORY_SEARCH_PILL_BG))
        radius = max(3, int(pill_rect.height() * 0.28))
        painter.drawRoundedRect(pill_rect, radius, radius)

        font = painter.font()
        font.setFamily(search_hit_count_font_family())
        base_size = font.pointSizeF() if font.pointSizeF() > 0 else 10.0
        font.setPointSizeF(max(6.5, base_size - 1.8))
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(self.DIRECTORY_SEARCH_PILL_FG))
        painter.drawText(pill_rect, int(Qt.AlignmentFlag.AlignCenter), count_text)

        cursor_x += self._SEARCH_SLOT_WIDTH + self._ICON_GAP
        if not base_icon.isNull():
            folder_pixmap = base_icon.pixmap(self._ICON_SIZE, self._ICON_SIZE)
            painter.drawPixmap(cursor_x, 0, folder_pixmap)
        painter.end()

        decorated = QIcon(canvas)
        self._decorated_icon_cache[cache_key] = decorated
        return decorated


class PdfTreeItemDelegate(ExtensionTreeItemDelegate):
    """Delegate that paints PDF rows using shared extension-row behavior.

    The class exists primarily as an explicit PDF explorer type boundary so
    future PDF-only row behavior can be added without touching markdown explorer
    delegates.
    """

    def paint(self, painter, option, index) -> None:
        """Use Qt's native directory painting and shared PDF-file painting."""
        model = index.model()
        try:
            info = model.fileInfo(index)
        except Exception:
            info = None
        if info is not None and info.isDir():
            QStyledItemDelegate.paint(self, painter, option, index)
            return
        super().paint(painter, option, index)
