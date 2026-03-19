"""Tree model and delegate helpers for the markdown file pane."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileSystemModel,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from .icons import build_markdown_icon, load_svg_icon
from .runtime import search_hit_count_font_family


class ColorizedMarkdownModel(QFileSystemModel):
    """Filesystem model with per-directory persisted file highlight colors."""

    COLOR_FILE_NAME = ".mdexplore-colors.json"
    _ICON_SIZE = 16
    _ICON_GAP = 2
    _VIEWS_ICON_SIZE = 12
    _MARKER_ICON_SIZE = 13
    _SEARCH_SLOT_WIDTH = 14
    _SEARCH_COUNT_TEXT_MAX = 99
    # Oversample the search-hit pill/count layer before downscaling it back
    # into the fixed gutter slot so the pill edges and digits stay sharper
    # than the surrounding small icons.
    _SEARCH_COUNT_OVERSAMPLE = 8
    _SEARCH_COUNT_TEXT_OVERSAMPLE = 12

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Cache both persistence data and the small finite set of tree icons so
        # the view stays responsive even in large directories.
        self._dir_color_map: dict[str, dict[str, str]] = {}
        self._loaded_dirs: set[str] = set()
        self._search_match_counts: dict[str, int] = {}
        self._multi_view_paths: set[str] = set()
        self._highlighted_preview_paths: set[str] = set()
        self._markdown_icon = load_svg_icon("markdown.svg", QColor("#bcc5d1"))
        if self._markdown_icon.isNull():
            self._markdown_icon = build_markdown_icon()
        # Render the views badge in a very light gray so it stays crisp on the
        # dark tree background without overpowering the markdown file icon.
        self._views_icon = load_svg_icon("views2.svg", QColor("#e3e7ee"))
        # Persistent preview highlights are represented by the same pastel
        # purple marker regardless of whether the file also has saved views.
        self._marker_icon = load_svg_icon("marker.svg", QColor("#c8b4f6"))
        self._decorated_icon_cache: dict[tuple[bool, bool, str], QIcon] = {}

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        # This model only customizes markdown files; directories continue to use
        # QFileSystemModel defaults so tree navigation behavior remains native.
        if role == Qt.ItemDataRole.DecorationRole:
            info = self.fileInfo(index)
            if info.isFile() and info.suffix().lower() == "md":
                path_key = self._path_key(Path(info.filePath()))
                search_hit_count = self._search_match_counts.get(path_key, 0)
                has_multi_view = path_key in self._multi_view_paths
                has_persistent_highlight = path_key in self._highlighted_preview_paths
                return self._decorated_markdown_icon(
                    has_multi_view,
                    has_persistent_highlight,
                    search_hit_count,
                )
        if role == Qt.ItemDataRole.ForegroundRole:
            info = self.fileInfo(index)
            if info.isFile() and info.suffix().lower() == "md":
                color_name = self._color_for_file(Path(info.filePath()))
                if color_name:
                    color = QColor(color_name)
                    if color.isValid():
                        luminance = (
                            (0.299 * color.redF())
                            + (0.587 * color.greenF())
                            + (0.114 * color.blueF())
                        )
                        return QBrush(
                            QColor("#101418") if luminance > 0.6 else QColor("#f8fafc")
                        )
        if role == Qt.ItemDataRole.FontRole:
            info = self.fileInfo(index)
            if info.isFile() and info.suffix().lower() == "md":
                if self._search_match_counts.get(
                    self._path_key(Path(info.filePath())), 0
                ) > 0:
                    base_font = super().data(index, role)
                    font = QFont(base_font) if isinstance(base_font, QFont) else QFont()
                    font.setBold(True)
                    font.setItalic(True)
                    return font
        return super().data(index, role)

    def set_color_for_file(self, path: Path, color_name: str | None) -> None:
        # Persist selected color immediately and notify the view.
        directory = path.parent
        color_map = self._load_directory_colors(directory)
        if color_name:
            color_map[path.name] = color_name
        else:
            color_map.pop(path.name, None)
        self._save_directory_colors(directory)

        index = self.index(str(path))
        if index.isValid():
            self.dataChanged.emit(
                index,
                index,
                [Qt.ItemDataRole.ForegroundRole],
            )

    def highlight_background_for_path(self, path: Path) -> QColor | None:
        color_name = self._color_for_file(path)
        if not color_name:
            return None
        color = QColor(color_name)
        return color if color.isValid() else None

    def highlight_foreground_for_path(self, path: Path) -> QColor | None:
        background = self.highlight_background_for_path(path)
        if background is None:
            return None
        luminance = (
            (0.299 * background.redF())
            + (0.587 * background.greenF())
            + (0.114 * background.blueF())
        )
        return QColor("#101418") if luminance > 0.6 else QColor("#f8fafc")

    def collect_files_with_color(self, root: Path, color_name: str) -> list[Path]:
        """Return files under root that are persisted with the requested color."""
        if not root.is_dir():
            return []
        normalized_color = color_name.lower()
        matches: list[Path] = []

        def on_walk_error(_err) -> None:
            # Permission errors are expected in some trees; skip quietly.
            return

        for dirpath, _dirnames, _filenames in os.walk(
            root, onerror=on_walk_error, followlinks=False
        ):
            directory = Path(dirpath)
            color_map = self._load_directory_colors(directory)
            for file_name, file_color in color_map.items():
                if file_color.lower() != normalized_color:
                    continue
                candidate = directory / file_name
                try:
                    if candidate.is_file():
                        matches.append(candidate.resolve())
                except Exception:
                    # Broken symlink or inaccessible file; ignore quietly.
                    pass

        matches.sort(key=str)
        return matches

    def clear_all_highlights(self, root: Path) -> int:
        """Clear all persisted highlights under root recursively."""
        if not root.is_dir():
            return 0

        cleared_entries = 0

        def on_walk_error(_err) -> None:
            # Permission errors are expected in some trees; skip quietly.
            return

        for dirpath, _dirnames, _filenames in os.walk(
            root, onerror=on_walk_error, followlinks=False
        ):
            directory = Path(dirpath)
            color_map = self._load_directory_colors(directory)
            if not color_map:
                continue
            cleared_entries += len(color_map)
            color_map.clear()
            self._save_directory_colors(directory)

        return cleared_entries

    def clear_directory_highlights(self, directory: Path) -> int:
        """Clear persisted highlights in one directory (non-recursive)."""
        if not directory.is_dir():
            return 0

        color_map = self._load_directory_colors(directory)
        if not color_map:
            return 0

        cleared_entries = len(color_map)
        color_map.clear()
        self._save_directory_colors(directory)
        return cleared_entries

    def _color_for_file(self, path: Path) -> str | None:
        color_map = self._load_directory_colors(path.parent)
        return color_map.get(path.name)

    def color_for_file(self, path: Path) -> str | None:
        """Return the persisted highlight color for one markdown file path."""
        return self._color_for_file(path)

    def set_search_match_paths(self, paths: set[Path]) -> None:
        self.set_search_match_counts({path: 1 for path in paths})

    def set_search_match_counts(self, match_counts: dict[Path, int]) -> None:
        """Replace active search-hit counts by path (non-positive counts omitted)."""
        next_counts: dict[str, int] = {}
        for path, raw_count in match_counts.items():
            try:
                count = int(raw_count)
            except Exception:
                continue
            if count <= 0:
                continue
            next_counts[self._path_key(path)] = count
        self._search_match_counts = next_counts
        self._decorated_icon_cache.clear()

    def clear_search_match_paths(self) -> None:
        if not self._search_match_counts:
            return
        self._search_match_counts.clear()
        self._decorated_icon_cache.clear()

    def set_multi_view_paths(self, paths: set[Path]) -> None:
        next_paths = {self._path_key(path) for path in paths}
        if next_paths == self._multi_view_paths:
            return
        self._multi_view_paths = next_paths
        self._decorated_icon_cache.clear()

    def set_multi_view_path_keys(self, path_keys: set[str]) -> None:
        """Replace active multi-view badge paths using precomputed path keys."""
        next_paths = {str(path_key) for path_key in path_keys if isinstance(path_key, str)}
        if next_paths == self._multi_view_paths:
            return
        self._multi_view_paths = next_paths
        self._decorated_icon_cache.clear()

    def clear_multi_view_paths(self) -> None:
        if not self._multi_view_paths:
            return
        self._multi_view_paths.clear()
        self._decorated_icon_cache.clear()

    def set_persistent_highlight_paths(self, paths: set[Path]) -> None:
        """Replace markdown paths that have persisted preview text highlights."""
        next_paths = {self._path_key(path) for path in paths}
        if next_paths == self._highlighted_preview_paths:
            return
        self._highlighted_preview_paths = next_paths
        self._decorated_icon_cache.clear()

    def set_persistent_highlight_path_keys(self, path_keys: set[str]) -> None:
        """Replace persisted-highlight badge paths using precomputed path keys."""
        next_paths = {str(path_key) for path_key in path_keys if isinstance(path_key, str)}
        if next_paths == self._highlighted_preview_paths:
            return
        self._highlighted_preview_paths = next_paths
        self._decorated_icon_cache.clear()

    def clear_persistent_highlight_paths(self) -> None:
        """Clear persisted-preview-highlight badges in the tree."""
        if not self._highlighted_preview_paths:
            return
        self._highlighted_preview_paths.clear()
        self._decorated_icon_cache.clear()

    def _path_key(self, path: Path) -> str:
        try:
            return str(path.resolve())
        except Exception:
            return str(path)

    def _directory_key(self, directory: Path) -> str:
        try:
            return str(directory.resolve())
        except Exception:
            return str(directory)

    def _load_directory_colors(self, directory: Path) -> dict[str, str]:
        # Load once per directory and cache the mapping in-memory.
        key = self._directory_key(directory)
        if key in self._loaded_dirs:
            return self._dir_color_map.setdefault(key, {})

        self._loaded_dirs.add(key)
        color_map: dict[str, str] = {}
        color_file = directory / self.COLOR_FILE_NAME
        try:
            raw = color_file.read_text(encoding="utf-8")
            payload = json.loads(raw)
            if isinstance(payload, dict):
                files_payload = payload.get("files", payload)
                if isinstance(files_payload, dict):
                    for name, color in files_payload.items():
                        if isinstance(name, str) and isinstance(color, str):
                            color_map[name] = color
        except Exception:
            # Missing file, access denied, or malformed JSON should not block browsing.
            pass

        self._dir_color_map[key] = color_map
        return color_map

    def _save_directory_colors(self, directory: Path) -> None:
        # Writes are intentionally best-effort; unwritable directories are
        # expected in some user environments.
        key = self._directory_key(directory)
        color_map = self._dir_color_map.get(key, {})
        color_file = directory / self.COLOR_FILE_NAME
        try:
            if color_map:
                payload = {"files": dict(sorted(color_map.items()))}
                color_file.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            elif color_file.exists():
                color_file.unlink()
        except Exception:
            # Requested behavior: fail quietly when persistence can't be written.
            pass

    @classmethod
    def decorated_icon_size(cls) -> QSize:
        return QSize(cls.max_decoration_width(), cls._ICON_SIZE)

    @classmethod
    def max_decoration_width(cls) -> int:
        """Return the widest markdown decoration strip used by the tree."""
        widths = [
            cls._SEARCH_SLOT_WIDTH,
            cls._MARKER_ICON_SIZE,
            cls._VIEWS_ICON_SIZE,
            cls._ICON_SIZE,
        ]
        return sum(widths) + (cls._ICON_GAP * max(0, len(widths) - 1))

    def _search_count_display_text(self, search_hit_count: int) -> str:
        """Render compact count text for the search slot."""
        if search_hit_count <= 0:
            return ""
        if search_hit_count <= self._SEARCH_COUNT_TEXT_MAX:
            return str(search_hit_count)
        return "++"

    def decoration_size_for_state(
        self,
        has_multi_view: bool,
        has_persistent_highlight: bool,
        search_hit_count: int,
    ) -> QSize:
        """Return packed decoration size for the active markdown row state."""
        search_count_text = self._search_count_display_text(search_hit_count)
        widths: list[int] = []
        if search_count_text:
            widths.append(self._SEARCH_SLOT_WIDTH)
        if has_persistent_highlight:
            widths.append(self._MARKER_ICON_SIZE)
        if has_multi_view:
            widths.append(self._VIEWS_ICON_SIZE)
        widths.append(self._ICON_SIZE)
        total_width = sum(widths) + (self._ICON_GAP * max(0, len(widths) - 1))
        return QSize(total_width, self._ICON_SIZE)

    def decoration_size_for_index(self, index) -> QSize:
        """Return packed decoration size for one markdown model index."""
        info = self.fileInfo(index)
        if not (info.isFile() and info.suffix().lower() == "md"):
            return QSize(self._ICON_SIZE, self._ICON_SIZE)
        path_key = self._path_key(Path(info.filePath()))
        search_hit_count = self._search_match_counts.get(path_key, 0)
        has_multi_view = path_key in self._multi_view_paths
        has_persistent_highlight = path_key in self._highlighted_preview_paths
        return self.decoration_size_for_state(
            has_multi_view, has_persistent_highlight, search_hit_count
        )

    def _decorated_markdown_icon(
        self,
        has_multi_view: bool,
        has_persistent_highlight: bool,
        search_hit_count: int,
    ) -> QIcon:
        search_count_text = self._search_count_display_text(search_hit_count)
        cache_key = (has_multi_view, has_persistent_highlight, search_count_text)
        cached = self._decorated_icon_cache.get(cache_key)
        if cached is not None:
            return cached

        # Keep a fixed gutter width for every markdown row so the markdown file
        # icon and the filename always start on the same vertical line. Inside
        # that strip, badges are packed tightly together and right-aligned so
        # any unused space appears only to the left of the icon cluster.
        cluster_size = self.decoration_size_for_state(
            has_multi_view, has_persistent_highlight, search_hit_count
        )
        total_width = self.max_decoration_width()
        total_height = self._ICON_SIZE
        canvas = QPixmap(total_width, total_height)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        cursor_x = max(0, total_width - cluster_size.width())
        if search_count_text:
            # Render the pill and the digits in separate passes so the text can
            # be oversampled more aggressively than the background shape while
            # still landing in the same fixed gutter slot.
            oversample = max(1, int(self._SEARCH_COUNT_OVERSAMPLE))
            text_oversample = max(oversample, int(self._SEARCH_COUNT_TEXT_OVERSAMPLE))
            slot_w = self._SEARCH_SLOT_WIDTH
            slot_h = self._ICON_SIZE
            hi_w = slot_w * oversample
            hi_h = slot_h * oversample

            count_canvas = QPixmap(hi_w, hi_h)
            count_canvas.fill(Qt.GlobalColor.transparent)
            count_painter = QPainter(count_canvas)
            count_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            count_painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            count_painter.setRenderHint(
                QPainter.RenderHint.SmoothPixmapTransform, True
            )
            font = count_painter.font()
            base_size = font.pointSizeF() if font.pointSizeF() > 0 else 10.0
            font.setFamily(search_hit_count_font_family())
            font.setPointSizeF(max(6.5, base_size - 1.8) * oversample)
            font.setBold(False)
            count_painter.setFont(font)
            metrics = QFontMetrics(font)
            text_rect = metrics.boundingRect(search_count_text)
            pill_padding_x = max(6 * oversample, int(metrics.height() * 0.42))
            pill_padding_y = max(2 * oversample, int(metrics.height() * 0.12))
            pill_width = min(
                hi_w - (2 * oversample),
                text_rect.width() + (2 * pill_padding_x),
            )
            pill_height = min(
                hi_h - (2 * oversample),
                text_rect.height() + (2 * pill_padding_y),
            )
            pill_x = max(0, (hi_w - pill_width) // 2)
            pill_y = max(0, (hi_h - pill_height) // 2)
            pill_rect = QRect(pill_x, pill_y, pill_width, pill_height)
            radius = max(oversample, int(pill_height * 0.28))

            count_painter.setPen(Qt.PenStyle.NoPen)
            count_painter.setBrush(QColor("#f7e27a"))
            count_painter.drawRoundedRect(pill_rect, radius, radius)
            count_painter.end()

            text_canvas = QPixmap(slot_w * text_oversample, slot_h * text_oversample)
            text_canvas.fill(Qt.GlobalColor.transparent)
            text_painter = QPainter(text_canvas)
            text_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            text_painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            text_painter.setRenderHint(
                QPainter.RenderHint.SmoothPixmapTransform, True
            )
            text_font = text_painter.font()
            text_base_size = (
                text_font.pointSizeF() if text_font.pointSizeF() > 0 else 10.0
            )
            text_font.setFamily(search_hit_count_font_family())
            text_font.setPointSizeF(max(6.5, text_base_size - 1.8) * text_oversample)
            text_font.setBold(False)
            text_painter.setFont(text_font)
            text_painter.setPen(QPen(QColor("#111111")))
            text_painter.drawText(
                QRect(0, 0, slot_w * text_oversample, slot_h * text_oversample),
                Qt.AlignmentFlag.AlignCenter,
                search_count_text,
            )
            text_painter.end()

            count_layer = count_canvas.scaled(
                slot_w,
                slot_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            text_layer = text_canvas.scaled(
                slot_w,
                slot_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(cursor_x, 0, count_layer)
            painter.drawPixmap(cursor_x, 0, text_layer)
            cursor_x += self._SEARCH_SLOT_WIDTH + self._ICON_GAP

        if has_persistent_highlight and not self._marker_icon.isNull():
            marker_pixmap = self._marker_icon.pixmap(
                self._MARKER_ICON_SIZE, self._MARKER_ICON_SIZE
            )
            marker_y = max(0, (self._ICON_SIZE - self._MARKER_ICON_SIZE) // 2)
            painter.drawPixmap(cursor_x, marker_y, marker_pixmap)
            cursor_x += self._MARKER_ICON_SIZE + self._ICON_GAP

        if has_multi_view:
            views_pixmap = self._views_icon.pixmap(
                self._VIEWS_ICON_SIZE, self._VIEWS_ICON_SIZE
            )
            views_y = max(0, (self._ICON_SIZE - self._VIEWS_ICON_SIZE) // 2)
            painter.drawPixmap(cursor_x, views_y, views_pixmap)
            cursor_x += self._VIEWS_ICON_SIZE + self._ICON_GAP

        icon_pixmap = self._markdown_icon.pixmap(self._ICON_SIZE, self._ICON_SIZE)
        painter.drawPixmap(cursor_x, 0, icon_pixmap)
        painter.end()

        decorated = QIcon(canvas)
        self._decorated_icon_cache[cache_key] = decorated
        return decorated


class MarkdownTreeItemDelegate(QStyledItemDelegate):
    """Paint filename-only highlight backgrounds for markdown rows."""

    def paint(self, painter: QPainter, option, index) -> None:
        model = index.model()
        if not isinstance(model, ColorizedMarkdownModel):
            super().paint(painter, option, index)
            return

        info = model.fileInfo(index)
        if not (info.isFile() and info.suffix().lower() == "md"):
            super().paint(painter, option, index)
            return

        file_path = Path(info.filePath())
        background = model.highlight_background_for_path(file_path)
        if background is None:
            super().paint(painter, option, index)
            return

        super().paint(painter, option, index)

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = opt.widget
        style = widget.style() if widget is not None else QApplication.style()

        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText, opt, widget
        )
        highlight_rect = text_rect.adjusted(-2, 1, 2, -1)
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRect(highlight_rect)
        painter.restore()

        foreground = model.highlight_foreground_for_path(file_path)
        text_color = (
            foreground
            if foreground is not None
            else opt.palette.color(QPalette.ColorRole.Text)
        )
        painter.save()
        painter.setFont(opt.font)
        painter.setPen(text_color)
        elided_text = opt.fontMetrics.elidedText(
            opt.text, opt.textElideMode, text_rect.width()
        )
        alignment = int(opt.displayAlignment | Qt.AlignmentFlag.AlignVCenter)
        painter.drawText(text_rect, alignment, elided_text)
        painter.restore()
