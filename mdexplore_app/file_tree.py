"""Generic file-tree model and delegate helpers shared across explorers."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
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

from .file_coordination import update_files_sidecar
from .icons import load_svg_icon, ui_asset_path
from .runtime import search_hit_count_font_family


class ColorizedExtensionModel(QFileSystemModel):
    """Filesystem model with per-directory persisted file highlight colors."""

    COLOR_FILE_NAME = ".mdexplore-colors.json"
    TARGET_EXTENSION = ""
    PRIMARY_ICON_NAME = ""
    PRIMARY_ICON_COLOR = "#bcc5d1"
    VIEWS_ICON_COLOR = "#e3e7ee"
    MARKER_ICON_COLOR = "#c8b4f6"
    SYMLINK_ICON_COLOR = "#a8bcc8"
    _ICON_SIZE = 16
    _ICON_GAP = 2
    _VIEWS_ICON_SIZE = 12
    _MARKER_ICON_SIZE = 13
    _CACHED_ICON_SIZE = 12
    _SEARCH_SLOT_WIDTH = 14
    _SEARCH_COUNT_TEXT_MAX = 99
    _SEARCH_COUNT_OVERSAMPLE = 8
    _SEARCH_COUNT_TEXT_OVERSAMPLE = 12
    _PATH_KEY_CACHE_MAX = 20000
    SEARCH_FILENAME_MATCH_COLOR = "#f7e27a"
    EFFECTIVE_SCOPE_DIRECTORY_COLOR = "#7fdfe8"
    EFFECTIVE_SCOPE_DIRECTORY_SEARCH_COLOR = SEARCH_FILENAME_MATCH_COLOR
    OUT_OF_SCOPE_FILENAME_BG = "#9ca3af"
    OUT_OF_SCOPE_FILENAME_BG_ALPHA = 46

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dir_color_map: dict[str, dict[str, str]] = {}
        self._loaded_dirs: set[str] = set()
        self._search_match_counts: dict[str, int] = {}
        self._directory_search_hit_counts: dict[str, int] = {}
        self._search_filename_match_paths: set[str] = set()
        self._multi_view_paths: set[str] = set()
        self._highlighted_preview_paths: set[str] = set()
        self._cached_text_paths: set[str] = set()
        self._effective_scope_root_key: str | None = None
        self._out_of_scope_background_enabled = False
        self._path_key_cache: dict[str, str] = {}
        self._reduce_paint_cost = False
        self._primary_icon = self._load_primary_icon()
        self._symlink_icon = self._load_symlink_icon()
        self._views_icon = load_svg_icon("views2.svg", QColor(self.VIEWS_ICON_COLOR))
        self._marker_icon = load_svg_icon("marker.svg", QColor(self.MARKER_ICON_COLOR))
        self._cached_icon = self._load_cached_icon()
        self._decorated_icon_cache: dict[tuple[object, ...], QIcon] = {}

    def _load_primary_icon(self) -> QIcon:
        icon_name = str(self.PRIMARY_ICON_NAME or "").strip()
        if icon_name:
            icon = load_svg_icon(icon_name, QColor(self.PRIMARY_ICON_COLOR))
            if not icon.isNull():
                return icon
        return self._fallback_primary_icon()

    def _fallback_primary_icon(self) -> QIcon:
        return QIcon()

    def _load_symlink_icon(self) -> QIcon:
        """Load dedicated icon used for target-extension symlink rows."""
        try:
            icon_path = ui_asset_path("symlink.png")
            if icon_path.is_file():
                pixmap = QPixmap(str(icon_path))
                if pixmap.isNull():
                    return self._fallback_primary_icon()
                tint = QColor(self.SYMLINK_ICON_COLOR)
                if not tint.isValid():
                    tint = QColor("#a8bcc8")
                tinted = QPixmap(pixmap.size())
                tinted.fill(Qt.GlobalColor.transparent)
                painter = QPainter(tinted)
                painter.drawPixmap(0, 0, pixmap)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                painter.fillRect(tinted.rect(), tint)
                painter.end()
                icon = QIcon(tinted)
                if not icon.isNull():
                    return icon
        except Exception:
            pass
        return self._fallback_primary_icon()

    def _load_cached_icon(self) -> QIcon:
        """Load small marker icon used for rows with extracted text cached."""
        try:
            preferred = Path(__file__).resolve().parent.parent / "pdfexplore" / "assets" / "cached.png"
            candidates = [preferred, ui_asset_path("cached.png")]
            for icon_path in candidates:
                if not icon_path.is_file():
                    continue
                pixmap = QPixmap(str(icon_path))
                if pixmap.isNull():
                    continue
                tinted = QPixmap(pixmap.size())
                tinted.fill(Qt.GlobalColor.transparent)
                painter = QPainter(tinted)
                painter.drawPixmap(0, 0, pixmap)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                painter.fillRect(tinted.rect(), QColor("#c9ced6"))
                painter.end()
                icon = QIcon(tinted)
                if not icon.isNull():
                    return icon
        except Exception:
            pass
        return QIcon()

    def supports_symlink_primary_icon(self) -> bool:
        """Return whether symlink target files should use dedicated icon state."""
        return False

    @staticmethod
    def _is_symlink_path(path: Path) -> bool:
        """Return whether one path is a symlink, swallowing filesystem errors."""
        try:
            return path.is_symlink()
        except OSError:
            return False

    @classmethod
    def _normalized_target_extension(cls) -> str:
        value = str(cls.TARGET_EXTENSION or "").strip().lower()
        if value and not value.startswith("."):
            value = "." + value
        return value

    @classmethod
    def matches_path(cls, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() == cls._normalized_target_extension()

    @classmethod
    def matches_file_info(cls, info) -> bool:
        return bool(
            info.isFile()
            and info.suffix().lower() == cls._normalized_target_extension().lstrip(".")
        )

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if self._reduce_paint_cost:
            if role == Qt.ItemDataRole.DecorationRole:
                info = self.fileInfo(index)
                if self.matches_file_info(info):
                    # Preserve text alignment during interaction mode by keeping
                    # a full-width decoration footprint from first paint.
                    return self._decorated_primary_icon(False, False, False, 0)
            if role in {Qt.ItemDataRole.ForegroundRole, Qt.ItemDataRole.FontRole}:
                info = self.fileInfo(index)
                if self.matches_file_info(info):
                    return super().data(index, role)

        if role == Qt.ItemDataRole.DecorationRole:
            info = self.fileInfo(index)
            if self.matches_file_info(info):
                raw_path = info.filePath()
                path_key = self._path_key_from_raw_path(raw_path)
                search_hit_count = self._search_match_counts.get(path_key, 0)
                if self.supports_symlink_primary_icon() and self._is_symlink_path(Path(raw_path)):
                    return self._decorated_symlink_icon(search_hit_count)
                has_multi_view = path_key in self._multi_view_paths
                has_persistent_highlight = path_key in self._highlighted_preview_paths
                has_cached_text = path_key in self._cached_text_paths
                return self._decorated_primary_icon(
                    has_multi_view,
                    has_persistent_highlight,
                    has_cached_text,
                    search_hit_count,
                )
        if role == Qt.ItemDataRole.ForegroundRole:
            info = self.fileInfo(index)
            if info.isDir():
                scope_key = self._effective_scope_root_key
                dir_path = Path(info.filePath())
                if scope_key and self._path_key_from_raw_path(info.filePath()) == scope_key:
                    hit_count = self.search_hit_count_for_directory(dir_path)
                    color_name = (
                        self.EFFECTIVE_SCOPE_DIRECTORY_SEARCH_COLOR
                        if hit_count > 0
                        else self.EFFECTIVE_SCOPE_DIRECTORY_COLOR
                    )
                    color = QColor(color_name)
                    if color.isValid():
                        return QBrush(color)
            if self.matches_file_info(info):
                raw_path = info.filePath()
                path_key = self._path_key_from_raw_path(raw_path)
                color_name = self._color_for_file(Path(raw_path))
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
                if path_key in self._search_filename_match_paths:
                    return QBrush(QColor(self.SEARCH_FILENAME_MATCH_COLOR))
        if role == Qt.ItemDataRole.FontRole:
            info = self.fileInfo(index)
            if info.isDir():
                scope_key = self._effective_scope_root_key
                if scope_key and self._path_key_from_raw_path(info.filePath()) == scope_key:
                    base_font = super().data(index, role)
                    font = QFont(base_font) if isinstance(base_font, QFont) else QFont()
                    font.setBold(True)
                    return font
            if self.matches_file_info(info):
                raw_path = info.filePath()
                if self._search_match_counts.get(
                    self._path_key_from_raw_path(raw_path), 0
                ) > 0:
                    base_font = super().data(index, role)
                    font = QFont(base_font) if isinstance(base_font, QFont) else QFont()
                    font.setBold(True)
                    font.setItalic(True)
                    return font
        return super().data(index, role)

    def set_color_for_file(self, path: Path, color_name: str | None) -> None:
        directory = path.parent
        color_map = self._load_directory_colors(directory)
        if color_name:
            color_map[path.name] = color_name
        else:
            color_map.pop(path.name, None)
        self._save_directory_colors(directory, changed_names={path.name})

        index = self.index(str(path))
        if index.isValid():
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.ForegroundRole])

    def highlight_background_for_path(self, path: Path) -> QColor | None:
        color_name = self._color_for_file(path)
        if not color_name:
            return None
        color = QColor(color_name)
        return color if color.isValid() else None

    def highlight_foreground_for_path(self, path: Path) -> QColor | None:
        if self._path_key(path) in self._search_filename_match_paths:
            return QColor(self.SEARCH_FILENAME_MATCH_COLOR)
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
        if not root.is_dir():
            return []
        normalized_color = color_name.lower()
        matches: list[Path] = []

        def on_walk_error(_err) -> None:
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
                    if self.matches_path(candidate):
                        matches.append(candidate.resolve())
                except Exception:
                    pass

        matches.sort(key=str)
        return matches

    def clear_all_highlights(self, root: Path) -> int:
        if not root.is_dir():
            return 0

        cleared_entries = 0

        def on_walk_error(_err) -> None:
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
        return self._color_for_file(path)

    def set_search_match_paths(self, paths: set[Path]) -> None:
        self.set_search_match_counts({path: 1 for path in paths})

    def set_search_match_counts(
        self,
        match_counts: dict[Path, int],
        *,
        filename_match_path_keys: set[str] | None = None,
        directory_match_paths: Iterable[Path] | None = None,
    ) -> None:
        previous_counts = self._search_match_counts
        previous_directory_counts = self._directory_search_hit_counts
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

        # File-row state uses canonical identity so symlink aliases share the
        # same extracted content and hit count. Directory aggregation uses the
        # lexical path shown in the tree, however: walking a symlink's resolved
        # target parents would put counts under an unrelated source directory
        # instead of under the visible alias folder. Keeping these keys lexical
        # also lets dataChanged address a symlink-directory index directly,
        # rather than waiting for a later click/scroll to repaint that row.
        directory_sources: list[Path] = []
        covered_canonical_keys: set[str] = set()
        seen_display_keys: set[str] = set()
        if directory_match_paths is not None:
            for raw_path in directory_match_paths:
                path = Path(raw_path)
                canonical_key = self._path_key(path)
                if canonical_key not in next_counts:
                    continue
                display_key = self._path_identity_without_io(path)
                if display_key in seen_display_keys:
                    continue
                seen_display_keys.add(display_key)
                covered_canonical_keys.add(canonical_key)
                directory_sources.append(Path(display_key))
        for canonical_key in next_counts:
            if canonical_key not in covered_canonical_keys:
                directory_sources.append(Path(canonical_key))

        directory_counts: dict[str, int] = {}
        for source_path in directory_sources:
            directory = source_path.parent
            while True:
                directory_key = self._path_identity_without_io(directory)
                directory_counts[directory_key] = (
                    directory_counts.get(directory_key, 0) + 1
                )
                parent = directory.parent
                if parent == directory:
                    break
                directory = parent
        self._directory_search_hit_counts = directory_counts
        if filename_match_path_keys is None:
            self._search_filename_match_paths = set(next_counts.keys())
        else:
            self._search_filename_match_paths = {
                str(path_key)
                for path_key in filename_match_path_keys
                if isinstance(path_key, str) and path_key in next_counts
            }
        self._decorated_icon_cache.clear()
        self._emit_search_state_changes(
            previous_counts,
            previous_directory_counts,
        )

    def _emit_search_state_changes(
        self,
        previous_counts: dict[str, int],
        previous_directory_counts: dict[str, int],
    ) -> None:
        """Invalidate loaded rows whose file or directory search state changed."""
        changed_file_keys = {
            path_key
            for path_key in set(previous_counts) | set(self._search_match_counts)
            if previous_counts.get(path_key, 0)
            != self._search_match_counts.get(path_key, 0)
        }
        changed_directory_keys = {
            path_key
            for path_key in set(previous_directory_counts)
            | set(self._directory_search_hit_counts)
            if previous_directory_counts.get(path_key, 0)
            != self._directory_search_hit_counts.get(path_key, 0)
        }

        file_roles = [
            Qt.ItemDataRole.DecorationRole,
            Qt.ItemDataRole.ForegroundRole,
            Qt.ItemDataRole.FontRole,
        ]
        directory_roles = [
            Qt.ItemDataRole.DecorationRole,
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.ForegroundRole,
            Qt.ItemDataRole.FontRole,
        ]
        for path_key in changed_file_keys:
            index = self.index(path_key)
            if index.isValid():
                self.dataChanged.emit(index, index, file_roles)
        for path_key in changed_directory_keys:
            index = self.index(path_key)
            if index.isValid():
                self.dataChanged.emit(index, index, directory_roles)

    def clear_search_match_paths(self) -> None:
        if (
            not self._search_match_counts
            and not self._directory_search_hit_counts
            and not self._search_filename_match_paths
        ):
            return
        previous_counts = self._search_match_counts
        previous_directory_counts = self._directory_search_hit_counts
        self._search_match_counts = {}
        self._directory_search_hit_counts = {}
        self._search_filename_match_paths.clear()
        self._decorated_icon_cache.clear()
        self._emit_search_state_changes(
            previous_counts,
            previous_directory_counts,
        )

    def set_multi_view_paths(self, paths: set[Path]) -> None:
        next_paths = {self._path_key(path) for path in paths}
        if next_paths == self._multi_view_paths:
            return
        self._multi_view_paths = next_paths
        self._decorated_icon_cache.clear()

    def set_multi_view_path_keys(self, path_keys: set[str]) -> None:
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
        next_paths = {self._path_key(path) for path in paths}
        if next_paths == self._highlighted_preview_paths:
            return
        self._highlighted_preview_paths = next_paths
        self._decorated_icon_cache.clear()

    def set_persistent_highlight_path_keys(self, path_keys: set[str]) -> None:
        next_paths = {str(path_key) for path_key in path_keys if isinstance(path_key, str)}
        if next_paths == self._highlighted_preview_paths:
            return
        self._highlighted_preview_paths = next_paths
        self._decorated_icon_cache.clear()

    def clear_persistent_highlight_paths(self) -> None:
        if not self._highlighted_preview_paths:
            return
        self._highlighted_preview_paths.clear()
        self._decorated_icon_cache.clear()

    def set_cached_path_keys(self, path_keys: set[str]) -> bool:
        next_paths = {
            str(path_key)
            for path_key in path_keys
            if isinstance(path_key, str)
        }
        if next_paths == self._cached_text_paths:
            return False
        self._cached_text_paths = next_paths
        self._decorated_icon_cache.clear()
        return True

    def clear_cached_path_keys(self) -> bool:
        if not self._cached_text_paths:
            return False
        self._cached_text_paths.clear()
        self._decorated_icon_cache.clear()
        return True

    def set_effective_scope_directory(self, directory: Path | None) -> bool:
        """Set active scope root used for out-of-scope row shading."""
        next_key: str | None = None
        if isinstance(directory, Path):
            next_key = self._path_key(directory)
        if next_key == self._effective_scope_root_key:
            return False
        self._effective_scope_root_key = next_key
        return True

    def set_reduce_paint_cost(self, enabled: bool) -> bool:
        next_value = bool(enabled)
        if next_value == self._reduce_paint_cost:
            return False
        self._reduce_paint_cost = next_value
        return True

    def is_reduce_paint_cost_enabled(self) -> bool:
        return bool(self._reduce_paint_cost)

    def search_hit_count_for_directory(self, directory: Path) -> int:
        """Return the number of matching descendant files for any directory."""
        display_key = self._path_identity_without_io(directory)
        count = self._directory_search_hit_counts.get(display_key)
        if count is None:
            # Retain compatibility for callers that pass an already-resolved
            # regular directory while keeping visible symlink aliases distinct.
            count = self._directory_search_hit_counts.get(self._path_key(directory), 0)
        return max(
            0,
            int(count),
        )

    def effective_scope_search_hit_count_for_directory(self, directory: Path) -> int:
        """Return the aggregated count only when directory is the active scope."""
        scope_key = self._effective_scope_root_key
        if not scope_key or self._path_key(directory) != scope_key:
            return 0
        return self.search_hit_count_for_directory(directory)

    def set_out_of_scope_background_enabled(self, enabled: bool) -> bool:
        """Toggle faint out-of-scope file backgrounds for non-effective-root rows."""
        next_value = bool(enabled)
        if next_value == self._out_of_scope_background_enabled:
            return False
        self._out_of_scope_background_enabled = next_value
        return True

    def out_of_scope_background_for_path(self, path: Path) -> QColor | None:
        """Return faint background color for visible files outside active scope."""
        if not self._out_of_scope_background_enabled:
            return None
        scope_key = self._effective_scope_root_key
        if not scope_key:
            return None
        path_key = self._path_key(path)
        if path_key == scope_key:
            return None
        scope_prefix = scope_key + os.sep
        if path_key.startswith(scope_prefix):
            return None
        color = QColor(self.OUT_OF_SCOPE_FILENAME_BG)
        if not color.isValid():
            return None
        color.setAlpha(max(0, min(255, int(self.OUT_OF_SCOPE_FILENAME_BG_ALPHA))))
        return color

    def _path_key(self, path: Path) -> str:
        return self._path_key_from_raw_path(str(path))

    @staticmethod
    def _path_identity_without_io(path: Path) -> str:
        """Return an absolute lexical identity without resolving symlinks."""
        try:
            return os.path.normcase(os.path.abspath(os.fspath(path)))
        except Exception:
            return str(path)

    def _path_key_from_raw_path(self, raw_path: str) -> str:
        cached = self._path_key_cache.get(raw_path)
        if cached is not None:
            return cached
        try:
            resolved = str(Path(raw_path).resolve())
        except Exception:
            resolved = str(raw_path)
        if len(self._path_key_cache) >= int(self._PATH_KEY_CACHE_MAX):
            self._path_key_cache.clear()
        self._path_key_cache[str(raw_path)] = resolved
        return resolved

    def _directory_key(self, directory: Path) -> str:
        return self._path_key_from_raw_path(str(directory))

    def _load_directory_colors(self, directory: Path) -> dict[str, str]:
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
            pass

        self._dir_color_map[key] = color_map
        return color_map

    def invalidate_persisted_color_cache(self) -> None:
        """Forget sidecar snapshots so another process's changes are reloaded."""
        self._loaded_dirs.clear()
        self._dir_color_map.clear()

    def _save_directory_colors(
        self,
        directory: Path,
        *,
        changed_names: set[str] | None = None,
    ) -> None:
        key = self._directory_key(directory)
        color_map = self._dir_color_map.get(key, {})
        color_file = directory / self.COLOR_FILE_NAME
        try:
            replace_all = changed_names is None
            names = set(color_map) if changed_names is None else set(changed_names)
            updates: dict[str, object | None] = {
                name: color_map.get(name)
                for name in names
            }
            committed = update_files_sidecar(
                color_file,
                updates,
                replace_all=replace_all,
            )
            self._dir_color_map[key] = {
                name: value
                for name, value in committed.items()
                if isinstance(name, str) and isinstance(value, str)
            }
            self._loaded_dirs.add(key)
        except Exception:
            pass

    @classmethod
    def decorated_icon_size(cls) -> QSize:
        return QSize(cls.max_decoration_width(), cls._ICON_SIZE)

    @classmethod
    def max_decoration_width(cls) -> int:
        widths = [
            cls._SEARCH_SLOT_WIDTH,
            cls._CACHED_ICON_SIZE,
            cls._MARKER_ICON_SIZE,
            cls._VIEWS_ICON_SIZE,
            cls._ICON_SIZE,
        ]
        return sum(widths) + (cls._ICON_GAP * max(0, len(widths) - 1))

    def _search_count_display_text(self, search_hit_count: int) -> str:
        if search_hit_count <= 0:
            return ""
        if search_hit_count <= self._SEARCH_COUNT_TEXT_MAX:
            return str(search_hit_count)
        return "++"

    def decoration_size_for_state(
        self,
        has_multi_view: bool,
        has_persistent_highlight: bool,
        has_cached_text: bool,
        search_hit_count: int,
    ) -> QSize:
        search_count_text = self._search_count_display_text(search_hit_count)
        widths: list[int] = []
        if search_count_text:
            widths.append(self._SEARCH_SLOT_WIDTH)
        if has_cached_text and not self._cached_icon.isNull():
            widths.append(self._CACHED_ICON_SIZE)
        if has_persistent_highlight:
            widths.append(self._MARKER_ICON_SIZE)
        if has_multi_view:
            widths.append(self._VIEWS_ICON_SIZE)
        widths.append(self._ICON_SIZE)
        total_width = sum(widths) + (self._ICON_GAP * max(0, len(widths) - 1))
        return QSize(total_width, self._ICON_SIZE)

    def decoration_size_for_index(self, index) -> QSize:
        info = self.fileInfo(index)
        if not self.matches_file_info(info):
            return QSize(self._ICON_SIZE, self._ICON_SIZE)
        if self._reduce_paint_cost:
            return QSize(self.max_decoration_width(), self._ICON_SIZE)
        raw_path = info.filePath()
        path_key = self._path_key_from_raw_path(raw_path)
        search_hit_count = self._search_match_counts.get(path_key, 0)
        if self.supports_symlink_primary_icon() and self._is_symlink_path(Path(raw_path)):
            widths: list[int] = []
            if self._search_count_display_text(search_hit_count):
                widths.append(self._SEARCH_SLOT_WIDTH)
            widths.append(self._ICON_SIZE)
            return QSize(
                sum(widths) + (self._ICON_GAP * max(0, len(widths) - 1)),
                self._ICON_SIZE,
            )
        has_multi_view = path_key in self._multi_view_paths
        has_persistent_highlight = path_key in self._highlighted_preview_paths
        has_cached_text = path_key in self._cached_text_paths
        return self.decoration_size_for_state(
            has_multi_view,
            has_persistent_highlight,
            has_cached_text,
            search_hit_count,
        )

    def _decorated_primary_icon(
        self,
        has_multi_view: bool,
        has_persistent_highlight: bool,
        has_cached_text: bool,
        search_hit_count: int,
    ) -> QIcon:
        search_count_text = self._search_count_display_text(search_hit_count)
        show_cached_icon = has_cached_text and not self._cached_icon.isNull()
        cache_key = (
            "primary",
            has_multi_view,
            has_persistent_highlight,
            show_cached_icon,
            search_count_text,
        )
        cached = self._decorated_icon_cache.get(cache_key)
        if cached is not None:
            return cached

        cluster_size = self.decoration_size_for_state(
            has_multi_view,
            has_persistent_highlight,
            show_cached_icon,
            search_hit_count,
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

        if show_cached_icon:
            cached_pixmap = self._cached_icon.pixmap(
                self._CACHED_ICON_SIZE,
                self._CACHED_ICON_SIZE,
            )
            cached_y = max(0, (self._ICON_SIZE - self._CACHED_ICON_SIZE) // 2)
            painter.drawPixmap(cursor_x, cached_y, cached_pixmap)
            cursor_x += self._CACHED_ICON_SIZE + self._ICON_GAP

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

        if not self._primary_icon.isNull():
            icon_pixmap = self._primary_icon.pixmap(self._ICON_SIZE, self._ICON_SIZE)
            painter.drawPixmap(cursor_x, 0, icon_pixmap)
        painter.end()

        decorated = QIcon(canvas)
        self._decorated_icon_cache[cache_key] = decorated
        return decorated

    def _decorated_symlink_icon(self, search_hit_count: int) -> QIcon:
        """Return symlink-only decoration cluster with optional search-hit pill."""
        search_count_text = self._search_count_display_text(search_hit_count)
        cache_key = ("symlink", search_count_text)
        cached = self._decorated_icon_cache.get(cache_key)
        if cached is not None:
            return cached

        widths: list[int] = []
        if search_count_text:
            widths.append(self._SEARCH_SLOT_WIDTH)
        widths.append(self._ICON_SIZE)
        cluster_width = sum(widths) + (self._ICON_GAP * max(0, len(widths) - 1))

        total_width = self.max_decoration_width()
        total_height = self._ICON_SIZE
        canvas = QPixmap(total_width, total_height)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        cursor_x = max(0, total_width - cluster_width)
        if search_count_text:
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

        if not self._symlink_icon.isNull():
            icon_pixmap = self._symlink_icon.pixmap(self._ICON_SIZE, self._ICON_SIZE)
            painter.drawPixmap(cursor_x, 0, icon_pixmap)
        painter.end()

        decorated = QIcon(canvas)
        self._decorated_icon_cache[cache_key] = decorated
        return decorated


class ExtensionTreeItemDelegate(QStyledItemDelegate):
    """Paint filename-only highlight backgrounds for target-extension rows."""

    _DIR_SEARCH_PILL_BG = "#f3c56b"
    _DIR_SEARCH_PILL_FG = "#111111"
    _DIR_SEARCH_PILL_GAP = 8

    def paint(self, painter: QPainter, option, index) -> None:
        model = index.model()
        if not isinstance(model, ColorizedExtensionModel):
            super().paint(painter, option, index)
            return

        info = model.fileInfo(index)
        if info.isDir():
            directory_path = Path(info.filePath())
            directory_hit_count = model.search_hit_count_for_directory(directory_path)
            if directory_hit_count > 0:
                self._paint_directory_with_search_count(
                    painter,
                    option,
                    index,
                    model,
                    directory_hit_count,
                )
                return

        if model.is_reduce_paint_cost_enabled():
            super().paint(painter, option, index)
            return

        if not model.matches_file_info(info):
            super().paint(painter, option, index)
            return

        file_path = Path(info.filePath())
        out_of_scope_background = model.out_of_scope_background_for_path(file_path)
        background = model.highlight_background_for_path(file_path)
        if out_of_scope_background is None and background is None:
            super().paint(painter, option, index)
            return

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = opt.widget
        style = widget.style() if widget is not None else QApplication.style()

        # Draw the standard row chrome/icon once, but suppress built-in text so
        # custom background rows do not render duplicated glyphs.
        base_opt = QStyleOptionViewItem(opt)
        base_opt.text = ""
        # Some Qt styles skip decorations when display text is suppressed.
        # Clear decoration in the style pass and paint it explicitly below.
        base_opt.icon = QIcon()
        base_opt.features &= ~QStyleOptionViewItem.ViewItemFeature.HasDecoration
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, base_opt, painter, widget)

        if not opt.icon.isNull():
            decoration_rect = style.subElementRect(
                QStyle.SubElement.SE_ItemViewItemDecoration, opt, widget
            )
            if decoration_rect.isValid() and decoration_rect.width() > 0:
                icon_mode = (
                    QIcon.Mode.Disabled
                    if not (opt.state & QStyle.StateFlag.State_Enabled)
                    else (
                        QIcon.Mode.Selected
                        if (opt.state & QStyle.StateFlag.State_Selected)
                        else QIcon.Mode.Normal
                    )
                )
                icon_state = (
                    QIcon.State.On
                    if (opt.state & QStyle.StateFlag.State_Open)
                    else QIcon.State.Off
                )
                pixmap_size = (
                    opt.decorationSize
                    if opt.decorationSize.isValid()
                    else QSize(16, 16)
                )
                decoration_pixmap = opt.icon.pixmap(
                    pixmap_size,
                    icon_mode,
                    icon_state,
                )
                if not decoration_pixmap.isNull():
                    draw_x = decoration_rect.x() + max(
                        0, (decoration_rect.width() - decoration_pixmap.width()) // 2
                    )
                    draw_y = decoration_rect.y() + max(
                        0, (decoration_rect.height() - decoration_pixmap.height()) // 2
                    )
                    painter.drawPixmap(draw_x, draw_y, decoration_pixmap)

        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText, opt, widget
        )
        highlight_rect = text_rect.adjusted(-2, 1, 2, -1)
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        if out_of_scope_background is not None:
            painter.setBrush(out_of_scope_background)
            painter.drawRect(highlight_rect)
        if background is not None:
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

    @staticmethod
    def _draw_item_decoration(
        painter: QPainter,
        opt: QStyleOptionViewItem,
        style: QStyle,
        widget,
    ) -> None:
        """Draw item icon/decoration after a text-suppressed style pass."""
        if opt.icon.isNull():
            return
        decoration_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemDecoration, opt, widget
        )
        if not decoration_rect.isValid() or decoration_rect.width() <= 0:
            return
        icon_mode = (
            QIcon.Mode.Disabled
            if not (opt.state & QStyle.StateFlag.State_Enabled)
            else (
                QIcon.Mode.Selected
                if (opt.state & QStyle.StateFlag.State_Selected)
                else QIcon.Mode.Normal
            )
        )
        icon_state = (
            QIcon.State.On if (opt.state & QStyle.StateFlag.State_Open) else QIcon.State.Off
        )
        pixmap_size = opt.decorationSize if opt.decorationSize.isValid() else QSize(16, 16)
        decoration_pixmap = opt.icon.pixmap(pixmap_size, icon_mode, icon_state)
        if decoration_pixmap.isNull():
            return
        draw_x = decoration_rect.x() + max(
            0, (decoration_rect.width() - decoration_pixmap.width()) // 2
        )
        draw_y = decoration_rect.y() + max(
            0, (decoration_rect.height() - decoration_pixmap.height()) // 2
        )
        painter.drawPixmap(draw_x, draw_y, decoration_pixmap)

    def _paint_directory_with_search_count(
        self,
        painter: QPainter,
        option,
        index,
        model: ColorizedExtensionModel,
        directory_hit_count: int,
    ) -> None:
        """Paint any directory row with its appended matching-file-count pill."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = opt.widget
        style = widget.style() if widget is not None else QApplication.style()

        base_opt = QStyleOptionViewItem(opt)
        base_opt.text = ""
        base_opt.icon = QIcon()
        base_opt.features &= ~QStyleOptionViewItem.ViewItemFeature.HasDecoration
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, base_opt, painter, widget)
        self._draw_item_decoration(painter, opt, style, widget)

        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText, opt, widget
        )
        if not text_rect.isValid() or text_rect.width() <= 2:
            return

        count_text = model._search_count_display_text(int(directory_hit_count))
        if not count_text:
            count_text = "1"

        pill_font = QFont(opt.font)
        pill_font.setFamily(search_hit_count_font_family())
        pill_font.setBold(False)
        pill_metrics = QFontMetrics(pill_font)
        pill_text_width = pill_metrics.horizontalAdvance(count_text)
        pill_height = max(14, min(text_rect.height() - 2, pill_metrics.height() + 4))
        pill_padding_x = max(6, int(pill_height * 0.42))
        pill_width = pill_text_width + (2 * pill_padding_x)
        pill_top = text_rect.y() + max(0, (text_rect.height() - pill_height) // 2)
        fitted_pill_width = min(pill_width, text_rect.width())
        maximum_pill_left = max(
            text_rect.left(),
            text_rect.right() - fitted_pill_width + 1,
        )
        natural_label_width = max(0, opt.fontMetrics.horizontalAdvance(opt.text))
        preferred_pill_left = (
            text_rect.left()
            + min(
                natural_label_width,
                max(
                    0,
                    text_rect.width()
                    - fitted_pill_width
                    - self._DIR_SEARCH_PILL_GAP,
                ),
            )
            + self._DIR_SEARCH_PILL_GAP
        )
        pill_left = min(maximum_pill_left, preferred_pill_left)
        pill_rect = QRect(
            pill_left,
            pill_top,
            fitted_pill_width,
            pill_height,
        )

        label_right = max(text_rect.left(), pill_rect.left() - self._DIR_SEARCH_PILL_GAP)
        label_rect = QRect(
            text_rect.left(),
            text_rect.y(),
            max(0, label_right - text_rect.left()),
            text_rect.height(),
        )

        foreground_data = model.data(index, Qt.ItemDataRole.ForegroundRole)
        if isinstance(foreground_data, QBrush):
            label_color = foreground_data.color()
        else:
            label_color = (
                opt.palette.color(QPalette.ColorRole.HighlightedText)
                if (opt.state & QStyle.StateFlag.State_Selected)
                else opt.palette.color(QPalette.ColorRole.Text)
            )

        painter.save()
        painter.setFont(opt.font)
        painter.setPen(label_color)
        label_text = opt.fontMetrics.elidedText(
            opt.text, opt.textElideMode, max(0, label_rect.width())
        )
        alignment = int(opt.displayAlignment | Qt.AlignmentFlag.AlignVCenter)
        painter.drawText(label_rect, alignment, label_text)
        painter.restore()

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._DIR_SEARCH_PILL_BG))
        radius = max(4, int(pill_rect.height() * 0.28))
        painter.drawRoundedRect(pill_rect, radius, radius)
        painter.setPen(QColor(self._DIR_SEARCH_PILL_FG))
        painter.setFont(pill_font)
        painter.drawText(pill_rect, int(Qt.AlignmentFlag.AlignCenter), count_text)
        painter.restore()
