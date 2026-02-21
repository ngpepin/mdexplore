#!/usr/bin/env python3
"""mdexplore: fast markdown browser/editor launcher for Ubuntu."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import zlib
from collections import deque
from pathlib import Path

from markdown_it import MarkdownIt
from PySide6.QtCore import QDir, QMimeData, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QBrush, QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QFileSystemModel,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


def _encode_plantuml_source(text: str) -> str:
    """Encode PlantUML text to the compact server format."""
    # PlantUML's URL-safe encoding is custom and uses raw DEFLATE bytes.
    # This helper mirrors PlantUML's own reference algorithm so diagrams can
    # be fetched directly from a PlantUML server without local Java tooling.

    def encode_6bit(value: int) -> str:
        if value < 10:
            return chr(48 + value)
        value -= 10
        if value < 26:
            return chr(65 + value)
        value -= 26
        if value < 26:
            return chr(97 + value)
        if value == 0:
            return "-"
        if value == 1:
            return "_"
        return "?"

    def append_3bytes(b1: int, b2: int, b3: int) -> str:
        c1 = b1 >> 2
        c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
        c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
        c4 = b3 & 0x3F
        return "".join((encode_6bit(c1 & 0x3F), encode_6bit(c2 & 0x3F), encode_6bit(c3 & 0x3F), encode_6bit(c4 & 0x3F)))

    compressor = zlib.compressobj(level=9, wbits=-15)
    compressed = compressor.compress(text.encode("utf-8")) + compressor.flush()

    out = []
    for i in range(0, len(compressed), 3):
        chunk = compressed[i : i + 3]
        if len(chunk) == 3:
            out.append(append_3bytes(chunk[0], chunk[1], chunk[2]))
        elif len(chunk) == 2:
            out.append(append_3bytes(chunk[0], chunk[1], 0))
        else:
            out.append(append_3bytes(chunk[0], 0, 0))
    return "".join(out)


def _build_markdown_icon() -> QIcon:
    """Return a standard markdown icon (theme icon with a drawn fallback)."""

    def color_distance(a: QColor, b: QColor) -> int:
        return abs(a.red() - b.red()) + abs(a.green() - b.green()) + abs(a.blue() - b.blue())

    def transparentize_outer_background(pixmap: QPixmap) -> QPixmap:
        # Convert border-connected background pixels to transparent while
        # preserving the interior icon artwork.
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        w = image.width()
        h = image.height()
        if w <= 0 or h <= 0:
            return pixmap

        corners = [
            image.pixelColor(0, 0),
            image.pixelColor(w - 1, 0),
            image.pixelColor(0, h - 1),
            image.pixelColor(w - 1, h - 1),
        ]

        # If corners are already transparent, keep the source image unchanged.
        if all(c.alpha() == 0 for c in corners):
            return pixmap

        threshold = 36

        def is_background(c: QColor) -> bool:
            for bg in corners:
                if color_distance(c, bg) <= threshold:
                    return True
            return False

        visited = bytearray(w * h)
        queue: deque[tuple[int, int]] = deque()

        def push(x: int, y: int) -> None:
            idx = y * w + x
            if visited[idx]:
                return
            color = image.pixelColor(x, y)
            if not is_background(color):
                return
            visited[idx] = 1
            queue.append((x, y))

        for x in range(w):
            push(x, 0)
            push(x, h - 1)
        for y in range(h):
            push(0, y)
            push(w - 1, y)

        while queue:
            x, y = queue.popleft()
            c = image.pixelColor(x, y)
            c.setAlpha(0)
            image.setPixelColor(x, y, c)

            if x > 0:
                push(x - 1, y)
            if x + 1 < w:
                push(x + 1, y)
            if y > 0:
                push(x, y - 1)
            if y + 1 < h:
                push(x, y + 1)

        return QPixmap.fromImage(image)

    def fit_icon_canvas(pixmap: QPixmap) -> QPixmap:
        # Trim excess transparent padding and place the art on a square icon
        # canvas so desktop launchers render it at a readable size.
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        w = image.width()
        h = image.height()
        if w <= 0 or h <= 0:
            return pixmap

        minx, miny = w, h
        maxx, maxy = -1, -1
        for y in range(h):
            for x in range(w):
                if image.pixelColor(x, y).alpha() > 20:
                    if x < minx:
                        minx = x
                    if y < miny:
                        miny = y
                    if x > maxx:
                        maxx = x
                    if y > maxy:
                        maxy = y

        if maxx < minx or maxy < miny:
            return pixmap

        cropped = QPixmap.fromImage(image.copy(minx, miny, maxx - minx + 1, maxy - miny + 1))

        size = 256
        # Keep only a tiny safety inset so icon art occupies as much of the
        # launcher tile as possible while still preserving aspect ratio.
        inset = 2
        max_w = size - (2 * inset)
        max_h = size - (2 * inset)
        scaled = cropped.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        canvas = QPixmap(size, size)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        x = (size - scaled.width()) // 2
        y = (size - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
        return canvas

    for icon_name in ("mdexplor-icon.png", "mdexplor-icon.webp", "mdexplore-icon.webp"):
        icon_path = Path(__file__).resolve().with_name(icon_name)
        if icon_path.exists():
            asset_pixmap = QPixmap(str(icon_path))
            if not asset_pixmap.isNull():
                cleaned = transparentize_outer_background(asset_pixmap)
                return QIcon(fit_icon_canvas(cleaned))

    themed = QIcon.fromTheme("text-markdown")
    if not themed.isNull():
        return themed

    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#2f81f7"))
    painter.drawRoundedRect(4, 8, 56, 48, 8, 8)

    pen = QPen(QColor("#ffffff"))
    pen.setWidth(4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    # Stylized "M"
    painter.drawLine(14, 42, 14, 22)
    painter.drawLine(14, 22, 22, 34)
    painter.drawLine(22, 34, 30, 22)
    painter.drawLine(30, 22, 30, 42)

    # Down arrow
    painter.drawLine(42, 22, 42, 38)
    painter.drawLine(37, 33, 42, 38)
    painter.drawLine(47, 33, 42, 38)
    painter.end()

    return QIcon(pixmap)


class ColorizedMarkdownModel(QFileSystemModel):
    """Filesystem model with per-directory persisted file highlight colors."""

    COLOR_FILE_NAME = ".mdexplore-colors.json"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dir_color_map: dict[str, dict[str, str]] = {}
        self._loaded_dirs: set[str] = set()

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        # Paint colorized markdown rows without affecting directories.
        if role in (Qt.ItemDataRole.BackgroundRole, Qt.ItemDataRole.ForegroundRole):
            info = self.fileInfo(index)
            if info.isFile() and info.suffix().lower() == "md":
                color_name = self._color_for_file(Path(info.filePath()))
                if color_name:
                    color = QColor(color_name)
                    if color.isValid():
                        if role == Qt.ItemDataRole.BackgroundRole:
                            return QBrush(color)
                        luminance = (0.299 * color.redF()) + (0.587 * color.greenF()) + (0.114 * color.blueF())
                        return QBrush(QColor("#101418") if luminance > 0.6 else QColor("#f8fafc"))
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
                [Qt.ItemDataRole.BackgroundRole, Qt.ItemDataRole.ForegroundRole],
            )

    def collect_files_with_color(self, root: Path, color_name: str) -> list[Path]:
        """Return files under root that are persisted with the requested color."""
        if not root.is_dir():
            return []
        normalized_color = color_name.lower()
        matches: list[Path] = []

        def on_walk_error(_err) -> None:
            # Permission errors are expected in some trees; skip quietly.
            return

        for dirpath, _dirnames, _filenames in os.walk(root, onerror=on_walk_error, followlinks=False):
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

        for dirpath, _dirnames, _filenames in os.walk(root, onerror=on_walk_error, followlinks=False):
            directory = Path(dirpath)
            color_map = self._load_directory_colors(directory)
            if not color_map:
                continue
            cleared_entries += len(color_map)
            color_map.clear()
            self._save_directory_colors(directory)

        return cleared_entries

    def _color_for_file(self, path: Path) -> str | None:
        color_map = self._load_directory_colors(path.parent)
        return color_map.get(path.name)

    def _directory_key(self, directory: Path) -> str:
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
                color_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            elif color_file.exists():
                color_file.unlink()
        except Exception:
            # Requested behavior: fail quietly when persistence can't be written.
            pass


class MarkdownRenderer:
    """Converts markdown to HTML with Mermaid, MathJax, and PlantUML support."""

    def __init__(self) -> None:
        self._plantuml_server = os.environ.get("PLANTUML_SERVER", "https://www.plantuml.com/plantuml").rstrip("/")
        self._md = MarkdownIt(
            "commonmark",
            {"html": True, "linkify": True, "typographer": True},
        ).enable("table").enable("strikethrough")

        default_fence = self._md.renderer.rules["fence"]

        def custom_fence(tokens, idx, options, env):
            # Intercept known diagram fences and delegate the rest to the
            # default fenced-code renderer.
            token = tokens[idx]
            info = token.info.strip().split(maxsplit=1)[0].lower() if token.info else ""
            code = token.content

            if info == "mermaid":
                return f'<div class="mermaid">\n{html.escape(code)}\n</div>\n'

            if info in {"plantuml", "puml", "uml"}:
                encoded = _encode_plantuml_source(code)
                src = f"{self._plantuml_server}/svg/{encoded}"
                return f'<img class="plantuml" src="{src}" alt="PlantUML diagram"/>\n'

            return default_fence(tokens, idx, options, env)

        self._md.renderer.rules["fence"] = custom_fence

    def render_document(self, markdown_text: str, title: str) -> str:
        body = self._md.render(markdown_text)
        escaped_title = html.escape(title)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --fg: #1f2937;
      --bg: #f9fafb;
      --code-bg: #e5e7eb;
      --border: #d1d5db;
      --link: #0b57d0;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --fg: #e5e7eb;
        --bg: #111827;
        --code-bg: #1f2937;
        --border: #374151;
        --link: #8ab4f8;
      }}
    }}
    html, body {{
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--fg);
      font-family: "Noto Sans", "DejaVu Sans", sans-serif;
      line-height: 1.55;
      font-size: 16px;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 1.1rem 1.4rem 4rem 1.4rem;
    }}
    a {{
      color: var(--link);
    }}
    pre, code {{
      font-family: "Noto Sans Mono", "DejaVu Sans Mono", monospace;
    }}
    code {{
      background: var(--code-bg);
      border-radius: 4px;
      padding: 0.1rem 0.35rem;
    }}
    pre {{
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.8rem;
      overflow: auto;
    }}
    pre > code {{
      background: transparent;
      padding: 0;
    }}
    table {{
      border-collapse: collapse;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 0.4rem 0.6rem;
    }}
    img.plantuml {{
      display: block;
      max-width: 100%;
      margin: 0.8rem 0;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
    }}
  </style>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
</head>
<body>
  <main>{body}</main>
  <script>
    window.addEventListener('DOMContentLoaded', async () => {{
      if (window.mermaid) {{
        mermaid.initialize({{ startOnLoad: false, securityLevel: 'loose' }});
        await mermaid.run({{ querySelector: '.mermaid' }});
      }}
      if (window.MathJax && MathJax.typesetPromise) {{
        await MathJax.typesetPromise();
      }}
    }});
  </script>
</body>
</html>
"""


class MdExploreWindow(QMainWindow):
    HIGHLIGHT_COLORS = [
        ("Yellow", "#f5d34f"),
        ("Green", "#78d389"),
        ("Blue", "#7bb9ff"),
        ("Orange", "#f6a05f"),
        ("Purple", "#bb9df5"),
        ("Red", "#ef7d7d"),
    ]

    def __init__(self, root: Path, app_icon: QIcon):
        super().__init__()
        self.root = root.resolve()
        self.renderer = MarkdownRenderer()
        self.current_file: Path | None = None
        self.cache: dict[str, tuple[int, int, str]] = {}
        self._initial_split_applied = False

        self.setWindowTitle("mdexplore")
        self.setWindowIcon(app_icon)
        self.resize(1300, 860)

        # Use a custom QFileSystemModel so highlight colors render directly
        # in the tree and persist beside files in each directory.
        self.model = ColorizedMarkdownModel(self)
        self.model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)
        self.model.setNameFilters(["*.md"])
        self.model.setNameFilterDisables(False)

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setHeaderHidden(True)
        self.tree.hideColumn(1)
        self.tree.hideColumn(2)
        self.tree.hideColumn(3)
        self.tree.setColumnWidth(0, 340)
        self.tree.setMinimumWidth(240)
        self.tree.setMaximumWidth(700)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_tree_context_menu)
        self.tree.selectionModel().currentChanged.connect(self._on_tree_selection_changed)

        self.preview = QWebEngineView()

        self.up_btn = QPushButton("^")
        self.up_btn.clicked.connect(self._go_up_directory)

        quit_btn = QPushButton("Quit")
        quit_btn.clicked.connect(self.close)

        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._edit_current_file)

        self.path_label = QLabel("")
        self.path_label.setTextInteractionFlags(self.path_label.textInteractionFlags())

        copy_label = QLabel("Copy to clipboard files matching: ")
        copy_buttons_widget = QWidget()
        copy_buttons_layout = QHBoxLayout(copy_buttons_widget)
        copy_buttons_layout.setContentsMargins(0, 0, 0, 0)
        copy_buttons_layout.setSpacing(4)
        copy_buttons_layout.addWidget(copy_label)
        for color_name, color_value in self.HIGHLIGHT_COLORS:
            color_btn = QPushButton("")
            color_btn.setFixedSize(18, 18)
            color_btn.setToolTip(f"Copy files highlighted with {color_name.lower()}")
            color_btn.setStyleSheet(
                f"background-color: {color_value}; border: 1px solid #4b5563; border-radius: 3px;"
            )
            color_btn.clicked.connect(
                lambda _checked=False, c=color_value, n=color_name: self._copy_highlighted_files_to_clipboard(c, n)
            )
            copy_buttons_layout.addWidget(color_btn)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.addWidget(self.up_btn)
        top_bar.addWidget(quit_btn)
        top_bar.addWidget(edit_btn)
        top_bar.addWidget(self.path_label, 1)
        top_bar.addWidget(copy_buttons_widget, 0, Qt.AlignmentFlag.AlignRight)

        top_bar_widget = QWidget()
        top_bar_widget.setLayout(top_bar)
        top_bar_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.tree)
        self.splitter.addWidget(self.preview)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(top_bar_widget)
        layout.addWidget(self.splitter, 1)

        self.setCentralWidget(central)
        # Root is initialized after widgets exist so view/model indexes are valid.
        self._set_root_directory(self.root)
        self._add_shortcuts()
        self.model.directoryLoaded.connect(self._maybe_apply_initial_split)
        QTimer.singleShot(0, self._maybe_apply_initial_split)

    def _add_shortcuts(self) -> None:
        """Register window-level keyboard shortcuts."""
        refresh_action = QAction("Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._refresh_current_preview)
        self.addAction(refresh_action)

    def _placeholder_html(self, message: str) -> str:
        """Render an empty-state page in the preview pane."""
        escaped = html.escape(message)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <style>
    html, body {{
      margin: 0;
      height: 100%;
      background: #0f172a;
      color: #cbd5e1;
      font-family: "Noto Sans", "DejaVu Sans", sans-serif;
    }}
    main {{
      height: 100%;
      display: grid;
      place-items: center;
      font-size: 1rem;
    }}
  </style>
</head>
<body><main>{escaped}</main></body>
</html>
"""

    def _set_root_directory(self, new_root: Path) -> None:
        """Re-root the tree view and reset file preview state."""
        self.root = new_root.resolve()
        self.current_file = None
        root_index = self.model.setRootPath(str(self.root))
        self.tree.setRootIndex(root_index)
        self.tree.clearSelection()
        self.path_label.setText("Select a markdown file")
        self.preview.setHtml(
            self._placeholder_html("Select a markdown file to preview"),
            QUrl.fromLocalFile(f"{self.root}/"),
        )
        self._initial_split_applied = False
        self._update_up_button_state()
        self._update_window_title()
        QTimer.singleShot(0, self._maybe_apply_initial_split)

    def _update_up_button_state(self) -> None:
        self.up_btn.setEnabled(self.root.parent != self.root)

    def _go_up_directory(self) -> None:
        """Navigate one level up from the current root."""
        parent = self.root.parent
        if parent == self.root:
            return
        self._set_root_directory(parent)

    def _maybe_apply_initial_split(self, *_args) -> None:
        # Qt may override splitter sizes during initial layout/model load.
        # Apply the intended 25/75 split once after real geometry is known.
        if self._initial_split_applied:
            return
        total_width = max(self.splitter.width(), self.width())
        if total_width <= 0:
            return
        left_width = max(260, min(700, total_width // 4))
        right_width = max(400, total_width - left_width)
        self.splitter.setSizes([left_width, right_width])
        self._initial_split_applied = True

    def _show_tree_context_menu(self, pos) -> None:
        """Show right-click menu for assigning a file highlight color."""
        index = self.tree.indexAt(pos)
        if not index.isValid():
            return
        self.tree.setCurrentIndex(index)
        self._update_window_title()
        path = Path(self.model.filePath(index))

        menu = QMenu(self)
        color_actions: dict[QAction, str] = {}
        clear_action: QAction | None = None

        if path.is_file() and path.suffix.lower() == ".md":
            for idx, (color_name, color_value) in enumerate(self.HIGHLIGHT_COLORS):
                label = f"Highlight {color_name}" if idx == 0 else f"... {color_name}"
                action = menu.addAction(label)
                action.setData(color_value)
                color_actions[action] = color_value

            menu.addSeparator()
            clear_action = menu.addAction("Clear Highlight")

        clear_all_action = menu.addAction("Clear All")
        chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == clear_all_action:
            self._confirm_and_clear_all_highlighting()
            self.tree.viewport().update()
            return

        if clear_action is not None and chosen == clear_action:
            self.model.set_color_for_file(path, None)
        elif chosen in color_actions:
            self.model.set_color_for_file(path, color_actions[chosen])
        self.tree.viewport().update()

    def _highlight_scope_directory(self) -> Path:
        """Use selected directory as scope when possible; otherwise current root."""
        index = self.tree.currentIndex()
        if index.isValid():
            selected = Path(self.model.filePath(index))
            if selected.is_dir():
                return selected
        return self.root

    def _update_window_title(self) -> None:
        """Show the effective root in the application window title."""
        scope = self._highlight_scope_directory()
        self.setWindowTitle(f"mdexplore - {scope}")

    def _confirm_and_clear_all_highlighting(self) -> None:
        """Prompt and clear all highlight metadata recursively under current scope."""
        scope = self._highlight_scope_directory()
        reply = QMessageBox.question(
            self,
            "Clear All Highlights",
            f"Clear all file highlights recursively under:\n{scope}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        cleared = self.model.clear_all_highlights(scope)
        self.statusBar().showMessage(
            f"Cleared {cleared} highlight assignment(s) under {scope}",
            4500,
        )

    def _copy_highlighted_files_to_clipboard(self, color_value: str, color_name: str) -> None:
        """Copy highlighted file paths for a color to the system clipboard."""
        scope = self._highlight_scope_directory()
        matches = self.model.collect_files_with_color(scope, color_value)
        clipboard = QApplication.clipboard()
        mime_data = QMimeData()

        urls = [QUrl.fromLocalFile(str(path)) for path in matches]
        mime_data.setUrls(urls)

        # Nemo/Nautilus paste support: this custom format marks clipboard data
        # as file copy operations rather than plain text.
        if urls:
            gnome_payload = "copy\n" + "\n".join(url.toString() for url in urls)
            mime_data.setData("x-special/gnome-copied-files", gnome_payload.encode("utf-8"))

        # Keep plain text for editors/terminals.
        mime_data.setText("\n".join(str(path) for path in matches))
        clipboard.setMimeData(mime_data)
        self.statusBar().showMessage(
            f"Copied {len(matches)} {color_name.lower()} highlighted file(s) from {scope}",
            4000,
        )

    def _on_tree_selection_changed(self, current, _previous) -> None:
        path = Path(self.model.filePath(current))
        self._update_window_title()
        if not path.is_file() or path.suffix.lower() != ".md":
            return
        self._load_preview(path)

    def _load_preview(self, path: Path) -> None:
        # Cache rendered HTML by size+mtime to keep repeated file switches fast.
        stat = path.stat()
        cache_key = str(path.resolve())
        cached = self.cache.get(cache_key)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            html_doc = cached[2]
        else:
            markdown_text = path.read_text(encoding="utf-8", errors="replace")
            html_doc = self.renderer.render_document(markdown_text, path.name)
            self.cache[cache_key] = (stat.st_mtime_ns, stat.st_size, html_doc)

        base_url = QUrl.fromLocalFile(f"{path.parent.resolve()}/")
        self.preview.setHtml(html_doc, base_url)
        self.current_file = path
        try:
            rel = path.relative_to(self.root)
            self.path_label.setText(str(rel))
        except ValueError:
            self.path_label.setText(str(path))

    def _refresh_current_preview(self) -> None:
        """Force re-render of the currently selected file."""
        if self.current_file is None:
            return
        self.cache.pop(str(self.current_file.resolve()), None)
        self._load_preview(self.current_file)

    def _edit_current_file(self) -> None:
        """Open the selected markdown file in VS Code."""
        if self.current_file is None:
            QMessageBox.information(self, "No file selected", "Select a markdown file before using Edit.")
            return
        try:
            subprocess.Popen(["code", str(self.current_file)])
        except FileNotFoundError:
            QMessageBox.critical(
                self,
                "VS Code not found",
                "Could not find 'code' in PATH. Install VS Code or add the 'code' launcher command.",
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mdexplore",
        description="Browse and preview markdown files with rich rendering.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Root directory to browse (default: current directory).",
    )
    args = parser.parse_args()

    root = Path(args.path).expanduser()
    if not root.exists():
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"Path is not a directory: {root}", file=sys.stderr)
        return 2

    app = QApplication(sys.argv)
    app.setApplicationName("mdexplore")
    app_icon = _build_markdown_icon()
    app.setWindowIcon(app_icon)

    window = MdExploreWindow(root, app_icon)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
