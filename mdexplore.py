#!/usr/bin/env python3
"""mdexplore: fast markdown browser/editor launcher for Ubuntu."""

from __future__ import annotations

import argparse
import base64
import html
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

from markdown_it import MarkdownIt
from PySide6.QtCore import QDir, QMimeData, QObject, QRunnable, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QBrush, QClipboard, QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QFileSystemModel,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

CONFIG_FILE_NAME = ".mdexplore.cfg"


def _config_file_path() -> Path:
    return Path.home() / CONFIG_FILE_NAME


def _load_default_root_from_config() -> Path:
    """Resolve default root when no CLI path is provided."""
    fallback = Path.home()
    cfg_path = _config_file_path()
    try:
        if not cfg_path.exists():
            return fallback
        raw = cfg_path.read_text(encoding="utf-8").strip()
        if not raw:
            return fallback
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
    except Exception:
        # Any read/parse/access issue should fall back to home directory.
        pass
    return fallback


def _extract_plantuml_error_details(stderr_text: str) -> str:
    """Parse PlantUML stderr into a readable, more detailed message."""
    raw = (stderr_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    if not lines:
        return "unknown error"

    # Common PlantUML stderr shape:
    # ERROR
    # <line-number>
    # <message>
    if len(lines) >= 3 and lines[0].upper() == "ERROR" and lines[1].isdigit():
        return f"line {lines[1]}: {lines[2]}"

    # Fallback: keep the first few lines for context.
    return "\n".join(lines[:8])


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
                print(f"mdexplore icon source: {icon_path}", file=sys.stderr)
                return QIcon(fit_icon_canvas(cleaned))

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

    print("mdexplore icon source: built-in fallback", file=sys.stderr)
    return QIcon(pixmap)


def _build_clear_x_icon() -> QIcon:
    """Return a small, explicit X icon for the search-field clear action."""
    pixmap = QPixmap(14, 14)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor("#9ca3af"))
    pen.setWidth(2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawLine(3, 3, 11, 11)
    painter.drawLine(11, 3, 3, 11)
    painter.end()
    return QIcon(pixmap)


class ColorizedMarkdownModel(QFileSystemModel):
    """Filesystem model with per-directory persisted file highlight colors."""

    COLOR_FILE_NAME = ".mdexplore-colors.json"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dir_color_map: dict[str, dict[str, str]] = {}
        self._loaded_dirs: set[str] = set()
        self._search_match_paths: set[str] = set()

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
        if role == Qt.ItemDataRole.FontRole:
            info = self.fileInfo(index)
            if info.isFile() and info.suffix().lower() == "md":
                if self._path_key(Path(info.filePath())) in self._search_match_paths:
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

    def set_search_match_paths(self, paths: set[Path]) -> None:
        self._search_match_paths = {self._path_key(path) for path in paths}

    def clear_search_match_paths(self) -> None:
        self._search_match_paths.clear()

    def _path_key(self, path: Path) -> str:
        try:
            return str(path.resolve())
        except Exception:
            return str(path)

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
        self._plantuml_jar_path = self._resolve_plantuml_jar_path()
        self._plantuml_setup_issue = self._plantuml_setup_error()
        self._plantuml_svg_cache: dict[str, str] = {}
        self._md = MarkdownIt(
            "commonmark",
            {"html": True, "linkify": True, "typographer": True},
        ).enable("table").enable("strikethrough")

        default_fence = self._md.renderer.rules["fence"]
        default_render_token = self._md.renderer.renderToken

        def custom_fence(tokens, idx, options, env):
            # Intercept known diagram fences and delegate the rest to the
            # default fenced-code renderer.
            token = tokens[idx]
            info = token.info.strip().split(maxsplit=1)[0].lower() if token.info else ""
            code = token.content
            line_attrs = ""
            if token.map and len(token.map) == 2:
                line_attrs = f' data-md-line-start="{token.map[0]}" data-md-line-end="{token.map[1]}"'

            if info == "mermaid":
                return (
                    f'<div class="mdexplore-fence"{line_attrs}>'
                    f'<div class="mermaid">\n{html.escape(code)}\n</div>'
                    "</div>\n"
                )

            if info in {"plantuml", "puml", "uml"}:
                resolver = env.get("plantuml_resolver") if isinstance(env, dict) else None
                if callable(resolver):
                    plantuml_index = int(env.get("plantuml_index", 0))
                    env["plantuml_index"] = plantuml_index + 1
                    return resolver(code, plantuml_index, line_attrs)

                data_uri, error_message = self._render_plantuml_data_uri(code)
                if data_uri is not None:
                    return (
                        f'<div class="mdexplore-fence"{line_attrs}>'
                        f'<img class="plantuml" src="{data_uri}" alt="PlantUML diagram"/>'
                        "</div>\n"
                    )

                escaped_error = html.escape(error_message or "PlantUML rendering failed")
                escaped_code = html.escape(code)
                return (
                    f'<div class="mdexplore-fence plantuml-error"{line_attrs}>'
                    f'<div class="plantuml-error-message">{escaped_error}</div>'
                    f'<pre><code class="language-plantuml">{escaped_code}</code></pre>'
                    "</div>\n"
                )

            return default_fence(tokens, idx, options, env)

        def custom_render_token(tokens, idx, options, env):
            # Attach source-line metadata so preview selections can map back
            # to source markdown ranges for copy operations.
            token = tokens[idx]
            if token.nesting == 1 and token.type.endswith("_open") and token.map and len(token.map) == 2:
                token.attrSet("data-md-line-start", str(token.map[0]))
                token.attrSet("data-md-line-end", str(token.map[1]))
            return default_render_token(tokens, idx, options, env)

        self._md.renderer.rules["fence"] = custom_fence
        self._md.renderer.renderToken = custom_render_token

    def _resolve_plantuml_jar_path(self) -> Path | None:
        """Locate plantuml.jar from env, app directory, or current directory."""
        env_value = os.environ.get("PLANTUML_JAR", "").strip()
        candidates: list[Path] = []
        if env_value:
            candidates.append(Path(env_value).expanduser())
        app_dir = Path(__file__).resolve().parent
        candidates.append(app_dir / "plantuml.jar")
        candidates.append(Path.cwd() / "plantuml.jar")

        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate.resolve()
            except Exception:
                continue
        return None

    def _plantuml_setup_error(self) -> str | None:
        """Return setup error text when local PlantUML execution is unavailable."""
        if self._plantuml_jar_path is None:
            return "plantuml.jar not found (set PLANTUML_JAR or place jar next to mdexplore.py)"
        if shutil.which("java") is None:
            return "Java runtime not found in PATH; install Java to render PlantUML diagrams"
        return None

    def _render_plantuml_data_uri(self, code: str) -> tuple[str | None, str | None]:
        """Render PlantUML source locally and return an SVG data URI."""
        prepared_code = self._prepare_plantuml_source(code)
        cache_key = hashlib.sha1(prepared_code.encode("utf-8", errors="replace")).hexdigest()
        cached = self._plantuml_svg_cache.get(cache_key)
        if cached is not None:
            return cached, None

        if self._plantuml_setup_issue is not None:
            return None, self._plantuml_setup_issue
        if self._plantuml_jar_path is None:
            return None, "plantuml.jar not available"

        command = [
            "java",
            "-Djava.awt.headless=true",
            "-jar",
            str(self._plantuml_jar_path),
            "-pipe",
            "-tsvg",
            "-charset",
            "UTF-8",
        ]

        try:
            result = subprocess.run(
                command,
                input=prepared_code,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            return None, "Local PlantUML render timed out"
        except Exception as exc:
            return None, f"Local PlantUML render failed: {exc}"

        if result.returncode != 0:
            details = _extract_plantuml_error_details(result.stderr or "")
            return None, f"Local PlantUML render failed: {details}"

        svg_text = (result.stdout or "").strip()
        if "<svg" not in svg_text.casefold():
            return None, "Local PlantUML did not return SVG output"

        encoded_svg = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
        data_uri = f"data:image/svg+xml;base64,{encoded_svg}"
        self._plantuml_svg_cache[cache_key] = data_uri
        return data_uri, None

    @staticmethod
    def _prepare_plantuml_source(code: str) -> str:
        """Normalize PlantUML fence content for local jar rendering."""
        normalized = code.replace("\r\n", "\n").strip("\n")
        if not normalized:
            return "@startuml\n@enduml\n"

        has_start_directive = any(
            line.strip().casefold().startswith("@start")
            for line in normalized.splitlines()
            if line.strip()
        )
        if has_start_directive:
            return normalized + "\n"

        # Support shorthand fenced blocks that omit @startuml/@enduml.
        return f"@startuml\n{normalized}\n@enduml\n"

    def render_document(self, markdown_text: str, title: str, plantuml_resolver=None) -> str:
        # `env` is passed through markdown-it and lets fence renderers call back
        # into window-level async PlantUML orchestration when available.
        env = {}
        if callable(plantuml_resolver):
            env["plantuml_resolver"] = plantuml_resolver
            env["plantuml_index"] = 0
        body = self._md.render(markdown_text, env)
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
    .plantuml-error-message {{
      margin: 0.8rem 0 0.4rem 0;
      color: #b91c1c;
      font-weight: 600;
    }}
    pre.plantuml-error-detail {{
      margin: 0 0 0.8rem 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: #b91c1c;
      background: color-mix(in srgb, var(--bg) 85%, #ef4444 15%);
      border: 1px solid #ef9a9a;
      border-radius: 6px;
      padding: 0.55rem 0.7rem;
      font-size: 0.92rem;
    }}
    .plantuml-pending {{
      margin: 0.8rem 0;
      padding: 0.55rem 0.7rem;
      border: 1px dashed var(--border);
      border-radius: 6px;
      font-style: italic;
      opacity: 0.9;
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
      // Mermaid/MathJax run after markdown HTML is mounted.
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


class PreviewRenderWorkerSignals(QObject):
    """Signals emitted by background preview rendering workers."""

    finished = Signal(int, str, str, object, object, str)


class PreviewRenderWorker(QRunnable):
    """Render markdown HTML in a worker thread to keep UI responsive."""

    def __init__(self, path: Path, request_id: int):
        super().__init__()
        self.path = path
        self.request_id = request_id
        self.signals = PreviewRenderWorkerSignals()

    def run(self) -> None:
        try:
            # Keep this fully self-contained so it is safe to run off-thread.
            resolved = self.path.resolve()
            stat = resolved.stat()
            markdown_text = resolved.read_text(encoding="utf-8", errors="replace")
            renderer = MarkdownRenderer()
            html_doc = renderer.render_document(markdown_text, resolved.name)
            self.signals.finished.emit(
                self.request_id,
                str(resolved),
                html_doc,
                stat.st_mtime_ns,
                stat.st_size,
                "",
            )
        except Exception as exc:
            path_text = str(self.path)
            self.signals.finished.emit(self.request_id, path_text, "", 0, 0, str(exc))


class PlantUmlRenderWorkerSignals(QObject):
    """Signals emitted by background PlantUML render workers."""

    finished = Signal(str, str, str)


class PlantUmlRenderWorker(QRunnable):
    """Render one PlantUML source block to SVG data URI in background."""

    def __init__(self, hash_key: str, prepared_code: str, jar_path: Path | None, setup_issue: str | None):
        super().__init__()
        self.hash_key = hash_key
        self.prepared_code = prepared_code
        self.jar_path = jar_path
        self.setup_issue = setup_issue
        self.signals = PlantUmlRenderWorkerSignals()

    def run(self) -> None:
        if self.setup_issue is not None:
            self.signals.finished.emit(self.hash_key, "error", self.setup_issue)
            return
        if self.jar_path is None:
            self.signals.finished.emit(self.hash_key, "error", "plantuml.jar not available")
            return

        command = [
            "java",
            "-Djava.awt.headless=true",
            "-jar",
            str(self.jar_path),
            "-pipe",
            "-tsvg",
            "-charset",
            "UTF-8",
        ]

        try:
            # PlantUML is CPU-heavy and can block; run in worker threads and
            # return compact status payloads via Qt signals.
            result = subprocess.run(
                command,
                input=self.prepared_code,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            self.signals.finished.emit(self.hash_key, "error", "Local PlantUML render timed out")
            return
        except Exception as exc:
            self.signals.finished.emit(self.hash_key, "error", f"Local PlantUML render failed: {exc}")
            return

        if result.returncode != 0:
            details = _extract_plantuml_error_details(result.stderr or "")
            self.signals.finished.emit(self.hash_key, "error", f"Local PlantUML render failed: {details}")
            return

        svg_text = (result.stdout or "").strip()
        if "<svg" not in svg_text.casefold():
            self.signals.finished.emit(self.hash_key, "error", "Local PlantUML did not return SVG output")
            return

        encoded_svg = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
        data_uri = f"data:image/svg+xml;base64,{encoded_svg}"
        self.signals.finished.emit(self.hash_key, "done", data_uri)


class MdExploreWindow(QMainWindow):
    HIGHLIGHT_COLORS = [
        ("Yellow", "#f5d34f"),
        ("Green", "#78d389"),
        ("Blue", "#7bb9ff"),
        ("Orange", "#f6a05f"),
        ("Purple", "#bb9df5"),
        ("Red", "#ef7d7d"),
    ]

    def __init__(self, root: Path, app_icon: QIcon, config_path: Path):
        super().__init__()
        self.root = root.resolve()
        self.config_path = config_path
        self.renderer = MarkdownRenderer()
        self.current_file: Path | None = None
        self.last_directory_selection: Path | None = self.root
        self.cache: dict[str, tuple[int, int, str]] = {}
        self.current_match_files: list[Path] = []
        self._pending_preview_search_terms: list[str] = []
        self._render_pool = QThreadPool(self)
        self._render_pool.setMaxThreadCount(1)
        self._render_request_id = 0
        self._active_render_workers: set[PreviewRenderWorker] = set()
        self._plantuml_pool = QThreadPool(self)
        self._plantuml_pool.setMaxThreadCount(2)
        # Global, in-process result cache for PlantUML blocks keyed by hash of
        # normalized source. This survives file navigation during this run.
        self._plantuml_results: dict[str, tuple[str, str]] = {}
        # Track active and dependent docs so completed jobs can invalidate only
        # affected cached HTML snapshots.
        self._plantuml_inflight: set[str] = set()
        self._plantuml_docs_by_hash: dict[str, set[str]] = {}
        self._plantuml_placeholders_by_doc: dict[str, dict[str, list[str]]] = {}
        self._active_plantuml_workers: set[PlantUmlRenderWorker] = set()
        # Per-file session scroll memory (not persisted to disk).
        self._preview_scroll_positions: dict[str, float] = {}
        self._match_input_text = ""
        self.match_timer = QTimer(self)
        self.match_timer.setSingleShot(True)
        self.match_timer.setInterval(1000)
        self.match_timer.timeout.connect(self._run_match_search)
        self._scroll_capture_timer = QTimer(self)
        self._scroll_capture_timer.setInterval(400)
        self._scroll_capture_timer.timeout.connect(self._capture_current_preview_scroll)
        self._scroll_capture_timer.start()
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
        self.tree.expanded.connect(self._on_tree_directory_expanded)

        self.preview = QWebEngineView()
        self.preview.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.preview.customContextMenuRequested.connect(self._show_preview_context_menu)
        self.preview.loadFinished.connect(self._on_preview_load_finished)

        self.up_btn = QPushButton("^")
        self.up_btn.clicked.connect(self._go_up_directory)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_current_preview)

        quit_btn = QPushButton("Quit")
        quit_btn.clicked.connect(self.close)

        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._edit_current_file)

        self.path_label = QLabel("")
        self.path_label.setTextInteractionFlags(self.path_label.textInteractionFlags())

        copy_label = QLabel("Copy to clipboard:")
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

        match_label = QLabel("Search: ")
        self.match_input = QLineEdit()
        self.match_input.setClearButtonEnabled(False)
        self.match_input.setPlaceholderText('words, "quoted phrases", AND/OR/NOT')
        self.match_input.setMinimumWidth(220)
        self.match_clear_action = self.match_input.addAction(
            _build_clear_x_icon(),
            QLineEdit.ActionPosition.TrailingPosition,
        )
        self.match_clear_action.setToolTip("Clear search")
        self.match_clear_action.triggered.connect(self._clear_match_input)
        self.match_clear_action.setVisible(False)
        self.match_input.textChanged.connect(self._on_match_text_changed)
        self.match_input.returnPressed.connect(self._run_match_search_now)

        match_buttons_widget = QWidget()
        match_buttons_layout = QHBoxLayout(match_buttons_widget)
        match_buttons_layout.setContentsMargins(0, 0, 0, 0)
        match_buttons_layout.setSpacing(4)
        match_buttons_layout.addWidget(match_label)
        match_buttons_layout.addWidget(self.match_input)
        for color_name, color_value in self.HIGHLIGHT_COLORS:
            color_btn = QPushButton("")
            color_btn.setFixedSize(18, 18)
            color_btn.setToolTip(f"Highlight current matches with {color_name.lower()}")
            color_btn.setStyleSheet(
                f"background-color: {color_value}; border: 1px solid #4b5563; border-radius: 3px;"
            )
            color_btn.clicked.connect(
                lambda _checked=False, c=color_value, n=color_name: self._apply_match_highlight_color(c, n)
            )
            match_buttons_layout.addWidget(color_btn)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.addWidget(self.up_btn)
        top_bar.addWidget(refresh_btn)
        top_bar.addWidget(quit_btn)
        top_bar.addWidget(edit_btn)
        top_bar.addWidget(self.path_label, 1)
        top_bar.addWidget(copy_buttons_widget, 0, Qt.AlignmentFlag.AlignRight)
        top_bar.addSpacing(16)
        top_bar.addWidget(match_buttons_widget, 0, Qt.AlignmentFlag.AlignRight)

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
        self._capture_current_preview_scroll()
        self.root = new_root.resolve()
        self.last_directory_selection = self.root
        self.current_file = None
        self._pending_preview_search_terms = []
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
        self._cancel_pending_preview_render()
        self._rerun_active_search_for_scope()
        QTimer.singleShot(0, self._maybe_apply_initial_split)

    def _on_preview_load_finished(self, ok: bool) -> None:
        """Apply deferred in-preview highlighting after a page finishes loading."""
        if not ok:
            return
        # PlantUML completions are patched in-place, but a full page load can
        # still happen from cache refreshes; re-apply any ready results.
        self._apply_all_ready_plantuml_to_current_preview()
        has_saved_scroll = self._has_saved_scroll_for_current_preview()
        if self._pending_preview_search_terms:
            # Search normally scrolls to first hit. If this file has a saved
            # scroll position, preserve that location instead.
            self._highlight_preview_search_terms(self._pending_preview_search_terms, scroll_to_first=not has_saved_scroll)
        if has_saved_scroll:
            # Small delay allows layout/MathJax/Mermaid updates to settle.
            QTimer.singleShot(90, self._restore_current_preview_scroll)

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

    def _on_match_text_changed(self, text: str) -> None:
        """Debounce free-form match input before running a new search."""
        self._match_input_text = text
        self.match_clear_action.setVisible(bool(text.strip()))
        if not text.strip():
            self.match_timer.stop()
            self._clear_match_results()
            return
        self.match_timer.start()

    def _clear_match_input(self) -> None:
        """Clear search text and immediately remove any active match styling."""
        self.match_timer.stop()
        self.match_input.clear()
        self._clear_match_results()

    def _run_match_search_now(self) -> None:
        """Run search immediately, bypassing debounce delay."""
        self.match_timer.stop()
        self._run_match_search()

    def _cancel_pending_preview_render(self) -> None:
        """Drop queued preview render jobs and invalidate running results."""
        self._render_request_id += 1
        self._render_pool.clear()

    def _rerun_active_search_for_scope(self) -> None:
        """Re-run search immediately when scope changes and query is active."""
        if not self.match_input.text().strip():
            return
        self.match_timer.stop()
        self._run_match_search()

    def _clear_match_results(self) -> None:
        """Clear bolded search matches without affecting persisted highlights."""
        self.current_match_files = []
        self.model.clear_search_match_paths()
        self.tree.viewport().update()
        self._remove_preview_search_highlights()

    def _run_match_search(self) -> None:
        """Search current scope non-recursively and bold matching markdown files."""
        query = self.match_input.text().strip()
        if not query:
            self._clear_match_results()
            return

        scope = self._highlight_scope_directory()
        predicate = self._compile_match_predicate(query)
        candidates = self._list_markdown_files_non_recursive(scope)
        matches: list[Path] = []

        for path in candidates:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                content = ""
            # Search over file name + body to support quick discovery.
            haystack = f"{path.name}\n{content}".casefold()
            if predicate(haystack):
                matches.append(path)

        self.current_match_files = matches
        self.model.set_search_match_paths(set(matches))
        self.tree.viewport().update()
        self.statusBar().showMessage(
            f"Matched {len(matches)} of {len(candidates)} markdown file(s) in {scope}",
            3500,
        )
        if self.current_file is not None:
            if self._is_path_in_current_matches(self.current_file):
                self._highlight_preview_search_terms(self._current_search_terms(), scroll_to_first=False)
            else:
                self._remove_preview_search_highlights()

    @staticmethod
    def _path_key(path: Path) -> str:
        try:
            return str(path.resolve())
        except Exception:
            return str(path)

    def _is_path_in_current_matches(self, path: Path) -> bool:
        """Return whether a file path is in the latest search match set."""
        target = self._path_key(path)
        for candidate in self.current_match_files:
            if self._path_key(candidate) == target:
                return True
        return False

    def _current_search_terms(self) -> list[str]:
        """Extract searchable terms from the current query (operators excluded)."""
        query = self.match_input.text().strip()
        if not query:
            return []
        terms: list[str] = []
        seen: set[str] = set()
        for token_type, token_value in self._tokenize_match_query(query):
            if token_type != "TERM":
                continue
            term = token_value.strip()
            if not term:
                continue
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
        terms.sort(key=len, reverse=True)
        return terms

    def _remove_preview_search_highlights(self) -> None:
        """Remove in-preview search highlight spans from the current page."""
        js = """
(() => {
  // Highlight spans are synthetic; remove them to restore original text nodes.
  const root = document.querySelector("main") || document.body;
  if (!root) return 0;
  const marks = root.querySelectorAll('span[data-mdexplore-search-mark="1"]');
  for (const mark of marks) {
    const parent = mark.parentNode;
    if (!parent) continue;
    parent.replaceChild(document.createTextNode(mark.textContent || ""), mark);
    parent.normalize();
  }
  return marks.length;
})();
"""
        # Mutates preview DOM to strip search marks; return value is not needed.
        self.preview.page().runJavaScript(js)

    def _highlight_preview_search_terms(self, terms: list[str], scroll_to_first: bool) -> None:
        """Highlight term matches in preview and optionally scroll to first one."""
        cleaned_terms = [term.strip() for term in terms if term.strip()]
        if not cleaned_terms:
            self._remove_preview_search_highlights()
            return

        terms_json = json.dumps(cleaned_terms)
        scroll_json = "true" if scroll_to_first else "false"
        js = f"""
(() => {{
  // Rebuild highlight spans from plain text each pass so updates are idempotent.
  const terms = {terms_json};
  const shouldScroll = {scroll_json};
  const root = document.querySelector("main") || document.body;
  if (!root || !terms.length) return 0;

  const markSelector = 'span[data-mdexplore-search-mark="1"]';
  for (const oldMark of root.querySelectorAll(markSelector)) {{
    const parent = oldMark.parentNode;
    if (!parent) continue;
    parent.replaceChild(document.createTextNode(oldMark.textContent || ""), oldMark);
    parent.normalize();
  }}

  const escaped = terms
    .map((term) => term.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&'))
    .filter((term) => term.length > 0);
  if (!escaped.length) return 0;

  // One regex across all terms keeps traversal cost low for large documents.
  const regex = new RegExp(escaped.join("|"), "gi");
  const skipTags = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA"]);
  const acceptedNodes = [];
  const walker = document.createTreeWalker(
    root,
    NodeFilter.SHOW_TEXT,
    {{
      acceptNode(node) {{
        if (!node || !node.nodeValue || !node.nodeValue.trim()) {{
          return NodeFilter.FILTER_REJECT;
        }}
        const parent = node.parentElement;
        if (!parent) {{
          return NodeFilter.FILTER_REJECT;
        }}
        if (skipTags.has(parent.tagName)) {{
          return NodeFilter.FILTER_REJECT;
        }}
        if (parent.closest(markSelector)) {{
          return NodeFilter.FILTER_REJECT;
        }}
        regex.lastIndex = 0;
        return regex.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }},
    }},
  );

  while (walker.nextNode()) {{
    acceptedNodes.push(walker.currentNode);
  }}

  let firstMark = null;
  let matchCount = 0;
  for (const textNode of acceptedNodes) {{
    const text = textNode.nodeValue || "";
    regex.lastIndex = 0;
    let cursor = 0;
    let changed = false;
    const fragment = document.createDocumentFragment();
    let match = null;
    while ((match = regex.exec(text)) !== null) {{
      changed = true;
      const start = match.index;
      const end = start + match[0].length;
      if (start > cursor) {{
        fragment.appendChild(document.createTextNode(text.slice(cursor, start)));
      }}
      const mark = document.createElement("span");
      mark.setAttribute("data-mdexplore-search-mark", "1");
      mark.style.backgroundColor = "#f5d34f";
      mark.style.color = "#111827";
      mark.style.padding = "0 1px";
      mark.style.borderRadius = "2px";
      mark.textContent = match[0];
      fragment.appendChild(mark);
      if (!firstMark) {{
        firstMark = mark;
      }}
      matchCount += 1;
      cursor = end;
      if (regex.lastIndex <= start) {{
        regex.lastIndex = start + 1;
      }}
    }}

    if (!changed) {{
      continue;
    }}
    if (cursor < text.length) {{
      fragment.appendChild(document.createTextNode(text.slice(cursor)));
    }}
    const parent = textNode.parentNode;
    if (parent) {{
      parent.replaceChild(fragment, textNode);
    }}
  }}

  if (firstMark && shouldScroll) {{
    firstMark.scrollIntoView({{ behavior: "auto", block: "center", inline: "nearest" }});
  }}
  return matchCount;
}})();
"""
        # Mutates preview DOM by inserting mark spans (and optional scroll).
        self.preview.page().runJavaScript(js)

    def _list_markdown_files_non_recursive(self, directory: Path) -> list[Path]:
        """Return direct child markdown files from a directory (non-recursive)."""
        if not directory.is_dir():
            return []

        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
        except Exception:
            return []

        files: list[Path] = []
        for entry in entries:
            try:
                if entry.is_file() and entry.suffix.lower() == ".md":
                    files.append(entry.resolve())
            except Exception:
                # Ignore files that disappear or become inaccessible mid-scan.
                pass
        return files

    def _compile_match_predicate(self, query: str):
        """Compile basic boolean query with implicit AND and quoted terms."""
        tokens = self._tokenize_match_query(query)
        if not tokens:
            return lambda _haystack: True

        # Insert implicit ANDs between adjacent terms/parentheses so users can
        # type natural queries like: foo "bar baz" NOT qux
        expanded_tokens: list[tuple[str, str]] = []
        previous_type: str | None = None
        for token_type, token_value in tokens:
            if previous_type in {"TERM", "RPAREN"} and token_type in {"TERM", "LPAREN"}:
                expanded_tokens.append(("OP", "AND"))
            elif previous_type in {"TERM", "RPAREN"} and token_type == "OP" and token_value == "NOT":
                expanded_tokens.append(("OP", "AND"))
            expanded_tokens.append((token_type, token_value))
            previous_type = token_type

        precedence = {"OR": 1, "AND": 2, "NOT": 3}
        output: list[tuple[str, str]] = []
        operators: list[tuple[str, str]] = []

        for token_type, token_value in expanded_tokens:
            if token_type == "TERM":
                output.append((token_type, token_value.casefold()))
                continue

            if token_type == "OP":
                # Shunting-yard conversion from infix -> postfix.
                while operators and operators[-1][0] == "OP":
                    top_op = operators[-1][1]
                    current_prec = precedence[token_value]
                    top_prec = precedence[top_op]
                    right_associative = token_value == "NOT"
                    if (not right_associative and current_prec <= top_prec) or (right_associative and current_prec < top_prec):
                        output.append(operators.pop())
                    else:
                        break
                operators.append((token_type, token_value))
                continue

            if token_type == "LPAREN":
                operators.append((token_type, token_value))
                continue

            if token_type == "RPAREN":
                found_lparen = False
                while operators:
                    top = operators.pop()
                    if top[0] == "LPAREN":
                        found_lparen = True
                        break
                    output.append(top)
                if not found_lparen:
                    return self._compile_simple_match_predicate(tokens)

        while operators:
            top = operators.pop()
            if top[0] == "LPAREN":
                return self._compile_simple_match_predicate(tokens)
            output.append(top)

        def predicate(haystack: str) -> bool:
            # Evaluate postfix expression against normalized haystack text.
            stack: list[bool] = []
            for token_type, token_value in output:
                if token_type == "TERM":
                    stack.append(token_value in haystack)
                    continue

                if token_type != "OP":
                    return False

                if token_value == "NOT":
                    if not stack:
                        return False
                    stack.append(not stack.pop())
                    continue

                if len(stack) < 2:
                    return False
                right = stack.pop()
                left = stack.pop()
                if token_value == "AND":
                    stack.append(left and right)
                elif token_value == "OR":
                    stack.append(left or right)
                else:
                    return False

            return len(stack) == 1 and stack[0]

        return predicate

    def _compile_simple_match_predicate(self, tokens: list[tuple[str, str]]):
        """Fallback matcher: all terms must appear in haystack."""
        terms = [value.casefold() for token_type, value in tokens if token_type == "TERM"]
        if not terms:
            return lambda _haystack: True
        return lambda haystack: all(term in haystack for term in terms)

    def _tokenize_match_query(self, query: str) -> list[tuple[str, str]]:
        """Tokenize query supporting AND/OR/NOT, quotes, and parentheses."""
        prepared = query.replace("(", " ( ").replace(")", " ) ")
        try:
            raw_tokens = shlex.split(prepared)
        except ValueError:
            raw_tokens = prepared.split()

        tokens: list[tuple[str, str]] = []
        for token in raw_tokens:
            upper = token.upper()
            if token == "(":
                tokens.append(("LPAREN", token))
            elif token == ")":
                tokens.append(("RPAREN", token))
            elif upper in {"AND", "OR", "NOT"}:
                tokens.append(("OP", upper))
            else:
                tokens.append(("TERM", token))
        return tokens

    def _apply_match_highlight_color(self, color_value: str, color_name: str) -> None:
        """Apply a highlight color to current match set, then clear bolding."""
        self.match_timer.stop()
        if self.match_input.text().strip():
            self._run_match_search()

        if not self.current_match_files:
            self.statusBar().showMessage("No matched files to highlight", 3000)
            return

        updated = 0
        for path in self.current_match_files:
            try:
                if path.is_file() and path.suffix.lower() == ".md":
                    self.model.set_color_for_file(path, color_value)
                    updated += 1
            except Exception:
                # Ignore files that are no longer available.
                pass

        self._clear_match_results()
        self.statusBar().showMessage(
            f"Applied {color_name.lower()} highlight to {updated} matched file(s)",
            4000,
        )

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

    def _on_tree_directory_expanded(self, index) -> None:
        """Treat expanded directories as active scope for quit persistence/title."""
        if not index.isValid():
            return
        path = Path(self.model.filePath(index))
        if not path.is_dir():
            return
        was_selected = self.tree.currentIndex() == index
        self.tree.setCurrentIndex(index)
        self.last_directory_selection = path.resolve()
        self._update_window_title()
        if was_selected:
            self._rerun_active_search_for_scope()

    def _show_preview_context_menu(self, pos) -> None:
        """Extend the preview context menu with a markdown copy action."""
        request = self.preview.lastContextMenuRequest()
        selected_text_hint = request.selectedText() if request is not None else ""
        click_x = int(pos.x())
        click_y = int(pos.y())

        # Capture selection mapping before context-menu interaction to avoid
        # losing selection state when the menu action is triggered.
        js = """
(() => {
  const sel = window.getSelection();

  function lineInfo(node) {
    if (!node) return null;
    if (node.nodeType === Node.TEXT_NODE) node = node.parentElement;
    if (!(node instanceof Element)) return null;
    const el = node.closest('[data-md-line-start][data-md-line-end]');
    if (!el) return null;
    const start = parseInt(el.getAttribute('data-md-line-start'), 10);
    const end = parseInt(el.getAttribute('data-md-line-end'), 10);
    if (Number.isNaN(start) || Number.isNaN(end)) return null;
    return { start, end };
  }

  function normalizeRange(startInfo, endInfo) {
    let start = startInfo ? startInfo.start : endInfo.start;
    let end = endInfo ? endInfo.end : startInfo.end;
    if (start > end) {
      const tmp = start;
      start = end;
      end = tmp;
    }
    if (end <= start) end = start + 1;
    return { start, end };
  }

  const hasSelection = !!(sel && sel.toString && sel.toString().trim());
  const selectedText = hasSelection ? sel.toString() : "";
  // Preferred path: map the active text selection to source line metadata.
  if (sel && sel.rangeCount > 0 && !sel.isCollapsed) {
    const range = sel.getRangeAt(0);
    const startInfo = lineInfo(range.startContainer);
    const endInfo = lineInfo(range.endContainer);
    if (startInfo || endInfo) {
      return { hasSelection: true, selectedText, ...normalizeRange(startInfo, endInfo), via: "selection" };
    }
  }

  // Fallback: map from right-clicked block location.
  const clicked = document.elementFromPoint(__CLICK_X__, __CLICK_Y__);
  const clickedInfo = lineInfo(clicked);
  if (clickedInfo) {
    return {
      hasSelection,
      selectedText,
      start: clickedInfo.start,
      end: clickedInfo.end,
      via: "click",
    };
  }

  return { hasSelection, selectedText };
})();
"""
        js = js.replace("__CLICK_X__", str(click_x)).replace("__CLICK_Y__", str(click_y))
        # Returns selection + line-range metadata used to build copy actions.
        self.preview.page().runJavaScript(
            js,
            lambda result: self._show_preview_context_menu_with_cached_selection(pos, result, selected_text_hint),
        )

    def _show_preview_context_menu_with_cached_selection(self, pos, selection_info, selected_text_hint: str) -> None:
        """Build preview menu and use cached selection metadata for copy action."""
        menu = self.preview.createStandardContextMenu()
        has_selection = bool(selected_text_hint.strip() or self.preview.selectedText().strip())
        if isinstance(selection_info, dict) and selection_info.get("hasSelection"):
            has_selection = True

        copy_source_action: QAction | None = None
        copy_rendered_action: QAction | None = None
        if has_selection:
            menu.addSeparator()
            copy_rendered_action = menu.addAction("Copy Rendered Text")
            copy_source_action = menu.addAction("Copy Source Markdown")

        chosen = menu.exec(self.preview.mapToGlobal(pos))
        if copy_rendered_action is not None and chosen == copy_rendered_action:
            self._copy_preview_selection_as_rendered_text(selection_info, selected_text_hint)
            menu.deleteLater()
            return
        if copy_source_action is not None and chosen == copy_source_action:
            self._copy_preview_selection_as_source_markdown(selection_info, selected_text_hint)
        menu.deleteLater()

    def _copy_preview_selection_as_rendered_text(self, selection_info, selected_text_hint: str) -> None:
        """Copy currently selected rendered preview text as plain text."""
        selected_text = ""
        if isinstance(selection_info, dict):
            selected_raw = selection_info.get("selectedText")
            if isinstance(selected_raw, str):
                selected_text = selected_raw
        if not selected_text.strip():
            selected_text = selected_text_hint
        if not selected_text.strip():
            selected_text = self.preview.selectedText()
        if not selected_text.strip():
            self.statusBar().showMessage("No selected rendered text to copy", 3000)
            return

        self._set_plain_text_clipboard(selected_text)
        self.statusBar().showMessage("Copied rendered text", 3000)

    def _copy_preview_selection_as_source_markdown(self, selection_info, selected_text_hint: str) -> None:
        """Copy source markdown lines that correspond to selected preview content."""
        if self.current_file is None:
            self.statusBar().showMessage("No markdown file selected", 3000)
            return
        source_path = self.current_file
        try:
            lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        except Exception:
            self.statusBar().showMessage("Could not read source markdown file", 3000)
            return

        if not lines:
            QApplication.clipboard().setText("")
            self.statusBar().showMessage("Copied source markdown (empty file)", 3000)
            return

        if isinstance(selection_info, dict):
            start_raw = selection_info.get("start")
            end_raw = selection_info.get("end")
            if isinstance(start_raw, (int, float)) and isinstance(end_raw, (int, float)):
                start = max(0, int(start_raw))
                end = max(start + 1, int(end_raw))
                start = min(start, len(lines) - 1)
                end = min(end, len(lines))
                snippet = "".join(lines[start:end])
                self._set_plain_text_clipboard(snippet)
                self.statusBar().showMessage(
                    f"Copied source markdown lines {start + 1}-{end}",
                    3500,
                )
                return

        # Fallback: find selected text in source and copy containing source lines.
        js_selected_text = ""
        if isinstance(selection_info, dict):
            selected_raw = selection_info.get("selectedText")
            if isinstance(selected_raw, str):
                js_selected_text = selected_raw.strip()

        query = js_selected_text or selected_text_hint.strip() or self.preview.selectedText().strip()
        if not query:
            query = QApplication.clipboard().text(QClipboard.Mode.Clipboard).strip()
        if query:
            source_text = "".join(lines)
            found_at = source_text.find(query)
            if found_at != -1:
                line_start = source_text.count("\n", 0, found_at)
                line_end = source_text.count("\n", 0, found_at + len(query)) + 1
                line_start = min(line_start, len(lines) - 1)
                line_end = min(max(line_end, line_start + 1), len(lines))
                snippet = "".join(lines[line_start:line_end])
                self._set_plain_text_clipboard(snippet)
                self.statusBar().showMessage(
                    f"Copied source markdown lines {line_start + 1}-{line_end} (fallback)",
                    4000,
                )
                return

            if self._copy_source_by_fuzzy_lines(lines, query):
                return

        self._set_plain_text_clipboard("".join(lines))
        self.statusBar().showMessage(
            "Could not map selection exactly; copied full source markdown",
            4500,
        )

    @staticmethod
    def _normalize_for_fuzzy_match(text: str) -> str:
        lowered = text.casefold()
        stripped = re.sub(r"[`*_~>#\\[\\](){}|!+-]", " ", lowered)
        return re.sub(r"\s+", " ", stripped).strip()

    def _copy_source_by_fuzzy_lines(self, lines: list[str], query_text: str) -> bool:
        """Fuzzy-match selected first/last lines against markdown source lines."""
        raw_query_lines = [line.strip() for line in query_text.splitlines() if line.strip()]
        if not raw_query_lines:
            return False

        normalized_lines = [self._normalize_for_fuzzy_match(line) for line in lines]
        meaningful_query_lines: list[str] = []
        for line in raw_query_lines:
            normalized = self._normalize_for_fuzzy_match(line)
            if normalized:
                meaningful_query_lines.append(normalized)
        if not meaningful_query_lines:
            return False

        def best_line_match(query_norm: str, start_index: int = 0) -> tuple[int, float]:
            best_idx = -1
            best_score = 0.0
            for idx in range(start_index, len(normalized_lines)):
                candidate = normalized_lines[idx]
                if not candidate:
                    continue
                if query_norm in candidate:
                    score = 0.90 + min(0.10, len(query_norm) / max(len(candidate), 1))
                elif candidate in query_norm and len(candidate) >= 8:
                    score = 0.60 + min(0.30, len(candidate) / max(len(query_norm), 1))
                else:
                    score = SequenceMatcher(None, query_norm, candidate).ratio()
                if score > best_score:
                    best_score = score
                    best_idx = idx
            return best_idx, best_score

        def find_anchor(candidates: list[str], start_index: int, min_score: float) -> tuple[int, float]:
            # Try candidate lines in order and stop at the first sufficiently
            # strong match so non-identifying boundary lines are skipped.
            best_idx = -1
            best_score = 0.0
            for query_norm in candidates:
                idx, score = best_line_match(query_norm, start_index)
                if score > best_score:
                    best_idx = idx
                    best_score = score
                if idx >= 0 and score >= min_score:
                    return idx, score
            return best_idx, best_score

        start_idx, start_score = find_anchor(meaningful_query_lines, 0, 0.45)
        if start_idx < 0 or start_score < 0.45:
            return False

        end_idx = start_idx + 1
        if len(meaningful_query_lines) > 1:
            end_candidates = list(reversed(meaningful_query_lines))
            end_match_idx, end_score = find_anchor(end_candidates, start_idx, 0.42)
            if end_match_idx >= start_idx and end_score >= 0.42:
                end_idx = end_match_idx + 1
            else:
                approx_span = max(1, len(meaningful_query_lines))
                end_idx = min(len(lines), start_idx + approx_span)

        end_idx = max(start_idx + 1, min(end_idx, len(lines)))
        snippet = "".join(lines[start_idx:end_idx])
        if not snippet.strip():
            return False

        self._set_plain_text_clipboard(snippet)
        self.statusBar().showMessage(
            f"Copied source markdown lines {start_idx + 1}-{end_idx} (fuzzy)",
            4000,
        )
        return True

    def _set_plain_text_clipboard(self, text: str) -> None:
        """Set clipboard text via Qt, with platform CLI fallback for reliability."""
        clipboard = QApplication.clipboard()
        clipboard.setText(text, QClipboard.Mode.Clipboard)
        clipboard.setText(text, QClipboard.Mode.Selection)
        QApplication.processEvents()

        try:
            if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
                subprocess.run(["wl-copy"], input=text, text=True, check=False)
                return
            if os.environ.get("DISPLAY") and shutil.which("xclip"):
                subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True, check=False)
                return
            if os.environ.get("DISPLAY") and shutil.which("xsel"):
                subprocess.run(["xsel", "--clipboard", "--input"], input=text, text=True, check=False)
        except Exception:
            # Qt clipboard already received the text; ignore CLI fallback errors.
            pass

    def _highlight_scope_directory(self) -> Path:
        """Resolve active scope from selected/last-visited directory, then root."""
        index = self.tree.currentIndex()
        if index.isValid():
            selected = Path(self.model.filePath(index))
            if selected.is_dir():
                try:
                    resolved = selected.resolve()
                except Exception:
                    resolved = selected
                self.last_directory_selection = resolved
                return resolved
        if self.last_directory_selection is not None and self.last_directory_selection.is_dir():
            return self.last_directory_selection
        return self.root

    def _update_window_title(self) -> None:
        """Show the effective root in the application window title."""
        scope = self._highlight_scope_directory()
        self.setWindowTitle(f"mdexplore - {scope}")

    def _effective_root_for_persistence(self) -> Path:
        """Resolve persisted root, preferring selected or last visited directory."""
        index = self.tree.currentIndex()
        if index.isValid():
            selected = Path(self.model.filePath(index))
            if selected.is_dir():
                try:
                    return selected.resolve()
                except Exception:
                    return selected
        if self.last_directory_selection is not None and self.last_directory_selection.is_dir():
            return self.last_directory_selection
        return self.root

    def _persist_effective_root(self) -> None:
        """Persist the effective root for future no-argument launches."""
        scope = self._effective_root_for_persistence()
        try:
            self.config_path.write_text(str(scope.resolve()) + "\n", encoding="utf-8")
        except Exception:
            # Requested behavior is resilience; persistence failure should not
            # block application exit.
            pass

    def closeEvent(self, event) -> None:  # noqa: N802
        self._persist_effective_root()
        super().closeEvent(event)

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
        if path.is_dir():
            try:
                self.last_directory_selection = path.resolve()
            except Exception:
                self.last_directory_selection = path
            self._rerun_active_search_for_scope()
            return
        if not path.is_file() or path.suffix.lower() != ".md":
            return
        self._load_preview(path)

    def _on_preview_render_finished(
        self,
        request_id: int,
        path_key: str,
        html_doc: str,
        mtime_ns: int,
        size: int,
        error_text: str,
    ) -> None:
        """Apply finished background render if it is still the active request."""
        worker_to_remove = None
        for worker in self._active_render_workers:
            if worker.request_id == request_id:
                worker_to_remove = worker
                break
        if worker_to_remove is not None:
            self._active_render_workers.remove(worker_to_remove)

        if request_id != self._render_request_id:
            return

        if self.current_file is None:
            return
        try:
            current_key = str(self.current_file.resolve())
        except Exception:
            current_key = str(self.current_file)
        if current_key != path_key:
            return

        if error_text:
            self.statusBar().showMessage(f"Preview render failed: {error_text}", 5000)
            html_doc = self._placeholder_html(f"Could not render preview for {self.current_file.name}: {error_text}")
        else:
            self.cache[path_key] = (mtime_ns, size, html_doc)

        try:
            base_url = QUrl.fromLocalFile(f"{self.current_file.parent.resolve()}/")
        except Exception:
            base_url = QUrl.fromLocalFile(f"{self.root}/")
        self.preview.setHtml(html_doc, base_url)

    @staticmethod
    def _doc_id_for_path(path_key: str) -> str:
        return hashlib.sha1(path_key.encode("utf-8", errors="replace")).hexdigest()[:12]

    def _current_preview_path_key(self) -> str | None:
        if self.current_file is None:
            return None
        try:
            return str(self.current_file.resolve())
        except Exception:
            return str(self.current_file)

    def _has_saved_scroll_for_current_preview(self) -> bool:
        key = self._current_preview_path_key()
        return key is not None and key in self._preview_scroll_positions

    def _capture_current_preview_scroll(self) -> None:
        """Capture current preview scroll position for the selected file."""
        path_key = self._current_preview_path_key()
        if path_key is None:
            return
        js = """
(() => {
  // Use window/document fallbacks to support different engines/modes.
  const y = window.scrollY ?? document.documentElement.scrollTop ?? document.body.scrollTop ?? 0;
  return Number.isFinite(y) ? y : 0;
})();
"""

        def on_result(value, expected_key=path_key):
            # Store against the file that triggered this request, even if the
            # user switched selection before JS finished.
            if isinstance(value, (int, float)):
                self._preview_scroll_positions[expected_key] = float(value)

        # Returns current scroll Y for this preview page.
        self.preview.page().runJavaScript(js, on_result)

    def _restore_current_preview_scroll(self) -> None:
        """Restore previously captured scroll position for the selected file."""
        path_key = self._current_preview_path_key()
        if path_key is None:
            return
        scroll_y = self._preview_scroll_positions.get(path_key)
        if scroll_y is None:
            return
        scroll_json = json.dumps(float(scroll_y))
        js = f"""
(() => {{
  const y = {scroll_json};
  // Apply twice (RAF + timeout) because late layout work can override scroll.
  requestAnimationFrame(() => window.scrollTo(0, y));
  setTimeout(() => window.scrollTo(0, y), 60);
}})();
"""
        # Mutates page scroll position (no returned data consumed).
        self.preview.page().runJavaScript(js)

    @staticmethod
    def _truncate_error_text(text: str, max_len: int = 1200) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(normalized) <= max_len:
            return normalized
        return normalized[: max_len - 4] + "\n..."

    def _plantuml_inner_html(self, status: str, payload: str) -> str:
        if status == "done":
            return f'<img class="plantuml" src="{payload}" alt="PlantUML diagram"/>'
        safe_error = html.escape(self._truncate_error_text(payload or "unknown error"))
        return (
            '<div class="plantuml-error-message">PlantUML render failed with error:</div>'
            f'<pre class="plantuml-error-detail">{safe_error}</pre>'
        )

    def _plantuml_block_html(self, placeholder_id: str, line_attrs: str, status: str, payload: str) -> str:
        inner = self._plantuml_inner_html(status, payload) if status in {"done", "error"} else "PlantUML rendering..."
        classes = ["mdexplore-fence", "plantuml-async"]
        if status == "pending":
            classes.append("plantuml-pending")
        elif status == "error":
            classes.append("plantuml-error")
        else:
            classes.append("plantuml-ready")
        class_attr = " ".join(classes)
        return (
            f'<div class="{class_attr}" id="{placeholder_id}"{line_attrs}>'
            f"{inner}"
            "</div>\n"
        )

    def _ensure_plantuml_render_started(self, hash_key: str, prepared_code: str) -> None:
        result = self._plantuml_results.get(hash_key)
        if result is not None and result[0] in {"done", "error"}:
            return
        if hash_key in self._plantuml_inflight:
            return

        # Deduplicate renders: identical diagram source only renders once.
        self._plantuml_inflight.add(hash_key)
        self._plantuml_results[hash_key] = ("pending", "")
        worker = PlantUmlRenderWorker(
            hash_key,
            prepared_code,
            self.renderer._plantuml_jar_path,
            self.renderer._plantuml_setup_issue,
        )
        self._active_plantuml_workers.add(worker)
        worker.signals.finished.connect(self._on_plantuml_render_finished)
        self._plantuml_pool.start(worker)

    def _on_plantuml_render_finished(self, hash_key: str, status: str, payload: str) -> None:
        worker_to_remove = None
        for worker in self._active_plantuml_workers:
            if worker.hash_key == hash_key:
                worker_to_remove = worker
                break
        if worker_to_remove is not None:
            self._active_plantuml_workers.remove(worker_to_remove)

        self._plantuml_inflight.discard(hash_key)
        final_status = "done" if status == "done" else "error"
        self._plantuml_results[hash_key] = (final_status, payload)

        # Any cached document snapshot that references this diagram is stale
        # until placeholders are replaced with the final result.
        for doc_key in self._plantuml_docs_by_hash.get(hash_key, set()):
            self.cache.pop(doc_key, None)

        self._apply_plantuml_result_to_current_preview(hash_key)

    def _apply_plantuml_result_to_current_preview(self, hash_key: str) -> None:
        if self.current_file is None:
            return
        try:
            current_key = str(self.current_file.resolve())
        except Exception:
            current_key = str(self.current_file)

        placeholders_by_hash = self._plantuml_placeholders_by_doc.get(current_key, {})
        placeholder_ids = placeholders_by_hash.get(hash_key, [])
        if not placeholder_ids:
            return

        status, payload = self._plantuml_results.get(hash_key, ("pending", ""))
        if status not in {"done", "error"}:
            return
        if status == "done":
            class_name = "plantuml-ready"
            inner_html = self._plantuml_inner_html("done", payload)
        else:
            class_name = "plantuml-error"
            inner_html = self._plantuml_inner_html("error", payload)

        ids_json = json.dumps(placeholder_ids)
        html_json = json.dumps(inner_html)
        class_json = json.dumps(class_name)
        # Patch nodes in place to preserve scroll position and selection state.
        js = f"""
(() => {{
  const ids = {ids_json};
  const inner = {html_json};
  const className = {class_json};
  // Only touch known placeholder nodes so unrelated page state is untouched.
  for (const id of ids) {{
    const node = document.getElementById(id);
    if (!node) continue;
    node.classList.remove("plantuml-pending", "plantuml-ready", "plantuml-error");
    node.classList.add(className);
    node.innerHTML = inner;
  }}
}})();
"""
        # Mutates only known PlantUML placeholder nodes in the active page.
        self.preview.page().runJavaScript(js)

    def _apply_all_ready_plantuml_to_current_preview(self) -> None:
        if self.current_file is None:
            return
        try:
            current_key = str(self.current_file.resolve())
        except Exception:
            current_key = str(self.current_file)

        placeholders_by_hash = self._plantuml_placeholders_by_doc.get(current_key, {})
        for hash_key in placeholders_by_hash:
            self._apply_plantuml_result_to_current_preview(hash_key)

    def _load_preview(self, path: Path) -> None:
        # Render markdown quickly with async PlantUML placeholders so the
        # document appears immediately while diagrams render in background.
        self._capture_current_preview_scroll()
        self._cancel_pending_preview_render()
        should_highlight_search = bool(self.match_input.text().strip()) and self._is_path_in_current_matches(path)
        self._pending_preview_search_terms = self._current_search_terms() if should_highlight_search else []

        self.current_file = path
        try:
            rel = path.relative_to(self.root)
            self.path_label.setText(str(rel))
        except ValueError:
            self.path_label.setText(str(path))

        try:
            base_url = QUrl.fromLocalFile(f"{path.parent.resolve()}/")
        except Exception:
            base_url = QUrl.fromLocalFile(f"{self.root}/")

        try:
            resolved = path.resolve()
            stat = resolved.stat()
            cache_key = str(resolved)
            cached = self.cache.get(cache_key)
            if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
                self.preview.setHtml(cached[2], base_url)
                return

            markdown_text = resolved.read_text(encoding="utf-8", errors="replace")
            doc_id = self._doc_id_for_path(cache_key)

            # Remove stale dependency links for this document before rebuilding.
            previous_placeholders = self._plantuml_placeholders_by_doc.get(cache_key, {})
            for hash_key in previous_placeholders:
                docs = self._plantuml_docs_by_hash.get(hash_key)
                if docs is not None:
                    docs.discard(cache_key)
                    if not docs:
                        self._plantuml_docs_by_hash.pop(hash_key, None)

            placeholders_by_hash: dict[str, list[str]] = {}

            def plantuml_resolver(code: str, index: int, line_attrs: str) -> str:
                # Stable hash key lets the same diagram result be reused across
                # multiple files and repeated blocks.
                prepared_code = self.renderer._prepare_plantuml_source(code)
                hash_key = hashlib.sha1(prepared_code.encode("utf-8", errors="replace")).hexdigest()
                placeholder_id = f"mdexplore-plantuml-{doc_id}-{index}"

                placeholders_by_hash.setdefault(hash_key, []).append(placeholder_id)
                self._plantuml_docs_by_hash.setdefault(hash_key, set()).add(cache_key)

                status, payload = self._plantuml_results.get(hash_key, ("pending", ""))
                if status in {"done", "error"}:
                    return self._plantuml_block_html(placeholder_id, line_attrs, status, payload)

                self._ensure_plantuml_render_started(hash_key, prepared_code)
                return self._plantuml_block_html(placeholder_id, line_attrs, "pending", "")

            html_doc = self.renderer.render_document(markdown_text, resolved.name, plantuml_resolver=plantuml_resolver)
            self._plantuml_placeholders_by_doc[cache_key] = placeholders_by_hash
            self.cache[cache_key] = (stat.st_mtime_ns, stat.st_size, html_doc)
            self.preview.setHtml(html_doc, base_url)
        except Exception as exc:
            self.statusBar().showMessage(f"Preview render failed: {exc}", 5000)
            self.preview.setHtml(
                self._placeholder_html(f"Could not render preview for {path.name}: {exc}"),
                base_url,
            )

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
        default=None,
        help="Root directory to browse (default: ~/.mdexplore.cfg path, or home directory).",
    )
    args = parser.parse_args()

    root = Path(args.path).expanduser() if args.path is not None else _load_default_root_from_config()
    if not root.exists():
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"Path is not a directory: {root}", file=sys.stderr)
        return 2

    app = QApplication(sys.argv)
    app.setApplicationName("mdexplore")
    # Explicit desktop file name improves Linux shell mapping between the
    # running window and mdexplore.desktop for icon/pinning behavior.
    app.setDesktopFileName("mdexplore")
    app_icon = _build_markdown_icon()
    app.setWindowIcon(app_icon)

    window = MdExploreWindow(root, app_icon, _config_file_path())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
