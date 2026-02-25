#!/usr/bin/env python3
"""mdexplore: fast markdown browser/editor launcher for Ubuntu."""

from __future__ import annotations

import argparse
import base64
from bisect import bisect_right
import html
import hashlib
from io import BytesIO
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from PySide6.QtCore import QDir, QMimeData, QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QBrush, QClipboard, QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWebEngineCore import QWebEngineSettings
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
    QStyle,
    QTabBar,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

CONFIG_FILE_NAME = ".mdexplore.cfg"
SEARCH_CLOSE_WORD_GAP = 50
PDF_EXPORT_PRECHECK_MAX_ATTEMPTS = 20
PDF_EXPORT_PRECHECK_INTERVAL_MS = 140


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


def _stamp_pdf_page_numbers(pdf_bytes: bytes) -> bytes:
    """Overlay centered `N of M` footers on every page of a PDF payload."""
    if not pdf_bytes:
        raise ValueError("Empty PDF payload")

    try:
        from pypdf import PageObject, PdfReader, PdfWriter, Transformation
    except Exception as exc:
        raise RuntimeError("Missing dependency 'pypdf' for PDF page numbering") from exc

    try:
        from reportlab.pdfgen import canvas
    except Exception as exc:
        raise RuntimeError("Missing dependency 'reportlab' for PDF page numbering") from exc

    reader = PdfReader(BytesIO(pdf_bytes))
    page_total = len(reader.pages)
    if page_total <= 0:
        raise RuntimeError("Generated PDF has no pages")

    def estimate_majority_font_size() -> float:
        """Infer dominant body font size from PDF text operators."""
        size_counts: dict[float, int] = {}
        for page in reader.pages[: min(5, page_total)]:
            try:
                contents = page.get_contents()
            except Exception:
                contents = None
            if contents is None:
                continue

            streams = contents if isinstance(contents, list) else [contents]
            for stream in streams:
                try:
                    raw = stream.get_data()
                except Exception:
                    continue
                if not raw:
                    continue
                text = raw.decode("latin-1", errors="ignore")
                for match in re.finditer(r"([0-9]+(?:\\.[0-9]+)?)\\s+Tf\\b", text):
                    try:
                        size = float(match.group(1))
                    except Exception:
                        continue
                    if 6.0 <= size <= 24.0:
                        bucket = round(size * 2.0) / 2.0
                        size_counts[bucket] = size_counts.get(bucket, 0) + 1

        if not size_counts:
            return 10.5
        return max(size_counts.items(), key=lambda item: (item[1], -abs(item[0] - 11.0)))[0]

    base_body_font_size = estimate_majority_font_size()

    writer = PdfWriter()
    for page_number, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if width <= 0 or height <= 0:
            writer.add_page(page)
            continue

        # Fit rendered content into a print-style content box with explicit
        # top/side margins plus a dedicated footer band for page numbers.
        side_margin = max(34.0, min(width * 0.12, base_body_font_size * 4.2))
        top_margin = max(30.0, min(height * 0.10, base_body_font_size * 3.8))
        footer_band_height = max(42.0, min(height * 0.16, base_body_font_size * 4.4))
        content_box_width = max(72.0, width - (2.0 * side_margin))
        content_box_height = max(72.0, height - top_margin - footer_band_height)

        content_scale = min(
            1.0,
            content_box_width / width,
            content_box_height / height,
        )
        content_width = width * content_scale
        content_height = height * content_scale
        content_translate_x = side_margin + max(0.0, (content_box_width - content_width) / 2.0)
        content_translate_y = footer_band_height + max(0.0, (content_box_height - content_height) / 2.0)

        composed_page = PageObject.create_blank_page(width=width, height=height)
        transform = Transformation().scale(content_scale, content_scale).translate(
            content_translate_x,
            content_translate_y,
        )
        composed_page.merge_transformed_page(page, transform, over=True)

        footer_font_size = max(8.0, min(14.0, base_body_font_size * content_scale))
        footer_baseline_y = max(12.0, (footer_band_height - footer_font_size) / 2.0)

        overlay_buffer = BytesIO()
        footer_canvas = canvas.Canvas(overlay_buffer, pagesize=(width, height))
        footer_canvas.setFont("Helvetica", footer_font_size)
        footer_text = f"{page_number} of {page_total}"
        footer_width = footer_canvas.stringWidth(footer_text, "Helvetica", footer_font_size)
        x = max(0.0, (width - footer_width) / 2.0)
        footer_canvas.drawString(x, footer_baseline_y, footer_text)
        footer_canvas.save()

        overlay_buffer.seek(0)
        overlay_pdf = PdfReader(overlay_buffer)
        if overlay_pdf.pages:
            composed_page.merge_page(overlay_pdf.pages[0])
        writer.add_page(composed_page)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


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
        self._mathjax_local_script = self._resolve_local_mathjax_script()
        self._mermaid_local_script = self._resolve_local_mermaid_script()
        self._plantuml_jar_path = self._resolve_plantuml_jar_path()
        self._plantuml_setup_issue = self._plantuml_setup_error()
        self._plantuml_svg_cache: dict[str, str] = {}
        self._md = MarkdownIt(
            "commonmark",
            {"html": True, "linkify": True, "typographer": True},
        ).enable("table").enable("strikethrough")
        # Parse $...$ / $$...$$ as dedicated math tokens before markdown
        # emphasis/underscore rules run, preventing TeX corruption.
        self._md.use(dollarmath_plugin)

        default_fence = self._md.renderer.rules["fence"]
        default_render_token = self._md.renderer.renderToken

        def custom_math_inline(tokens, idx, options, env):
            token = tokens[idx]
            # Keep TeX content raw for MathJax, only HTML-escape unsafe chars.
            return f"${html.escape(token.content)}$"

        def custom_math_block(tokens, idx, options, env):
            token = tokens[idx]
            line_attrs = ""
            if token.map and len(token.map) == 2:
                line_attrs = f' data-md-line-start="{token.map[0]}" data-md-line-end="{token.map[1]}"'
            math_body = (token.content or "").strip("\n")
            return f'<div class="mdexplore-math-block"{line_attrs}>$$\n{html.escape(math_body)}\n$$</div>\n'

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
        self._md.renderer.rules["math_inline"] = custom_math_inline
        self._md.renderer.rules["math_block"] = custom_math_block
        self._md.renderer.renderToken = custom_render_token

    def _resolve_local_mathjax_script(self) -> Path | None:
        """Locate a local MathJax bundle, preferring SVG output quality."""
        env_value = os.environ.get("MDEXPLORE_MATHJAX_JS", "").strip()
        candidates: list[Path] = []
        if env_value:
            candidates.append(Path(env_value).expanduser())

        app_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                app_dir / "mathjax" / "es5" / "tex-svg.js",
                app_dir / "mathjax" / "tex-svg.js",
                app_dir / "assets" / "mathjax" / "es5" / "tex-svg.js",
                app_dir / "vendor" / "mathjax" / "es5" / "tex-svg.js",
                Path("/usr/share/javascript/mathjax/es5/tex-svg.js"),
                Path("/usr/share/mathjax/es5/tex-svg.js"),
                Path("/usr/share/nodejs/mathjax/es5/tex-svg.js"),
                app_dir / "mathjax" / "es5" / "tex-mml-chtml.js",
                app_dir / "mathjax" / "tex-mml-chtml.js",
                app_dir / "assets" / "mathjax" / "es5" / "tex-mml-chtml.js",
                app_dir / "vendor" / "mathjax" / "es5" / "tex-mml-chtml.js",
                Path("/usr/share/javascript/mathjax/es5/tex-mml-chtml.js"),
                Path("/usr/share/mathjax/es5/tex-mml-chtml.js"),
                Path("/usr/share/nodejs/mathjax/es5/tex-mml-chtml.js"),
            ]
        )

        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate.resolve()
            except Exception:
                continue
        return None

    def _mathjax_script_sources(self) -> list[str]:
        """Return local-first MathJax script URLs with CDN fallback."""
        sources: list[str] = []
        if self._mathjax_local_script is not None:
            try:
                sources.append(self._mathjax_local_script.as_uri())
            except Exception:
                pass
        # Prefer SVG output (closer to Obsidian quality), keep CHTML as fallback.
        sources.append("https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js")
        sources.append("https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")
        # Keep order while dropping duplicates.
        return list(dict.fromkeys(sources))

    def _resolve_local_mermaid_script(self) -> Path | None:
        """Locate a local Mermaid bundle to use before CDN fallback."""
        env_value = os.environ.get("MDEXPLORE_MERMAID_JS", "").strip()
        candidates: list[Path] = []
        if env_value:
            candidates.append(Path(env_value).expanduser())

        app_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                app_dir / "mermaid" / "mermaid.min.js",
                app_dir / "mermaid" / "dist" / "mermaid.min.js",
                app_dir / "assets" / "mermaid" / "mermaid.min.js",
                app_dir / "assets" / "mermaid" / "dist" / "mermaid.min.js",
                app_dir / "vendor" / "mermaid" / "mermaid.min.js",
                app_dir / "vendor" / "mermaid" / "dist" / "mermaid.min.js",
                Path("/usr/share/javascript/mermaid/mermaid.min.js"),
                Path("/usr/share/nodejs/mermaid/dist/mermaid.min.js"),
            ]
        )

        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate.resolve()
            except Exception:
                continue
        return None

    def _mermaid_script_sources(self) -> list[str]:
        """Return local-first Mermaid script URLs with CDN fallback."""
        sources: list[str] = []
        if self._mermaid_local_script is not None:
            try:
                sources.append(self._mermaid_local_script.as_uri())
            except Exception:
                pass
        sources.append("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js")
        # Keep order while dropping duplicates.
        return list(dict.fromkeys(sources))

    def _resolve_plantuml_jar_path(self) -> Path | None:
        """Locate plantuml.jar from env, vendor directory, app directory, or current directory."""
        env_value = os.environ.get("PLANTUML_JAR", "").strip()
        candidates: list[Path] = []
        if env_value:
            candidates.append(Path(env_value).expanduser())
        app_dir = Path(__file__).resolve().parent
        # Prefer the vendored jar location, but keep legacy fallbacks for compatibility.
        candidates.append(app_dir / "vendor" / "plantuml" / "plantuml.jar")
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
            return (
                "plantuml.jar not found "
                "(set PLANTUML_JAR or place jar at vendor/plantuml/plantuml.jar)"
            )
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
        mathjax_sources_json = json.dumps(self._mathjax_script_sources())
        mermaid_sources_json = json.dumps(self._mermaid_script_sources())
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
      --callout-note-border: #2563eb;
      --callout-note-bg: rgba(37, 99, 235, 0.12);
      --callout-tip-border: #16a34a;
      --callout-tip-bg: rgba(22, 163, 74, 0.12);
      --callout-important-border: #7c3aed;
      --callout-important-bg: rgba(124, 58, 237, 0.12);
      --callout-warning-border: #d97706;
      --callout-warning-bg: rgba(217, 119, 6, 0.14);
      --callout-caution-border: #dc2626;
      --callout-caution-bg: rgba(220, 38, 38, 0.12);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --fg: #e5e7eb;
        --bg: #111827;
        --code-bg: #1f2937;
        --border: #374151;
        --link: #8ab4f8;
        --callout-note-border: #60a5fa;
        --callout-note-bg: rgba(96, 165, 250, 0.16);
        --callout-tip-border: #4ade80;
        --callout-tip-bg: rgba(74, 222, 128, 0.16);
        --callout-important-border: #a78bfa;
        --callout-important-bg: rgba(167, 139, 250, 0.18);
        --callout-warning-border: #fbbf24;
        --callout-warning-bg: rgba(251, 191, 36, 0.2);
        --callout-caution-border: #f87171;
        --callout-caution-bg: rgba(248, 113, 113, 0.18);
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
    .mermaid svg {{
      max-width: 100%;
      height: auto;
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
    .mdexplore-callout {{
      margin: 0.9rem 0;
      padding: 0.72rem 0.9rem 0.78rem 0.95rem;
      border-left: 0.32rem solid var(--callout-note-border);
      background: var(--callout-note-bg);
      border-radius: 0.45rem;
      break-inside: avoid;
      page-break-inside: avoid;
    }}
    .mdexplore-callout-title {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin: 0 0 0.38rem 0;
      font-weight: 700;
      letter-spacing: 0.01em;
    }}
    .mdexplore-callout-icon {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 1.05rem;
      height: 1.05rem;
      border-radius: 999px;
      border: 1px solid currentColor;
      font-size: 0.73rem;
      line-height: 1;
      font-family: "Noto Sans", "DejaVu Sans", sans-serif;
      font-weight: 700;
      flex: 0 0 auto;
    }}
    .mdexplore-callout-content > :first-child {{
      margin-top: 0;
    }}
    .mdexplore-callout-content > :last-child {{
      margin-bottom: 0;
    }}
    .mdexplore-callout-note {{
      border-left-color: var(--callout-note-border);
      background: var(--callout-note-bg);
    }}
    .mdexplore-callout-tip {{
      border-left-color: var(--callout-tip-border);
      background: var(--callout-tip-bg);
    }}
    .mdexplore-callout-important {{
      border-left-color: var(--callout-important-border);
      background: var(--callout-important-bg);
    }}
    .mdexplore-callout-warning {{
      border-left-color: var(--callout-warning-border);
      background: var(--callout-warning-bg);
    }}
    .mdexplore-callout-caution {{
      border-left-color: var(--callout-caution-border);
      background: var(--callout-caution-bg);
    }}
    @media print {{
      .mdexplore-callout {{
        border-left-width: 4px;
      }}
    }}
    mjx-container[display="true"] {{
      margin: 0.9em 0 1.05em 0 !important;
    }}
    mjx-container[jax="SVG"] {{
      font-size: 1.07em;
    }}
    mjx-container[jax="SVG"] > svg {{
      max-width: 100%;
      height: auto;
      overflow: visible;
    }}
    mjx-container[jax="SVG"][display="false"] {{
      vertical-align: -0.08em;
    }}
  </style>
  <script>
    window.MathJax = {{
      startup: {{
        typeset: false
      }},
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      svg: {{
        fontCache: "global",
        scale: 1.05
      }},
      chtml: {{
        scale: 1.05,
        matchFontHeight: false
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }}
    }};
    window.__mdexploreMathReady = false;
    window.__mdexploreMathInFlight = false;
    window.__mdexploreMathJaxSource = "";
    window.__mdexploreMathJaxSources = {mathjax_sources_json};
    window.__mdexploreMathJaxLoadPromise = null;
    window.__mdexploreMermaidReady = false;
    window.__mdexploreMermaidSource = "";
    window.__mdexploreMermaidSources = {mermaid_sources_json};
    window.__mdexploreMermaidLoadPromise = null;
    window.__mdexploreMermaidAttempted = false;
    window.__mdexploreMermaidPaletteMode = "auto";
    window.__mdexploreMermaidRenderPromise = null;
    window.__mdexploreMermaidRenderMode = "";

    window.__mdexploreLoadMathJaxScript = () => {{
      if (window.__mdexploreMathJaxLoadPromise) {{
        return window.__mdexploreMathJaxLoadPromise;
      }}
      window.__mdexploreMathJaxLoadPromise = (async () => {{
        const sources = Array.isArray(window.__mdexploreMathJaxSources) ? window.__mdexploreMathJaxSources : [];
        for (const src of sources) {{
          try {{
            await new Promise((resolve, reject) => {{
              const script = document.createElement("script");
              script.src = src;
              script.defer = true;
              script.onload = () => resolve(true);
              script.onerror = () => reject(new Error(`Failed to load ${{src}}`));
              document.head.appendChild(script);
            }});
            if (window.MathJax && MathJax.typesetPromise) {{
              window.__mdexploreMathJaxSource = src;
              return true;
            }}
          }} catch (error) {{
            console.error("mdexplore MathJax script load failed:", src, error);
          }}
        }}
        return false;
      }})();
      return window.__mdexploreMathJaxLoadPromise;
    }};

    window.__mdexploreLoadMermaidScript = () => {{
      if (window.__mdexploreMermaidLoadPromise) {{
        return window.__mdexploreMermaidLoadPromise;
      }}
      window.__mdexploreMermaidLoadPromise = (async () => {{
        const sources = Array.isArray(window.__mdexploreMermaidSources) ? window.__mdexploreMermaidSources : [];
        for (const src of sources) {{
          try {{
            await new Promise((resolve, reject) => {{
              const script = document.createElement("script");
              script.src = src;
              script.defer = true;
              script.onload = () => resolve(true);
              script.onerror = () => reject(new Error(`Failed to load ${{src}}`));
              document.head.appendChild(script);
            }});
            if (window.mermaid) {{
              window.__mdexploreMermaidSource = src;
              return true;
            }}
          }} catch (error) {{
            console.error("mdexplore Mermaid script load failed:", src, error);
          }}
        }}
        return false;
      }})();
      return window.__mdexploreMermaidLoadPromise;
    }};

    window.__mdexploreParseCssRgb = (colorText) => {{
      const text = String(colorText || "").trim();
      const match = text.match(/^rgba?\\(([^)]+)\\)$/i);
      if (!match) {{
        return null;
      }}
      const parts = match[1].split(",").map((part) => Number.parseFloat(part.trim()));
      if (parts.length < 3 || parts.some((value) => Number.isNaN(value))) {{
        return null;
      }}
      return [parts[0], parts[1], parts[2]];
    }};

    window.__mdexploreIsDarkBackground = () => {{
      try {{
        const body = document.body || document.documentElement;
        if (!body) {{
          return true;
        }}
        const bg = getComputedStyle(body).backgroundColor || "";
        const rgb = window.__mdexploreParseCssRgb(bg);
        if (rgb) {{
          const [r, g, b] = rgb.map((value) => Math.max(0, Math.min(255, value)) / 255);
          const luminance = (0.2126 * r) + (0.7152 * g) + (0.0722 * b);
          return luminance < 0.56;
        }}
      }} catch (_error) {{
        // Fall through to media query fallback.
      }}
      return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
    }};

    window.__mdexploreMermaidInitConfig = (mode = "auto") => {{
      const usePdfMode = String(mode || "").toLowerCase() === "pdf";
      const dark = !usePdfMode && window.__mdexploreIsDarkBackground();
      const shared = {{
        startOnLoad: false,
        securityLevel: "loose",
        fontFamily: "Noto Sans, DejaVu Sans, sans-serif",
        flowchart: {{ htmlLabels: true, useMaxWidth: true }},
        sequence: {{ useMaxWidth: true }},
        gantt: {{ useMaxWidth: true }},
      }};

      if (dark) {{
        return {{
          ...shared,
          theme: "base",
          darkMode: true,
          themeVariables: {{
            background: "#0f172a",
            primaryColor: "#1e293b",
            primaryBorderColor: "#93c5fd",
            primaryTextColor: "#e5e7eb",
            secondaryColor: "#172554",
            secondaryBorderColor: "#93c5fd",
            secondaryTextColor: "#e5e7eb",
            tertiaryColor: "#111827",
            tertiaryBorderColor: "#94a3b8",
            tertiaryTextColor: "#e5e7eb",
            lineColor: "#d1d5db",
            textColor: "#e5e7eb",
            mainBkg: "#1e293b",
            nodeBkg: "#1e293b",
            clusterBkg: "#1f2937",
            clusterBorder: "#94a3b8",
            edgeLabelBackground: "#0f172a",
            titleColor: "#e5e7eb",
            actorBkg: "#1e293b",
            actorBorder: "#93c5fd",
            actorTextColor: "#e5e7eb",
            actorLineColor: "#d1d5db",
            signalColor: "#d1d5db",
            noteBkgColor: "#1f2937",
            noteTextColor: "#e5e7eb",
            noteBorderColor: "#93c5fd",
            labelTextColor: "#e5e7eb",
            cScale0: "#60a5fa",
            cScale1: "#93c5fd",
            cScale2: "#bfdbfe",
            cScale3: "#e2e8f0",
            cScale4: "#cbd5e1",
            cScale5: "#94a3b8",
            cScale6: "#64748b",
            cScale7: "#475569",
            cScale8: "#334155",
            cScale9: "#1f2937",
          }},
        }};
      }}

      return {{
        ...shared,
        theme: "base",
        darkMode: false,
        themeVariables: {{
          background: "#f8fafc",
          primaryColor: "#ffffff",
          primaryBorderColor: "#1f2937",
          primaryTextColor: "#111827",
          secondaryColor: "#f1f5f9",
          secondaryBorderColor: "#334155",
          secondaryTextColor: "#111827",
          tertiaryColor: "#e2e8f0",
          tertiaryBorderColor: "#334155",
          tertiaryTextColor: "#111827",
          lineColor: "#1f2937",
          textColor: "#111827",
          edgeLabelBackground: "#f8fafc",
          noteBkgColor: "#eef2ff",
          noteTextColor: "#111827",
          noteBorderColor: "#334155",
          actorBkg: "#ffffff",
          actorBorder: "#1f2937",
          actorTextColor: "#111827",
          actorLineColor: "#1f2937",
          signalColor: "#1f2937",
          clusterBkg: "#f1f5f9",
          clusterBorder: "#334155",
          labelTextColor: "#111827",
        }},
      }};
    }};

    window.__mdexploreTransformCallouts = () => {{
      const root = document.querySelector("main") || document.body;
      if (!root) {{
        return 0;
      }}
      const quotes = Array.from(root.querySelectorAll("blockquote"));
      if (!quotes.length) {{
        return 0;
      }}

      const titleByType = {{
        note: "Note",
        info: "Info",
        tip: "Tip",
        important: "Important",
        warning: "Warning",
        caution: "Caution",
      }};
      const iconByStyleType = {{
        note: "i",
        tip: "+",
        important: "!",
        warning: "!",
        caution: "!",
      }};
      const aliasByType = {{
        info: "note",
      }};

      const markerLineRegex = /^\\s*\\[!([A-Za-z0-9_-]+)\\]\\s*([+-])?(?:\\s+(.*))?\\s*$/;
      let transformed = 0;

      for (const quote of quotes) {{
        const firstBlock = quote.firstElementChild;
        if (!firstBlock) {{
          continue;
        }}
        const firstLine = ((firstBlock.textContent || "").split(/\\r?\\n/, 1)[0] || "").trim();
        const markerMatch = firstLine.match(markerLineRegex);
        if (!markerMatch) {{
          continue;
        }}

        const rawType = (markerMatch[1] || "note").trim().toLowerCase();
        const customTitle = (markerMatch[3] || "").trim();
        const normalizedType = aliasByType[rawType] || rawType;
        const styleType = {{
          note: "note",
          tip: "tip",
          important: "important",
          warning: "warning",
          caution: "caution",
        }}[normalizedType] || "note";
        const titleText =
          customTitle ||
          titleByType[rawType] ||
          titleByType[normalizedType] ||
          (rawType ? rawType.charAt(0).toUpperCase() + rawType.slice(1) : "Note");

        // Remove only the first marker line from leading text nodes while
        // preserving nested inline markup in the callout body.
        const firstBlockText = firstBlock.textContent || "";
        let charsToRemove = 0;
        const firstLineBoundary = firstBlockText.match(/^\\s*[^\\r\\n]*/);
        if (firstLineBoundary) {{
          charsToRemove = firstLineBoundary[0].length;
        }}
        const newlineBoundary = firstBlockText.slice(charsToRemove).match(/^\\r?\\n/);
        if (newlineBoundary) {{
          charsToRemove += newlineBoundary[0].length;
        }}

        if (charsToRemove > 0) {{
          const textWalker = document.createTreeWalker(firstBlock, NodeFilter.SHOW_TEXT);
          const textNodes = [];
          while (textWalker.nextNode()) {{
            textNodes.push(textWalker.currentNode);
          }}
          for (const node of textNodes) {{
            if (charsToRemove <= 0) {{
              break;
            }}
            const nodeText = node.nodeValue || "";
            if (nodeText.length <= charsToRemove) {{
              charsToRemove -= nodeText.length;
              node.nodeValue = "";
            }} else {{
              node.nodeValue = nodeText.slice(charsToRemove);
              charsToRemove = 0;
            }}
          }}
        }}

        // Drop leading blank text nodes and <br> left behind by marker stripping.
        while (firstBlock.firstChild && firstBlock.firstChild.nodeType === Node.TEXT_NODE) {{
          const value = firstBlock.firstChild.nodeValue || "";
          if (!value.trim()) {{
            firstBlock.removeChild(firstBlock.firstChild);
            continue;
          }}
          break;
        }}
        while (firstBlock.firstChild && firstBlock.firstChild.nodeType === Node.ELEMENT_NODE) {{
          const element = firstBlock.firstChild;
          if (element.tagName === "BR") {{
            firstBlock.removeChild(element);
            continue;
          }}
          break;
        }}

        const hasBodyText = ((firstBlock.textContent || "").trim().length > 0) || firstBlock.children.length > 0;
        if (!hasBodyText) {{
          firstBlock.remove();
        }}

        const callout = document.createElement("div");
        callout.className = "mdexplore-callout mdexplore-callout-" + styleType;
        callout.setAttribute("data-callout", rawType);
        for (const attrName of ["data-md-line-start", "data-md-line-end"]) {{
          if (quote.hasAttribute(attrName)) {{
            callout.setAttribute(attrName, quote.getAttribute(attrName) || "");
          }}
        }}

        const header = document.createElement("div");
        header.className = "mdexplore-callout-title";
        const icon = document.createElement("span");
        icon.className = "mdexplore-callout-icon";
        icon.setAttribute("aria-hidden", "true");
        icon.textContent = iconByStyleType[styleType] || "i";
        const label = document.createElement("span");
        label.className = "mdexplore-callout-title-text";
        label.textContent = titleText;
        header.appendChild(icon);
        header.appendChild(label);

        const body = document.createElement("div");
        body.className = "mdexplore-callout-content";
        while (quote.firstChild) {{
          body.appendChild(quote.firstChild);
        }}

        callout.appendChild(header);
        callout.appendChild(body);
        quote.replaceWith(callout);
        transformed += 1;
      }}

      return transformed;
    }};

    window.__mdexploreNormalizeMathText = () => {{
      // markdown-it treats \\# as an escape and can strip the backslash before
      // MathJax runs. Re-escape bare # inside $...$ / $$...$$ text so cardinality
      // expressions like #{{...}} remain valid TeX.
      const root = document.querySelector("main") || document.body;
      if (!root) {{
        return 0;
      }}

      const skipTags = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA", "PRE", "CODE"]);

      const escapeHashes = (mathText) => {{
        let out = "";
        for (let i = 0; i < mathText.length; i += 1) {{
          const ch = mathText[i];
          if (ch !== "#") {{
            out += ch;
            continue;
          }}
          const prev = i > 0 ? mathText[i - 1] : "";
          const next = i + 1 < mathText.length ? mathText[i + 1] : "";
          if (prev !== "\\\\" && !(next >= "0" && next <= "9")) {{
            out += "\\\\#";
          }} else {{
            out += ch;
          }}
        }}
        return out;
      }};

      const findClosingDouble = (text, start) => {{
        for (let i = start; i + 1 < text.length; i += 1) {{
          if (text[i] === "$" && text[i + 1] === "$" && (i === 0 || text[i - 1] !== "\\\\")) {{
            return i;
          }}
        }}
        return -1;
      }};

      const findClosingSingle = (text, start) => {{
        for (let i = start; i < text.length; i += 1) {{
          if (text[i] !== "$" || (i > 0 && text[i - 1] === "\\\\")) {{
            continue;
          }}
          // Treat $$ as double-delimiter, not the end of an inline span.
          if (i + 1 < text.length && text[i + 1] === "$") {{
            continue;
          }}
          return i;
        }}
        return -1;
      }};

      const normalizeNodeText = (text) => {{
        if (!text || text.indexOf("$") === -1 || text.indexOf("#") === -1) {{
          return text;
        }}
        let out = "";
        let i = 0;
        while (i < text.length) {{
          if (text[i] !== "$") {{
            out += text[i];
            i += 1;
            continue;
          }}

          if (i + 1 < text.length && text[i + 1] === "$") {{
            const end = findClosingDouble(text, i + 2);
            if (end < 0) {{
              out += text[i];
              i += 1;
              continue;
            }}
            const body = text.slice(i + 2, end);
            out += "$$" + escapeHashes(body) + "$$";
            i = end + 2;
            continue;
          }}

          const end = findClosingSingle(text, i + 1);
          if (end < 0) {{
            out += text[i];
            i += 1;
            continue;
          }}
          const body = text.slice(i + 1, end);
          out += "$" + escapeHashes(body) + "$";
          i = end + 1;
        }}
        return out;
      }};

      let updated = 0;
      const walker = document.createTreeWalker(
        root,
        NodeFilter.SHOW_TEXT,
        {{
          acceptNode(node) {{
            if (!node || !node.nodeValue || node.nodeValue.indexOf("$") === -1) {{
              return NodeFilter.FILTER_REJECT;
            }}
            const parent = node.parentElement;
            if (!parent || skipTags.has(parent.tagName)) {{
              return NodeFilter.FILTER_REJECT;
            }}
            return NodeFilter.FILTER_ACCEPT;
          }},
        }},
      );

      const nodes = [];
      while (walker.nextNode()) {{
        nodes.push(walker.currentNode);
      }}
      for (const node of nodes) {{
        const nextText = normalizeNodeText(node.nodeValue || "");
        if (nextText !== node.nodeValue) {{
          node.nodeValue = nextText;
          updated += 1;
        }}
      }}
      return updated;
    }};

    window.__mdexploreRunMermaidWithMode = async (mode = "auto", force = false) => {{
      const normalizedMode = String(mode || "").toLowerCase() === "pdf" ? "pdf" : "auto";
      if (
        !force &&
        window.__mdexploreMermaidReady &&
        window.__mdexploreMermaidPaletteMode === normalizedMode
      ) {{
        return window.__mdexploreMermaidReady;
      }}
      if (window.__mdexploreMermaidRenderPromise) {{
        if (window.__mdexploreMermaidRenderMode === normalizedMode) {{
          return await window.__mdexploreMermaidRenderPromise;
        }}
        try {{
          await window.__mdexploreMermaidRenderPromise;
        }} catch (_error) {{
          // Re-render attempt below with requested mode.
        }}
      }}
      window.__mdexploreMermaidAttempted = true;
      window.__mdexploreMermaidRenderMode = normalizedMode;
      window.__mdexploreMermaidRenderPromise = (async () => {{
        try {{
          const loaded = await window.__mdexploreLoadMermaidScript();
          if (!loaded || !window.mermaid) {{
            throw new Error("Mermaid script failed to load from local/CDN sources");
          }}
          const mermaidBlocks = Array.from(document.querySelectorAll(".mermaid"));
          for (const block of mermaidBlocks) {{
            if (!(block instanceof HTMLElement)) {{
              continue;
            }}
            const existingSource = (block.dataset && block.dataset.mdexploreMermaidSource) || "";
            if (!existingSource) {{
              const rawSource = (block.textContent || "").trim();
              if (rawSource) {{
                block.dataset.mdexploreMermaidSource = rawSource;
              }}
            }}
            const sourceText = (block.dataset && block.dataset.mdexploreMermaidSource) || "";
            if (!sourceText) {{
              continue;
            }}
            block.removeAttribute("data-processed");
            block.textContent = sourceText;
          }}
          const mermaidConfig = window.__mdexploreMermaidInitConfig(normalizedMode);
          mermaid.initialize(mermaidConfig);
          await mermaid.run({{ querySelector: '.mermaid' }});
          window.__mdexploreMermaidReady = true;
          window.__mdexploreMermaidPaletteMode = normalizedMode;
        }} catch (error) {{
          window.__mdexploreMermaidReady = false;
          window.__mdexploreMermaidPaletteMode = normalizedMode;
          console.error("mdexplore Mermaid render failed:", error);
        }} finally {{
          window.__mdexploreMermaidRenderPromise = null;
          window.__mdexploreMermaidRenderMode = "";
        }}
        return window.__mdexploreMermaidReady;
      }})();
      return await window.__mdexploreMermaidRenderPromise;
    }};

    window.__mdexploreRunMermaid = async () => {{
      return window.__mdexploreRunMermaidWithMode("auto", false);
    }};

    window.__mdexploreTryTypesetMath = async () => {{
      if (window.__mdexploreMathInFlight) {{
        return window.__mdexploreMathReady;
      }}
      window.__mdexploreMathInFlight = true;
      try {{
        window.__mdexploreNormalizeMathText();
        const loaded = await window.__mdexploreLoadMathJaxScript();
        if (!loaded) {{
          throw new Error("MathJax script failed to load from local/CDN sources");
        }}
        if (!(window.MathJax && MathJax.typesetPromise)) {{
          throw new Error("MathJax runtime not available yet");
        }}
        if (MathJax.startup && MathJax.startup.promise) {{
          await MathJax.startup.promise;
        }}
        await MathJax.typesetPromise();
        window.__mdexploreMathReady = true;
        window.__mdexploreMathError = "";
      }} catch (error) {{
        window.__mdexploreMathReady = false;
        window.__mdexploreMathError = (error && error.message) ? error.message : String(error);
        console.error("mdexplore MathJax render failed:", error);
      }} finally {{
        window.__mdexploreMathInFlight = false;
      }}
      return window.__mdexploreMathReady;
    }};

    window.__mdexploreScheduleMathRetries = () => {{
      // External script load timing can lag; retry shortly before giving up.
      for (const delayMs of [160, 420, 900, 1700]) {{
        window.setTimeout(() => {{
          if (!window.__mdexploreMathReady) {{
            window.__mdexploreTryTypesetMath();
          }}
        }}, delayMs);
      }}
    }};

    window.__mdexploreRunClientRenderers = async (options = null) => {{
      if (window.__mdexploreTransformCallouts) {{
        window.__mdexploreTransformCallouts();
      }}
      const mermaidMode =
        options && typeof options === "object" && String(options.mermaidMode || "").toLowerCase() === "pdf"
          ? "pdf"
          : "auto";
      // Keep Mermaid failures isolated so math rendering is never blocked.
      if (
        !window.__mdexploreMermaidReady ||
        window.__mdexploreMermaidPaletteMode !== mermaidMode
      ) {{
        await window.__mdexploreRunMermaidWithMode(mermaidMode, false);
      }}
      const mathReady = await window.__mdexploreTryTypesetMath();
      if (!mathReady) {{
        window.__mdexploreScheduleMathRetries();
      }}
    }};

    window.__mdexploreLoadMathJaxScript();
    window.__mdexploreLoadMermaidScript();
  </script>
</head>
<body>
  <main>{body}</main>
  <script>
    window.addEventListener('DOMContentLoaded', () => {{
      // Start client-side renderers once content is mounted.
      if (window.__mdexploreRunClientRenderers) {{
        window.__mdexploreRunClientRenderers();
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


class PdfExportWorkerSignals(QObject):
    """Signals emitted by background PDF export workers."""

    finished = Signal(str, str)


class PdfExportWorker(QRunnable):
    """Apply footer page numbers and write exported PDF in background."""

    def __init__(self, output_path: Path, pdf_bytes: bytes):
        super().__init__()
        self.output_path = output_path
        self.pdf_bytes = pdf_bytes
        self.signals = PdfExportWorkerSignals()

    def run(self) -> None:
        try:
            stamped_pdf = _stamp_pdf_page_numbers(self.pdf_bytes)
            self.output_path.write_bytes(stamped_pdf)
            self.signals.finished.emit(str(self.output_path), "")
        except Exception as exc:
            self.signals.finished.emit(str(self.output_path), str(exc))


class ViewTabBar(QTabBar):
    """Custom tab bar that paints dark-theme-friendly pastel tab backgrounds."""

    PASTEL_SEQUENCE = [
        "#8fb8ff",
        "#9fd8c9",
        "#d7b8ff",
        "#f6c89f",
        "#f4b8c9",
        "#c8d8a0",
        "#b5d5f4",
        "#e8c6a7",
    ]
    WIDTH_TEMPLATE_TEXT = "999999"
    WIDTH_SIDE_PADDING = 10
    POSITION_BAR_WIDTH = 26
    POSITION_BAR_HEIGHT = 8
    POSITION_BAR_TEXT_GAP = 7
    POSITION_BAR_SEGMENTS = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_candidate_index = -1
        self._drag_start_pos = None
        self._dragging_index = -1
        self._dragging_tab_x = 0
        self._dragging_tab_offset_x = 0

    @staticmethod
    def _event_pos(event):
        """Return event position as QPoint across Qt6 API variants."""
        try:
            return event.position().toPoint()
        except Exception:
            return event.pos()

    def _is_close_button_hit(self, tab_index: int, pos) -> bool:
        """Detect whether a pointer position targets a tab close button."""
        button = self.tabButton(tab_index, QTabBar.ButtonPosition.RightSide)
        return bool(button is not None and button.isVisible() and button.geometry().contains(pos))

    def _set_all_close_buttons_visible(self, visible: bool) -> None:
        """Show/hide close buttons while custom drag ghost is active."""
        for index in range(self.count()):
            button = self.tabButton(index, QTabBar.ButtonPosition.RightSide)
            if button is not None:
                button.setVisible(visible)

    def _begin_tab_drag(self, tab_index: int, pos) -> None:
        """Start custom full-tab drag ghost for smoother visual feedback."""
        if tab_index < 0 or tab_index >= self.count():
            return
        rect = self.tabRect(tab_index)
        if not rect.isValid():
            return

        self._dragging_index = tab_index
        self._dragging_tab_offset_x = max(0, min(rect.width() - 1, pos.x() - rect.x()))
        self._dragging_tab_x = rect.x()
        self._set_all_close_buttons_visible(False)
        self.update()

    def _target_index_for_x(self, center_x: int) -> int:
        """Map cursor X position to closest insertion tab index."""
        if self.count() <= 0:
            return -1
        for index in range(self.count()):
            if center_x <= self.tabRect(index).center().x():
                return index
        return self.count() - 1

    def _update_tab_drag(self, pos) -> None:
        """Move drag ghost and reorder tabs as cursor crosses boundaries."""
        if self._dragging_index < 0:
            return
        self._dragging_tab_x = pos.x() - self._dragging_tab_offset_x
        target_index = self._target_index_for_x(pos.x())
        if target_index >= 0 and target_index != self._dragging_index:
            self.moveTab(self._dragging_index, target_index)
            self._dragging_index = target_index
        self.update()

    def _end_tab_drag(self) -> None:
        """Finish drag mode and restore normal close button visibility."""
        self._drag_candidate_index = -1
        self._drag_start_pos = None
        self._dragging_index = -1
        self._dragging_tab_x = 0
        self._dragging_tab_offset_x = 0
        self._set_all_close_buttons_visible(True)
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Record potential drag start while preserving regular tab behavior."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self._event_pos(event)
            tab_index = self.tabAt(pos)
            if tab_index >= 0 and not self._is_close_button_hit(tab_index, pos):
                self._drag_candidate_index = tab_index
                self._drag_start_pos = pos
            else:
                self._drag_candidate_index = -1
                self._drag_start_pos = None
        else:
            self._drag_candidate_index = -1
            self._drag_start_pos = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """Drive custom tab drag animation and reorder behavior."""
        if (
            self._drag_candidate_index >= 0
            and self._drag_start_pos is not None
            and (event.buttons() & Qt.MouseButton.LeftButton)
        ):
            pos = self._event_pos(event)
            if self._dragging_index < 0:
                if (pos - self._drag_start_pos).manhattanLength() >= QApplication.startDragDistance():
                    self._begin_tab_drag(self._drag_candidate_index, pos)
            if self._dragging_index >= 0:
                self._update_tab_drag(pos)
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """End custom drag mode on release and continue normal processing."""
        if event.button() == Qt.MouseButton.LeftButton and self._dragging_index >= 0:
            self._end_tab_drag()
            super().mouseReleaseEvent(event)
            return
        self._drag_candidate_index = -1
        self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def _color_slot_for_index(self, tab_index: int) -> int:
        """Resolve palette slot from tab metadata, with sequence fallback."""
        palette_size = len(self.PASTEL_SEQUENCE)
        data = self.tabData(tab_index)
        if isinstance(data, dict):
            raw_slot = data.get("color_slot")
            try:
                slot = int(raw_slot)
            except Exception:
                slot = -1
            if 0 <= slot < palette_size:
                return slot
        sequence = self._sequence_for_index(tab_index)
        return (max(1, sequence) - 1) % palette_size

    def _sequence_for_index(self, tab_index: int) -> int:
        """Extract open-sequence index from tab data with backward compatibility."""
        data = self.tabData(tab_index)
        if isinstance(data, dict):
            raw = data.get("sequence")
            if isinstance(raw, int) and raw > 0:
                return raw
        if isinstance(data, int) and data > 0:
            return data
        return tab_index + 1

    def _base_color_for_index(self, tab_index: int) -> QColor:
        """Return base color for a tab from the configured pastel sequence."""
        color_slot = self._color_slot_for_index(tab_index)
        color_hex = self.PASTEL_SEQUENCE[color_slot]
        return QColor(color_hex)

    def _paint_single_tab(self, painter: QPainter, tab_index: int, rect, *, selected: bool, force_opaque: bool) -> None:
        """Paint one tab using shared logic for static and drag-ghost rendering."""
        base = self._base_color_for_index(tab_index)
        fill = QColor(base)
        if selected:
            fill = fill.lighter(107)
            if force_opaque:
                fill.setAlpha(255)
            else:
                fill.setAlpha(236)
            border = QColor(fill).darker(130)
        else:
            if force_opaque:
                fill.setAlpha(244)
            else:
                fill.setAlpha(172)
            border = QColor(fill).darker(155)

        painter.setPen(QPen(border, 1.1))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # Draw a compact segmented bargraph at the left to indicate each
        # tab's approximate position within the current document.
        bar_x = rect.left() + self.WIDTH_SIDE_PADDING - 1
        bar_y = rect.center().y() - (self.POSITION_BAR_HEIGHT // 2)
        bar_w = self.POSITION_BAR_WIDTH
        bar_h = self.POSITION_BAR_HEIGHT
        track_fill = QColor("#1f2937" if selected else "#182233")
        if force_opaque:
            track_fill.setAlpha(236 if selected else 214)
        else:
            track_fill.setAlpha(188 if selected else 152)
        track_border = QColor("#314156")
        painter.setPen(QPen(track_border, 0.9))
        painter.setBrush(track_fill)
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 2.2, 2.2)

        inner_x = bar_x + 1
        inner_y = bar_y + 1
        inner_w = max(1, bar_w - 2)
        inner_h = max(1, bar_h - 2)
        segments = self.POSITION_BAR_SEGMENTS
        segment_gap = 1
        segment_w = max(1, (inner_w - ((segments - 1) * segment_gap)) // segments)
        used_w = (segment_w * segments) + ((segments - 1) * segment_gap)
        start_x = inner_x + max(0, (inner_w - used_w) // 2)
        progress = self._progress_for_index(tab_index)
        filled_segments = int(round(progress * segments))
        if progress > 0.0 and filled_segments <= 0:
            filled_segments = 1
        filled_segments = max(0, min(segments, filled_segments))
        segment_active = QColor(base).darker(116 if selected else 128)
        segment_inactive = QColor("#425066")
        if force_opaque:
            segment_inactive.setAlpha(148 if selected else 126)
        else:
            segment_inactive.setAlpha(115 if selected else 92)
        painter.setPen(Qt.PenStyle.NoPen)
        for segment_index in range(segments):
            segment_x = start_x + (segment_index * (segment_w + segment_gap))
            segment_color = segment_active if segment_index < filled_segments else segment_inactive
            painter.setBrush(segment_color)
            painter.drawRect(segment_x, inner_y, segment_w, inner_h)

        text_left = bar_x + bar_w + self.POSITION_BAR_TEXT_GAP
        text_rect = rect.adjusted(text_left - rect.left(), 0, -9, 0)
        close_button = self.tabButton(tab_index, QTabBar.ButtonPosition.RightSide)
        if close_button is not None and close_button.isVisible():
            text_rect.setRight(min(text_rect.right(), close_button.geometry().left() - 4))

        text_color = QColor("#0b1220" if selected else "#1b2436")
        if not self.isTabEnabled(tab_index):
            text_color.setAlpha(130)
        painter.setPen(text_color)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.tabText(tab_index))

    def _progress_for_index(self, tab_index: int) -> float:
        """Read normalized document-position progress (0..1) from tab metadata."""
        data = self.tabData(tab_index)
        if isinstance(data, dict):
            raw = data.get("progress")
            try:
                value = float(raw)
            except Exception:
                value = 0.0
            if math.isfinite(value):
                return max(0.0, min(1.0, value))
        return 0.0

    def _constant_tab_width(self) -> int:
        """Compute a stable tab width sized for six digits plus close button."""
        text_width = self.fontMetrics().horizontalAdvance(self.WIDTH_TEMPLATE_TEXT)
        close_width = 0
        if self.tabsClosable():
            close_width = self.style().pixelMetric(QStyle.PixelMetric.PM_TabCloseIndicatorWidth, None, self) + 8
        return (
            text_width
            + (self.WIDTH_SIDE_PADDING * 2)
            + self.POSITION_BAR_WIDTH
            + self.POSITION_BAR_TEXT_GAP
            + close_width
        )

    def tabSizeHint(self, index: int) -> QSize:  # noqa: N802
        """Return fixed-width tab sizing so all view tabs align uniformly."""
        base = super().tabSizeHint(index)
        return QSize(self._constant_tab_width(), base.height())

    def minimumTabSizeHint(self, index: int) -> QSize:  # noqa: N802
        """Enforce same fixed width as minimum to prevent style shrink."""
        base = super().minimumTabSizeHint(index)
        return QSize(self._constant_tab_width(), base.height())

    def paintEvent(self, event) -> None:  # noqa: N802
        """Draw rounded pastel tabs while preserving built-in tab close buttons."""
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        for tab_index in range(self.count()):
            if self._dragging_index >= 0 and tab_index == self._dragging_index:
                # Draw active dragged tab as a floating ghost after static tabs.
                continue
            rect = self.tabRect(tab_index).adjusted(2, 2, -2, -1)
            if rect.width() <= 2 or rect.height() <= 2:
                continue

            selected = tab_index == self.currentIndex()
            self._paint_single_tab(painter, tab_index, rect, selected=selected, force_opaque=False)

        if self._dragging_index >= 0 and self.count() > 0:
            current_rect = self.tabRect(max(0, min(self._dragging_index, self.count() - 1)))
            ghost_rect = current_rect.adjusted(2, 2, -2, -1)
            if ghost_rect.width() <= 2 or ghost_rect.height() <= 2:
                painter.end()
                return
            draw_y = max(2, current_rect.y() + 2)
            draw_x = int(self._dragging_tab_x + 2)
            max_x = max(2, self.width() - ghost_rect.width() - 2)
            draw_x = max(2, min(max_x, draw_x))
            ghost_rect.moveLeft(draw_x)
            ghost_rect.moveTop(draw_y)
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(5, 10, 18, 108))
            painter.drawRoundedRect(ghost_rect.adjusted(1, 1, 3, 3), 6.0, 6.0)
            selected = self._dragging_index == self.currentIndex()
            self._paint_single_tab(painter, self._dragging_index, ghost_rect, selected=selected, force_opaque=True)
            painter.restore()

        painter.end()


class MdExploreWindow(QMainWindow):
    MAX_DOCUMENT_VIEWS = 8
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
        self._pending_preview_search_close_groups: list[list[tuple[str, bool]]] = []
        self._render_pool = QThreadPool(self)
        self._render_pool.setMaxThreadCount(1)
        self._render_request_id = 0
        self._active_render_workers: set[PreviewRenderWorker] = set()
        self._plantuml_pool = QThreadPool(self)
        self._plantuml_pool.setMaxThreadCount(2)
        self._pdf_pool = QThreadPool(self)
        self._pdf_pool.setMaxThreadCount(1)
        self._active_pdf_workers: set[PdfExportWorker] = set()
        self._pdf_export_in_progress = False
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
        # Signature of the currently previewed markdown file so we can detect
        # on-disk edits and auto-refresh when content changes externally.
        self._current_preview_signature_key: str | None = None
        self._current_preview_signature: tuple[int, int] | None = None
        self._preview_capture_enabled = False
        self._scroll_restore_block_until = 0.0
        self._view_states: dict[int, dict[str, float | int]] = {}
        self._active_view_id: int | None = None
        self._next_view_id = 1
        self._next_view_sequence = 1
        self._next_tab_color_index = 0
        # In-memory per-document tab/view sessions for this app run only.
        self._document_view_sessions: dict[str, dict] = {}
        self._document_line_counts: dict[str, int] = {}
        self._current_document_total_lines = 1
        self._view_line_probe_pending = False
        self._last_view_line_probe_at = 0.0
        self._match_input_text = ""
        self.match_timer = QTimer(self)
        self.match_timer.setSingleShot(True)
        self.match_timer.setInterval(1000)
        self.match_timer.timeout.connect(self._run_match_search)
        self._scroll_capture_timer = QTimer(self)
        self._scroll_capture_timer.setInterval(200)
        self._scroll_capture_timer.timeout.connect(self._capture_current_preview_scroll)
        self._scroll_capture_timer.start()
        self._default_status_text = "Ready"
        self._status_idle_timer = QTimer(self)
        self._status_idle_timer.setInterval(900)
        self._status_idle_timer.timeout.connect(self._ensure_non_empty_status_message)
        self._status_idle_timer.start()
        self._file_change_watch_timer = QTimer(self)
        self._file_change_watch_timer.setInterval(1200)
        self._file_change_watch_timer.timeout.connect(self._on_file_change_watch_tick)
        self._file_change_watch_timer.start()
        # Keep user-adjusted tree/preview pane widths for this app run.
        self._session_splitter_sizes: list[int] | None = None
        self._initial_split_applied = False

        self.setWindowTitle("mdexplore")
        self.setWindowIcon(app_icon)
        # Give the top control bar a bit more horizontal/vertical room by default.
        self.resize(1540, 980)

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
        # Preview pages are loaded as local HTML. Allow remote JS/CSS so CDN
        # assets (MathJax/Mermaid) can load and render as expected.
        preview_settings = self.preview.settings()
        preview_settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        preview_settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        if hasattr(QWebEngineSettings.WebAttribute, "PrintElementBackgrounds"):
            # Keep PDF output visually closer to what users see in the preview.
            preview_settings.setAttribute(QWebEngineSettings.WebAttribute.PrintElementBackgrounds, True)
        self.preview.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.preview.customContextMenuRequested.connect(self._show_preview_context_menu)
        self.preview.loadFinished.connect(self._on_preview_load_finished)

        self.view_tabs = ViewTabBar()
        self.view_tabs.setDocumentMode(True)
        self.view_tabs.setMovable(False)
        self.view_tabs.setDrawBase(False)
        self.view_tabs.setExpanding(False)
        self.view_tabs.setUsesScrollButtons(False)
        self.view_tabs.setTabsClosable(True)
        self.view_tabs.setElideMode(Qt.TextElideMode.ElideNone)
        self.view_tabs.currentChanged.connect(self._on_view_tab_changed)
        self.view_tabs.tabCloseRequested.connect(self._on_view_tab_close_requested)
        self.view_tabs.setVisible(False)
        self._reset_document_views()

        self.up_btn = QPushButton("^")
        self.up_btn.clicked.connect(self._go_up_directory)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_directory_view)

        self.pdf_btn = QPushButton("PDF")
        self.pdf_btn.clicked.connect(self._export_current_preview_pdf)

        self.add_view_btn = QPushButton("Add View")
        self.add_view_btn.clicked.connect(self._add_document_view)

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

        copy_current_btn = QPushButton("")
        copy_current_btn.setFixedSize(18, 18)
        copy_current_btn.setToolTip("Copy currently previewed markdown file")
        copy_current_btn.setStyleSheet("border: 1px solid #4b5563; border-radius: 3px; padding: 0px;")
        pin_icon_path = Path(__file__).resolve().parent / "pin.png"
        if pin_icon_path.is_file():
            pin_icon = QIcon(str(pin_icon_path))
            copy_current_btn.setIcon(pin_icon)
            copy_current_btn.setIconSize(QSize(16, 16))
        else:
            copy_current_btn.setText("P")
        copy_current_btn.clicked.connect(self._copy_current_preview_file_to_clipboard)
        copy_buttons_layout.addWidget(copy_current_btn)

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

        match_label = QLabel("Search and highlight: ")
        self.match_input = QLineEdit()
        self.match_input.setClearButtonEnabled(False)
        self.match_input.setPlaceholderText('words, "quoted phrases", AND/OR/NOT, CLOSE(...)')
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
        top_bar.addWidget(self.pdf_btn)
        top_bar.addWidget(self.add_view_btn)
        top_bar.addWidget(edit_btn)
        top_bar.addWidget(self.path_label, 1)
        top_bar.addWidget(copy_buttons_widget, 0, Qt.AlignmentFlag.AlignRight)
        top_bar.addSpacing(16)
        top_bar.addWidget(match_buttons_widget, 0, Qt.AlignmentFlag.AlignRight)

        top_bar_widget = QWidget()
        top_bar_widget.setLayout(top_bar)
        top_bar_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)
        preview_layout.addWidget(self.view_tabs)
        preview_layout.addWidget(self.preview, 1)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.tree)
        self.splitter.addWidget(preview_container)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.splitterMoved.connect(self._on_splitter_moved)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(top_bar_widget)
        layout.addWidget(self.splitter, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage(self._default_status_text)
        # Root is initialized after widgets exist so view/model indexes are valid.
        self._set_root_directory(self.root)
        self._add_shortcuts()
        self.model.directoryLoaded.connect(self._maybe_apply_initial_split)
        QTimer.singleShot(0, self._maybe_apply_initial_split)

    def _add_shortcuts(self) -> None:
        """Register window-level keyboard shortcuts."""
        refresh_action = QAction("Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._refresh_directory_view)
        self.addAction(refresh_action)

    @staticmethod
    def _view_tab_label_for_line(line_number: int) -> str:
        """Return compact tab text for a view anchored near a source line."""
        return str(max(1, int(line_number)))

    @staticmethod
    def _count_markdown_lines(markdown_text: str) -> int:
        """Return total source lines, treating empty content as one line."""
        return max(1, markdown_text.count("\n") + 1)

    @staticmethod
    def _line_progress(line_number: int, total_lines: int) -> float:
        """Convert top visible line number into normalized 0..1 document progress."""
        line_value = max(1, int(line_number))
        total_value = max(1, int(total_lines))
        if total_value <= 1:
            return 0.0
        return max(0.0, min(1.0, (line_value - 1) / (total_value - 1)))

    def _tab_view_id(self, tab_index: int) -> int | None:
        """Resolve a tab index into an internal view id."""
        if tab_index < 0 or tab_index >= self.view_tabs.count():
            return None
        try:
            value = self.view_tabs.tabData(tab_index)
            if value is None:
                return None
            if isinstance(value, dict):
                raw = value.get("view_id")
                if raw is None:
                    return None
                return int(raw)
            return int(value)
        except Exception:
            return None

    def _used_tab_color_slots(self) -> set[int]:
        """Collect palette slots currently assigned to open tabs."""
        used: set[int] = set()
        palette_size = len(ViewTabBar.PASTEL_SEQUENCE)
        for index in range(self.view_tabs.count()):
            data = self.view_tabs.tabData(index)
            if not isinstance(data, dict):
                continue
            raw_slot = data.get("color_slot")
            try:
                slot = int(raw_slot)
            except Exception:
                continue
            if 0 <= slot < palette_size:
                used.add(slot)
        return used

    def _allocate_next_tab_color_slot(self) -> int:
        """Pick next palette slot in rotation, skipping slots already in use."""
        palette_size = len(ViewTabBar.PASTEL_SEQUENCE)
        if palette_size <= 0:
            return 0
        used = self._used_tab_color_slots()
        start = self._next_tab_color_index % palette_size

        if len(used) < palette_size:
            for offset in range(palette_size):
                slot = (start + offset) % palette_size
                if slot in used:
                    continue
                self._next_tab_color_index = (slot + 1) % palette_size
                return slot

        # Fallback when every slot is occupied (should not happen with max views == palette size).
        slot = start
        self._next_tab_color_index = (slot + 1) % palette_size
        return slot

    def _current_view_state(self) -> dict[str, float | int] | None:
        """Return active view state dictionary when available."""
        if self._active_view_id is None:
            return None
        return self._view_states.get(self._active_view_id)

    def _save_document_view_session(self, path_key: str | None = None) -> None:
        """Snapshot current tab/view state for one document path key."""
        if path_key is None:
            path_key = self._current_preview_path_key()
        if not path_key:
            return

        self._capture_current_preview_scroll(force=True)

        sanitized_states: dict[int, dict[str, float | int]] = {}
        for raw_view_id, raw_state in self._view_states.items():
            try:
                view_id = int(raw_view_id)
            except Exception:
                continue
            if not isinstance(raw_state, dict):
                continue
            try:
                scroll_y = float(raw_state.get("scroll_y", 0.0))
            except Exception:
                scroll_y = 0.0
            if not math.isfinite(scroll_y):
                scroll_y = 0.0
            try:
                top_line = max(1, int(raw_state.get("top_line", 1)))
            except Exception:
                top_line = 1
            sanitized_states[view_id] = {"scroll_y": scroll_y, "top_line": top_line}

        palette_size = max(1, len(ViewTabBar.PASTEL_SEQUENCE))
        tabs: list[dict[str, int]] = []
        max_sequence = 0
        max_view_id = 0
        for index in range(self.view_tabs.count()):
            view_id = self._tab_view_id(index)
            if view_id is None:
                continue
            data = self.view_tabs.tabData(index)
            sequence = index + 1
            color_slot = (sequence - 1) % palette_size
            if isinstance(data, dict):
                raw_sequence = data.get("sequence")
                raw_color_slot = data.get("color_slot")
                try:
                    sequence = max(1, int(raw_sequence))
                except Exception:
                    sequence = index + 1
                try:
                    color_slot = int(raw_color_slot)
                except Exception:
                    color_slot = (sequence - 1) % palette_size
            if color_slot < 0 or color_slot >= palette_size:
                color_slot = (sequence - 1) % palette_size
            state = sanitized_states.get(view_id)
            if state is None:
                state = {"scroll_y": 0.0, "top_line": 1}
                sanitized_states[view_id] = state
            tabs.append({"view_id": view_id, "sequence": sequence, "color_slot": color_slot})
            max_sequence = max(max_sequence, sequence)
            max_view_id = max(max_view_id, view_id)

        active_view_id = self._active_view_id
        if active_view_id is None or all(entry["view_id"] != active_view_id for entry in tabs):
            current_index = self.view_tabs.currentIndex()
            active_view_id = self._tab_view_id(current_index)
        if active_view_id is None and tabs:
            active_view_id = tabs[0]["view_id"]

        next_view_id = max(self._next_view_id, max_view_id + 1)
        next_sequence = max(self._next_view_sequence, max_sequence + 1)
        try:
            next_color_index = int(self._next_tab_color_index) % palette_size
        except Exception:
            next_color_index = 0

        self._document_view_sessions[path_key] = {
            "view_states": sanitized_states,
            "tabs": tabs,
            "active_view_id": active_view_id,
            "next_view_id": next_view_id,
            "next_view_sequence": next_sequence,
            "next_tab_color_index": next_color_index,
        }

    def _restore_document_view_session(self, path_key: str) -> bool:
        """Restore tab/view state for one document if a session snapshot exists."""
        session = self._document_view_sessions.get(path_key)
        if not isinstance(session, dict):
            return False

        raw_states = session.get("view_states")
        raw_tabs = session.get("tabs")
        if not isinstance(raw_states, dict) or not isinstance(raw_tabs, list):
            return False

        view_states: dict[int, dict[str, float | int]] = {}
        for raw_view_id, raw_state in raw_states.items():
            try:
                view_id = int(raw_view_id)
            except Exception:
                continue
            if not isinstance(raw_state, dict):
                continue
            try:
                scroll_y = float(raw_state.get("scroll_y", 0.0))
            except Exception:
                scroll_y = 0.0
            if not math.isfinite(scroll_y):
                scroll_y = 0.0
            try:
                top_line = max(1, int(raw_state.get("top_line", 1)))
            except Exception:
                top_line = 1
            view_states[view_id] = {"scroll_y": scroll_y, "top_line": top_line}

        palette_size = max(1, len(ViewTabBar.PASTEL_SEQUENCE))
        normalized_tabs: list[dict[str, int]] = []
        seen_view_ids: set[int] = set()
        max_sequence = 0
        max_view_id = 0
        for entry in raw_tabs:
            if not isinstance(entry, dict):
                continue
            try:
                view_id = int(entry.get("view_id"))
            except Exception:
                continue
            if view_id in seen_view_ids:
                continue
            seen_view_ids.add(view_id)
            if view_id not in view_states:
                view_states[view_id] = {"scroll_y": 0.0, "top_line": 1}
            try:
                sequence = max(1, int(entry.get("sequence", len(normalized_tabs) + 1)))
            except Exception:
                sequence = len(normalized_tabs) + 1
            try:
                color_slot = int(entry.get("color_slot", (sequence - 1) % palette_size))
            except Exception:
                color_slot = (sequence - 1) % palette_size
            if color_slot < 0 or color_slot >= palette_size:
                color_slot = (sequence - 1) % palette_size
            normalized_tabs.append({"view_id": view_id, "sequence": sequence, "color_slot": color_slot})
            max_sequence = max(max_sequence, sequence)
            max_view_id = max(max_view_id, view_id)

        if not normalized_tabs:
            return False

        blocked = self.view_tabs.blockSignals(True)
        while self.view_tabs.count() > 0:
            self.view_tabs.removeTab(0)
        self._view_states = view_states

        total_lines = max(1, int(self._current_document_total_lines))
        for tab_entry in normalized_tabs:
            view_id = tab_entry["view_id"]
            state = self._view_states.get(view_id, {"scroll_y": 0.0, "top_line": 1})
            try:
                top_line = max(1, int(state.get("top_line", 1)))
            except Exception:
                top_line = 1
            progress = self._line_progress(top_line, total_lines)
            index = self.view_tabs.addTab(self._view_tab_label_for_line(top_line))
            self.view_tabs.setTabData(
                index,
                {
                    "view_id": view_id,
                    "sequence": tab_entry["sequence"],
                    "color_slot": tab_entry["color_slot"],
                    "progress": progress,
                },
            )
            self.view_tabs.setTabToolTip(index, f"Top visible line: {top_line} / {total_lines}")

        try:
            wanted_active = int(session.get("active_view_id"))
        except Exception:
            wanted_active = normalized_tabs[0]["view_id"]
        if all(entry["view_id"] != wanted_active for entry in normalized_tabs):
            wanted_active = normalized_tabs[0]["view_id"]

        active_index = 0
        for index in range(self.view_tabs.count()):
            if self._tab_view_id(index) == wanted_active:
                active_index = index
                break
        self.view_tabs.setCurrentIndex(active_index)
        self._active_view_id = self._tab_view_id(active_index)

        try:
            next_view_id = int(session.get("next_view_id", max_view_id + 1))
        except Exception:
            next_view_id = max_view_id + 1
        try:
            next_sequence = int(session.get("next_view_sequence", max_sequence + 1))
        except Exception:
            next_sequence = max_sequence + 1
        try:
            next_color_index = int(session.get("next_tab_color_index", 0))
        except Exception:
            next_color_index = 0

        self._next_view_id = max(next_view_id, max_view_id + 1)
        self._next_view_sequence = max(next_sequence, max_sequence + 1)
        self._next_tab_color_index = next_color_index % palette_size
        self.view_tabs.blockSignals(blocked)
        self._sync_all_view_tab_progress()
        self._update_view_tabs_visibility()
        self._update_add_view_button_state()
        return True

    def _set_view_tab_line(self, view_id: int, line_number: int) -> None:
        """Update one tab label/tooltip to match its top visible line."""
        line_value = max(1, int(line_number))
        label = self._view_tab_label_for_line(line_value)
        total_lines = max(1, int(self._current_document_total_lines))
        progress = self._line_progress(line_value, total_lines)
        for index in range(self.view_tabs.count()):
            if self._tab_view_id(index) != view_id:
                continue
            if self.view_tabs.tabText(index) != label:
                self.view_tabs.setTabText(index, label)
            data = self.view_tabs.tabData(index)
            if isinstance(data, dict):
                updated_data = dict(data)
                updated_data["progress"] = progress
                self.view_tabs.setTabData(index, updated_data)
            self.view_tabs.setTabToolTip(index, f"Top visible line: {line_value} / {total_lines}")
            self.view_tabs.update(self.view_tabs.tabRect(index))
            break

    def _sync_all_view_tab_progress(self) -> None:
        """Refresh per-tab progress metadata against current document line count."""
        total_lines = max(1, int(self._current_document_total_lines))
        for index in range(self.view_tabs.count()):
            view_id = self._tab_view_id(index)
            if view_id is None:
                continue
            state = self._view_states.get(view_id)
            if state is None:
                line_value = 1
            else:
                try:
                    line_value = max(1, int(state.get("top_line", 1)))
                except Exception:
                    line_value = 1
            progress = self._line_progress(line_value, total_lines)
            data = self.view_tabs.tabData(index)
            if isinstance(data, dict):
                updated_data = dict(data)
                updated_data["progress"] = progress
                self.view_tabs.setTabData(index, updated_data)
            self.view_tabs.setTabToolTip(index, f"Top visible line: {line_value} / {total_lines}")
        self.view_tabs.update()

    def _current_preview_scroll_key(self) -> str | None:
        """Return scroll cache key for current file + active view."""
        path_key = self._current_preview_path_key()
        if path_key is None:
            return None
        view_id = self._active_view_id
        if view_id is None:
            return path_key
        return f"{path_key}::view:{view_id}"

    def _update_add_view_button_state(self) -> None:
        """Enable Add View only when a file is open and tab budget remains."""
        if not hasattr(self, "add_view_btn"):
            return
        can_add = self.current_file is not None and self.view_tabs.count() < self.MAX_DOCUMENT_VIEWS
        self.add_view_btn.setEnabled(can_add)

    def _update_view_tabs_visibility(self) -> None:
        """Show tab strip only when there are multiple views for an open file."""
        if not hasattr(self, "view_tabs"):
            return
        self.view_tabs.setVisible(self.current_file is not None and self.view_tabs.count() > 1)

    def _create_document_view(self, scroll_y: float, top_line: int, *, make_current: bool) -> int:
        """Create a new view tab/state entry and optionally activate it."""
        view_id = self._next_view_id
        self._next_view_id += 1
        sequence = self._next_view_sequence
        self._next_view_sequence += 1
        color_slot = self._allocate_next_tab_color_slot()
        try:
            safe_scroll = float(scroll_y)
        except Exception:
            safe_scroll = 0.0
        if not math.isfinite(safe_scroll):
            safe_scroll = 0.0
        safe_line = max(1, int(top_line))
        self._view_states[view_id] = {"scroll_y": safe_scroll, "top_line": safe_line}
        total_lines = max(1, int(self._current_document_total_lines))
        progress = self._line_progress(safe_line, total_lines)

        tab_index = self.view_tabs.addTab(self._view_tab_label_for_line(safe_line))
        self.view_tabs.setTabData(
            tab_index,
            {"view_id": view_id, "sequence": sequence, "color_slot": color_slot, "progress": progress},
        )
        self.view_tabs.setTabToolTip(tab_index, f"Top visible line: {safe_line} / {total_lines}")

        if make_current:
            blocked = self.view_tabs.blockSignals(True)
            self.view_tabs.setCurrentIndex(tab_index)
            self.view_tabs.blockSignals(blocked)
            self._active_view_id = view_id

        return view_id

    def _reset_document_views(self, initial_scroll: float = 0.0, initial_line: int = 1) -> None:
        """Reset per-document view tabs back to a single base view."""
        self._view_states.clear()
        self._active_view_id = None
        self._next_view_id = 1
        self._next_view_sequence = 1
        self._next_tab_color_index = 0
        self._view_line_probe_pending = False
        self._last_view_line_probe_at = 0.0

        blocked = self.view_tabs.blockSignals(True)
        while self.view_tabs.count() > 0:
            self.view_tabs.removeTab(0)
        self.view_tabs.blockSignals(blocked)
        self._create_document_view(initial_scroll, initial_line, make_current=True)
        self._update_view_tabs_visibility()
        self._update_add_view_button_state()

    def _add_document_view(self) -> None:
        """Create another view tab for the current document at current top line."""
        if self.current_file is None:
            self.statusBar().showMessage("Open a markdown file before adding a view", 3000)
            return
        if self.view_tabs.count() >= self.MAX_DOCUMENT_VIEWS:
            self.statusBar().showMessage(f"Maximum of {self.MAX_DOCUMENT_VIEWS} views reached", 3500)
            return

        self._capture_current_preview_scroll(force=True)
        current_state = self._current_view_state() or {"scroll_y": 0.0, "top_line": 1}
        scroll_y = float(current_state.get("scroll_y", 0.0))
        top_line = int(current_state.get("top_line", 1))
        new_view_id = self._create_document_view(scroll_y, top_line, make_current=False)

        for index in range(self.view_tabs.count()):
            if self._tab_view_id(index) != new_view_id:
                continue
            self.view_tabs.setCurrentIndex(index)
            break

        self._update_view_tabs_visibility()
        self._update_add_view_button_state()
        self.statusBar().showMessage(
            f"Added view {self.view_tabs.count()} of {self.MAX_DOCUMENT_VIEWS} at line {top_line}",
            3000,
        )

    def _on_view_tab_close_requested(self, tab_index: int) -> None:
        """Close one saved view tab while keeping at least one active view."""
        if self.view_tabs.count() <= 1:
            self.statusBar().showMessage("At least one view must remain open", 2500)
            return

        view_id = self._tab_view_id(tab_index)
        if view_id is None:
            return

        self._capture_current_preview_scroll(force=True)
        self._view_states.pop(view_id, None)
        path_key = self._current_preview_path_key()
        if path_key is not None:
            self._preview_scroll_positions.pop(f"{path_key}::view:{view_id}", None)

        self.view_tabs.removeTab(tab_index)
        if self._active_view_id == view_id:
            self._active_view_id = self._tab_view_id(self.view_tabs.currentIndex())

        self._update_view_tabs_visibility()
        self._update_add_view_button_state()

    def _on_view_tab_changed(self, tab_index: int) -> None:
        """Switch active view and restore its own saved scroll position."""
        new_view_id = self._tab_view_id(tab_index)
        if new_view_id is None:
            return

        previous_view_id = self._active_view_id
        if previous_view_id is not None and previous_view_id != new_view_id:
            self._capture_current_preview_scroll(force=True)
        self._active_view_id = new_view_id

        if self.current_file is None:
            self._update_add_view_button_state()
            return

        self._preview_capture_enabled = False
        self._scroll_restore_block_until = time.monotonic() + 0.9
        expected_key = self._current_preview_path_key()
        if expected_key is None:
            return

        QTimer.singleShot(0, lambda key=expected_key: self._restore_current_preview_scroll(key))
        QTimer.singleShot(180, lambda key=expected_key: self._restore_current_preview_scroll(key))
        QTimer.singleShot(520, lambda key=expected_key: self._restore_current_preview_scroll(key))
        QTimer.singleShot(900, lambda key=expected_key: self._enable_preview_scroll_capture_for(key))
        self._request_active_view_top_line_update(force=True)
        self._update_add_view_button_state()

    def _request_active_view_top_line_update(self, force: bool = False) -> None:
        """Probe top-most visible source line and update active tab label."""
        if self.current_file is None or self._active_view_id is None:
            return
        now = time.monotonic()
        if not force:
            if self._view_line_probe_pending:
                return
            if now - self._last_view_line_probe_at < 0.35:
                return

        expected_key = self._current_preview_path_key()
        expected_view_id = self._active_view_id
        if expected_key is None:
            return

        self._view_line_probe_pending = True
        self._last_view_line_probe_at = now
        js = """
(() => {
  const probeX = Math.max(14, Math.floor(window.innerWidth * 0.42));
  const probeYs = [10, 20, 34, 50, 72, 96];
  for (const y of probeYs) {
    const el = document.elementFromPoint(probeX, y);
    if (!el) continue;
    const tagged = el.closest('[data-md-line-start]');
    if (!tagged) continue;
    const value = parseInt(tagged.getAttribute('data-md-line-start') || "", 10);
    if (!Number.isNaN(value)) return value + 1;
  }
  const taggedNodes = document.querySelectorAll('[data-md-line-start]');
  for (const node of taggedNodes) {
    const rect = node.getBoundingClientRect();
    if (rect.bottom < 0) continue;
    const value = parseInt(node.getAttribute('data-md-line-start') || "", 10);
    if (!Number.isNaN(value)) return value + 1;
  }
  return 1;
})();
"""
        self.preview.page().runJavaScript(
            js,
            lambda result, key=expected_key, view_id=expected_view_id: self._on_active_view_line_probe_result(
                key,
                view_id,
                result,
            ),
        )

    def _on_active_view_line_probe_result(self, expected_key: str, expected_view_id: int, result) -> None:
        """Apply top-line probe result to active view tab when still current."""
        self._view_line_probe_pending = False
        if self._current_preview_path_key() != expected_key:
            return
        if self._active_view_id != expected_view_id:
            return

        try:
            line_number = max(1, int(result))
        except Exception:
            line_number = 1

        state = self._view_states.get(expected_view_id)
        if state is None:
            return
        state["top_line"] = line_number
        self._set_view_tab_line(expected_view_id, line_number)

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

    def _ensure_non_empty_status_message(self) -> None:
        """Keep a non-empty status bar message instead of blank idle periods."""
        current = self.statusBar().currentMessage().strip()
        if current:
            return
        self.statusBar().showMessage(self._default_status_text)

    def _preview_plantuml_progress(self) -> tuple[int, int, int] | None:
        """Return (done, pending, failed) PlantUML counts for current preview."""
        path_key = self._current_preview_path_key()
        if path_key is None:
            return None
        placeholders = self._plantuml_placeholders_by_doc.get(path_key, {})
        if not placeholders:
            return (0, 0, 0)

        done = 0
        pending = 0
        failed = 0
        for hash_key in placeholders:
            status = self._plantuml_results.get(hash_key, ("pending", ""))[0]
            if status == "done":
                done += 1
            elif status == "error":
                failed += 1
            else:
                pending += 1
        return (done, pending, failed)

    def _show_preview_progress_status(self) -> None:
        """Publish a detailed current-file status while preview assets settle."""
        if self.current_file is None:
            self.statusBar().showMessage(self._default_status_text)
            return

        progress = self._preview_plantuml_progress()
        if progress is None:
            self.statusBar().showMessage(self._default_status_text)
            return

        done, pending, failed = progress
        if done == 0 and pending == 0 and failed == 0:
            self.statusBar().showMessage(f"Preview ready: {self.current_file.name}", 3500)
            return

        total = done + pending + failed
        if pending > 0:
            message = (
                f"Preview shown: {self.current_file.name} "
                f"(PlantUML {done}/{total} ready, {pending} rendering"
            )
            if failed > 0:
                message += f", {failed} failed"
            message += ")"
            self.statusBar().showMessage(message)
            return

        if failed > 0:
            self.statusBar().showMessage(
                f"Preview ready: {self.current_file.name} "
                f"(PlantUML {done}/{total} ready, {failed} failed)",
                5000,
            )
            return

        self.statusBar().showMessage(
            f"Preview ready: {self.current_file.name} (PlantUML {done}/{total} ready)",
            3500,
        )

    def _set_root_directory(self, new_root: Path) -> None:
        """Re-root the tree view and reset file preview state."""
        self._capture_current_preview_scroll(force=True)
        self._save_document_view_session()
        self._capture_splitter_sizes_for_session()
        self.root = new_root.resolve()
        self.statusBar().showMessage(f"Root changed to {self.root}", 3000)
        self.last_directory_selection = self.root
        self.current_file = None
        self._current_document_total_lines = 1
        self._reset_document_views()
        self._clear_current_preview_signature()
        self._preview_capture_enabled = False
        self._scroll_restore_block_until = 0.0
        self._pending_preview_search_terms = []
        self._pending_preview_search_close_groups = []
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
        self._update_add_view_button_state()
        QTimer.singleShot(0, self._maybe_apply_initial_split)

    def _on_preview_load_finished(self, ok: bool) -> None:
        """Apply deferred in-preview highlighting after a page finishes loading."""
        if not ok:
            self.statusBar().showMessage("Preview load failed", 5000)
            return
        current_key = self._current_preview_path_key()
        if current_key is None:
            return
        # Kick client-side renderer startup now and a bit later to tolerate
        # delayed external script availability (MathJax/Mermaid).
        self._trigger_client_renderers_for(current_key)
        QTimer.singleShot(450, lambda key=current_key: self._trigger_client_renderers_for(key))
        QTimer.singleShot(1500, lambda key=current_key: self._trigger_client_renderers_for(key))
        # PlantUML completions are patched in-place, but a full page load can
        # still happen from cache refreshes; re-apply any ready results.
        self._apply_all_ready_plantuml_to_current_preview()
        has_saved_scroll = self._has_saved_scroll_for_current_preview()
        if self._pending_preview_search_terms:
            # Search normally scrolls to first hit. If this file has a saved
            # scroll position, preserve that location instead.
            self._highlight_preview_search_terms(
                self._pending_preview_search_terms,
                scroll_to_first=not has_saved_scroll,
                close_term_groups=self._pending_preview_search_close_groups,
            )
        if has_saved_scroll:
            # Re-apply a few times because late MathJax/Mermaid/layout work can
            # shift content after the initial load.
            self._preview_capture_enabled = False
            self._scroll_restore_block_until = time.monotonic() + 1.2
            QTimer.singleShot(90, lambda key=current_key: self._restore_current_preview_scroll(key))
            QTimer.singleShot(320, lambda key=current_key: self._restore_current_preview_scroll(key))
            QTimer.singleShot(900, lambda key=current_key: self._restore_current_preview_scroll(key))
            QTimer.singleShot(1250, lambda key=current_key: self._enable_preview_scroll_capture_for(key))
        else:
            self._preview_capture_enabled = True
            self._scroll_restore_block_until = 0.0
        self._request_active_view_top_line_update(force=True)
        self._show_preview_progress_status()

    def _trigger_client_renderers_for(self, expected_key: str) -> None:
        """Run in-page renderer helpers only if the same preview is still active."""
        if self._current_preview_path_key() != expected_key:
            return
        js = """
(() => {
  if (window.__mdexploreRunClientRenderers) {
    window.__mdexploreRunClientRenderers();
  } else if (window.__mdexploreTryTypesetMath) {
    window.__mdexploreTryTypesetMath();
  }
})();
"""
        self.preview.page().runJavaScript(js)

    def _update_up_button_state(self) -> None:
        self.up_btn.setEnabled(self.root.parent != self.root)

    def _go_up_directory(self) -> None:
        """Navigate one level up from the current root."""
        parent = self.root.parent
        if parent == self.root:
            return
        self._set_root_directory(parent)

    def _expanded_directory_paths(self) -> list[str]:
        """Capture currently expanded directory paths under the visible root."""
        root_index = self.tree.rootIndex()
        if not root_index.isValid():
            return []

        stack = [root_index]
        expanded_paths: list[str] = []
        seen: set[str] = set()

        while stack:
            index = stack.pop()
            if not index.isValid():
                continue

            path_text = self.model.filePath(index)
            path = Path(path_text)
            try:
                is_dir = path.is_dir()
            except Exception:
                is_dir = False
            if not is_dir:
                continue

            if self.tree.isExpanded(index):
                path_key = self._path_key(path)
                if path_key not in seen:
                    seen.add(path_key)
                    expanded_paths.append(path_key)

                # Ensure child indexes are available so nested expanded paths
                # can be captured/restored.
                if self.model.canFetchMore(index):
                    self.model.fetchMore(index)
                for row in range(self.model.rowCount(index)):
                    child_index = self.model.index(row, 0, index)
                    if child_index.isValid():
                        stack.append(child_index)

        return expanded_paths

    def _restore_expanded_directory_paths(self, paths: list[str]) -> None:
        """Restore expanded directories that still exist after a model refresh."""
        for path_text in paths:
            index = self.model.index(path_text)
            if index.isValid():
                self.tree.expand(index)

    def _refresh_directory_view(self, _checked: bool = False) -> None:
        """Refresh tree contents to detect newly created/deleted markdown files."""
        self.statusBar().showMessage("Refreshing directory view...")
        selected_path: Path | None = None
        current_index = self.tree.currentIndex()
        if current_index.isValid():
            try:
                selected_path = Path(self.model.filePath(current_index)).resolve()
            except Exception:
                selected_path = Path(self.model.filePath(current_index))

        expanded_paths = self._expanded_directory_paths()

        # Force QFileSystemModel to re-scan root by toggling root path.
        self.model.setRootPath("")
        root_index = self.model.setRootPath(str(self.root))
        self.tree.setRootIndex(root_index)

        if expanded_paths:
            self._restore_expanded_directory_paths(expanded_paths)

        restored_selection = False
        if selected_path is not None:
            selected_index = self.model.index(str(selected_path))
            if selected_index.isValid():
                self.tree.setCurrentIndex(selected_index)
                restored_selection = True

        if self.current_file is not None:
            try:
                current_exists = self.current_file.is_file()
            except Exception:
                current_exists = False
            if not current_exists:
                self.current_file = None
                self._reset_document_views()
                self._clear_current_preview_signature()
                self.path_label.setText("Select a markdown file")
                self.preview.setHtml(
                    self._placeholder_html("Select a markdown file to preview"),
                    QUrl.fromLocalFile(f"{self.root}/"),
                )
                if not restored_selection:
                    self.tree.clearSelection()
                self.statusBar().showMessage("Directory view refreshed; preview file no longer exists", 4500)
            else:
                self.statusBar().showMessage("Directory view refreshed", 2500)
        else:
            self.statusBar().showMessage("Directory view refreshed", 2500)

        self._update_window_title()
        self._update_up_button_state()
        self._update_add_view_button_state()
        self._rerun_active_search_for_scope()

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        """Persist current pane split after any manual divider movement."""
        self._capture_splitter_sizes_for_session()

    def _capture_splitter_sizes_for_session(self) -> None:
        """Store non-zero splitter sizes for reuse while app stays open."""
        sizes = self.splitter.sizes()
        if len(sizes) != 2:
            return
        left_width = int(sizes[0])
        right_width = int(sizes[1])
        if left_width <= 0 or right_width <= 0:
            return
        self._session_splitter_sizes = [left_width, right_width]

    def _maybe_apply_initial_split(self, *_args) -> None:
        # Qt may override splitter sizes during initial layout/model load.
        # Re-apply either user-adjusted session split or initial 25/75 once
        # real geometry is known.
        if self._initial_split_applied:
            return
        total_width = max(self.splitter.width(), self.width())
        if total_width <= 0:
            return

        left_min = max(200, self.tree.minimumWidth())
        left_max = max(left_min, self.tree.maximumWidth())
        if self._session_splitter_sizes and len(self._session_splitter_sizes) == 2:
            previous_left, previous_right = self._session_splitter_sizes
            previous_total = max(1, int(previous_left) + int(previous_right))
            left_width = int(round(total_width * (int(previous_left) / previous_total)))
        else:
            left_width = total_width // 4

        left_width = max(left_min, min(left_max, left_width))
        right_width = max(400, total_width - left_width)
        left_width = max(left_min, min(left_max, total_width - right_width))
        self.splitter.setSizes([left_width, right_width])
        self._capture_splitter_sizes_for_session()
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
        self.statusBar().showMessage(
            f"Searching {len(candidates)} markdown file(s) in {scope}...",
        )
        matches: list[Path] = []

        for path in candidates:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                content = ""
            # Search over file name + body to support quick discovery.
            if predicate(path.name, content):
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
                self._highlight_preview_search_terms(
                    self._current_search_terms(),
                    scroll_to_first=False,
                    close_term_groups=self._current_close_term_groups(),
                )
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
        for token_type, token_value, is_quoted in self._tokenize_match_query(query):
            if token_type != "TERM":
                continue
            term = token_value.strip()
            if not term:
                continue
            key = f"Q:{term}" if is_quoted else f"I:{term.casefold()}"
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
        terms.sort(key=len, reverse=True)
        return terms

    def _current_close_term_groups(self) -> list[list[tuple[str, bool]]]:
        """Extract CLOSE(...) argument groups from the current query."""
        query = self.match_input.text().strip()
        if not query:
            return []

        tokens = self._tokenize_match_query(query)
        groups: list[list[tuple[str, bool]]] = []
        i = 0
        token_count = len(tokens)

        while i < token_count:
            token_type, _token_value, _token_quoted = tokens[i]
            if token_type != "CLOSE":
                i += 1
                continue

            # Require function-style CLOSE (...) call shape.
            if i + 1 >= token_count or tokens[i + 1][0] != "LPAREN":
                i += 1
                continue

            j = i + 2
            group: list[tuple[str, bool]] = []
            is_valid = False

            while j < token_count:
                part_type, part_value, part_quoted = tokens[j]
                if part_type == "RPAREN":
                    is_valid = True
                    break
                if part_type == "COMMA":
                    j += 1
                    continue
                if part_type == "TERM":
                    cleaned = part_value.strip()
                    if cleaned:
                        group.append((cleaned, part_quoted))
                    j += 1
                    continue
                # Any nested expression token invalidates this CLOSE group.
                is_valid = False
                break

            if is_valid and len(group) >= 2:
                groups.append(group)
            i = j + 1 if j > i else i + 1

        return groups

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

    def _highlight_preview_search_terms(
        self,
        terms: list[str],
        scroll_to_first: bool,
        close_term_groups: list[list[tuple[str, bool]]] | None = None,
    ) -> None:
        """Highlight term matches in preview and optionally scroll to first one."""
        cleaned_terms = [term.strip() for term in terms if term.strip()]
        if not cleaned_terms:
            self._remove_preview_search_highlights()
            return

        terms_json = json.dumps(cleaned_terms)
        scroll_json = "true" if scroll_to_first else "false"
        close_word_gap_json = str(int(SEARCH_CLOSE_WORD_GAP))
        close_groups_payload: list[list[dict[str, object]]] = []
        for group in close_term_groups or []:
            payload_group: list[dict[str, object]] = []
            for term_text, is_quoted in group:
                cleaned = term_text.strip()
                if cleaned:
                    payload_group.append({"text": cleaned, "quoted": bool(is_quoted)})
            if len(payload_group) >= 2:
                close_groups_payload.append(payload_group)
        close_groups_json = json.dumps(close_groups_payload)
        js = """
(() => {
  // Rebuild highlight spans from plain text each pass so updates are idempotent.
  const terms = __TERMS_JSON__;
  const shouldScroll = __SCROLL_BOOL__;
  const closeWordGap = __CLOSE_WORD_GAP__;
  const closeTermGroups = __CLOSE_GROUPS_JSON__;
  const root = document.querySelector("main") || document.body;
  if (!root || !terms.length) return 0;

  const markSelector = 'span[data-mdexplore-search-mark="1"]';
  for (const oldMark of root.querySelectorAll(markSelector)) {
    const parent = oldMark.parentNode;
    if (!parent) continue;
    parent.replaceChild(document.createTextNode(oldMark.textContent || ""), oldMark);
    parent.normalize();
  }

  const skipTags = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA"]);
  const walker = document.createTreeWalker(
    root,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        if (!node || !node.nodeValue || !node.nodeValue.trim()) {
          return NodeFilter.FILTER_REJECT;
        }
        const parent = node.parentElement;
        if (!parent) {
          return NodeFilter.FILTER_REJECT;
        }
        if (skipTags.has(parent.tagName)) {
          return NodeFilter.FILTER_REJECT;
        }
        if (parent.closest(markSelector)) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    },
  );

  const segments = [];
  let fullText = "";
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const value = node.nodeValue || "";
    if (!value) {
      continue;
    }
    const start = fullText.length;
    fullText += value;
    const end = fullText.length;
    segments.push({ node, text: value, start, end });
    // Separate nodes to avoid accidental cross-node token merging.
    fullText += "\\n";
  }
  if (!segments.length) return 0;

  function escapeRegExp(input) {
    return String(input || "").replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&");
  }

  function upperBound(values, target) {
    let lo = 0;
    let hi = values.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (values[mid] <= target) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  }

  function normalizeCloseGroups(groups) {
    if (!Array.isArray(groups)) return [];
    const normalized = [];
    for (const group of groups) {
      if (!Array.isArray(group)) continue;
      const next = [];
      for (const item of group) {
        if (!item || typeof item.text !== "string") continue;
        const text = item.text.trim();
        if (!text) continue;
        next.push({ text, quoted: !!item.quoted });
      }
      if (next.length >= 2) normalized.push(next);
    }
    return normalized;
  }

  const normalizedCloseGroups = normalizeCloseGroups(closeTermGroups);
  let closeFocusRange = null;
  let closeFocusTerms = null;

  if (normalizedCloseGroups.length) {
    const wordMatches = [];
    const wordRegex = /\\S+/g;
    let wordMatch = null;
    while ((wordMatch = wordRegex.exec(fullText)) !== null) {
      wordMatches.push({ start: wordMatch.index, end: wordMatch.index + wordMatch[0].length });
      if (wordRegex.lastIndex <= wordMatch.index) {
        wordRegex.lastIndex = wordMatch.index + 1;
      }
    }

    if (wordMatches.length) {
      const wordStarts = wordMatches.map((item) => item.start);

      function bestWindowForGroup(group) {
        const occurrences = [];
        const found = new Array(group.length).fill(false);

        for (let termIndex = 0; termIndex < group.length; termIndex += 1) {
          const termInfo = group[termIndex];
          const pattern = new RegExp(escapeRegExp(termInfo.text), termInfo.quoted ? "g" : "gi");
          let m = null;
          while ((m = pattern.exec(fullText)) !== null) {
            const startChar = m.index;
            const wordIndex = upperBound(wordStarts, startChar) - 1;
            if (wordIndex >= 0) {
              occurrences.push({ word: wordIndex, term: termIndex });
              found[termIndex] = true;
            }
            if (pattern.lastIndex <= startChar) {
              pattern.lastIndex = startChar + 1;
            }
          }
        }

        if (!occurrences.length || found.some((value) => !value)) {
          return null;
        }

        occurrences.sort((a, b) => a.word - b.word);
        const counts = new Array(group.length).fill(0);
        let have = 0;
        let left = 0;
        let best = null;

        for (let right = 0; right < occurrences.length; right += 1) {
          const rightOcc = occurrences[right];
          if (counts[rightOcc.term] === 0) {
            have += 1;
          }
          counts[rightOcc.term] += 1;

          while (have === group.length && left <= right) {
            const leftOcc = occurrences[left];
            const span = rightOcc.word - leftOcc.word;
            if (span <= closeWordGap) {
              if (!best || span < best.span || (span === best.span && leftOcc.word < best.leftWord)) {
                best = { span, leftWord: leftOcc.word, rightWord: rightOcc.word };
              }
            }
            counts[leftOcc.term] -= 1;
            if (counts[leftOcc.term] === 0) {
              have -= 1;
            }
            left += 1;
          }
        }

        if (!best) {
          return null;
        }
        const startWord = wordMatches[best.leftWord];
        const endWord = wordMatches[best.rightWord];
        if (!startWord || !endWord) {
          return null;
        }
        return {
          span: best.span,
          startChar: startWord.start,
          endChar: endWord.end,
          terms: group,
        };
      }

      let chosenWindow = null;
      for (const group of normalizedCloseGroups) {
        const candidate = bestWindowForGroup(group);
        if (!candidate) continue;
        if (!chosenWindow || candidate.span < chosenWindow.span || (candidate.span === chosenWindow.span && candidate.startChar < chosenWindow.startChar)) {
          chosenWindow = candidate;
        }
      }

      if (chosenWindow) {
        closeFocusRange = { startChar: chosenWindow.startChar, endChar: chosenWindow.endChar };
        closeFocusTerms = chosenWindow.terms;
      }
    }
  }

  // If CLOSE(...) is present, only highlight the matched CLOSE window.
  if (normalizedCloseGroups.length && !closeFocusRange) {
    return 0;
  }

  const targetTerms = closeFocusTerms || terms.map((text) => ({ text, quoted: false }));
  if (!targetTerms.length) return 0;

  function collectRanges(segment) {
    const ranges = [];
    for (const termInfo of targetTerms) {
      const rawText = termInfo && typeof termInfo.text === "string" ? termInfo.text : "";
      const termText = rawText.trim();
      if (!termText) continue;
      const pattern = new RegExp(escapeRegExp(termText), termInfo.quoted ? "g" : "gi");
      let m = null;
      while ((m = pattern.exec(segment.text)) !== null) {
        const localStart = m.index;
        const localEnd = localStart + m[0].length;
        const absoluteStart = segment.start + localStart;
        const absoluteEnd = segment.start + localEnd;
        if (closeFocusRange) {
          if (absoluteStart < closeFocusRange.startChar || absoluteEnd > closeFocusRange.endChar) {
            if (pattern.lastIndex <= localStart) {
              pattern.lastIndex = localStart + 1;
            }
            continue;
          }
        }
        ranges.push({ start: localStart, end: localEnd });
        if (pattern.lastIndex <= localStart) {
          pattern.lastIndex = localStart + 1;
        }
      }
    }

    if (!ranges.length) {
      return [];
    }

    ranges.sort((a, b) => {
      if (a.start !== b.start) return a.start - b.start;
      return (b.end - b.start) - (a.end - a.start);
    });

    const deduped = [];
    let lastEnd = -1;
    for (const item of ranges) {
      if (item.start < lastEnd) continue;
      deduped.push(item);
      lastEnd = item.end;
    }
    return deduped;
  }

  let firstMark = null;
  let matchCount = 0;
  for (const segment of segments) {
    const ranges = collectRanges(segment);
    if (!ranges.length) continue;

    const text = segment.text;
    let cursor = 0;
    const fragment = document.createDocumentFragment();
    for (const range of ranges) {
      if (range.start > cursor) {
        fragment.appendChild(document.createTextNode(text.slice(cursor, range.start)));
      }
      const mark = document.createElement("span");
      mark.setAttribute("data-mdexplore-search-mark", "1");
      mark.style.backgroundColor = "#f5d34f";
      mark.style.color = "#111827";
      mark.style.padding = "0 1px";
      mark.style.borderRadius = "2px";
      mark.textContent = text.slice(range.start, range.end);
      fragment.appendChild(mark);
      if (!firstMark) {
        firstMark = mark;
      }
      matchCount += 1;
      cursor = range.end;
    }
    if (cursor < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(cursor)));
    }
    const parent = segment.node.parentNode;
    if (parent) {
      parent.replaceChild(fragment, segment.node);
    }
  }

  if (firstMark && shouldScroll) {
    firstMark.scrollIntoView({ behavior: "auto", block: "center", inline: "nearest" });
  }
  return matchCount;
})();
"""
        js = js.replace("__TERMS_JSON__", terms_json)
        js = js.replace("__SCROLL_BOOL__", scroll_json)
        js = js.replace("__CLOSE_WORD_GAP__", close_word_gap_json)
        js = js.replace("__CLOSE_GROUPS_JSON__", close_groups_json)
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
        """Compile boolean query with implicit AND, quotes, and CLOSE(...)."""
        tokens = self._tokenize_match_query(query)
        if not tokens:
            return lambda _name, _content: True

        class QueryParseError(Exception):
            pass

        idx = 0

        def peek(offset: int = 0) -> tuple[str, str, bool] | None:
            token_index = idx + offset
            if 0 <= token_index < len(tokens):
                return tokens[token_index]
            return None

        def token_starts_expression(token_index: int) -> bool:
            if token_index < 0 or token_index >= len(tokens):
                return False
            token_type, token_value, _token_quoted = tokens[token_index]
            if token_type in {"TERM", "LPAREN", "CLOSE"}:
                return True
            if token_type == "OP" and token_value == "NOT":
                return True
            if token_type == "OP" and token_value in {"AND", "OR"}:
                next_token = tokens[token_index + 1] if token_index + 1 < len(tokens) else None
                return bool(next_token and next_token[0] == "LPAREN")
            return False

        def consume(expected_type: str | None = None, expected_value: str | None = None) -> tuple[str, str, bool]:
            nonlocal idx
            token = peek()
            if token is None:
                raise QueryParseError("Unexpected end of query")
            token_type, token_value, token_quoted = token
            if expected_type is not None and token_type != expected_type:
                raise QueryParseError(f"Expected {expected_type} but found {token_type}")
            if expected_value is not None and token_value != expected_value:
                raise QueryParseError(f"Expected {expected_value} but found {token_value}")
            idx += 1
            return token_type, token_value, token_quoted

        def parse_expression(allow_implicit_and: bool = True):
            return parse_or(allow_implicit_and)

        def parse_or(allow_implicit_and: bool = True):
            node = parse_and(allow_implicit_and)
            while True:
                token = peek()
                if token is None or token[0] != "OP" or token[1] != "OR":
                    break
                consume("OP", "OR")
                right = parse_and(allow_implicit_and)
                node = ("OR", node, right)
            return node

        def parse_and(allow_implicit_and: bool = True):
            node = parse_not(allow_implicit_and)
            while True:
                token = peek()
                if token is not None and token[0] == "OP" and token[1] == "AND":
                    consume("OP", "AND")
                    right = parse_not(allow_implicit_and)
                    node = ("AND", node, right)
                    continue
                # Implicit AND between adjacent expressions.
                if allow_implicit_and and token_starts_expression(idx):
                    right = parse_not(allow_implicit_and)
                    node = ("AND", node, right)
                    continue
                break
            return node

        def parse_not(allow_implicit_and: bool = True):
            token = peek()
            if token is not None and token[0] == "OP" and token[1] == "NOT":
                consume("OP", "NOT")
                return ("NOT", parse_not(allow_implicit_and))
            return parse_primary(allow_implicit_and)

        def parse_logic_call(operator_name: str):
            consume("OP", operator_name)
            consume("LPAREN")
            while True:
                token = peek()
                if token is None:
                    raise QueryParseError(f"Unterminated {operator_name}(...)")
                if token[0] == "RPAREN":
                    break
                if token[0] == "COMMA":
                    consume("COMMA")
                    continue
                if not token_starts_expression(idx):
                    raise QueryParseError(f"{operator_name}() accepts expression arguments only")
                break

            items: list[tuple] = []
            while True:
                token = peek()
                if token is None:
                    raise QueryParseError(f"Unterminated {operator_name}(...)")
                if token[0] == "RPAREN":
                    break
                if token[0] == "COMMA":
                    consume("COMMA")
                    continue

                # Function-style args accept comma, whitespace, or a mix.
                items.append(parse_expression(allow_implicit_and=False))
                token = peek()
                if token is None:
                    raise QueryParseError(f"Unterminated {operator_name}(...)")
                if token[0] == "COMMA":
                    consume("COMMA")
                    continue
                if token[0] == "RPAREN":
                    break
                if token_starts_expression(idx):
                    continue
                raise QueryParseError(f"Unexpected token in {operator_name}(...)")

            consume("RPAREN")
            if not items:
                raise QueryParseError(f"{operator_name}() requires at least one argument")
            combined = items[0]
            for item in items[1:]:
                combined = (operator_name, combined, item)
            return combined

        def parse_close_call():
            consume("CLOSE", "CLOSE")
            consume("LPAREN")
            terms: list[tuple[str, bool]] = []
            while True:
                token = peek()
                if token is None:
                    raise QueryParseError("Unterminated CLOSE(...)")
                token_type, token_value, token_quoted = token
                if token_type == "RPAREN":
                    break
                if token_type == "COMMA":
                    consume("COMMA")
                    continue
                if token_type == "TERM":
                    consume("TERM")
                    cleaned = token_value.strip()
                    if cleaned:
                        terms.append((cleaned, token_quoted))
                    continue
                raise QueryParseError("CLOSE(...) accepts terms only")
            consume("RPAREN")
            if len(terms) < 2:
                raise QueryParseError("CLOSE(...) requires at least two terms")
            return ("CLOSE", terms)

        def parse_primary(_allow_implicit_and: bool = True):
            token = peek()
            if token is None:
                raise QueryParseError("Missing query operand")
            token_type, token_value, token_quoted = token
            if token_type == "TERM":
                consume("TERM")
                return ("TERM", token_value, token_quoted)
            if token_type == "CLOSE":
                return parse_close_call()
            if token_type == "OP" and token_value in {"AND", "OR"} and peek(1) is not None and peek(1)[0] == "LPAREN":
                return parse_logic_call(token_value)
            if token_type == "LPAREN":
                consume("LPAREN")
                node = parse_expression()
                consume("RPAREN")
                return node
            raise QueryParseError(f"Unexpected token: {token_type}")

        def term_matches(
            term: str,
            is_quoted: bool,
            file_name: str,
            file_content: str,
            file_name_folded: str,
            file_content_folded: str,
        ) -> bool:
            if not term:
                return False
            if is_quoted:
                return term in file_name or term in file_content
            term_folded = term.casefold()
            return term_folded in file_name_folded or term_folded in file_content_folded

        def close_terms_match(terms: list[tuple[str, bool]], file_content: str) -> bool:
            content = file_content or ""
            word_matches = list(re.finditer(r"\S+", content))
            if not word_matches:
                return False
            # CLOSE() uses whitespace-delimited token positions, not line indexes.
            word_starts = [match.start() for match in word_matches]
            content_folded = content.casefold()
            occurrences: list[tuple[int, int]] = []
            term_found = [False] * len(terms)

            for term_index, (term_text, is_quoted) in enumerate(terms):
                if not term_text:
                    return False
                needle = term_text if is_quoted else term_text.casefold()
                haystack = content if is_quoted else content_folded
                search_start = 0
                while True:
                    char_index = haystack.find(needle, search_start)
                    if char_index < 0:
                        break
                    word_index = bisect_right(word_starts, char_index) - 1
                    if word_index >= 0:
                        occurrences.append((word_index, term_index))
                        term_found[term_index] = True
                    search_start = char_index + 1

            if not all(term_found) or not occurrences:
                return False

            occurrences.sort(key=lambda item: item[0])
            counts = [0] * len(terms)
            have = 0
            left = 0

            for right, (right_word, right_term_index) in enumerate(occurrences):
                if counts[right_term_index] == 0:
                    have += 1
                counts[right_term_index] += 1

                while have == len(terms) and left <= right:
                    left_word, left_term_index = occurrences[left]
                    if right_word - left_word <= SEARCH_CLOSE_WORD_GAP:
                        return True
                    counts[left_term_index] -= 1
                    if counts[left_term_index] == 0:
                        have -= 1
                    left += 1

            return False

        def evaluate(node, file_name: str, file_content: str, file_name_folded: str, file_content_folded: str) -> bool:
            node_type = node[0]
            if node_type == "TERM":
                _kind, term_text, is_quoted = node
                return term_matches(term_text, bool(is_quoted), file_name, file_content, file_name_folded, file_content_folded)
            if node_type == "CLOSE":
                _kind, close_terms = node
                return close_terms_match(close_terms, file_content)
            if node_type == "NOT":
                _kind, operand = node
                return not evaluate(operand, file_name, file_content, file_name_folded, file_content_folded)
            if node_type == "AND":
                _kind, left_node, right_node = node
                return evaluate(left_node, file_name, file_content, file_name_folded, file_content_folded) and evaluate(
                    right_node,
                    file_name,
                    file_content,
                    file_name_folded,
                    file_content_folded,
                )
            if node_type == "OR":
                _kind, left_node, right_node = node
                return evaluate(left_node, file_name, file_content, file_name_folded, file_content_folded) or evaluate(
                    right_node,
                    file_name,
                    file_content,
                    file_name_folded,
                    file_content_folded,
                )
            return False

        try:
            expression = parse_expression()
            if idx != len(tokens):
                raise QueryParseError("Unexpected trailing query tokens")
        except QueryParseError:
            return self._compile_simple_match_predicate(tokens)

        def predicate(file_name: str, file_content: str) -> bool:
            name_text = file_name or ""
            content_text = file_content or ""
            return evaluate(
                expression,
                name_text,
                content_text,
                name_text.casefold(),
                content_text.casefold(),
            )

        return predicate

    def _compile_simple_match_predicate(self, tokens: list[tuple[str, str, bool]]):
        """Fallback matcher: all terms must appear in filename or content."""
        terms = [(value.strip(), is_quoted) for token_type, value, is_quoted in tokens if token_type == "TERM" and value.strip()]
        if not terms:
            return lambda _file_name, _file_content: True

        def predicate(file_name: str, file_content: str) -> bool:
            name_text = file_name or ""
            content_text = file_content or ""
            name_folded = name_text.casefold()
            content_folded = content_text.casefold()
            for term_text, is_quoted in terms:
                if is_quoted:
                    if term_text not in name_text and term_text not in content_text:
                        return False
                else:
                    folded = term_text.casefold()
                    if folded not in name_folded and folded not in content_folded:
                        return False
            return True

        return predicate

    def _tokenize_match_query(self, query: str) -> list[tuple[str, str, bool]]:
        """Tokenize query supporting AND/OR/NOT, CLOSE(...), quotes, and commas."""
        tokens: list[tuple[str, str, bool]] = []
        i = 0
        length = len(query)

        while i < length:
            ch = query[i]
            if ch.isspace():
                i += 1
                continue
            if ch == "(":
                tokens.append(("LPAREN", ch, False))
                i += 1
                continue
            if ch == ")":
                tokens.append(("RPAREN", ch, False))
                i += 1
                continue
            if ch == ",":
                tokens.append(("COMMA", ch, False))
                i += 1
                continue
            if ch == '"':
                i += 1
                buffer: list[str] = []
                while i < length:
                    current = query[i]
                    if current == "\\" and i + 1 < length:
                        next_char = query[i + 1]
                        if next_char in {'"', "\\"}:
                            buffer.append(next_char)
                            i += 2
                            continue
                    if current == '"':
                        i += 1
                        break
                    buffer.append(current)
                    i += 1
                tokens.append(("TERM", "".join(buffer), True))
                continue

            start = i
            while i < length and not query[i].isspace() and query[i] not in "(),":
                i += 1
            token_value = query[start:i]
            if not token_value:
                continue
            upper = token_value.upper()
            if upper in {"AND", "OR", "NOT"}:
                tokens.append(("OP", upper, False))
            elif upper == "CLOSE":
                tokens.append(("CLOSE", "CLOSE", False))
            else:
                tokens.append(("TERM", token_value, False))

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

    def _copy_files_to_clipboard(self, files: list[Path]) -> int:
        """Copy file paths to clipboard with file-manager compatible MIME payloads."""
        normalized: list[Path] = []
        seen: set[str] = set()
        for path in files:
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(resolved)

        clipboard = QApplication.clipboard()
        mime_data = QMimeData()
        urls = [QUrl.fromLocalFile(str(path)) for path in normalized]
        mime_data.setUrls(urls)

        # Nemo/Nautilus paste support: this custom format marks clipboard data
        # as file copy operations rather than plain text.
        if urls:
            gnome_payload = "copy\n" + "\n".join(url.toString() for url in urls)
            mime_data.setData("x-special/gnome-copied-files", gnome_payload.encode("utf-8"))

        # Keep plain text for editors/terminals.
        mime_data.setText("\n".join(str(path) for path in normalized))
        clipboard.setMimeData(mime_data)
        return len(normalized)

    def _copy_current_preview_file_to_clipboard(self) -> None:
        """Copy the currently previewed markdown file path to clipboard."""
        if self.current_file is None:
            self.statusBar().showMessage("No previewed markdown file to copy", 3000)
            return

        try:
            target = self.current_file.resolve()
        except Exception:
            target = self.current_file

        if not target.is_file():
            self.statusBar().showMessage("Previewed file is unavailable", 3000)
            return

        copied = self._copy_files_to_clipboard([target])
        if copied:
            self.statusBar().showMessage(f"Copied previewed file to clipboard: {target.name}", 4000)

    def _copy_highlighted_files_to_clipboard(self, color_value: str, color_name: str) -> None:
        """Copy highlighted file paths for a color to the system clipboard."""
        scope = self._highlight_scope_directory()
        matches = self.model.collect_files_with_color(scope, color_value)
        copied = self._copy_files_to_clipboard(matches)
        self.statusBar().showMessage(
            f"Copied {copied} {color_name.lower()} highlighted file(s) from {scope}",
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
        self.statusBar().showMessage(f"Loading preview: {path.name}...")
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
            self._set_current_preview_signature(path_key, int(mtime_ns), int(size))
            self.statusBar().showMessage(f"Preview rendered: {self.current_file.name}")

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

    def _clear_current_preview_signature(self) -> None:
        """Drop tracked mtime/size signature for the active preview file."""
        self._current_preview_signature_key = None
        self._current_preview_signature = None

    def _set_current_preview_signature(self, path_key: str, mtime_ns: int, size: int) -> None:
        """Record the latest observed on-disk signature for a previewed file."""
        self._current_preview_signature_key = path_key
        self._current_preview_signature = (int(mtime_ns), int(size))

    def _on_file_change_watch_tick(self) -> None:
        """Auto-refresh preview when the active markdown file changes on disk."""
        if self.current_file is None:
            return

        try:
            resolved = self.current_file.resolve()
            path_key = str(resolved)
        except Exception:
            resolved = self.current_file
            path_key = str(self.current_file)

        try:
            stat = resolved.stat()
        except Exception:
            # File may be temporarily inaccessible while external tools save.
            return

        current_sig = (int(stat.st_mtime_ns), int(stat.st_size))
        if self._current_preview_signature_key != path_key or self._current_preview_signature is None:
            self._set_current_preview_signature(path_key, current_sig[0], current_sig[1])
            return
        if current_sig == self._current_preview_signature:
            return

        # Update baseline first so repeated timer ticks during one save do not
        # trigger duplicate refresh cycles.
        self._set_current_preview_signature(path_key, current_sig[0], current_sig[1])
        self._refresh_current_preview(reason="file changed on disk")

    def _has_saved_scroll_for_current_preview(self) -> bool:
        key = self._current_preview_scroll_key()
        if key is None:
            return False
        if key in self._preview_scroll_positions:
            return True
        # Backward compatibility with pre-view-tab scroll cache entries.
        path_key = self._current_preview_path_key()
        return bool(path_key and path_key in self._preview_scroll_positions)

    def _capture_current_preview_scroll(self, force: bool = False) -> None:
        """Capture current preview scroll position for the selected file."""
        scroll_key = self._current_preview_scroll_key()
        if scroll_key is None:
            return
        if not force:
            if not self._preview_capture_enabled:
                return
            if time.monotonic() < self._scroll_restore_block_until:
                return
        # Use Qt's synchronous scrollPosition() to avoid async JS race conditions
        # during rapid file switches.
        try:
            pos = self.preview.page().scrollPosition()
            y = float(pos.y())
        except Exception:
            return
        if math.isfinite(y):
            self._preview_scroll_positions[scroll_key] = y
            state = self._current_view_state()
            if state is not None:
                state["scroll_y"] = y
            self._request_active_view_top_line_update(force=force)

    def _enable_preview_scroll_capture_for(self, expected_key: str) -> None:
        """Re-enable periodic scroll capture for the currently displayed file."""
        if self._current_preview_path_key() != expected_key:
            return
        self._preview_capture_enabled = True
        self._scroll_restore_block_until = 0.0
        self._capture_current_preview_scroll(force=True)
        self._request_active_view_top_line_update(force=True)

    def _restore_current_preview_scroll(self, expected_key: str | None = None) -> None:
        """Restore previously captured scroll position for the selected file."""
        path_key = self._current_preview_path_key()
        if path_key is None:
            return
        if expected_key is not None and path_key != expected_key:
            return
        scroll_key = self._current_preview_scroll_key()
        scroll_y = self._preview_scroll_positions.get(scroll_key) if scroll_key is not None else None
        if scroll_y is None:
            state = self._current_view_state()
            if state is not None:
                try:
                    scroll_y = float(state.get("scroll_y", 0.0))
                except Exception:
                    scroll_y = 0.0
        if scroll_y is None:
            # Backward compatibility with pre-view-tab scroll cache entries.
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
        self._request_active_view_top_line_update(force=True)

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
        current_key = self._current_preview_path_key()
        if current_key is not None:
            current_placeholders = self._plantuml_placeholders_by_doc.get(current_key, {})
            if hash_key in current_placeholders:
                self._show_preview_progress_status()

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
        previous_path_key = self._current_preview_path_key()
        next_path_key = self._path_key(path)
        self._capture_current_preview_scroll(force=True)
        if previous_path_key is not None and previous_path_key != next_path_key:
            self._save_document_view_session(previous_path_key)
        self._cancel_pending_preview_render()
        self._preview_capture_enabled = False
        self._scroll_restore_block_until = 0.0
        if previous_path_key != next_path_key:
            restored = self._restore_document_view_session(next_path_key)
            if not restored:
                self._reset_document_views(initial_scroll=0.0, initial_line=1)
        should_highlight_search = bool(self.match_input.text().strip()) and self._is_path_in_current_matches(path)
        self._pending_preview_search_terms = self._current_search_terms() if should_highlight_search else []
        self._pending_preview_search_close_groups = self._current_close_term_groups() if should_highlight_search else []
        self.statusBar().showMessage(f"Loading preview content: {path.name}...")

        self.current_file = path
        self._current_document_total_lines = max(1, int(self._document_line_counts.get(next_path_key, 1)))
        self._sync_all_view_tab_progress()
        self._update_view_tabs_visibility()
        self._update_add_view_button_state()
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
            self._set_current_preview_signature(cache_key, int(stat.st_mtime_ns), int(stat.st_size))
            cached = self.cache.get(cache_key)
            if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
                self.statusBar().showMessage(f"Using cached preview: {resolved.name}...")
                self.preview.setHtml(cached[2], base_url)
                return

            self.statusBar().showMessage(f"Rendering markdown: {resolved.name}...")
            markdown_text = resolved.read_text(encoding="utf-8", errors="replace")
            total_lines = self._count_markdown_lines(markdown_text)
            self._document_line_counts[cache_key] = total_lines
            self._current_document_total_lines = total_lines
            self._sync_all_view_tab_progress()
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
            self.statusBar().showMessage(f"Preview rendered, loading in viewer: {resolved.name}...")
            self.preview.setHtml(html_doc, base_url)
        except Exception as exc:
            self.statusBar().showMessage(f"Preview render failed: {exc}", 5000)
            self.preview.setHtml(
                self._placeholder_html(f"Could not render preview for {path.name}: {exc}"),
                base_url,
            )

    def _refresh_current_preview(self, _checked: bool = False, *, reason: str | None = None) -> None:
        """Force re-render of the currently selected file."""
        if self.current_file is None:
            return
        try:
            cache_key = str(self.current_file.resolve())
        except Exception:
            cache_key = str(self.current_file)
        self.cache.pop(cache_key, None)
        self.statusBar().showMessage(f"Refreshing preview: {self.current_file.name}...")
        self._load_preview(self.current_file)
        if reason:
            self.statusBar().showMessage(
                f"Auto-refreshed preview: {self.current_file.name} ({reason})",
                4500,
            )

    def _set_pdf_export_busy(self, busy: bool) -> None:
        """Toggle PDF export UI state while async export is running."""
        self._pdf_export_in_progress = busy
        self.pdf_btn.setEnabled(not busy)

    def _export_current_preview_pdf(self) -> None:
        """Export the currently previewed markdown rendering to a numbered PDF."""
        if self.current_file is None:
            QMessageBox.information(self, "No file selected", "Select a markdown file before exporting to PDF.")
            return
        if self._pdf_export_in_progress:
            self.statusBar().showMessage("PDF export already in progress", 3000)
            return

        try:
            source_path = self.current_file.resolve()
        except Exception:
            source_path = self.current_file
        output_path = source_path.with_suffix(".pdf")

        self._set_pdf_export_busy(True)
        self.statusBar().showMessage(f"Preparing PDF for {source_path.name}...")
        self._prepare_preview_for_pdf_export(output_path, attempt=0)

    def _prepare_preview_for_pdf_export(self, output_path: Path, attempt: int) -> None:
        """Wait for math/Mermaid/fonts readiness and inject print style before export."""
        js = """
(() => {
  // Print-only math tuning to avoid cramped/squished glyph appearance in PDF.
  if (!document.getElementById("__mdexplore_pdf_math_style")) {
    const style = document.createElement("style");
    style.id = "__mdexplore_pdf_math_style";
    style.textContent = `
@media print {
  mjx-container[jax="SVG"] {
    font-size: 1.08em !important;
    text-rendering: geometricPrecision;
    page-break-inside: avoid;
    break-inside: avoid;
  }
  mjx-container[jax="SVG"] > svg {
    overflow: visible;
    shape-rendering: geometricPrecision;
    text-rendering: geometricPrecision;
  }
  mjx-container[jax="SVG"][display="true"] {
    margin: 0.9em 0 1.05em 0 !important;
  }
  mjx-container[jax="CHTML"] {
    font-family: "STIX Two Math", "STIXGeneral", "Cambria Math", "Noto Sans Math", "Latin Modern Math", serif !important;
    font-kerning: normal !important;
    text-rendering: geometricPrecision;
  }
  mjx-container[jax="CHTML"] mjx-mi,
  mjx-container[jax="CHTML"] mjx-mo,
  mjx-container[jax="CHTML"] mjx-mn,
  mjx-container[jax="CHTML"] mjx-mtext {
    letter-spacing: 0.01em !important;
  }
}
`;
    document.head.appendChild(style);
  }

  if (window.__mdexploreRunClientRenderers) {
    window.__mdexploreRunClientRenderers({ mermaidMode: "pdf" });
  }

  const hasMath = !!document.querySelector("mjx-container, .MathJax");
  const hasMermaid = !!document.querySelector(".mermaid");
  const mathReady = !hasMath || !!window.__mdexploreMathReady;
  const mermaidReady =
    !hasMermaid ||
    (!!window.__mdexploreMermaidReady && String(window.__mdexploreMermaidPaletteMode || "") === "pdf");
  const fontsReady = !document.fonts || document.fonts.status === "loaded";

  return { mathReady, mermaidReady, fontsReady, hasMath, hasMermaid };
})();
"""
        self.preview.page().runJavaScript(
            js,
            lambda result, target=output_path, tries=attempt: self._on_pdf_precheck_result(target, tries, result),
        )

    def _on_pdf_precheck_result(self, output_path: Path, attempt: int, result) -> None:
        """Continue waiting until print assets are ready, then trigger print."""
        math_ready = False
        mermaid_ready = False
        fonts_ready = False
        if isinstance(result, dict):
            math_ready = bool(result.get("mathReady"))
            mermaid_ready = bool(result.get("mermaidReady"))
            fonts_ready = bool(result.get("fontsReady"))

        if math_ready and mermaid_ready and fonts_ready:
            self._trigger_pdf_print(output_path)
            return

        if attempt < PDF_EXPORT_PRECHECK_MAX_ATTEMPTS:
            if attempt == 0:
                self.statusBar().showMessage("Waiting for math/Mermaid/fonts before PDF export...")
            QTimer.singleShot(
                PDF_EXPORT_PRECHECK_INTERVAL_MS,
                lambda target=output_path, tries=attempt + 1: self._prepare_preview_for_pdf_export(target, tries),
            )
            return

        # Fall back instead of blocking export indefinitely if some assets never settle.
        self.statusBar().showMessage(
            "Proceeding with PDF export before all preview assets reported ready",
            3500,
        )
        self._trigger_pdf_print(output_path)

    def _trigger_pdf_print(self, output_path: Path) -> None:
        """Start Qt WebEngine PDF generation for the active preview page."""
        self.statusBar().showMessage(f"Rendering PDF snapshot: {output_path.name}...")
        try:
            self.preview.page().printToPdf(
                lambda pdf_data, target=output_path: self._on_pdf_render_ready(target, pdf_data)
            )
        except Exception as exc:
            self._set_pdf_export_busy(False)
            self._restore_preview_mermaid_palette()
            error_text = self._truncate_error_text(str(exc), 500)
            QMessageBox.critical(self, "PDF export failed", f"Could not start PDF rendering:\n{error_text}")
            self.statusBar().showMessage(f"PDF export failed: {error_text}", 5000)

    def _restore_preview_mermaid_palette(self) -> None:
        """Switch Mermaid back to preview palette after PDF export attempts."""
        js = """
(() => {
  if (window.__mdexploreRunClientRenderers) {
    window.__mdexploreRunClientRenderers({ mermaidMode: "auto" });
    return true;
  }
  if (window.__mdexploreRunMermaidWithMode) {
    window.__mdexploreRunMermaidWithMode("auto", false);
    return true;
  }
  return false;
})();
"""
        self.preview.page().runJavaScript(js)

    def _on_pdf_render_ready(self, output_path: Path, pdf_data) -> None:
        """Receive raw PDF bytes from WebEngine and start footer stamping."""
        try:
            raw_pdf = bytes(pdf_data)
        except Exception:
            raw_pdf = b""

        if not raw_pdf:
            self._set_pdf_export_busy(False)
            self._restore_preview_mermaid_palette()
            message = "Qt WebEngine returned an empty PDF payload"
            QMessageBox.critical(self, "PDF export failed", message)
            self.statusBar().showMessage(f"PDF export failed: {message}", 5000)
            return

        worker = PdfExportWorker(output_path, raw_pdf)
        self._active_pdf_workers.add(worker)
        worker.signals.finished.connect(
            lambda path_text, error_text, current_worker=worker: self._on_pdf_export_finished(
                current_worker,
                path_text,
                error_text,
            )
        )
        self._pdf_pool.start(worker)
        self.statusBar().showMessage(f"Writing numbered PDF: {output_path.name}...")

    def _on_pdf_export_finished(self, worker: PdfExportWorker, output_path_text: str, error_text: str) -> None:
        """Finalize async PDF export and report result."""
        self._active_pdf_workers.discard(worker)
        self._set_pdf_export_busy(False)
        self._restore_preview_mermaid_palette()

        if error_text:
            short_error = self._truncate_error_text(error_text, 500)
            QMessageBox.critical(
                self,
                "PDF export failed",
                f"Could not create PDF:\n{output_path_text}\n\n{short_error}",
            )
            self.statusBar().showMessage(f"PDF export failed: {short_error}", 5000)
            return

        self.statusBar().showMessage(f"Exported PDF: {output_path_text}", 5000)

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
