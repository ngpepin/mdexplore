"""Local icon-building helpers for mdexplore."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from collections import deque
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer

from .constants import (
    PROJECT_ROOT,
    SVG_ICON_ALPHA_CUTOFF,
    SVG_ICON_RENDER_OVERSAMPLE,
    UI_ASSET_DIR,
)

_ICON_CACHE_SCHEMA_VERSION = "v1"


def ui_asset_path(filename: str) -> Path:
    """Resolve a UI asset from the canonical asset directory."""
    asset_path = UI_ASSET_DIR / filename
    if asset_path.is_file():
        return asset_path
    return PROJECT_ROOT / filename


def _icon_cache_root() -> Path:
    """Return per-user icon cache directory used for generated icon assets."""
    xdg_cache_home = str(os.environ.get("XDG_CACHE_HOME", "") or "").strip()
    base_dir = (
        Path(xdg_cache_home).expanduser()
        if xdg_cache_home
        else (Path.home() / ".cache")
    )
    return base_dir / "mdexplore" / "icon-cache"


def _build_icon_cache_key(*parts: object) -> str:
    """Return stable SHA1 cache key from arbitrary icon-generation inputs."""
    joined = "\n".join(str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8", errors="replace")).hexdigest()


def _source_identity(path: Path) -> tuple[str, int, int]:
    """Return `(resolved_path, mtime_ns, size)` for cache keying."""
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    try:
        stat = resolved.stat()
        return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)
    except Exception:
        return str(resolved), 0, 0


def _icon_cache_file_path(cache_key: str) -> Path | None:
    """Return cache path for one icon key, creating directory as needed."""
    try:
        cache_root = _icon_cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / f"{cache_key}.png"
    except Exception:
        return None


def _load_cached_icon(cache_key: str) -> QIcon | None:
    """Load one generated icon from disk cache when present and valid."""
    cache_path = _icon_cache_file_path(cache_key)
    if cache_path is None or not cache_path.is_file():
        return None
    try:
        pixmap = QPixmap(str(cache_path))
    except Exception:
        return None
    if pixmap.isNull():
        return None
    return QIcon(pixmap)


def _persist_generated_icon(cache_key: str, pixmap: QPixmap) -> None:
    """Persist generated icon pixmap atomically for future launches."""
    if pixmap.isNull():
        return
    cache_path = _icon_cache_file_path(cache_key)
    if cache_path is None:
        return
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(cache_path.parent),
            prefix=f".{cache_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
        if not pixmap.save(str(tmp_path), "PNG"):
            return
        tmp_path.replace(cache_path)
    except Exception:
        pass
    finally:
        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass


def _color_signature(color: QColor) -> str:
    """Return stable RGBA signature for cache key generation."""
    c = QColor(color)
    return f"{c.red()}:{c.green()}:{c.blue()}:{c.alpha()}"


def _two_tone_icon_cache_key(
    *,
    source_path: Path,
    render_kind: str,
    dark_color: QColor,
    light_color: QColor,
    size: int,
) -> str:
    """Return cache key for one two-tone icon render."""
    source_id, mtime_ns, byte_size = _source_identity(source_path)
    return _build_icon_cache_key(
        "two-tone-icon",
        _ICON_CACHE_SCHEMA_VERSION,
        render_kind,
        source_id,
        mtime_ns,
        byte_size,
        f"size:{int(size)}",
        f"oversample:{int(SVG_ICON_RENDER_OVERSAMPLE)}",
        f"alpha-cutoff:{int(SVG_ICON_ALPHA_CUTOFF)}",
        f"dark:{_color_signature(dark_color)}",
        f"light:{_color_signature(light_color)}",
    )


def build_markdown_icon() -> QIcon:
    """Return a standard markdown icon (asset with drawn fallback)."""

    def color_distance(a: QColor, b: QColor) -> int:
        return (
            abs(a.red() - b.red())
            + abs(a.green() - b.green())
            + abs(a.blue() - b.blue())
        )

    def transparentize_outer_background(pixmap: QPixmap) -> QPixmap:
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
        if all(c.alpha() == 0 for c in corners):
            return pixmap

        threshold = 36

        def is_background(c: QColor) -> bool:
            return any(color_distance(c, bg) <= threshold for bg in corners)

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
                    minx = min(minx, x)
                    miny = min(miny, y)
                    maxx = max(maxx, x)
                    maxy = max(maxy, y)

        if maxx < minx or maxy < miny:
            return pixmap

        cropped = QPixmap.fromImage(
            image.copy(minx, miny, maxx - minx + 1, maxy - miny + 1)
        )

        size = 256
        inset = 2
        max_w = size - (2 * inset)
        max_h = size - (2 * inset)
        scaled = cropped.scaled(
            max_w,
            max_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

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
        icon_path = PROJECT_ROOT / icon_name
        if icon_path.exists():
            source_id, mtime_ns, byte_size = _source_identity(icon_path)
            cache_key = _build_icon_cache_key(
                "markdown-app-icon",
                _ICON_CACHE_SCHEMA_VERSION,
                source_id,
                mtime_ns,
                byte_size,
                "canvas:256",
                "inset:2",
            )
            cached_icon = _load_cached_icon(cache_key)
            if cached_icon is not None:
                print(
                    f"mdexplore icon source: {icon_path} (cached)",
                    file=sys.stderr,
                )
                return cached_icon
            asset_pixmap = QPixmap(str(icon_path))
            if not asset_pixmap.isNull():
                cleaned = transparentize_outer_background(asset_pixmap)
                fitted = fit_icon_canvas(cleaned)
                _persist_generated_icon(cache_key, fitted)
                print(f"mdexplore icon source: {icon_path}", file=sys.stderr)
                return QIcon(fitted)

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
    painter.drawLine(14, 42, 14, 22)
    painter.drawLine(14, 22, 22, 34)
    painter.drawLine(22, 34, 30, 22)
    painter.drawLine(30, 22, 30, 42)
    painter.drawLine(42, 22, 42, 38)
    painter.drawLine(37, 33, 42, 38)
    painter.drawLine(47, 33, 42, 38)
    painter.end()

    print("mdexplore icon source: built-in fallback", file=sys.stderr)
    return QIcon(pixmap)


def build_clear_x_icon() -> QIcon:
    """Return a small X icon for the search-field clear action."""
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


def _render_svg_icon_image(renderer: QSvgRenderer, icon_size: int) -> QImage:
    """Render an SVG through a supersampled surface to reduce jagged edges."""
    target_size = max(1, int(icon_size))
    oversample = max(1, int(SVG_ICON_RENDER_OVERSAMPLE))
    render_size = target_size * oversample

    render_pixmap = QPixmap(render_size, render_size)
    render_pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(render_pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    renderer.render(painter)
    painter.end()

    if oversample > 1:
        render_pixmap = render_pixmap.scaled(
            target_size,
            target_size,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    return render_pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)


def load_svg_icon(filename: str, color: QColor, size: int = 16) -> QIcon:
    """Load and recolor a local SVG icon to a fixed flat color."""
    icon_path = ui_asset_path(filename)
    if icon_path.is_file():
        renderer = QSvgRenderer(str(icon_path))
        if not renderer.isValid():
            return QIcon()
        image = _render_svg_icon_image(renderer, size)
        target = QColor(color)
        for y in range(image.height()):
            for x in range(image.width()):
                pixel = image.pixelColor(x, y)
                alpha = pixel.alpha()
                if alpha < SVG_ICON_ALPHA_CUTOFF:
                    image.setPixelColor(x, y, QColor(0, 0, 0, 0))
                    continue
                image.setPixelColor(
                    x, y, QColor(target.red(), target.green(), target.blue(), alpha)
                )
        return QIcon(QPixmap.fromImage(image))
    return QIcon()


def load_svg_icon_two_tone(
    filename: str,
    dark_color: QColor,
    light_color: QColor,
    size: int = 16,
) -> QIcon:
    """Load local SVG and recolor dark/light tones while preserving alpha edges."""
    icon_path = ui_asset_path(filename)
    if not icon_path.is_file():
        return QIcon()
    cache_key = _two_tone_icon_cache_key(
        source_path=icon_path,
        render_kind="svg",
        dark_color=dark_color,
        light_color=light_color,
        size=size,
    )
    cached_icon = _load_cached_icon(cache_key)
    if cached_icon is not None:
        return cached_icon
    renderer = QSvgRenderer(str(icon_path))
    if not renderer.isValid():
        return QIcon()
    image = _render_svg_icon_image(renderer, size)
    dark = QColor(dark_color)
    light = QColor(light_color)
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixelColor(x, y)
            alpha = pixel.alpha()
            if alpha < SVG_ICON_ALPHA_CUTOFF:
                image.setPixelColor(x, y, QColor(0, 0, 0, 0))
                continue
            luminance = (
                (0.299 * pixel.redF())
                + (0.587 * pixel.greenF())
                + (0.114 * pixel.blueF())
            )
            luminance = max(0.0, min(1.0, luminance))
            red = int(
                round((dark.red() * (1.0 - luminance)) + (light.red() * luminance))
            )
            green = int(
                round((dark.green() * (1.0 - luminance)) + (light.green() * luminance))
            )
            blue = int(
                round((dark.blue() * (1.0 - luminance)) + (light.blue() * luminance))
            )
            image.setPixelColor(x, y, QColor(red, green, blue, alpha))
    final_pixmap = QPixmap.fromImage(image)
    _persist_generated_icon(cache_key, final_pixmap)
    return QIcon(final_pixmap)


def load_png_icon_two_tone(
    filename: str,
    dark_color: QColor,
    light_color: QColor,
    size: int = 16,
) -> QIcon:
    """Load local PNG and map dark/light tones while preserving alpha edges."""
    icon_path = ui_asset_path(filename)
    if not icon_path.is_file():
        return QIcon()
    cache_key = _two_tone_icon_cache_key(
        source_path=icon_path,
        render_kind="png",
        dark_color=dark_color,
        light_color=light_color,
        size=size,
    )
    cached_icon = _load_cached_icon(cache_key)
    if cached_icon is not None:
        return cached_icon

    source = QImage(str(icon_path)).convertToFormat(QImage.Format.Format_ARGB32)
    if source.isNull():
        return QIcon()

    target_size = max(1, int(size))
    oversample = max(1, int(SVG_ICON_RENDER_OVERSAMPLE))
    working_size = target_size * oversample

    working = QPixmap(working_size, working_size)
    working.fill(Qt.GlobalColor.transparent)
    painter = QPainter(working)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    fitted = QPixmap.fromImage(source).scaled(
        working_size,
        working_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    x = (working_size - fitted.width()) // 2
    y = (working_size - fitted.height()) // 2
    painter.drawPixmap(x, y, fitted)
    painter.end()

    image = working.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    dark = QColor(dark_color)
    light = QColor(light_color)
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixelColor(x, y)
            alpha = pixel.alpha()
            if alpha < SVG_ICON_ALPHA_CUTOFF:
                image.setPixelColor(x, y, QColor(0, 0, 0, 0))
                continue

            luminance = (
                (0.299 * pixel.redF())
                + (0.587 * pixel.greenF())
                + (0.114 * pixel.blueF())
            )
            luminance = max(0.0, min(1.0, luminance))
            red = int(
                round((dark.red() * (1.0 - luminance)) + (light.red() * luminance))
            )
            green = int(
                round((dark.green() * (1.0 - luminance)) + (light.green() * luminance))
            )
            blue = int(
                round((dark.blue() * (1.0 - luminance)) + (light.blue() * luminance))
            )
            image.setPixelColor(x, y, QColor(red, green, blue, alpha))

    final_pixmap = QPixmap.fromImage(image).scaled(
        target_size,
        target_size,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    _persist_generated_icon(cache_key, final_pixmap)
    return QIcon(final_pixmap)
