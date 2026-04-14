"""Runtime helpers that are independent of the main window implementation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import QMarginsF
from PySide6.QtGui import (
    QFontDatabase,
    QOffscreenSurface,
    QOpenGLContext,
    QPageLayout,
    QPageSize,
    QSurfaceFormat,
)

from .constants import (
    CONFIG_FILE_NAME,
    MAX_PRINT_DIAGRAM_FONT_PT,
    MIN_PRINT_DIAGRAM_FONT_PT,
    PDF_PRINT_HEADING_TO_DIAGRAM_GAP_PX,
    PDF_PRINT_HORIZONTAL_MARGIN_PX,
    PDF_PRINT_KEEP_MIN_HEIGHT_PX,
    PDF_PRINT_LANDSCAPE_LETTER_HEIGHT_PX,
    PDF_PRINT_LANDSCAPE_LETTER_WIDTH_PX,
    PDF_PRINT_LANDSCAPE_MIN_HEIGHT_PX,
    PDF_PRINT_LANDSCAPE_MIN_WIDTH_PX,
    PDF_PRINT_LAYOUT_SAFETY_PX,
    PDF_PRINT_PLANTUML_LANDSCAPE_ASPECT_RATIO,
    PDF_PRINT_PORTRAIT_LETTER_HEIGHT_PX,
    PDF_PRINT_PORTRAIT_LETTER_WIDTH_PX,
    PDF_PRINT_PORTRAIT_MIN_HEIGHT_PX,
    PDF_PRINT_PORTRAIT_MIN_WIDTH_PX,
    PDF_PRINT_VERTICAL_MARGIN_PX,
    PDF_PRINT_WIDE_DIAGRAM_ASPECT_RATIO,
    PDF_PRINT_WIDE_DIAGRAM_LANDSCAPE_GAIN,
    SEARCH_HIT_COUNT_FONT_PATHS,
)

_SEARCH_HIT_COUNT_FONT_FAMILY_CACHE: str | None = None


def config_file_path() -> Path:
    """Return the per-user config file path for the persisted default root."""
    return Path.home() / CONFIG_FILE_NAME


def letter_pdf_page_layout() -> QPageLayout:
    """Return an explicit US Letter page layout with zero margins."""
    return QPageLayout(
        QPageSize(QPageSize.PageSizeId.Letter),
        QPageLayout.Orientation.Portrait,
        QMarginsF(0.0, 0.0, 0.0, 0.0),
    )


def search_hit_count_font_family() -> str:
    """Resolve the narrow font used by tree-gutter search hit pills."""
    global _SEARCH_HIT_COUNT_FONT_FAMILY_CACHE
    preferred_family = "Liberation Sans Narrow"
    fallback_family = "Arial Narrow"

    if (
        _SEARCH_HIT_COUNT_FONT_FAMILY_CACHE
        and _SEARCH_HIT_COUNT_FONT_FAMILY_CACHE != fallback_family
    ):
        return _SEARCH_HIT_COUNT_FONT_FAMILY_CACHE

    try:
        families = set(QFontDatabase.families())
    except Exception:
        families = set()
    if preferred_family in families:
        _SEARCH_HIT_COUNT_FONT_FAMILY_CACHE = preferred_family
        return _SEARCH_HIT_COUNT_FONT_FAMILY_CACHE

    base_dir = Path(__file__).resolve().parent.parent
    for candidate in SEARCH_HIT_COUNT_FONT_PATHS:
        raw_path = Path(candidate)
        font_path = raw_path if raw_path.is_absolute() else (base_dir / raw_path)
        if not font_path.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id == -1:
            continue
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            _SEARCH_HIT_COUNT_FONT_FAMILY_CACHE = families[0]
            return _SEARCH_HIT_COUNT_FONT_FAMILY_CACHE

    _SEARCH_HIT_COUNT_FONT_FAMILY_CACHE = fallback_family
    return _SEARCH_HIT_COUNT_FONT_FAMILY_CACHE


def configure_qt_graphics_fallback() -> None:
    """Force software rendering env vars for fallback launch paths."""
    if not os.environ.get("QT_QUICK_BACKEND"):
        os.environ["QT_QUICK_BACKEND"] = "software"
    if not os.environ.get("QSG_RHI_BACKEND"):
        os.environ["QSG_RHI_BACKEND"] = "software"
    if not os.environ.get("QT_OPENGL"):
        os.environ["QT_OPENGL"] = "software"
    chromium_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    if "--disable-gpu" not in chromium_flags.split():
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
            f"{chromium_flags} --disable-gpu".strip()
        )


def gpu_context_available() -> bool:
    """Return whether a usable GPU OpenGL context can be created."""
    qt_quick_backend = os.environ.get("QT_QUICK_BACKEND", "").strip().lower()
    qsg_rhi_backend = os.environ.get("QSG_RHI_BACKEND", "").strip().lower()
    qt_opengl = os.environ.get("QT_OPENGL", "").strip().lower()
    chromium_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").split()

    if (
        qt_quick_backend == "software"
        or qsg_rhi_backend == "software"
        or qt_opengl == "software"
    ):
        return False
    if "--disable-gpu" in chromium_flags:
        return False

    try:
        surface = QOffscreenSurface()
        surface.setFormat(QSurfaceFormat.defaultFormat())
        surface.create()
        if not surface.isValid():
            return False

        context = QOpenGLContext()
        context.setFormat(surface.format())
        if not context.create():
            return False
        if not context.makeCurrent(surface):
            return False
        context.doneCurrent()
        return True
    except Exception:
        return False


def pdf_print_layout_knobs() -> dict[str, float | int]:
    """Expose print-layout policy knobs to the in-page PDF preflight JS."""
    return {
        "headingToDiagramGapPx": PDF_PRINT_HEADING_TO_DIAGRAM_GAP_PX,
        "layoutSafetyPx": PDF_PRINT_LAYOUT_SAFETY_PX,
        "keepMinHeightPx": PDF_PRINT_KEEP_MIN_HEIGHT_PX,
        "minPrintDiagramFontPt": MIN_PRINT_DIAGRAM_FONT_PT,
        "maxPrintDiagramFontPt": MAX_PRINT_DIAGRAM_FONT_PT,
        "portraitLetterWidthPx": PDF_PRINT_PORTRAIT_LETTER_WIDTH_PX,
        "portraitLetterHeightPx": PDF_PRINT_PORTRAIT_LETTER_HEIGHT_PX,
        "landscapeLetterWidthPx": PDF_PRINT_LANDSCAPE_LETTER_WIDTH_PX,
        "landscapeLetterHeightPx": PDF_PRINT_LANDSCAPE_LETTER_HEIGHT_PX,
        "horizontalMarginPx": PDF_PRINT_HORIZONTAL_MARGIN_PX,
        "verticalMarginPx": PDF_PRINT_VERTICAL_MARGIN_PX,
        "portraitMinWidthPx": PDF_PRINT_PORTRAIT_MIN_WIDTH_PX,
        "portraitMinHeightPx": PDF_PRINT_PORTRAIT_MIN_HEIGHT_PX,
        "landscapeMinWidthPx": PDF_PRINT_LANDSCAPE_MIN_WIDTH_PX,
        "landscapeMinHeightPx": PDF_PRINT_LANDSCAPE_MIN_HEIGHT_PX,
        "wideDiagramAspectRatio": PDF_PRINT_WIDE_DIAGRAM_ASPECT_RATIO,
        "wideDiagramLandscapeGain": PDF_PRINT_WIDE_DIAGRAM_LANDSCAPE_GAIN,
        "plantumlLandscapeAspectRatio": PDF_PRINT_PLANTUML_LANDSCAPE_ASPECT_RATIO,
    }


def load_default_root_from_config() -> Path:
    """Resolve default root when no CLI path is provided."""
    fallback = Path.home()
    cfg_path = config_file_path()
    try:
        if not cfg_path.exists():
            return fallback
        raw = cfg_path.read_text(encoding="utf-8").strip()
        if not raw:
            return fallback
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None

        candidates: list[str] = []
        if isinstance(parsed, dict):
            default_root = parsed.get("default_root")
            if isinstance(default_root, str) and default_root.strip():
                candidates.append(default_root.strip())
            recent_roots = parsed.get("recent_roots")
            if isinstance(recent_roots, list):
                for entry in recent_roots:
                    if isinstance(entry, str) and entry.strip():
                        candidates.append(entry.strip())
        elif isinstance(parsed, str) and parsed.strip():
            candidates.append(parsed.strip())
        else:
            candidates.append(raw)

        for entry in candidates:
            candidate = Path(entry).expanduser()
            if candidate.is_dir():
                return candidate.resolve()
    except Exception:
        pass
    return fallback
