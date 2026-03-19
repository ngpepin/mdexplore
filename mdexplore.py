#!/usr/bin/env python3
"""mdexplore: fast markdown browser/editor launcher for Ubuntu."""

from __future__ import annotations

import argparse
import base64
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
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from PySide6.QtCore import (
    QDir,
    QEventLoop,
    QMimeData,
    QObject,
    QPoint,
    QRect,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QClipboard,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QIcon,
    QImage,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QPolygon,
)
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from mdexplore_app.constants import (
    CONFIG_FILE_NAME,
    DIAGRAM_VIEW_STATE_JSON_TOKEN,
    MAX_PRINT_DIAGRAM_FONT_PT,
    MERMAID_BACKEND_JS,
    MERMAID_BACKEND_RUST,
    MERMAID_CACHE_JSON_TOKEN,
    MERMAID_CACHE_RESTORE_BATCH_SIZE,
    MERMAID_SVG_CACHE_MAX_ENTRIES,
    MERMAID_SVG_MAX_CHARS,
    MIN_PRINT_DIAGRAM_FONT_PT,
    PDF_EXPORT_PRECHECK_INTERVAL_MS,
    PDF_EXPORT_PRECHECK_MAX_ATTEMPTS,
    PLANTUML_RESTORE_BATCH_SIZE,
    PREVIEW_HIGHLIGHT_KIND_IMPORTANT,
    PREVIEW_HIGHLIGHT_KIND_NORMAL,
    PREVIEW_PERSISTENT_HIGHLIGHT_COLOR,
    PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_COLOR,
    PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR,
    PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_TEXT_COLOR,
    PREVIEW_PERSISTENT_HIGHLIGHT_MARKER_COLOR,
    PREVIEW_SETHTML_MAX_BYTES,
    PREVIEW_ZOOM_MAX,
    PREVIEW_ZOOM_MIN,
    PREVIEW_ZOOM_OVERLAY_TIMEOUT_MS,
    PREVIEW_ZOOM_RESET,
    PREVIEW_ZOOM_STEP,
    RESTORE_OVERLAY_MAX_VISIBLE_SECONDS,
    RESTORE_OVERLAY_SHOW_DELAY_MS,
    RESTORE_OVERLAY_TIMEOUT_SECONDS,
    PDF_LANDSCAPE_PAGE_TOKEN,
    SEARCH_CLOSE_WORD_GAP,
)
from mdexplore_app.icons import (
    build_clear_x_icon as _build_clear_x_icon,
    build_markdown_icon as _build_markdown_icon,
    load_png_icon_two_tone as _load_png_icon_two_tone,
    load_svg_icon as _load_svg_icon,
    load_svg_icon_two_tone as _load_svg_icon_two_tone,
    ui_asset_path as _ui_asset_path,
)
from mdexplore_app.js import (
    preload_js_assets as _preload_js_assets,
    render_js_asset as _render_js_asset,
)
from mdexplore_app.templates import (
    preload_template_assets as _preload_template_assets,
    render_template_asset as _render_template_asset,
)
from mdexplore_app.pdf import (
    extract_plantuml_error_details as _extract_plantuml_error_details,
    stamp_pdf_page_numbers as _stamp_pdf_page_numbers,
)
import mdexplore_app.search as _search_query
from mdexplore_app.runtime import (
    config_file_path as _config_file_path,
    configure_qt_graphics_fallback as _configure_qt_graphics_fallback,
    gpu_context_available as _gpu_context_available,
    letter_pdf_page_layout as _letter_pdf_page_layout,
    load_default_root_from_config as _load_default_root_from_config,
    pdf_print_layout_knobs as _pdf_print_layout_knobs,
    search_hit_count_font_family as _search_hit_count_font_family,
)
from mdexplore_app.tabs import ViewTabBar
from mdexplore_app.tree import ColorizedMarkdownModel, MarkdownTreeItemDelegate
from mdexplore_app.workers import (
    PdfExportWorker,
    PlantUmlRenderWorker,
    PreviewRenderWorker,
    TreeMarkerScanWorker,
)


class MarkdownRenderer:
    """Converts markdown to HTML with Mermaid, MathJax, and PlantUML support."""

    def __init__(self, mermaid_backend: str = MERMAID_BACKEND_JS) -> None:
        # Backend selection is resolved once per renderer so the markdown fence
        # callbacks can stay simple and deterministic.
        self._mermaid_backend_requested = (
            str(mermaid_backend or MERMAID_BACKEND_JS).strip().lower()
        )
        if self._mermaid_backend_requested not in {
            MERMAID_BACKEND_JS,
            MERMAID_BACKEND_RUST,
        }:
            self._mermaid_backend_requested = MERMAID_BACKEND_JS
        self._mermaid_rs_binary = self._resolve_mermaid_rs_binary()
        self._mermaid_rs_setup_issue = self._mermaid_rs_setup_error()
        if (
            self._mermaid_backend_requested == MERMAID_BACKEND_RUST
            and self._mermaid_rs_setup_issue is None
        ):
            self._mermaid_backend = MERMAID_BACKEND_RUST
        else:
            self._mermaid_backend = MERMAID_BACKEND_JS
        self._mermaid_svg_cache: dict[str, str] = {}
        self._last_mermaid_pdf_svg_by_hash: dict[str, str] = {}
        self._mathjax_local_script = self._resolve_local_mathjax_script()
        self._mermaid_local_script = self._resolve_local_mermaid_script()
        self._plantuml_jar_path = self._resolve_plantuml_jar_path()
        self._plantuml_setup_issue = self._plantuml_setup_error()
        self._plantuml_svg_cache: dict[str, str] = {}
        self._md = (
            MarkdownIt(
                "commonmark",
                {"html": True, "linkify": True, "typographer": True},
            )
            .enable("table")
            .enable("strikethrough")
        )
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
            # default fenced-code renderer. This is the only place where raw
            # markdown fence language is turned into preview/PDF placeholders.
            token = tokens[idx]
            info = token.info.strip().split(maxsplit=1)[0].lower() if token.info else ""
            code = token.content
            line_attrs = ""
            if token.map and len(token.map) == 2:
                line_attrs = f' data-md-line-start="{token.map[0]}" data-md-line-end="{token.map[1]}"'

            if info == "mermaid":
                try:
                    # All Mermaid flows start by normalizing source and hashing
                    # it. The hash is the cache key shared by preview, restore,
                    # and PDF-specific SVG variants.
                    prepared_source = self._prepare_mermaid_source(code)
                    mermaid_hash = hashlib.sha1(
                        prepared_source.encode("utf-8", errors="replace")
                    ).hexdigest()
                    mermaid_index = (
                        int(env.get("mermaid_index", 0)) if isinstance(env, dict) else 0
                    )
                    if isinstance(env, dict):
                        env["mermaid_index"] = mermaid_index + 1
                    if self._mermaid_backend == MERMAID_BACKEND_RUST:
                        # Rust preview and PDF SVGs intentionally fork here:
                        # preview gets mdexplore's GUI profile, while PDF gets
                        # a separate vanilla render cached for export only.
                        svg_markup, error_message = self._render_mermaid_svg_markup(
                            prepared_source, "preview"
                        )
                        if isinstance(env, dict):
                            pdf_svg_map = env.get("mermaid_pdf_svg_by_hash")
                            if (
                                isinstance(pdf_svg_map, dict)
                                and mermaid_hash not in pdf_svg_map
                            ):
                                pdf_svg, _pdf_error = self._render_mermaid_svg_markup(
                                    prepared_source, "pdf"
                                )
                                if pdf_svg:
                                    pdf_svg_map[mermaid_hash] = pdf_svg
                        source_attr = html.escape(prepared_source, quote=True)
                        if svg_markup is not None:
                            return (
                                f'<div class="mdexplore-fence"{line_attrs}>'
                                f'<div class="mermaid mermaid-ready" data-mdexplore-mermaid-backend="rust" '
                                f'data-mdexplore-mermaid-hash="{mermaid_hash}" '
                                f'data-mdexplore-mermaid-index="{mermaid_index}" '
                                f'data-mdexplore-mermaid-source="{source_attr}">\n{svg_markup}\n</div>'
                                "</div>\n"
                            )
                        safe_error_attr = html.escape(
                            error_message or "Rust Mermaid rendering failed", quote=True
                        )
                        return (
                            f'<div class="mdexplore-fence"{line_attrs}>'
                            f'<div class="mermaid mermaid-rust-fallback" data-mdexplore-mermaid-backend="rust" '
                            f'data-mdexplore-mermaid-hash="{mermaid_hash}" '
                            f'data-mdexplore-mermaid-index="{mermaid_index}" '
                            f'data-mdexplore-mermaid-source="{source_attr}" '
                            f'data-mdexplore-rust-error="{safe_error_attr}">'
                            "Mermaid rendering..."
                            "</div>"
                            "</div>\n"
                        )
                    return (
                        f'<div class="mdexplore-fence"{line_attrs}>'
                        f'<div class="mermaid" data-mdexplore-mermaid-hash="{mermaid_hash}" '
                        f'data-mdexplore-mermaid-index="{mermaid_index}">\n{html.escape(code)}\n</div>'
                        "</div>\n"
                    )
                except Exception as exc:
                    safe_error = html.escape(
                        str(exc) or "unexpected Mermaid rendering error"
                    )
                    return (
                        f'<div class="mdexplore-fence mermaid-error"{line_attrs}>'
                        f'<div class="mermaid mermaid-error">Mermaid render failed: {safe_error}</div>'
                        "</div>\n"
                    )

            if info in {"plantuml", "puml", "uml"}:
                # PlantUML takes a different path because preview rendering is
                # progressive and may complete after the rest of the markdown is
                # already visible.
                resolver = (
                    env.get("plantuml_resolver") if isinstance(env, dict) else None
                )
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

                escaped_error = html.escape(
                    error_message or "PlantUML rendering failed"
                )
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
            if (
                token.nesting == 1
                and token.type.endswith("_open")
                and token.map
                and len(token.map) == 2
            ):
                token.attrSet("data-md-line-start", str(token.map[0]))
                token.attrSet("data-md-line-end", str(token.map[1]))
            return default_render_token(tokens, idx, options, env)

        self._md.renderer.rules["fence"] = custom_fence
        self._md.renderer.rules["math_inline"] = custom_math_inline
        self._md.renderer.rules["math_block"] = custom_math_block
        self._md.renderer.renderToken = custom_render_token

    @property
    def mermaid_backend(self) -> str:
        """Return active Mermaid backend (`js` or `rust`)."""
        return self._mermaid_backend

    @property
    def mermaid_backend_requested(self) -> str:
        """Return requested Mermaid backend from CLI/config."""
        return self._mermaid_backend_requested

    def mermaid_backend_warning(self) -> str | None:
        """Describe why requested Mermaid backend could not be activated."""
        if (
            self._mermaid_backend_requested == MERMAID_BACKEND_RUST
            and self._mermaid_backend != MERMAID_BACKEND_RUST
        ):
            return self._mermaid_rs_setup_issue or "Rust Mermaid backend unavailable"
        return None

    def _resolve_mermaid_rs_binary(self) -> Path | None:
        """Locate mmdr executable for Rust Mermaid rendering."""
        env_value = os.environ.get("MDEXPLORE_MERMAID_RS_BIN", "").strip()
        candidates: list[Path] = []
        if env_value:
            candidates.append(Path(env_value).expanduser())

        app_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                Path.home() / ".cargo" / "bin" / "mmdr",
                app_dir
                / "vendor"
                / "mermaid-rs-renderer"
                / "target"
                / "release"
                / "mmdr",
                app_dir / "vendor" / "mermaid-rs-renderer" / "mmdr",
                app_dir / "vendor" / "mermaid-rs-renderer" / "bin" / "mmdr",
                app_dir / "mermaid-rs-renderer" / "target" / "release" / "mmdr",
                app_dir / "mmdr",
            ]
        )

        for candidate in candidates:
            try:
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate.resolve()
            except Exception:
                continue

        for name in ("mmdr", "mermaid-rs-renderer"):
            found = shutil.which(name)
            if found:
                try:
                    return Path(found).resolve()
                except Exception:
                    return Path(found)
        return None

    def _mermaid_rs_setup_error(self) -> str | None:
        """Return setup issue text when Rust Mermaid backend is unavailable."""
        if self._mermaid_rs_binary is None:
            return (
                "mmdr executable not found "
                "(set MDEXPLORE_MERMAID_RS_BIN or install mermaid-rs-renderer)"
            )
        return None

    def _render_mermaid_svg_markup(
        self, code: str, render_profile: str = "preview"
    ) -> tuple[str | None, str | None]:
        """Render Mermaid source through Rust mmdr backend and return raw SVG."""
        if self._mermaid_rs_setup_issue is not None:
            return None, self._mermaid_rs_setup_issue
        if self._mermaid_rs_binary is None:
            return None, "mmdr executable not available"

        profile = str(render_profile or "preview").strip().lower()
        if profile not in {"preview", "pdf"}:
            profile = "preview"
        prepared_source = self._prepare_mermaid_source(code)
        if profile == "preview":
            rust_theme_config = self._rust_mermaid_theme_config()
            config_signature = json.dumps(
                rust_theme_config,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        else:
            # PDF-mode Rust rendering should stay vanilla/default.
            rust_theme_config = None
            config_signature = "__MDEXPLORE_RUST_DEFAULT_THEME__"
        cache_key = hashlib.sha1(
            (
                profile
                + "\n__MDEXPLORE_RUST_PROFILE__\n"
                + prepared_source
                + "\n__MDEXPLORE_RUST_CFG__\n"
                + config_signature
            ).encode("utf-8", errors="replace")
        ).hexdigest()
        cached = self._mermaid_svg_cache.get(cache_key)
        if cached is not None:
            return cached, None

        tmp_input_path = None
        tmp_output_path = None
        tmp_config_path = None
        try:
            input_file = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=".mmd", delete=False
            )
            tmp_input_path = Path(input_file.name)
            input_file.write(prepared_source)
            input_file.flush()
            input_file.close()

            output_file = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=".svg", delete=False
            )
            tmp_output_path = Path(output_file.name)
            output_file.close()

            candidate_commands = []
            if profile == "preview":
                config_file = tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", suffix=".json", delete=False
                )
                tmp_config_path = Path(config_file.name)
                config_file.write(config_signature)
                config_file.flush()
                config_file.close()

                # `mmdr` CLI signatures vary by build. Prefer the current
                # flag-based form (-i/-o/-e), then fall back to positional.
                candidate_commands.extend(
                    [
                        [
                            str(self._mermaid_rs_binary),
                            "-i",
                            str(tmp_input_path),
                            "-o",
                            str(tmp_output_path),
                            "-e",
                            "svg",
                            "-c",
                            str(tmp_config_path),
                        ],
                        [
                            str(self._mermaid_rs_binary),
                            "-i",
                            str(tmp_input_path),
                            "-o",
                            str(tmp_output_path),
                            "-e",
                            "svg",
                        ],
                        [
                            str(self._mermaid_rs_binary),
                            str(tmp_input_path),
                            str(tmp_output_path),
                            "--output-format",
                            "svg",
                        ],
                    ]
                )
            else:
                candidate_commands.extend(
                    [
                        [
                            str(self._mermaid_rs_binary),
                            "-i",
                            str(tmp_input_path),
                            "-o",
                            str(tmp_output_path),
                            "-e",
                            "svg",
                        ],
                        [
                            str(self._mermaid_rs_binary),
                            str(tmp_input_path),
                            str(tmp_output_path),
                            "--output-format",
                            "svg",
                        ],
                    ]
                )
            result = None
            for command in candidate_commands:
                result = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=20,
                )
                if result.returncode == 0:
                    break

            if result is None or result.returncode != 0:
                error_text = (
                    (result.stderr if result is not None else "")
                    or (result.stdout if result is not None else "")
                    or ""
                ).strip()
                if not error_text:
                    code = result.returncode if result is not None else "unknown"
                    error_text = f"mmdr exited with code {code}"
                return None, error_text

            svg_markup = tmp_output_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
            if "<svg" not in svg_markup.casefold():
                return None, "mmdr did not produce SVG output"
            cleaned_svg = (
                self._sanitize_rust_mermaid_svg_markup(svg_markup)
                if profile == "preview"
                else svg_markup
            )
            self._mermaid_svg_cache[cache_key] = cleaned_svg
            return cleaned_svg, None
        except subprocess.TimeoutExpired:
            return None, "mmdr render timed out"
        except Exception as exc:
            return None, f"Rust Mermaid render failed: {exc}"
        finally:
            if tmp_input_path is not None:
                try:
                    tmp_input_path.unlink(missing_ok=True)
                except Exception:
                    pass
            if tmp_output_path is not None:
                try:
                    tmp_output_path.unlink(missing_ok=True)
                except Exception:
                    pass
            if tmp_config_path is not None:
                try:
                    tmp_config_path.unlink(missing_ok=True)
                except Exception:
                    pass

    @staticmethod
    def _rust_mermaid_theme_config() -> dict[str, object]:
        """Dark-theme palette for Rust Mermaid output in GUI preview mode."""
        return {
            "theme": "base",
            "themeVariables": {
                "background": "#0f172a",
                "primaryColor": "#1e293b",
                "primaryBorderColor": "#93c5fd",
                "primaryTextColor": "#e5e7eb",
                "secondaryColor": "#172554",
                "tertiaryColor": "#111827",
                "lineColor": "#d1d5db",
                "textColor": "#e5e7eb",
                "edgeLabelBackground": "#0f172a",
                "clusterBkg": "#1f2937",
                "clusterBorder": "#94a3b8",
                "actorBkg": "#1e293b",
                "actorBorder": "#93c5fd",
                "actorLine": "#d1d5db",
                "noteBkg": "#1f2937",
                "noteBorderColor": "#93c5fd",
                "fontFamily": "Noto Sans, DejaVu Sans, sans-serif",
            },
        }

    @staticmethod
    def _parse_svg_length(value: str | None) -> float | None:
        """Parse numeric SVG lengths from values like '123', '123px', '123.4%'."""
        if not value:
            return None
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
        if not match:
            return None
        try:
            return float(match.group(0))
        except Exception:
            return None

    @classmethod
    def _sanitize_rust_mermaid_svg_markup(cls, svg_markup: str) -> str:
        """Remove opaque white canvas background from Rust Mermaid SVG output."""
        if not svg_markup:
            return svg_markup
        try:
            root = ET.fromstring(svg_markup)
        except Exception:
            return svg_markup

        def local_name(tag: str) -> str:
            return tag.rsplit("}", 1)[-1] if "}" in tag else tag

        if local_name(root.tag).lower() != "svg":
            return svg_markup

        # Some renderers add root-level background styles; strip them for
        # dark preview transparency.
        style_attr = str(root.attrib.get("style", "")).strip()
        if style_attr:
            style_parts = [
                part.strip() for part in style_attr.split(";") if part.strip()
            ]
            kept_parts = [
                part for part in style_parts if "background" not in part.casefold()
            ]
            if kept_parts:
                root.set("style", "; ".join(kept_parts))
            else:
                root.attrib.pop("style", None)

        view_w = None
        view_h = None
        view_box = str(root.attrib.get("viewBox", "")).strip()
        if view_box:
            numbers = [
                float(part)
                for part in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", view_box)
            ]
            if len(numbers) == 4:
                view_w = numbers[2]
                view_h = numbers[3]
        if view_w is None or view_h is None:
            view_w = cls._parse_svg_length(root.attrib.get("width"))
            view_h = cls._parse_svg_length(root.attrib.get("height"))

        white_fill_values = {
            "#fff",
            "#ffffff",
            "white",
            "rgb(255,255,255)",
            "rgba(255,255,255,1)",
            "rgba(255,255,255,1.0)",
        }

        # Rust output commonly injects a first full-canvas white <rect>.
        # Remove only when it clearly covers the canvas.
        for child in list(root):
            if local_name(child.tag).lower() != "rect":
                continue
            fill_value = (
                str(child.attrib.get("fill", "")).strip().casefold().replace(" ", "")
            )
            if not fill_value:
                style_text = str(child.attrib.get("style", ""))
                style_match = re.search(
                    r"(?:^|;)\s*fill\s*:\s*([^;]+)", style_text, flags=re.IGNORECASE
                )
                if style_match:
                    fill_value = (
                        style_match.group(1).strip().casefold().replace(" ", "")
                    )
            if fill_value not in white_fill_values:
                continue

            x = cls._parse_svg_length(child.attrib.get("x")) or 0.0
            y = cls._parse_svg_length(child.attrib.get("y")) or 0.0
            w = cls._parse_svg_length(child.attrib.get("width"))
            h = cls._parse_svg_length(child.attrib.get("height"))
            if view_w and view_h and w and h:
                covers_canvas = (
                    x <= 1.0
                    and y <= 1.0
                    and w >= (view_w * 0.97)
                    and h >= (view_h * 0.97)
                )
            else:
                covers_canvas = x <= 1.0 and y <= 1.0
            if not covers_canvas:
                continue
            root.remove(child)
            break

        try:
            if root.tag.startswith("{") and "}" in root.tag:
                namespace_uri = root.tag[1:].split("}", 1)[0]
                ET.register_namespace("", namespace_uri)
            return ET.tostring(root, encoding="unicode")
        except Exception:
            return svg_markup

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
        cache_key = hashlib.sha1(
            prepared_code.encode("utf-8", errors="replace")
        ).hexdigest()
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

    @staticmethod
    def _prepare_mermaid_source(code: str) -> str:
        """Normalize Mermaid source for stable hashing/caching."""
        return code.replace("\r\n", "\n").strip("\n")

    def take_last_mermaid_pdf_svg_by_hash(self) -> dict[str, str]:
        """Return and clear PDF-mode Rust Mermaid SVG seed map for latest render."""
        payload = dict(self._last_mermaid_pdf_svg_by_hash)
        self._last_mermaid_pdf_svg_by_hash = {}
        return payload

    def render_document(
        self,
        markdown_text: str,
        title: str,
        total_lines: int | None = None,
        plantuml_resolver=None,
    ) -> str:
        # `env` is passed through markdown-it and lets fence renderers call back
        # into window-level async PlantUML orchestration when available.
        env = {}
        env["mermaid_index"] = 0
        env["mermaid_pdf_svg_by_hash"] = {}
        if callable(plantuml_resolver):
            env["plantuml_resolver"] = plantuml_resolver
            env["plantuml_index"] = 0
        body = self._md.render(markdown_text, env)
        if isinstance(env.get("mermaid_pdf_svg_by_hash"), dict):
            self._last_mermaid_pdf_svg_by_hash = dict(
                env.get("mermaid_pdf_svg_by_hash") or {}
            )
        else:
            self._last_mermaid_pdf_svg_by_hash = {}
        escaped_title = html.escape(title)
        mermaid_cache_token = MERMAID_CACHE_JSON_TOKEN
        diagram_state_token = DIAGRAM_VIEW_STATE_JSON_TOKEN
        mathjax_sources_json = json.dumps(self._mathjax_script_sources())
        mermaid_sources_json = json.dumps(self._mermaid_script_sources())
        mermaid_backend_json = json.dumps(self._mermaid_backend)
        total_source_lines_json = json.dumps(
            max(1, int(total_lines or (markdown_text.count("\n") + 1)))
        )
        return _render_template_asset(
            "preview/document.html",
            {
                "__ESCAPED_TITLE__": escaped_title,
                "__MATHJAX_SOURCES_JSON__": mathjax_sources_json,
                "__MERMAID_SOURCES_JSON__": mermaid_sources_json,
                "__MERMAID_BACKEND_JSON__": mermaid_backend_json,
                "__TOTAL_SOURCE_LINES_JSON__": total_source_lines_json,
                "__PERSISTENT_HIGHLIGHT_MARKER_COLOR_JSON__": json.dumps(
                    PREVIEW_PERSISTENT_HIGHLIGHT_MARKER_COLOR
                ),
                "__PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR_JSON__": json.dumps(
                    PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR
                ),
                "__PREVIEW_HIGHLIGHT_KIND_NORMAL_JSON__": json.dumps(
                    PREVIEW_HIGHLIGHT_KIND_NORMAL
                ),
                "__PREVIEW_HIGHLIGHT_KIND_IMPORTANT_JSON__": json.dumps(
                    PREVIEW_HIGHLIGHT_KIND_IMPORTANT
                ),
                "__MERMAID_CACHE_RESTORE_BATCH_SIZE__": str(
                    MERMAID_CACHE_RESTORE_BATCH_SIZE
                ),
                "__MERMAID_CACHE_TOKEN_JSON__": json.dumps(mermaid_cache_token),
                "__DIAGRAM_VIEW_STATE_TOKEN_JSON__": json.dumps(diagram_state_token),
                "__BODY_HTML__": body,
            },
        )



class PreviewPage(QWebEnginePage):
    """Handle preview-side custom navigation actions emitted by injected JS."""

    namedViewRequested = Signal(int)

    def acceptNavigationRequest(self, url, nav_type, is_main_frame):  # type: ignore[override]
        if isinstance(url, QUrl) and url.scheme() == "mdexplore":
            if url.host() == "view":
                try:
                    view_id = int(str(url.path() or "").lstrip("/"))
                except Exception:
                    return False
                if view_id > 0:
                    QTimer.singleShot(
                        0,
                        lambda requested_view_id=view_id: self.namedViewRequested.emit(
                            requested_view_id
                        ),
                    )
                return False
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)


class MdExploreWindow(QMainWindow):
    MAX_DOCUMENT_VIEWS = 8
    VIEWS_FILE_NAME = ".mdexplore-views.json"
    HIGHLIGHTING_FILE_NAME = ".mdexplore-highlighting.json"
    PREVIEW_HIGHLIGHT_COLOR = PREVIEW_PERSISTENT_HIGHLIGHT_COLOR
    PREVIEW_HIGHLIGHT_IMPORTANT_COLOR = PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_COLOR
    DEBUG_LOG_FILE_NAME = "mdexplore.log"
    DEBUG_LOG_MAX_LINES = 10_000
    HIGHLIGHT_COLORS = [
        ("Yellow", "#f5d34f"),
        ("Green", "#78d389"),
        ("Blue", "#7bb9ff"),
        ("Orange", "#f6a05f"),
        ("Purple", "#bb9df5"),
        ("Light Gray", "#d1d5db"),
        ("Medium Gray", "#9ca3af"),
        ("Red", "#ef7d7d"),
    ]

    def __init__(
        self,
        root: Path,
        app_icon: QIcon,
        config_path: Path,
        mermaid_backend: str = MERMAID_BACKEND_JS,
        gpu_context_available: bool = False,
        debug_mode: bool = False,
    ):
        super().__init__()
        # Persistent document/session state is split deliberately:
        # - cache: rendered HTML keyed by file + stat signature
        # - preview scrolls: per-run only
        # - document view sessions: per-run
        # - persisted view sessions: on-disk per-directory sidecars
        self.root = root.resolve()
        self.config_path = config_path
        self.renderer = MarkdownRenderer(mermaid_backend=mermaid_backend)
        self.current_file: Path | None = None
        self.last_directory_selection: Path | None = self.root
        self.cache: dict[str, tuple[int, int, str]] = {}
        self.current_match_files: list[Path] = []
        self._pending_preview_search_terms: list[tuple[str, bool]] = []
        self._pending_preview_search_close_groups: list[list[tuple[str, bool]]] = []
        self._render_pool = QThreadPool(self)
        self._render_pool.setMaxThreadCount(1)
        self._render_request_id = 0
        self._active_render_workers: set[PreviewRenderWorker] = set()
        self._tree_marker_scan_pool = QThreadPool(self)
        self._tree_marker_scan_pool.setMaxThreadCount(1)
        self._tree_marker_scan_request_id = 0
        self._active_tree_marker_scan_workers: set[TreeMarkerScanWorker] = set()
        self._tree_marker_scan_dirty_paths: set[str] = set()
        self._plantuml_pool = QThreadPool(self)
        # Let independent PlantUML blocks render concurrently; keep a modest
        # upper bound to avoid CPU saturation on large documents.
        self._plantuml_pool.setMaxThreadCount(max(2, min(6, os.cpu_count() or 2)))
        self._pdf_pool = QThreadPool(self)
        self._pdf_pool.setMaxThreadCount(1)
        self._active_pdf_workers: set[PdfExportWorker] = set()
        self._pdf_export_in_progress = False
        self._pdf_export_source_key: str | None = None
        self._preview_load_in_progress = False
        self._pdf_diagram_ready_by_key: dict[str, bool] = {}
        self._pdf_diagram_settle_deadline_by_key: dict[str, float] = {}
        self._pdf_diagram_probe_expected_key: str | None = None
        self._pdf_diagram_probe_inflight = False
        self._pdf_diagram_probe_started_at = 0.0
        # PDF export should ignore preview-only zoom. Capture the user's
        # current QWebEngine zoom factor so export can temporarily reset to
        # 100% and then restore the original interactive scale afterward.
        self._pdf_export_saved_preview_zoom: float | None = None
        self._pending_pdf_layout_hints: dict[str, object] = {}
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
        self._view_line_probe_block_until = 0.0
        self._view_states: dict[int, dict[str, float | int]] = {}
        self._active_view_id: int | None = None
        self._next_view_id = 1
        self._next_view_sequence = 1
        self._next_tab_color_index = 0
        self._mermaid_svg_cache_by_mode: dict[str, dict[str, str]] = {
            "auto": {},
            "pdf": {},
        }
        # Per-document, in-memory diagram viewport state (zoom/pan) for this run.
        self._diagram_view_state_by_doc: dict[
            str, dict[str, dict[str, float | bool]]
        ] = {}
        # In-memory per-document tab/view sessions for this app run only.
        self._document_view_sessions: dict[str, dict] = {}
        # Per-directory disk-backed view session cache loaded on demand from
        # .mdexplore-views.json files.
        self._persisted_view_sessions_by_dir: dict[str, dict[str, dict]] = {}
        # Tree gutter badges are cached by root so navigation and tab changes
        # can update only the affected file instead of recursively rescanning
        # the whole tree on every selection.
        self._tree_multi_view_marker_paths: set[str] = set()
        self._tree_highlight_marker_paths: set[str] = set()
        self._tree_marker_cache_root_key: str | None = None
        self._document_line_counts: dict[str, int] = {}
        self._current_document_total_lines = 1
        # Cache per-document rich-render feature flags so post-load startup can
        # skip renderer retries and diagram restore passes for plain markdown.
        self._preview_feature_flags_by_key: dict[str, tuple[bool, bool, bool]] = {}
        self._persisted_text_highlights_by_dir: dict[str, dict[str, list[dict]]] = {}
        self._current_preview_text_highlights: list[dict[str, int | str]] = []
        self._next_text_highlight_id = 1
        self._last_named_view_marker_payload_key: str | None = None
        self._last_named_view_marker_payload_json: str | None = None
        self._debug_enabled = bool(debug_mode)
        self._debug_log_path = Path(__file__).resolve().parent / self.DEBUG_LOG_FILE_NAME
        self._debug_log_line_count = 0
        if self._debug_enabled:
            self._debug_log_line_count = self._count_debug_log_lines()
            self._trim_debug_log()
            self._debug_log(
                f"session-start root={self.root} mermaid_backend={mermaid_backend} "
                f"gpu_context={bool(gpu_context_available)}"
            )
        self._preview_html_temp_files: deque[Path] = deque()
        self._view_line_probe_pending = False
        self._last_view_line_probe_at = 0.0
        self._view_tab_restore_request_id = 0
        self._preview_action_poll_inflight = False
        self._match_input_text = ""
        self._copy_destination_directory: Path | None = None
        # Search is debounced so filesystem/content scans do not run on every
        # keystroke while the user is still typing an expression.
        self.match_timer = QTimer(self)
        self.match_timer.setSingleShot(True)
        self.match_timer.setInterval(1000)
        self.match_timer.timeout.connect(self._run_match_search)
        self._scroll_capture_timer = QTimer(self)
        self._scroll_capture_timer.setInterval(200)
        self._scroll_capture_timer.timeout.connect(self._capture_current_preview_scroll)
        self._scroll_capture_timer.start()
        # Diagram view-state capture is kept separate from normal scroll capture
        # because Mermaid/PlantUML maintain their own zoom/pan state inside the
        # embedded page.
        self._diagram_state_capture_timer = QTimer(self)
        self._diagram_state_capture_timer.setInterval(250)
        self._diagram_state_capture_timer.timeout.connect(
            self._on_diagram_state_capture_tick
        )
        self._preview_action_poll_timer = QTimer(self)
        self._preview_action_poll_timer.setInterval(120)
        self._preview_action_poll_timer.timeout.connect(self._poll_preview_actions)
        self._preview_action_poll_timer.start()
        self._diagram_state_capture_timer.start()
        self._default_status_text = "Ready"
        self._gpu_context_available = bool(gpu_context_available)
        self._status_idle_timer = QTimer(self)
        self._status_idle_timer.setInterval(900)
        self._status_idle_timer.timeout.connect(self._ensure_non_empty_status_message)
        self._status_idle_timer.start()
        self._file_change_watch_timer = QTimer(self)
        self._file_change_watch_timer.setInterval(1200)
        self._file_change_watch_timer.timeout.connect(self._on_file_change_watch_tick)
        self._file_change_watch_timer.start()
        # Centered overlay for long-running preview restore operations.
        self._restore_overlay_expected_key: str | None = None
        self._restore_overlay_needs_math = False
        self._restore_overlay_needs_mermaid = False
        self._restore_overlay_needs_plantuml = False
        self._restore_overlay_deadline = 0.0
        self._restore_overlay_probe_inflight = False
        self._restore_overlay_probe_started_at = 0.0
        self._restore_overlay_pending_show = False
        self._restore_overlay_shown_at = 0.0
        self._restore_overlay_poll_timer = QTimer(self)
        self._restore_overlay_poll_timer.setInterval(170)
        self._restore_overlay_poll_timer.timeout.connect(
            self._check_restore_overlay_progress
        )
        self._restore_overlay_show_timer = QTimer(self)
        self._restore_overlay_show_timer.setSingleShot(True)
        self._restore_overlay_show_timer.setInterval(RESTORE_OVERLAY_SHOW_DELAY_MS)
        self._restore_overlay_show_timer.timeout.connect(self._show_restore_overlay_now)
        self._pdf_diagram_probe_timer = QTimer(self)
        self._pdf_diagram_probe_timer.setInterval(220)
        self._pdf_diagram_probe_timer.timeout.connect(
            self._probe_pdf_diagram_readiness
        )
        # Keep user-adjusted tree/preview pane widths for this app run.
        self._session_splitter_sizes: list[int] | None = None
        self._initial_split_applied = False

        self.setWindowTitle("mdexplore")
        self.setWindowIcon(app_icon)
        _preload_js_assets()
        _preload_template_assets()
        # Give the top control bar a bit more horizontal/vertical room by default.
        self.resize(1848, 980)

        # Use a custom QFileSystemModel so highlight colors render directly
        # in the tree and persist beside files in each directory.
        self.model = ColorizedMarkdownModel(self)
        self.model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)
        self.model.setNameFilters(["*.md"])
        self.model.setNameFilterDisables(False)

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setItemDelegate(MarkdownTreeItemDelegate(self.tree))
        self.tree.setIconSize(ColorizedMarkdownModel.decorated_icon_size())
        # Reduce branch indentation so gutter markers/counts sit closer to the
        # left pane edge.
        self.tree.setIndentation(14)
        self.tree.setHeaderHidden(True)
        self.tree.hideColumn(1)
        self.tree.hideColumn(2)
        self.tree.hideColumn(3)
        self.tree.setColumnWidth(0, 340)
        self.tree.setMinimumWidth(240)
        self.tree.setMaximumWidth(700)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_tree_context_menu)
        self.tree.selectionModel().currentChanged.connect(
            self._on_tree_selection_changed
        )
        self.tree.expanded.connect(self._on_tree_directory_expanded)

        self.preview = QWebEngineView()
        self._preview_page = PreviewPage(self.preview)
        self.preview.setPage(self._preview_page)
        # Preview pages are loaded as local HTML. Allow remote JS/CSS so CDN
        # assets (MathJax/Mermaid) can load and render as expected.
        preview_settings = self.preview.settings()
        preview_settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        preview_settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True
        )
        if hasattr(QWebEngineSettings.WebAttribute, "PrintElementBackgrounds"):
            # Keep PDF output visually closer to what users see in the preview.
            preview_settings.setAttribute(
                QWebEngineSettings.WebAttribute.PrintElementBackgrounds, True
            )
        self.preview.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.preview.customContextMenuRequested.connect(self._show_preview_context_menu)
        self.preview.loadFinished.connect(self._on_preview_load_finished)
        self.preview.urlChanged.connect(self._on_preview_url_changed)
        self._preview_page.namedViewRequested.connect(
            self._on_preview_named_view_requested
        )

        self.view_tabs = ViewTabBar()
        self.view_tabs.setDocumentMode(True)
        self.view_tabs.setMovable(False)
        self.view_tabs.setDrawBase(False)
        self.view_tabs.setExpanding(False)
        self.view_tabs.setUsesScrollButtons(True)
        self.view_tabs.setTabsClosable(True)
        self.view_tabs.setElideMode(Qt.TextElideMode.ElideNone)
        self.view_tabs.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view_tabs.currentChanged.connect(self._on_view_tab_changed)
        self.view_tabs.tabCloseRequested.connect(self._on_view_tab_close_requested)
        self.view_tabs.customContextMenuRequested.connect(
            self._show_view_tab_context_menu
        )
        self.view_tabs.homeRequested.connect(self._on_view_tab_home_requested)
        self.view_tabs.beginningResetRequested.connect(
            self._on_view_tab_beginning_reset_requested
        )
        self.view_tabs.setVisible(False)
        self._reset_document_views()

        # Top-left document controls operate on directory scope and current file
        # scope; the right side hosts clipboard/search operations.
        self.up_btn = QPushButton("^")
        self.up_btn.clicked.connect(self._go_up_directory)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_directory_view)

        self.pdf_btn = QPushButton("PDF")
        self.pdf_btn.setToolTip(
            "Export the currently previewed markdown rendering to PDF"
        )
        self.pdf_btn.clicked.connect(self._export_current_preview_pdf)

        self.add_view_btn = QPushButton("Add View")
        self.add_view_btn.clicked.connect(self._add_document_view)

        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._edit_current_file)

        self.path_label = QLabel("")
        self.path_label.setTextInteractionFlags(self.path_label.textInteractionFlags())

        copy_label = QLabel("Copy to:")
        copy_buttons_widget = QWidget()
        copy_buttons_layout = QHBoxLayout(copy_buttons_widget)
        copy_buttons_layout.setContentsMargins(0, 0, 0, 0)
        copy_buttons_layout.setSpacing(4)
        copy_buttons_layout.addWidget(copy_label)
        self.copy_clipboard_radio = QRadioButton("Clipboard")
        self.copy_directory_radio = QRadioButton("Directory")
        self.copy_clipboard_radio.setChecked(True)
        copy_mode_radio_style = """
            QRadioButton::indicator {
                width: 12px;
                height: 12px;
                border-radius: 6px;
            }
            QRadioButton::indicator:unchecked {
                background-color: #0f1218;
                border: 1px solid #6b7280;
            }
            QRadioButton::indicator:checked {
                background-color: #60a5fa;
                border: 1px solid #93c5fd;
            }
        """
        self.copy_clipboard_radio.setStyleSheet(copy_mode_radio_style)
        self.copy_directory_radio.setStyleSheet(copy_mode_radio_style)
        copy_buttons_layout.addWidget(self.copy_clipboard_radio)
        copy_buttons_layout.addWidget(self.copy_directory_radio)

        copy_current_btn = QPushButton("")
        copy_current_btn.setFixedSize(18, 18)
        copy_current_btn.setToolTip(
            "Copy currently previewed markdown file to selected destination"
        )
        copy_current_btn.setStyleSheet(
            "border: 1px solid #4b5563; border-radius: 3px; padding: 0px;"
        )
        pin_icon_path = _ui_asset_path("pin.png")
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
            color_btn.setToolTip(
                f"Copy files highlighted with {color_name.lower()} to selected destination"
            )
            color_btn.setStyleSheet(
                f"background-color: {color_value}; border: 1px solid #4b5563; border-radius: 3px;"
            )
            color_btn.clicked.connect(
                lambda _checked=False, c=color_value, n=color_name: self._copy_highlighted_files_to_clipboard(
                    c, n
                )
            )
            copy_buttons_layout.addWidget(color_btn)

        # Search UI stays in the toolbar so tree filtering/highlighting remains
        # visible while the preview reacts in the right pane.
        match_label = QLabel("Search and highlight: ")
        self.match_input = QLineEdit()
        self.match_input.setClearButtonEnabled(False)
        self.match_input.setPlaceholderText(
            'words, "phrases", \'case-sensitive\', AND/OR/NOT, NEAR(...)'
        )
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
                lambda _checked=False, c=color_value, n=color_name: self._apply_match_highlight_color(
                    c, n
                )
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
        self.preview_container = preview_container
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)
        preview_layout.addWidget(self.view_tabs)
        preview_layout.addWidget(self.preview, 1)

        # This badge is purely transient UI feedback for preview-wide zoom
        # changes; it is not tied to Mermaid/PlantUML internal zoom state.
        self._preview_zoom_overlay = QLabel("", self.preview_container)
        self._preview_zoom_overlay.setObjectName("mdexplore-preview-zoom-overlay")
        self._preview_zoom_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_zoom_overlay.setStyleSheet(
            """
            QLabel#mdexplore-preview-zoom-overlay {
                background-color: rgba(11, 20, 38, 210);
                color: #e5e7eb;
                border: 1px solid rgba(148, 163, 184, 0.55);
                border-radius: 8px;
                padding: 3px 9px;
                font-weight: 600;
            }
            """
        )
        self._preview_zoom_overlay.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._preview_zoom_overlay.hide()
        self._preview_zoom_overlay_timer = QTimer(self)
        self._preview_zoom_overlay_timer.setSingleShot(True)
        self._preview_zoom_overlay_timer.setInterval(PREVIEW_ZOOM_OVERLAY_TIMEOUT_MS)
        self._preview_zoom_overlay_timer.timeout.connect(self._preview_zoom_overlay.hide)

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
        self._restore_overlay = QLabel(
            "One moment please...",
            self,
        )
        self._restore_overlay.setObjectName("mdexplore-restore-overlay")
        self._restore_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._restore_overlay.setWordWrap(True)
        self._restore_overlay.setStyleSheet(
            """
            QLabel#mdexplore-restore-overlay {
                background-color: rgba(147, 197, 253, 238);
                color: #000000;
                border: 1px solid #60a5fa;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 600;
            }
            """
        )
        self._restore_overlay.hide()
        self._position_restore_overlay()
        self.statusBar().showMessage(self._default_status_text)
        self._gpu_status_label = QLabel("GPU")
        self._gpu_status_label.setStyleSheet("color: rgba(156, 163, 175, 0.45);")
        self._gpu_status_label.setVisible(self._gpu_context_available)
        self.statusBar().addPermanentWidget(self._gpu_status_label)
        backend_warning = self.renderer.mermaid_backend_warning()
        if backend_warning:
            self.statusBar().showMessage(
                f"Mermaid backend fallback to JS: {backend_warning}",
                7000,
            )
        # Root is initialized after widgets exist so view/model indexes are valid.
        self._set_root_directory(self.root)
        self._update_pdf_button_state()
        self._add_shortcuts()
        self.model.directoryLoaded.connect(self._maybe_apply_initial_split)
        QTimer.singleShot(0, self._maybe_apply_initial_split)

    def _add_shortcuts(self) -> None:
        """Register window-level keyboard shortcuts."""
        refresh_action = QAction("Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._refresh_directory_view)
        self.addAction(refresh_action)

        preview_zoom_in_action = QAction("Preview Zoom In", self)
        preview_zoom_in_action.setShortcuts(
            ["Ctrl++", "Ctrl+=", "Ctrl+Shift+="]
        )
        preview_zoom_in_action.triggered.connect(self._zoom_preview_in)
        self.addAction(preview_zoom_in_action)

        preview_zoom_out_action = QAction("Preview Zoom Out", self)
        preview_zoom_out_action.setShortcuts(["Ctrl+-", "Ctrl+_"])
        preview_zoom_out_action.triggered.connect(self._zoom_preview_out)
        self.addAction(preview_zoom_out_action)

        preview_zoom_reset_action = QAction("Preview Zoom Reset", self)
        preview_zoom_reset_action.setShortcuts(["Ctrl+0", "Ctrl+Shift+0"])
        preview_zoom_reset_action.triggered.connect(self._reset_preview_zoom)
        self.addAction(preview_zoom_reset_action)

    def _set_preview_zoom_factor(self, factor: float) -> None:
        """Set preview-only QWebEngine zoom factor with clamped bounds."""
        clamped = max(PREVIEW_ZOOM_MIN, min(PREVIEW_ZOOM_MAX, float(factor)))
        self.preview.setZoomFactor(clamped)
        percent_text = f"{int(round(clamped * 100))}%"
        self.statusBar().showMessage(f"Preview zoom: {percent_text}", 1500)
        self._show_preview_zoom_overlay(percent_text)

    def _zoom_preview_in(self) -> None:
        """Increase preview pane zoom level."""
        current = float(self.preview.zoomFactor())
        self._set_preview_zoom_factor(current + PREVIEW_ZOOM_STEP)

    def _zoom_preview_out(self) -> None:
        """Decrease preview pane zoom level."""
        current = float(self.preview.zoomFactor())
        self._set_preview_zoom_factor(current - PREVIEW_ZOOM_STEP)

    def _reset_preview_zoom(self) -> None:
        """Reset preview pane zoom level to default scale."""
        self._set_preview_zoom_factor(PREVIEW_ZOOM_RESET)

    def _show_preview_zoom_overlay(self, percent_text: str) -> None:
        """Show a short-lived zoom percentage label at top of preview pane."""
        if not hasattr(self, "_preview_zoom_overlay"):
            return
        self._preview_zoom_overlay.setText(percent_text)
        self._preview_zoom_overlay.adjustSize()
        self._position_preview_zoom_overlay()
        self._preview_zoom_overlay.raise_()
        self._preview_zoom_overlay.show()
        self._preview_zoom_overlay_timer.start()

    def _position_preview_zoom_overlay(self) -> None:
        """Keep zoom overlay anchored to the top center of the preview view."""
        if not hasattr(self, "_preview_zoom_overlay") or not hasattr(self, "preview"):
            return
        overlay = self._preview_zoom_overlay
        target = self.preview
        top_left = target.mapTo(self.preview_container, QPoint(0, 0))
        target_width = max(1, target.width())
        x = top_left.x() + max(8, (target_width - overlay.width()) // 2)
        y = top_left.y() + 8
        overlay.move(x, y)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Keep centered overlays aligned when the main window is resized."""
        super().resizeEvent(event)
        self._position_restore_overlay()
        self._position_preview_zoom_overlay()

    def _position_restore_overlay(self) -> None:
        """Center the restore popup label within the visible window area."""
        if not hasattr(self, "_restore_overlay"):
            return
        available_width = max(320, self.width() - 80)
        target_width = min(760, available_width)
        self._restore_overlay.setFixedWidth(target_width)
        self._restore_overlay.adjustSize()
        x = max(20, (self.width() - self._restore_overlay.width()) // 2)
        y = max(20, (self.height() - self._restore_overlay.height()) // 2)
        self._restore_overlay.move(x, y)

    @staticmethod
    def _detect_special_features_from_markdown(
        markdown_text: str,
    ) -> tuple[bool, bool, bool]:
        """Return (has_math, has_mermaid, has_plantuml) from raw markdown."""
        text = markdown_text or ""
        has_mermaid = bool(re.search(r"(?im)^\s*```+\s*mermaid\b", text))
        has_plantuml = bool(re.search(r"(?im)^\s*```+\s*(?:plantuml|puml|uml)\b", text))
        has_math = bool(
            re.search(r"(?s)(?<!\\)\$\$(.+?)(?<!\\)\$\$", text)
            or re.search(r"(?<!\\)\$(?!\$)(?:[^$\n]|\\\$){1,400}?(?<!\\)\$", text)
        )
        return has_math, has_mermaid, has_plantuml

    @staticmethod
    def _detect_special_features_from_html(html_doc: str) -> tuple[bool, bool, bool]:
        """Return (has_math, has_mermaid, has_plantuml) from rendered HTML."""
        text = html_doc or ""
        has_mermaid = 'class="mermaid"' in text
        has_plantuml = ("plantuml-async" in text) or ('class="plantuml"' in text)
        has_math = (
            "mdexplore-math-block" in text
            or bool(re.search(r"(?<!\\)\$(?!\$)(?:[^$\n]|\\\$){1,400}?(?<!\\)\$", text))
            or bool(re.search(r"(?s)(?<!\\)\$\$(.+?)(?<!\\)\$\$", text))
        )
        return has_math, has_mermaid, has_plantuml

    def _detect_special_features_for_path(
        self,
        path: Path,
        *,
        cached_html: str | None = None,
    ) -> tuple[bool, bool, bool]:
        """Detect rich-render features from source markdown, with HTML fallback."""
        try:
            markdown_text = path.read_text(encoding="utf-8", errors="replace")
            return self._detect_special_features_from_markdown(markdown_text)
        except Exception:
            if cached_html is not None:
                return self._detect_special_features_from_html(cached_html)
            return (False, False, False)

    def _begin_restore_overlay_monitor(
        self,
        expected_key: str,
        *,
        needs_math: bool,
        needs_mermaid: bool,
        needs_plantuml: bool,
        phase: str,
    ) -> None:
        """Start delayed restore popup and readiness polling for rich previews."""
        # Disabled by request: this overlay could interfere with restore UX.
        self._stop_restore_overlay_monitor()
        return

    def _show_restore_overlay_now(self) -> None:
        """Show centered popup after delay if work is still in-flight."""
        if not self._restore_overlay_pending_show:
            return
        expected_key = self._restore_overlay_expected_key
        if not expected_key or self._current_preview_path_key() != expected_key:
            return
        self._position_restore_overlay()
        self._restore_overlay_shown_at = time.monotonic()
        self._restore_overlay.raise_()
        self._restore_overlay.show()

    def _stop_restore_overlay_monitor(self) -> None:
        """Stop restore polling and hide the centered progress popup."""
        self._restore_overlay_expected_key = None
        self._restore_overlay_needs_math = False
        self._restore_overlay_needs_mermaid = False
        self._restore_overlay_needs_plantuml = False
        self._restore_overlay_deadline = 0.0
        self._restore_overlay_probe_inflight = False
        self._restore_overlay_probe_started_at = 0.0
        self._restore_overlay_pending_show = False
        self._restore_overlay_shown_at = 0.0
        self._restore_overlay_show_timer.stop()
        self._restore_overlay_poll_timer.stop()
        if hasattr(self, "_restore_overlay"):
            self._restore_overlay.hide()

    def _check_restore_overlay_progress(self) -> None:
        """Poll current preview restore readiness and dismiss popup when ready."""
        expected_key = self._restore_overlay_expected_key
        if not expected_key:
            self._stop_restore_overlay_monitor()
            return
        if self._current_preview_path_key() != expected_key:
            self._stop_restore_overlay_monitor()
            return
        if time.monotonic() >= self._restore_overlay_deadline:
            self._stop_restore_overlay_monitor()
            return
        if self._restore_overlay_shown_at > 0.0:
            if (
                time.monotonic() - self._restore_overlay_shown_at
            ) >= RESTORE_OVERLAY_MAX_VISIBLE_SECONDS:
                self._stop_restore_overlay_monitor()
                return

        plantuml_ready = True
        if self._restore_overlay_needs_plantuml:
            progress = self._preview_plantuml_progress()
            pending = bool(progress and progress[1] > 0)
            plantuml_ready = not pending

        needs_js_probe = (
            self._restore_overlay_needs_math or self._restore_overlay_needs_mermaid
        )
        if not needs_js_probe:
            if plantuml_ready:
                self._stop_restore_overlay_monitor()
            return
        if self._restore_overlay_probe_inflight:
            if (time.monotonic() - self._restore_overlay_probe_started_at) < 1.8:
                return
            # A prior JS callback can be dropped during rapid page switches.
            # Clear stale in-flight state and re-issue probe.
            self._restore_overlay_probe_inflight = False
            self._restore_overlay_probe_started_at = 0.0

        self._restore_overlay_probe_inflight = True
        self._restore_overlay_probe_started_at = time.monotonic()
        js = _render_js_asset("preview/probe_restore_overlay_readiness.js")
        self.preview.page().runJavaScript(
            js,
            lambda result, key=expected_key, plantuml_ok=plantuml_ready: self._on_restore_overlay_probe_result(
                key,
                plantuml_ok,
                result,
            ),
        )

    def _on_restore_overlay_probe_result(
        self, expected_key: str, plantuml_ready: bool, result
    ) -> None:
        """Handle JS readiness probe for math/mermaid restore completion."""
        self._restore_overlay_probe_inflight = False
        self._restore_overlay_probe_started_at = 0.0
        if self._restore_overlay_expected_key != expected_key:
            return
        if self._current_preview_path_key() != expected_key:
            self._stop_restore_overlay_monitor()
            return
        if time.monotonic() >= self._restore_overlay_deadline:
            self._stop_restore_overlay_monitor()
            return

        math_ready = True
        mermaid_ready = True
        if self._restore_overlay_needs_math:
            math_ready = bool(isinstance(result, dict) and result.get("mathReady"))
        if self._restore_overlay_needs_mermaid:
            mermaid_ready = bool(
                isinstance(result, dict) and result.get("mermaidReady")
            )

        if math_ready and mermaid_ready and plantuml_ready:
            self._stop_restore_overlay_monitor()

    @staticmethod
    def _view_tab_label_for_line(line_number: int) -> str:
        """Return compact tab text for a view anchored near a source line."""
        return str(max(1, int(line_number)))

    @staticmethod
    def _normalize_custom_view_label(raw_value) -> str | None:
        """Normalize a custom tab label; blank resets, long labels are truncated."""
        if not isinstance(raw_value, str):
            return None
        if not raw_value.strip():
            return None
        cleaned = raw_value.replace("\r", " ").replace("\n", " ")
        if len(cleaned) > ViewTabBar.MAX_LABEL_CHARS:
            cleaned = cleaned[: ViewTabBar.MAX_LABEL_CHARS]
        return cleaned

    def _display_label_for_view(
        self, line_number: int, custom_label: str | None = None
    ) -> str:
        """Return visible tab label, preferring a validated custom label override."""
        normalized = self._normalize_custom_view_label(custom_label)
        if normalized is not None:
            return normalized
        return self._view_tab_label_for_line(line_number)

    def _tab_custom_label(self, tab_index: int) -> str | None:
        """Read a tab's persisted custom label override from tab metadata."""
        if tab_index < 0 or tab_index >= self.view_tabs.count():
            return None
        data = self.view_tabs.tabData(tab_index)
        if not isinstance(data, dict):
            return None
        return self._normalize_custom_view_label(data.get("custom_label"))

    def _tab_label_anchor(self, tab_index: int) -> tuple[float, int] | None:
        """Read the saved custom-label bookmark for one tab, if present."""
        if tab_index < 0 or tab_index >= self.view_tabs.count():
            return None
        data = self.view_tabs.tabData(tab_index)
        if not isinstance(data, dict):
            return None
        if self._normalize_custom_view_label(data.get("custom_label")) is None:
            return None
        try:
            scroll_y = float(data.get("custom_label_anchor_scroll_y", 0.0))
        except Exception:
            scroll_y = 0.0
        try:
            top_line = max(1, int(data.get("custom_label_anchor_top_line", 1)))
        except Exception:
            top_line = 1
        if not math.isfinite(scroll_y):
            scroll_y = 0.0
        return scroll_y, top_line

    def _current_named_view_marker_entries(self) -> list[dict[str, int | str | float]]:
        """Return named-view gutter markers for the active document."""
        palette = ViewTabBar.PASTEL_SEQUENCE
        palette_size = max(1, len(palette))
        markers: list[dict[str, int | str | float]] = []
        seen_view_ids: set[int] = set()
        for tab_index in range(self.view_tabs.count()):
            anchor = self._tab_label_anchor(tab_index)
            if anchor is None:
                continue
            view_id = self._tab_view_id(tab_index)
            if view_id is None or view_id in seen_view_ids:
                continue
            seen_view_ids.add(view_id)
            anchor_scroll_y, anchor_top_line = anchor
            state = self._view_states.get(view_id) or {}
            try:
                scroll_y = float(state.get("scroll_y", anchor_scroll_y))
            except Exception:
                scroll_y = anchor_scroll_y
            try:
                top_line = max(1, int(state.get("top_line", anchor_top_line)))
            except Exception:
                top_line = anchor_top_line
            if not math.isfinite(scroll_y):
                scroll_y = anchor_scroll_y if math.isfinite(anchor_scroll_y) else 0.0
            data = self.view_tabs.tabData(tab_index)
            color_slot = tab_index % palette_size
            if isinstance(data, dict):
                try:
                    color_slot = int(data.get("color_slot", color_slot))
                except Exception:
                    color_slot = tab_index % palette_size
            if color_slot < 0 or color_slot >= palette_size:
                color_slot = tab_index % palette_size
            markers.append(
                {
                    "view_id": int(view_id),
                    "top_line": max(1, int(top_line)),
                    "scroll_y": max(0.0, float(scroll_y)),
                    "color": str(palette[color_slot]),
                }
            )
        return markers

    def _refresh_named_view_markers_in_preview(
        self, expected_key: str | None = None, *, force: bool = False
    ) -> None:
        """Push named-view home markers into the live preview without reloading."""
        current_key = self._current_preview_path_key()
        if current_key is None:
            self._last_named_view_marker_payload_key = None
            self._last_named_view_marker_payload_json = None
            return
        if expected_key is not None and expected_key != current_key:
            return
        payload = self._current_named_view_marker_entries()
        payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        if (
            not force
            and self._last_named_view_marker_payload_key == current_key
            and self._last_named_view_marker_payload_json == payload_json
        ):
            return
        self._last_named_view_marker_payload_key = current_key
        self._last_named_view_marker_payload_json = payload_json
        js = _render_js_asset(
            "preview/update_named_view_markers.js",
            {"__PAYLOAD_JSON__": payload_json},
        )
        # Marker placement is derived entirely from persisted line anchors, so
        # a small payload update is enough; avoid full preview reloads here.
        self.preview.page().runJavaScript(js)

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

    def _tab_index_for_view_id(self, view_id: int) -> int | None:
        """Resolve one view id back to its current tab index."""
        for index in range(self.view_tabs.count()):
            if self._tab_view_id(index) == view_id:
                return index
        return None

    def _begin_preview_restore_block(self, duration_seconds: float) -> None:
        """Pause capture/probe writes while a scripted restore is settling."""
        try:
            duration = max(0.0, float(duration_seconds))
        except Exception:
            duration = 0.0
        block_until = time.monotonic() + duration
        self._preview_capture_enabled = False
        self._scroll_restore_block_until = block_until
        self._view_line_probe_block_until = max(
            self._view_line_probe_block_until, block_until
        )

    def _schedule_view_restore(
        self, expected_key: str, expected_view_id: int, *, needs_settle_restore: bool
    ) -> None:
        """Queue the guarded restore sequence shared by tabs and gutter markers."""
        self._view_tab_restore_request_id += 1
        restore_request_id = self._view_tab_restore_request_id
        self._begin_preview_restore_block(1.0 if needs_settle_restore else 0.55)
        QTimer.singleShot(
            0,
            lambda key=expected_key, view_id=expected_view_id, request_id=restore_request_id: self._restore_current_preview_scroll_if_active(
                key, view_id, request_id, stabilize=False
            ),
        )
        if needs_settle_restore:
            QTimer.singleShot(
                360,
                lambda key=expected_key, view_id=expected_view_id, request_id=restore_request_id: self._restore_current_preview_scroll_if_active(
                    key, view_id, request_id, stabilize=True
                ),
            )
        QTimer.singleShot(
            980 if needs_settle_restore else 520,
            lambda key=expected_key, view_id=expected_view_id, request_id=restore_request_id: self._enable_preview_scroll_capture_if_active(
                key, view_id, request_id
            ),
        )

    def _save_document_view_session(
        self,
        path_key: str | None = None,
        *,
        capture_current: bool = True,
    ) -> None:
        """Snapshot current tab/view state for one document path key."""
        if path_key is None:
            path_key = self._current_preview_path_key()
        if not path_key:
            return

        if capture_current:
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
        tabs: list[dict[str, int | float | str]] = []
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
                custom_label = self._normalize_custom_view_label(
                    data.get("custom_label")
                )
                try:
                    custom_label_anchor_scroll_y = float(
                        data.get("custom_label_anchor_scroll_y", 0.0)
                    )
                except Exception:
                    custom_label_anchor_scroll_y = 0.0
                try:
                    custom_label_anchor_top_line = max(
                        1, int(data.get("custom_label_anchor_top_line", 1))
                    )
                except Exception:
                    custom_label_anchor_top_line = 1
                try:
                    sequence = max(1, int(raw_sequence))
                except Exception:
                    sequence = index + 1
                try:
                    color_slot = int(raw_color_slot)
                except Exception:
                    color_slot = (sequence - 1) % palette_size
            else:
                custom_label = None
                custom_label_anchor_scroll_y = 0.0
                custom_label_anchor_top_line = 1
            if color_slot < 0 or color_slot >= palette_size:
                color_slot = (sequence - 1) % palette_size
            state = sanitized_states.get(view_id)
            if state is None:
                state = {"scroll_y": 0.0, "top_line": 1}
                sanitized_states[view_id] = state
            tab_entry: dict[str, int | float | str] = {
                "view_id": view_id,
                "sequence": sequence,
                "color_slot": color_slot,
            }
            if custom_label is not None:
                tab_entry["custom_label"] = custom_label
                if not math.isfinite(custom_label_anchor_scroll_y):
                    custom_label_anchor_scroll_y = 0.0
                tab_entry["custom_label_anchor_scroll_y"] = custom_label_anchor_scroll_y
                tab_entry["custom_label_anchor_top_line"] = custom_label_anchor_top_line
            tabs.append(tab_entry)
            max_sequence = max(max_sequence, sequence)
            max_view_id = max(max_view_id, view_id)

        active_view_id = self._active_view_id
        if active_view_id is None or all(
            entry["view_id"] != active_view_id for entry in tabs
        ):
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

    @staticmethod
    def _clone_json_compatible_dict(payload: dict) -> dict:
        """Return a detached copy of a JSON-compatible dictionary."""
        try:
            cloned = json.loads(json.dumps(payload, ensure_ascii=False))
        except Exception:
            return {}
        return cloned if isinstance(cloned, dict) else {}

    @staticmethod
    def _clone_json_compatible_list(payload: list) -> list:
        """Return a detached copy of a JSON-compatible list."""
        try:
            cloned = json.loads(json.dumps(payload, ensure_ascii=False))
        except Exception:
            return []
        return cloned if isinstance(cloned, list) else []

    def _count_debug_log_lines(self) -> int:
        """Return current debug log line count (best effort)."""
        if not getattr(self, "_debug_enabled", False):
            return 0
        try:
            with self._debug_log_path.open("r", encoding="utf-8", errors="replace") as fh:
                return sum(1 for _ in fh)
        except Exception:
            return 0

    def _trim_debug_log(self) -> None:
        """Keep the debug log capped to the most recent configured line count."""
        if not getattr(self, "_debug_enabled", False):
            return
        if self._debug_log_line_count <= self.DEBUG_LOG_MAX_LINES:
            return
        try:
            recent = deque(maxlen=self.DEBUG_LOG_MAX_LINES)
            with self._debug_log_path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    recent.append(line)
            with self._debug_log_path.open("w", encoding="utf-8") as fh:
                fh.writelines(recent)
            self._debug_log_line_count = len(recent)
        except Exception:
            # Debug logging is non-critical and should never block UI behavior.
            pass

    def _debug_log(self, message: str) -> None:
        """Append one timestamped debug line to mdexplore.log in project root."""
        if not getattr(self, "_debug_enabled", False):
            return
        try:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with self._debug_log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[{stamp}] {message}\n")
            self._debug_log_line_count += 1
            self._trim_debug_log()
        except Exception:
            pass

    @staticmethod
    def _path_directory_and_name(path_key: str | None) -> tuple[Path, str] | None:
        """Resolve a persisted path key into its directory and markdown filename."""
        if not isinstance(path_key, str) or not path_key:
            return None
        path = Path(path_key)
        name = path.name
        if not name:
            return None
        directory = path.parent
        try:
            directory = directory.resolve()
        except Exception:
            pass
        return directory, name

    def _views_file_path(self, directory: Path) -> Path:
        """Return the sidecar JSON path that stores persisted document views."""
        return directory / self.VIEWS_FILE_NAME

    def _highlighting_file_path(self, directory: Path) -> Path:
        """Return the sidecar JSON path that stores preview text highlights."""
        return directory / self.HIGHLIGHTING_FILE_NAME

    @staticmethod
    def _normalize_text_highlight_entries(raw_entries) -> list[dict[str, int | str]]:
        """Sanitize and merge persistent text-highlight ranges by kind."""
        if not isinstance(raw_entries, list):
            return []

        sanitized: list[dict[str, int | str]] = []
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            raw_id = item.get("id")
            if not isinstance(raw_id, str) or not raw_id.strip():
                continue
            try:
                start = int(item.get("start", -1))
                end = int(item.get("end", -1))
            except Exception:
                continue
            raw_kind = item.get("kind", PREVIEW_HIGHLIGHT_KIND_NORMAL)
            kind = (
                str(raw_kind).strip().lower()
                if isinstance(raw_kind, str)
                else PREVIEW_HIGHLIGHT_KIND_NORMAL
            )
            if kind not in {
                PREVIEW_HIGHLIGHT_KIND_NORMAL,
                PREVIEW_HIGHLIGHT_KIND_IMPORTANT,
            }:
                kind = PREVIEW_HIGHLIGHT_KIND_NORMAL
            if start < 0 or end <= start:
                continue
            sanitized.append(
                {"id": raw_id, "start": start, "end": end, "kind": kind}
            )

        if not sanitized:
            return []

        sanitized.sort(
            key=lambda entry: (
                int(entry["start"]),
                int(entry["end"]),
                str(entry.get("kind", PREVIEW_HIGHLIGHT_KIND_NORMAL)),
            )
        )
        merged: list[dict[str, int | str]] = []
        for entry in sanitized:
            if not merged:
                merged.append(entry)
                continue
            prev = merged[-1]
            prev_start = int(prev["start"])
            prev_end = int(prev["end"])
            cur_start = int(entry["start"])
            cur_end = int(entry["end"])
            prev_kind = str(prev.get("kind", PREVIEW_HIGHLIGHT_KIND_NORMAL))
            cur_kind = str(entry.get("kind", PREVIEW_HIGHLIGHT_KIND_NORMAL))
            # Merge overlap/adjacent ranges only when they share the same visual
            # style; this preserves normal-vs-important transitions.
            if cur_kind == prev_kind and cur_start <= prev_end:
                prev["start"] = min(prev_start, cur_start)
                prev["end"] = max(prev_end, cur_end)
                continue
            merged.append(entry)
        return merged

    @staticmethod
    def _normalize_preview_highlight_kind(kind: str | None) -> str:
        """Return a supported persistent preview-highlight kind."""
        raw = str(kind or PREVIEW_HIGHLIGHT_KIND_NORMAL).strip().lower()
        if raw == PREVIEW_HIGHLIGHT_KIND_IMPORTANT:
            return PREVIEW_HIGHLIGHT_KIND_IMPORTANT
        return PREVIEW_HIGHLIGHT_KIND_NORMAL

    def _new_text_highlight_id(self) -> str:
        """Return a unique id for a persistent preview text-highlight range."""
        token = self._next_text_highlight_id
        self._next_text_highlight_id += 1
        return f"h{int(time.time() * 1000):x}-{token:x}"

    def _directory_text_highlights(self, directory: Path) -> dict[str, list[dict]]:
        """Load or return cached text highlights for one directory sidecar."""
        try:
            resolved_directory = directory.resolve()
        except Exception:
            resolved_directory = directory
        directory_key = str(resolved_directory)
        cached = self._persisted_text_highlights_by_dir.get(directory_key)
        if cached is not None:
            return cached

        highlights_by_file: dict[str, list[dict]] = {}
        file_path = self._highlighting_file_path(resolved_directory)
        try:
            raw_payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            raw_payload = None

        file_map = (
            raw_payload.get("files") if isinstance(raw_payload, dict) else raw_payload
        )
        if isinstance(file_map, dict):
            for raw_name, raw_entries in file_map.items():
                if not isinstance(raw_name, str):
                    continue
                file_name = Path(raw_name).name
                if not file_name.lower().endswith(".md"):
                    continue
                entries = self._normalize_text_highlight_entries(raw_entries)
                if entries:
                    highlights_by_file[file_name] = entries

        self._persisted_text_highlights_by_dir[directory_key] = highlights_by_file
        return highlights_by_file

    def _save_directory_text_highlights(self, directory: Path) -> None:
        """Persist one directory's text-highlight sidecar, failing quietly on IO errors."""
        try:
            resolved_directory = directory.resolve()
        except Exception:
            resolved_directory = directory
        directory_key = str(resolved_directory)
        highlights_by_file = self._persisted_text_highlights_by_dir.get(
            directory_key, {}
        )
        file_path = self._highlighting_file_path(resolved_directory)

        serializable_files: dict[str, list[dict]] = {}
        if isinstance(highlights_by_file, dict):
            for file_name in sorted(highlights_by_file):
                raw_entries = highlights_by_file.get(file_name)
                entries = self._normalize_text_highlight_entries(raw_entries)
                if entries:
                    serializable_files[file_name] = entries

        try:
            if not serializable_files:
                if file_path.exists():
                    file_path.unlink()
                return
            payload = {"files": serializable_files}
            file_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except Exception:
            # Requested behavior for sidecars is best-effort persistence.
            pass

    def _load_text_highlights_for_path_key(self, path_key: str | None) -> list[dict]:
        """Return persisted text-highlight entries for one markdown path key."""
        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return []
        directory, file_name = resolved
        entries = self._directory_text_highlights(directory).get(file_name, [])
        return self._normalize_text_highlight_entries(entries)

    def _persist_text_highlights_for_path_key(
        self, path_key: str | None, entries: list[dict]
    ) -> None:
        """Persist one markdown file's text-highlight entries into its directory sidecar."""
        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return
        directory, file_name = resolved
        normalized = self._normalize_text_highlight_entries(entries)
        highlights_by_file = self._directory_text_highlights(directory)
        if normalized:
            highlights_by_file[file_name] = self._clone_json_compatible_list(normalized)
        else:
            highlights_by_file.pop(file_name, None)
        self._save_directory_text_highlights(directory)
        root = getattr(self, "root", None)
        if isinstance(root, Path):
            try:
                target_path = (directory / file_name).resolve()
            except Exception:
                target_path = directory / file_name
            target_key = self._path_key(target_path)
            root_key = self._path_key(root)
            if target_key == root_key or target_key.startswith(root_key + os.sep):
                self._refresh_tree_multi_view_markers(changed_path_key=target_key)

    def _directory_view_sessions(self, directory: Path) -> dict[str, dict]:
        """Load or return cached persisted view sessions for one directory."""
        try:
            resolved_directory = directory.resolve()
        except Exception:
            resolved_directory = directory
        directory_key = str(resolved_directory)
        cached = self._persisted_view_sessions_by_dir.get(directory_key)
        if cached is not None:
            return cached

        sessions: dict[str, dict] = {}
        file_path = self._views_file_path(resolved_directory)
        try:
            raw_payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            raw_payload = None

        file_map = (
            raw_payload.get("files") if isinstance(raw_payload, dict) else raw_payload
        )
        if isinstance(file_map, dict):
            for raw_name, raw_session in file_map.items():
                if not isinstance(raw_name, str) or not isinstance(raw_session, dict):
                    continue
                file_name = Path(raw_name).name
                if not file_name.lower().endswith(".md"):
                    continue
                cloned = self._clone_json_compatible_dict(raw_session)
                if cloned:
                    sessions[file_name] = cloned

        self._persisted_view_sessions_by_dir[directory_key] = sessions
        return sessions

    def _save_directory_view_sessions(self, directory: Path) -> None:
        """Persist one directory's view-session sidecar, failing quietly on IO errors."""
        try:
            resolved_directory = directory.resolve()
        except Exception:
            resolved_directory = directory
        directory_key = str(resolved_directory)
        sessions = self._persisted_view_sessions_by_dir.get(directory_key, {})
        file_path = self._views_file_path(resolved_directory)

        serializable_files: dict[str, dict] = {}
        if isinstance(sessions, dict):
            for file_name in sorted(sessions):
                raw_session = sessions.get(file_name)
                if not isinstance(file_name, str) or not isinstance(raw_session, dict):
                    continue
                cloned = self._clone_json_compatible_dict(raw_session)
                if cloned:
                    serializable_files[file_name] = cloned

        try:
            if not serializable_files:
                if file_path.exists():
                    file_path.unlink()
                return
            payload = {"files": serializable_files}
            file_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except Exception:
            # View persistence is best-effort only and must not interrupt UI flow.
            pass

    @staticmethod
    def _should_persist_document_view_session(session: dict | None) -> bool:
        """Persist documents that have multi-view or custom-label state to restore."""
        if not isinstance(session, dict):
            return False
        tabs = session.get("tabs")
        if not isinstance(tabs, list):
            return False
        if len(tabs) > 1:
            return True
        for entry in tabs:
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("custom_label"), str)
                and entry.get("custom_label").strip()
            ):
                return True
        return False

    def _load_persisted_document_view_session(self, path_key: str) -> None:
        """Populate in-memory session cache from directory sidecar when needed."""
        if not path_key or path_key in self._document_view_sessions:
            return
        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return
        directory, file_name = resolved
        session = self._directory_view_sessions(directory).get(file_name)
        if not isinstance(session, dict):
            return
        cloned = self._clone_json_compatible_dict(session)
        if cloned:
            self._document_view_sessions[path_key] = cloned

    def _persist_document_view_session(
        self,
        path_key: str | None = None,
        *,
        capture_current: bool = True,
    ) -> None:
        """Flush one document's in-memory view session snapshot into its directory sidecar."""
        if path_key is None:
            path_key = self._current_preview_path_key()
        if not path_key:
            return

        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return
        directory, file_name = resolved

        self._save_document_view_session(path_key, capture_current=capture_current)
        sessions = self._directory_view_sessions(directory)
        session = self._document_view_sessions.get(path_key)
        if self._should_persist_document_view_session(session):
            sessions[file_name] = self._clone_json_compatible_dict(session)
        else:
            sessions.pop(file_name, None)
        self._save_directory_view_sessions(directory)
        self._refresh_tree_multi_view_markers(changed_path_key=path_key)

    def _serialized_mermaid_cache_json(self) -> str:
        """Serialize in-memory Mermaid SVG cache for template injection."""
        try:
            return json.dumps(
                self._mermaid_svg_cache_by_mode,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except Exception:
            return "{}"

    def _merge_renderer_pdf_mermaid_cache_seed(
        self, payload: dict[str, str] | None = None
    ) -> None:
        """Merge renderer-produced Rust PDF Mermaid SVGs into runtime cache."""
        if payload is None:
            try:
                payload = self.renderer.take_last_mermaid_pdf_svg_by_hash()
            except Exception:
                return
        if not isinstance(payload, dict) or not payload:
            return

        target = self._mermaid_svg_cache_by_mode.setdefault("pdf", {})
        for raw_hash, raw_svg in payload.items():
            if not isinstance(raw_hash, str) or not isinstance(raw_svg, str):
                continue
            hash_key = raw_hash.strip().lower()
            if not re.fullmatch(r"[0-9a-f]{40}", hash_key):
                continue
            if "<svg" not in raw_svg.casefold():
                continue
            if len(raw_svg) > MERMAID_SVG_MAX_CHARS:
                continue
            target[hash_key] = raw_svg
        while len(target) > MERMAID_SVG_CACHE_MAX_ENTRIES:
            target.pop(next(iter(target)))

    def _build_preview_render_payload(
        self, resolved: Path
    ) -> tuple[str, int, int, dict[str, object]]:
        """Render one markdown file to HTML and return side metadata for the UI thread."""
        stat = resolved.stat()
        cache_key = str(resolved)
        markdown_text = resolved.read_text(encoding="utf-8", errors="replace")
        has_math, has_mermaid, has_plantuml = (
            self._detect_special_features_from_markdown(markdown_text)
        )
        total_lines = self._count_markdown_lines(markdown_text)
        doc_id = self._doc_id_for_path(cache_key)
        placeholders_by_hash: dict[str, list[str]] = {}
        prepared_plantuml_sources_by_hash: dict[str, str] = {}
        worker_renderer = MarkdownRenderer(
            mermaid_backend=self.renderer._mermaid_backend_requested
        )

        def plantuml_resolver(code: str, index: int, line_attrs: str) -> str:
            prepared_code = worker_renderer._prepare_plantuml_source(code)
            hash_key = hashlib.sha1(
                prepared_code.encode("utf-8", errors="replace")
            ).hexdigest()
            placeholder_id = f"mdexplore-plantuml-{doc_id}-{index}"
            placeholders_by_hash.setdefault(hash_key, []).append(placeholder_id)
            prepared_plantuml_sources_by_hash.setdefault(hash_key, prepared_code)
            return self._plantuml_block_html(
                placeholder_id, line_attrs, "pending", "", hash_key=hash_key
            )

        html_doc = worker_renderer.render_document(
            markdown_text,
            resolved.name,
            total_lines=total_lines,
            plantuml_resolver=plantuml_resolver,
        )
        metadata: dict[str, object] = {
            "total_lines": total_lines,
            "has_math": has_math,
            "has_mermaid": has_mermaid,
            "has_plantuml": has_plantuml,
            "placeholders_by_hash": placeholders_by_hash,
            "prepared_plantuml_sources_by_hash": prepared_plantuml_sources_by_hash,
            "pdf_mermaid_by_hash": worker_renderer.take_last_mermaid_pdf_svg_by_hash(),
        }
        return html_doc, int(stat.st_mtime_ns), int(stat.st_size), metadata

    def _preview_feature_flags_for_key(self, path_key: str | None) -> tuple[bool, bool, bool]:
        """Return cached (has_math, has_mermaid, has_plantuml) flags for one preview."""
        if not path_key:
            return (False, False, False)
        flags = self._preview_feature_flags_by_key.get(path_key)
        if not isinstance(flags, tuple) or len(flags) != 3:
            return (False, False, False)
        return bool(flags[0]), bool(flags[1]), bool(flags[2])

    def _has_diagram_features_for_key(self, path_key: str | None) -> bool:
        """Return whether a document can produce interactive diagram viewport state."""
        _has_math, has_mermaid, has_plantuml = self._preview_feature_flags_for_key(
            path_key
        )
        return bool(has_mermaid or has_plantuml)

    def _set_preview_feature_flags(
        self,
        path_key: str | None,
        *,
        has_math: bool,
        has_mermaid: bool,
        has_plantuml: bool,
    ) -> None:
        """Cache rich-render feature flags for one previewed markdown document."""
        if not path_key:
            return
        self._preview_feature_flags_by_key[path_key] = (
            bool(has_math),
            bool(has_mermaid),
            bool(has_plantuml),
        )

    def _has_persistent_preview_highlights_for_key(self, path_key: str | None) -> bool:
        """Return whether the active file has persisted preview text highlights."""
        if not path_key:
            return False
        return bool(self._normalize_text_highlight_entries(self._current_preview_text_highlights))

    def _has_named_view_markers_for_key(self, path_key: str | None) -> bool:
        """Return whether the active document currently exposes named-view markers."""
        current_key = self._current_preview_path_key()
        if not path_key or current_key != path_key:
            return False
        return bool(self._current_named_view_marker_entries())

    def _has_saved_diagram_view_state_for_key(self, path_key: str | None) -> bool:
        """Return whether a document currently has cached diagram zoom/pan state."""
        if not path_key:
            return False
        payload = self._diagram_view_state_by_doc.get(path_key, {})
        return isinstance(payload, dict) and bool(payload)

    def _has_ready_plantuml_for_key(self, path_key: str | None) -> bool:
        """Return whether the active document has any completed PlantUML jobs to patch in."""
        if not path_key:
            return False
        placeholders_by_hash = self._plantuml_placeholders_by_doc.get(path_key, {})
        if not isinstance(placeholders_by_hash, dict) or not placeholders_by_hash:
            return False
        for hash_key in placeholders_by_hash:
            status, _payload = self._plantuml_results.get(hash_key, ("pending", ""))
            if status in {"done", "error"}:
                return True
        return False

    def _serialized_diagram_view_state_json(self, path_key: str | None) -> str:
        """Serialize per-document diagram view state for HTML seed injection."""
        if not path_key:
            return "{}"
        payload = self._diagram_view_state_by_doc.get(path_key, {})
        if not isinstance(payload, dict):
            return "{}"
        try:
            return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            return "{}"

    def _inject_mermaid_cache_seed(
        self, html_doc: str, path_key: str | None = None
    ) -> str:
        """Inject runtime cache/state payloads into HTML template seed tokens."""
        resolved_key = (
            path_key if path_key is not None else self._current_preview_path_key()
        )
        token_literal = json.dumps(MERMAID_CACHE_JSON_TOKEN)
        cache_literal = json.dumps(self._serialized_mermaid_cache_json())
        out = html_doc
        if token_literal in out:
            out = out.replace(token_literal, cache_literal, 1)

        state_token_literal = json.dumps(DIAGRAM_VIEW_STATE_JSON_TOKEN)
        state_literal = json.dumps(
            self._serialized_diagram_view_state_json(resolved_key)
        )
        if state_token_literal in out:
            out = out.replace(state_token_literal, state_literal, 1)
        return out

    def _capture_current_diagram_view_state(
        self, expected_key: str | None = None
    ) -> None:
        """Snapshot in-page diagram zoom/pan state into per-document runtime cache."""
        key = expected_key or self._current_preview_path_key()
        if key is None:
            return
        if not self._has_diagram_features_for_key(key):
            return
        js = _render_js_asset("preview/collect_diagram_view_state.js")
        self.preview.page().runJavaScript(
            js,
            lambda result, path_key=key: self._on_diagram_view_state_snapshot(
                path_key, result
            ),
        )

    def _reapply_diagram_view_state_for(self, expected_key: str) -> None:
        """Push cached diagram zoom/pan state into the active page and reapply."""
        if self._current_preview_path_key() != expected_key:
            return
        payload = self._diagram_view_state_by_doc.get(expected_key, {})
        if not isinstance(payload, dict) or not payload:
            return
        payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        js = f"""
(() => {{
  const incoming = {payload_json};
  if (!incoming || typeof incoming !== "object") {{
    return 0;
  }}
  if (!window.__mdexploreDiagramViewState || typeof window.__mdexploreDiagramViewState !== "object") {{
    window.__mdexploreDiagramViewState = {{}};
  }}
  for (const [key, value] of Object.entries(incoming)) {{
    window.__mdexploreSetDiagramViewState(key, value || {{}});
  }}
  let applied = 0;
  for (const shell of Array.from(document.querySelectorAll(".mdexplore-mermaid-shell"))) {{
    const fn = shell && shell.__mdexploreReapplySavedState;
    if (typeof fn !== "function") {{
      continue;
    }}
    try {{
      fn();
      applied += 1;
    }} catch (_error) {{
      // Ignore per-shell restore failures.
    }}
  }}
  return applied;
}})();
"""
        self.preview.page().runJavaScript(js)

    def _capture_current_diagram_view_state_blocking(
        self, expected_key: str, timeout_ms: int = 300
    ) -> None:
        """Synchronously capture diagram zoom/pan state before preview navigation."""
        if not expected_key or self._current_preview_path_key() != expected_key:
            return
        if not self._has_diagram_features_for_key(expected_key):
            return
        js = _render_js_asset("preview/collect_diagram_view_state.js")
        loop = QEventLoop(self)
        completed = {"done": False}

        def on_result(result) -> None:
            if completed["done"]:
                return
            completed["done"] = True
            self._on_diagram_view_state_snapshot(expected_key, result)
            if loop.isRunning():
                loop.quit()

        self.preview.page().runJavaScript(js, on_result)

        timeout_timer = QTimer(self)
        timeout_timer.setSingleShot(True)
        timeout_timer.timeout.connect(loop.quit)
        timeout_timer.start(max(40, int(timeout_ms)))
        loop.exec()
        timeout_timer.stop()
        if not completed["done"]:
            # Fall back to an async capture if the blocking window times out.
            self._capture_current_diagram_view_state(expected_key)

    def _on_diagram_state_capture_tick(self) -> None:
        """Periodically mirror diagram zoom/pan state into Python memory."""
        path_key = self._current_preview_path_key()
        if path_key is None:
            return
        if not self._has_diagram_features_for_key(path_key):
            return
        self._capture_current_diagram_view_state(path_key)

    def _on_diagram_view_state_snapshot(self, path_key: str, result) -> None:
        """Merge diagram view state snapshot from JS into process cache."""
        if not isinstance(path_key, str) or not path_key:
            return
        if not isinstance(result, dict):
            return
        sanitized: dict[str, dict[str, float | bool]] = {}
        for raw_key, raw_state in result.items():
            if not isinstance(raw_key, str) or not isinstance(raw_state, dict):
                continue
            state_key = raw_key.strip()
            if not state_key or len(state_key) > 240:
                continue
            try:
                zoom = float(raw_state.get("zoom", 1.0))
            except Exception:
                zoom = 1.0
            try:
                scroll_left = float(raw_state.get("scrollLeft", 0.0))
            except Exception:
                scroll_left = 0.0
            try:
                scroll_top = float(raw_state.get("scrollTop", 0.0))
            except Exception:
                scroll_top = 0.0
            if not math.isfinite(zoom):
                zoom = 1.0
            if not math.isfinite(scroll_left):
                scroll_left = 0.0
            if not math.isfinite(scroll_top):
                scroll_top = 0.0
            sanitized[state_key] = {
                "zoom": max(0.2, min(8.0, zoom)),
                "scrollLeft": max(0.0, scroll_left),
                "scrollTop": max(0.0, scroll_top),
                "dirty": bool(raw_state.get("dirty")),
            }
        existing = self._diagram_view_state_by_doc.get(path_key, {})
        if not sanitized:
            # Ignore transient empty snapshots (for example during page
            # teardown/navigation) so a good saved zoom/pan state is not lost.
            if isinstance(existing, dict) and existing:
                return
            self._diagram_view_state_by_doc[path_key] = {}
            return
        merged: dict[str, dict[str, float | bool]] = {}
        if isinstance(existing, dict):
            for existing_key, existing_state in existing.items():
                if isinstance(existing_key, str) and isinstance(existing_state, dict):
                    merged[existing_key] = existing_state
        merged.update(sanitized)
        self._diagram_view_state_by_doc[path_key] = merged

    def _schedule_mermaid_cache_harvest_for(self, expected_key: str) -> None:
        """Collect Mermaid SVG cache snapshots after client rendering settles."""
        for delay_ms in (380, 980, 2100):
            QTimer.singleShot(
                delay_ms, lambda key=expected_key: self._harvest_mermaid_cache_for(key)
            )

    def _harvest_mermaid_cache_for(self, expected_key: str) -> None:
        """Fetch Mermaid SVG cache snapshot from active preview page."""
        if self._current_preview_path_key() != expected_key:
            return
        js = _render_js_asset("preview/harvest_mermaid_cache.js")
        self.preview.page().runJavaScript(
            js,
            lambda result, key=expected_key: self._on_mermaid_cache_snapshot(
                key, result
            ),
        )

    def _on_mermaid_cache_snapshot(self, expected_key: str, result) -> None:
        """Merge Mermaid SVG cache snapshot from JS into Python process cache."""
        if self._current_preview_path_key() != expected_key:
            return
        if not isinstance(result, dict):
            return
        for mode_name in ("auto", "pdf"):
            mode_payload = result.get(mode_name)
            if not isinstance(mode_payload, dict):
                continue
            target = self._mermaid_svg_cache_by_mode.setdefault(mode_name, {})
            for raw_hash, raw_svg in mode_payload.items():
                if not isinstance(raw_hash, str) or not isinstance(raw_svg, str):
                    continue
                hash_key = raw_hash.strip().lower()
                if not re.fullmatch(r"[0-9a-f]{40}", hash_key):
                    continue
                if "<svg" not in raw_svg.casefold():
                    continue
                if len(raw_svg) > MERMAID_SVG_MAX_CHARS:
                    continue
                target[hash_key] = raw_svg
            while len(target) > MERMAID_SVG_CACHE_MAX_ENTRIES:
                target.pop(next(iter(target)))

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
        normalized_tabs: list[dict[str, int | float | str]] = []
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
            normalized_entry: dict[str, int | float | str] = {
                "view_id": view_id,
                "sequence": sequence,
                "color_slot": color_slot,
            }
            custom_label = self._normalize_custom_view_label(entry.get("custom_label"))
            if custom_label is not None:
                try:
                    custom_label_anchor_scroll_y = float(
                        entry.get("custom_label_anchor_scroll_y", 0.0)
                    )
                except Exception:
                    custom_label_anchor_scroll_y = 0.0
                if not math.isfinite(custom_label_anchor_scroll_y):
                    custom_label_anchor_scroll_y = 0.0
                try:
                    custom_label_anchor_top_line = max(
                        1, int(entry.get("custom_label_anchor_top_line", 1))
                    )
                except Exception:
                    custom_label_anchor_top_line = 1
                normalized_entry["custom_label"] = custom_label
                normalized_entry["custom_label_anchor_scroll_y"] = (
                    custom_label_anchor_scroll_y
                )
                normalized_entry["custom_label_anchor_top_line"] = (
                    custom_label_anchor_top_line
                )
            normalized_tabs.append(normalized_entry)
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
            custom_label = self._normalize_custom_view_label(
                tab_entry.get("custom_label")
            )
            progress = self._line_progress(top_line, total_lines)
            index = self.view_tabs.addTab(
                self._display_label_for_view(top_line, custom_label)
            )
            self.view_tabs.setTabData(
                index,
                {
                    "view_id": view_id,
                    "sequence": tab_entry["sequence"],
                    "color_slot": tab_entry["color_slot"],
                    "progress": progress,
                    "custom_label": custom_label,
                    "custom_label_anchor_scroll_y": float(
                        tab_entry.get("custom_label_anchor_scroll_y", 0.0)
                    ),
                    "custom_label_anchor_top_line": max(
                        1, int(tab_entry.get("custom_label_anchor_top_line", top_line))
                    ),
                },
            )
            self.view_tabs.setTabToolTip(
                index, f"Top visible line: {top_line} / {total_lines}"
            )

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

        # Seed per-view scroll cache immediately so first-load restore can use
        # persisted positions before any tab switch occurs.
        for view_id, state in self._view_states.items():
            try:
                scroll_y = float(state.get("scroll_y", 0.0))
            except Exception:
                scroll_y = 0.0
            if not math.isfinite(scroll_y):
                scroll_y = 0.0
            self._preview_scroll_positions[f"{path_key}::view:{view_id}"] = max(
                0.0, scroll_y
            )

        self._sync_all_view_tab_progress()
        self._update_view_tabs_visibility()
        self._update_add_view_button_state()
        return True

    def _set_view_tab_line(self, view_id: int, line_number: int) -> None:
        """Update one tab label/tooltip to match its top visible line."""
        line_value = max(1, int(line_number))
        total_lines = max(1, int(self._current_document_total_lines))
        progress = self._line_progress(line_value, total_lines)
        for index in range(self.view_tabs.count()):
            if self._tab_view_id(index) != view_id:
                continue
            display_label = self._display_label_for_view(
                line_value, self._tab_custom_label(index)
            )
            if self.view_tabs.tabText(index) != display_label:
                self.view_tabs.setTabText(index, display_label)
                self.view_tabs.updateGeometry()
            data = self.view_tabs.tabData(index)
            if isinstance(data, dict):
                updated_data = dict(data)
                updated_data["progress"] = progress
                self.view_tabs.setTabData(index, updated_data)
            self.view_tabs.setTabToolTip(
                index, f"Top visible line: {line_value} / {total_lines}"
            )
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
            display_label = self._display_label_for_view(
                line_value, self._tab_custom_label(index)
            )
            if self.view_tabs.tabText(index) != display_label:
                self.view_tabs.setTabText(index, display_label)
            if isinstance(data, dict):
                updated_data = dict(data)
                updated_data["progress"] = progress
                self.view_tabs.setTabData(index, updated_data)
            self.view_tabs.setTabToolTip(
                index, f"Top visible line: {line_value} / {total_lines}"
            )
        self.view_tabs.updateGeometry()
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
        can_add = (
            self.current_file is not None
            and self.view_tabs.count() < self.MAX_DOCUMENT_VIEWS
        )
        self.add_view_btn.setEnabled(can_add)

    def _update_view_tabs_visibility(self) -> None:
        """Show tab strip for multi-view docs, or for a single custom-labeled view."""
        if not hasattr(self, "view_tabs"):
            return
        visible = False
        if self.current_file is not None:
            if self.view_tabs.count() > 1:
                visible = True
            elif self.view_tabs.count() == 1 and self._tab_custom_label(0) is not None:
                visible = True
        self.view_tabs.setVisible(visible)

    def _create_document_view(
        self, scroll_y: float, top_line: int, *, make_current: bool
    ) -> int:
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
            {
                "view_id": view_id,
                "sequence": sequence,
                "color_slot": color_slot,
                "progress": progress,
                "custom_label": None,
                "custom_label_anchor_scroll_y": 0.0,
                "custom_label_anchor_top_line": safe_line,
            },
        )
        self.view_tabs.setTabToolTip(
            tab_index, f"Top visible line: {safe_line} / {total_lines}"
        )

        if make_current:
            blocked = self.view_tabs.blockSignals(True)
            self.view_tabs.setCurrentIndex(tab_index)
            self.view_tabs.blockSignals(blocked)
            self._active_view_id = view_id

        return view_id

    def _reset_document_views(
        self, initial_scroll: float = 0.0, initial_line: int = 1
    ) -> None:
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
            self.statusBar().showMessage(
                "Open a markdown file before adding a view", 3000
            )
            return
        if self.view_tabs.count() >= self.MAX_DOCUMENT_VIEWS:
            self.statusBar().showMessage(
                f"Maximum of {self.MAX_DOCUMENT_VIEWS} views reached", 3500
            )
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
        path_key = self._current_preview_path_key()
        if path_key is not None:
            self._refresh_tree_multi_view_markers(changed_path_key=path_key)
        self.statusBar().showMessage(
            f"Added view {self.view_tabs.count()} of {self.MAX_DOCUMENT_VIEWS} at line {top_line}",
            3000,
        )

    def _on_view_tab_close_requested(self, tab_index: int) -> None:
        """Close one saved view tab while keeping at least one active view."""
        if self.view_tabs.count() <= 1:
            if self._tab_custom_label(tab_index) is not None:
                view_id = self._tab_view_id(tab_index)
                if view_id is None:
                    return
                state = self._view_states.get(view_id) or {
                    "scroll_y": 0.0,
                    "top_line": 1,
                }
                try:
                    top_line = max(1, int(state.get("top_line", 1)))
                except Exception:
                    top_line = 1
                data = self.view_tabs.tabData(tab_index)
                updated_data = (
                    dict(data) if isinstance(data, dict) else {"view_id": view_id}
                )
                updated_data["custom_label"] = None
                updated_data["custom_label_anchor_scroll_y"] = 0.0
                updated_data["custom_label_anchor_top_line"] = top_line
                self.view_tabs.setTabData(tab_index, updated_data)
                self.view_tabs.setTabText(
                    tab_index, self._display_label_for_view(top_line, None)
                )
                self.view_tabs.updateGeometry()
                self.view_tabs.update()
                self._update_view_tabs_visibility()
                self._persist_document_view_session()
                self._refresh_named_view_markers_in_preview()
                self.statusBar().showMessage(
                    "Closed labeled tab and returned to hidden default view", 3000
                )
                return
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
        if path_key is not None:
            self._refresh_tree_multi_view_markers(changed_path_key=path_key)
        self._refresh_named_view_markers_in_preview()

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

        expected_key = self._current_preview_path_key()
        if expected_key is None:
            return

        has_math, has_mermaid, has_plantuml = self._preview_feature_flags_for_key(
            expected_key
        )
        self._schedule_view_restore(
            expected_key,
            new_view_id,
            needs_settle_restore=bool(has_math or has_mermaid or has_plantuml),
        )
        self._update_add_view_button_state()

    def _restore_current_preview_scroll_if_active(
        self,
        expected_key: str,
        expected_view_id: int,
        restore_request_id: int,
        *,
        stabilize: bool,
    ) -> None:
        """Restore preview scroll only if tab switch request is still current."""
        if restore_request_id != self._view_tab_restore_request_id:
            return
        if self._current_preview_path_key() != expected_key:
            return
        if self._active_view_id != expected_view_id:
            return
        self._restore_current_preview_scroll(expected_key, stabilize=stabilize)

    def _enable_preview_scroll_capture_if_active(
        self, expected_key: str, expected_view_id: int, restore_request_id: int
    ) -> None:
        """Re-enable scroll capture only for the active tab switch request."""
        if restore_request_id != self._view_tab_restore_request_id:
            return
        if self._current_preview_path_key() != expected_key:
            return
        if self._active_view_id != expected_view_id:
            return
        self._enable_preview_scroll_capture_for(expected_key)

    def _on_preview_named_view_requested(self, view_id: int) -> None:
        """Activate a left-gutter view marker via the same restore path as tabs."""
        tab_index = self._tab_index_for_view_id(view_id)
        if tab_index is None:
            return
        if self.view_tabs.currentIndex() != tab_index:
            self.view_tabs.setCurrentIndex(tab_index)
            return

        expected_key = self._current_preview_path_key()
        if expected_key is None:
            return

        state = self._view_states.get(view_id)
        if isinstance(state, dict):
            scroll_key = self._current_preview_scroll_key()
            if scroll_key is not None:
                try:
                    scroll_y = float(state.get("scroll_y", 0.0))
                except Exception:
                    scroll_y = 0.0
                if math.isfinite(scroll_y):
                    self._preview_scroll_positions[scroll_key] = max(0.0, scroll_y)

        has_math, has_mermaid, has_plantuml = self._preview_feature_flags_for_key(
            expected_key
        )
        self._schedule_view_restore(
            expected_key,
            view_id,
            needs_settle_restore=bool(has_math or has_mermaid or has_plantuml),
        )

    def _on_preview_url_changed(self, url: QUrl) -> None:
        """Handle hash-based preview actions emitted from injected marker JS."""
        fragment = str(url.fragment() or "").strip()
        match = re.match(r"^mdexplore-view-(\d+)(?:-\d+)?$", fragment)
        if not match:
            return
        try:
            view_id = int(match.group(1))
        except Exception:
            return
        self.preview.page().runJavaScript(
            """
(() => {
  try {
    const cleanUrl = window.location.pathname + window.location.search;
    window.history.replaceState(null, "", cleanUrl);
  } catch (_err) {
    window.location.hash = "";
  }
})();
"""
        )
        self._on_preview_named_view_requested(view_id)

    def _poll_preview_actions(self) -> None:
        """Poll one-shot preview actions requested by injected marker scripts."""
        if self.current_file is None or self._preview_load_in_progress:
            return
        if self._preview_action_poll_inflight:
            return
        self._preview_action_poll_inflight = True
        self.preview.page().runJavaScript(
            """
(() => {
  const raw = Number(window.__mdexploreRequestedViewId || 0);
  if (!Number.isFinite(raw) || raw <= 0) {
    return 0;
  }
  window.__mdexploreRequestedViewId = 0;
  return Math.floor(raw);
})();
""",
            self._on_preview_action_poll_result,
        )

    def _on_preview_action_poll_result(self, result) -> None:
        """Handle one preview action request returned by the JS poll loop."""
        self._preview_action_poll_inflight = False
        try:
            view_id = int(result)
        except Exception:
            view_id = 0
        if view_id > 0:
            self._on_preview_named_view_requested(view_id)

    def _show_view_tab_context_menu(self, pos) -> None:
        """Offer custom-label editing for document view tabs."""
        tab_index = self.view_tabs.tabAt(pos)
        if tab_index < 0:
            return

        menu = QMenu(self)
        edit_action = menu.addAction("Edit Tab Label...")
        return_action = None
        if self._tab_label_anchor(tab_index) is not None:
            return_action = menu.addAction("Return to beginning")
        chosen = menu.exec(self.view_tabs.mapToGlobal(pos))
        if chosen != edit_action:
            if chosen == return_action:
                self._return_view_tab_to_beginning(tab_index)
            return
        self._edit_view_tab_label(tab_index)

    def _on_view_tab_home_requested(self, tab_index: int) -> None:
        """Jump a labeled tab back to its saved beginning when house icon is clicked."""
        if tab_index < 0 or tab_index >= self.view_tabs.count():
            return
        if self._tab_label_anchor(tab_index) is None:
            return
        if self.view_tabs.currentIndex() != tab_index:
            self.view_tabs.setCurrentIndex(tab_index)
        self._return_view_tab_to_beginning(tab_index)

    def _on_view_tab_beginning_reset_requested(self, tab_index: int) -> None:
        """Reset a labeled tab's saved beginning to its current scroll position."""
        if tab_index < 0 or tab_index >= self.view_tabs.count():
            return
        if self._tab_label_anchor(tab_index) is None:
            return
        if self.current_file is None:
            return

        if self.view_tabs.currentIndex() != tab_index:
            self.view_tabs.setCurrentIndex(tab_index)
            expected_key = self._current_preview_path_key()
            if expected_key is None:
                return
            # Tab switching restores scroll in staged timers; wait until that
            # settles before taking the "new beginning" snapshot.
            QTimer.singleShot(
                980,
                lambda idx=tab_index, key=expected_key: self._reset_view_tab_beginning_to_current(
                    idx, expected_key=key
                ),
            )
            return

        self._reset_view_tab_beginning_to_current(tab_index)

    def _reset_view_tab_beginning_to_current(
        self, tab_index: int, *, expected_key: str | None = None
    ) -> None:
        """Save the selected tab's current state as its new named beginning."""
        if tab_index < 0 or tab_index >= self.view_tabs.count():
            return
        if self._tab_label_anchor(tab_index) is None:
            return
        path_key = self._current_preview_path_key()
        if path_key is None:
            return
        if expected_key is not None and path_key != expected_key:
            return

        view_id = self._tab_view_id(tab_index)
        if view_id is None:
            return

        if self._active_view_id == view_id:
            self._capture_current_preview_scroll(force=True)

        state = self._view_states.get(view_id) or {"scroll_y": 0.0, "top_line": 1}
        try:
            current_scroll_y = float(state.get("scroll_y", 0.0))
        except Exception:
            current_scroll_y = 0.0
        if not math.isfinite(current_scroll_y):
            current_scroll_y = 0.0
        try:
            current_top_line = max(1, int(state.get("top_line", 1)))
        except Exception:
            current_top_line = 1

        data = self.view_tabs.tabData(tab_index)
        updated_data = dict(data) if isinstance(data, dict) else {"view_id": view_id}
        updated_data["custom_label_anchor_scroll_y"] = current_scroll_y
        updated_data["custom_label_anchor_top_line"] = current_top_line
        self.view_tabs.setTabData(tab_index, updated_data)
        self._persist_document_view_session(path_key, capture_current=False)
        self._refresh_named_view_markers_in_preview()
        self.statusBar().showMessage(
            f"Reset tab beginning to line {current_top_line}", 2800
        )

    def _edit_view_tab_label(self, tab_index: int) -> None:
        """Prompt for a custom tab label; blank restores the dynamic line number."""
        if tab_index < 0 or tab_index >= self.view_tabs.count():
            return

        current_custom = self._tab_custom_label(tab_index) or ""
        label_text, accepted = QInputDialog.getText(
            self,
            "Edit Tab Label",
            "Enter a custom tab label (blank restores the line number):",
            text=current_custom,
        )
        if not accepted:
            return
        was_truncated = len(label_text) > ViewTabBar.MAX_LABEL_CHARS
        custom_label = self._normalize_custom_view_label(label_text)
        view_id = self._tab_view_id(tab_index)
        if view_id is None:
            return

        data = self.view_tabs.tabData(tab_index)
        updated_data = dict(data) if isinstance(data, dict) else {"view_id": view_id}
        previous_custom_label = self._normalize_custom_view_label(
            updated_data.get("custom_label")
        )
        anchor_scroll_y, anchor_top_line = self._tab_label_anchor(tab_index) or (0.0, 1)
        state = self._view_states.get(view_id) or {"scroll_y": 0.0, "top_line": 1}
        try:
            current_scroll_y = float(state.get("scroll_y", 0.0))
        except Exception:
            current_scroll_y = 0.0
        if not math.isfinite(current_scroll_y):
            current_scroll_y = 0.0
        try:
            current_top_line = max(1, int(state.get("top_line", 1)))
        except Exception:
            current_top_line = 1
        if custom_label is None:
            anchor_scroll_y = 0.0
            anchor_top_line = current_top_line
        elif previous_custom_label != custom_label:
            anchor_scroll_y = current_scroll_y
            anchor_top_line = current_top_line
        updated_data["custom_label"] = custom_label
        updated_data["custom_label_anchor_scroll_y"] = anchor_scroll_y
        updated_data["custom_label_anchor_top_line"] = anchor_top_line
        self.view_tabs.setTabData(tab_index, updated_data)

        try:
            top_line = max(1, int(state.get("top_line", 1)))
        except Exception:
            top_line = 1
        self.view_tabs.setTabText(
            tab_index, self._display_label_for_view(top_line, custom_label)
        )
        self.view_tabs.updateGeometry()
        self.view_tabs.update()
        self._persist_document_view_session()
        self._refresh_named_view_markers_in_preview()
        if custom_label is None:
            self.statusBar().showMessage("Restored dynamic line-number tab label", 2500)
        elif was_truncated:
            self.statusBar().showMessage(
                f"Tab label updated and truncated to {ViewTabBar.MAX_LABEL_CHARS} characters",
                3000,
            )
        else:
            self.statusBar().showMessage(f"Tab label updated to '{custom_label}'", 2500)

    def _return_view_tab_to_beginning(self, tab_index: int) -> None:
        """Restore a custom-labeled view to the scroll position captured when labeled."""
        anchor = self._tab_label_anchor(tab_index)
        view_id = self._tab_view_id(tab_index)
        path_key = self._current_preview_path_key()
        if anchor is None or view_id is None or path_key is None:
            return

        anchor_scroll_y, anchor_top_line = anchor
        state = self._view_states.setdefault(view_id, {"scroll_y": 0.0, "top_line": 1})
        state["scroll_y"] = anchor_scroll_y
        state["top_line"] = anchor_top_line
        self._preview_scroll_positions[f"{path_key}::view:{view_id}"] = anchor_scroll_y
        self._set_view_tab_line(view_id, anchor_top_line)
        self._persist_document_view_session(path_key, capture_current=False)

        if self._active_view_id != view_id:
            self.view_tabs.setCurrentIndex(tab_index)
        expected_key = self._current_preview_path_key()
        if expected_key is not None:
            has_math, has_mermaid, has_plantuml = self._preview_feature_flags_for_key(
                expected_key
            )
            self._schedule_view_restore(
                expected_key,
                view_id,
                needs_settle_restore=bool(has_math or has_mermaid or has_plantuml),
            )
        self.statusBar().showMessage(
            f"Returned tab to labeled beginning at line {anchor_top_line}", 3000
        )

    def _request_active_view_top_line_update(self, force: bool = False) -> None:
        """Probe top-most visible source line and update active tab label."""
        if self.current_file is None or self._active_view_id is None:
            return
        now = time.monotonic()
        if now < self._view_line_probe_block_until:
            return
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
        js = _render_js_asset("preview/probe_active_top_line.js")
        self.preview.page().runJavaScript(
            js,
            lambda result, key=expected_key, view_id=expected_view_id: self._on_active_view_line_probe_result(
                key,
                view_id,
                result,
            ),
        )

    def _on_active_view_line_probe_result(
        self, expected_key: str, expected_view_id: int, result
    ) -> None:
        """Apply top-line probe result to active view tab when still current."""
        self._view_line_probe_pending = False
        if time.monotonic() < self._view_line_probe_block_until:
            return
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

    @staticmethod
    def _html_with_base_href(html_doc: str, base_url: QUrl) -> str:
        """Inject `<base href=...>` so temp-file preview keeps relative links."""
        if not html_doc:
            return html_doc
        href = html.escape(base_url.toString(), quote=True)
        if not href:
            return html_doc
        closing_head = re.search(r"</head\s*>", html_doc, flags=re.IGNORECASE)
        if closing_head:
            head_content = html_doc[: closing_head.start()]
            if re.search(r"<base\s+[^>]*href=", head_content, flags=re.IGNORECASE):
                return html_doc
        open_head = re.search(r"<head\b[^>]*>", html_doc, flags=re.IGNORECASE)
        if not open_head:
            return html_doc
        base_tag = f'\n  <base href="{href}"/>'
        insert_at = open_head.end()
        return html_doc[:insert_at] + base_tag + html_doc[insert_at:]

    def _cleanup_preview_temp_files(self) -> None:
        """Delete temporary preview files used for oversized HTML payloads."""
        while self._preview_html_temp_files:
            path = self._preview_html_temp_files.popleft()
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    def _track_preview_temp_file(self, temp_path: Path) -> None:
        """Track temp preview files and clean up old entries eagerly."""
        self._preview_html_temp_files.append(temp_path)
        while len(self._preview_html_temp_files) > 6:
            stale = self._preview_html_temp_files.popleft()
            try:
                stale.unlink(missing_ok=True)
            except Exception:
                pass

    def _set_preview_html(self, html_doc: str, base_url: QUrl) -> None:
        """Load preview HTML with file-based fallback for large documents."""
        # Any fresh page load drops in-page marker state, so force the next
        # named-view marker push to repopulate the overlay after loadFinished.
        self._last_named_view_marker_payload_key = None
        self._last_named_view_marker_payload_json = None
        try:
            encoded_size = len(html_doc.encode("utf-8", errors="replace"))
        except Exception:
            encoded_size = len(html_doc)
        if encoded_size <= PREVIEW_SETHTML_MAX_BYTES:
            self.preview.setHtml(html_doc, base_url)
            return
        try:
            prepared = self._html_with_base_href(html_doc, base_url)
            temp_dir = Path(tempfile.gettempdir()) / "mdexplore-preview"
            temp_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".html",
                prefix="preview-",
                dir=temp_dir,
                delete=False,
            ) as temp_file:
                temp_file.write(prepared)
                temp_path = Path(temp_file.name)
            self._track_preview_temp_file(temp_path)
            self.preview.load(QUrl.fromLocalFile(str(temp_path)))
            return
        except Exception:
            # If temp-file load setup fails, fall back to direct setHtml.
            self.preview.setHtml(html_doc, base_url)

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

    def _has_preview_render_inflight_for_key(self, path_key: str | None) -> bool:
        """Return whether the current preview request is still rendering in background."""
        if not path_key:
            return False
        active_request_id = int(self._render_request_id)
        for worker in tuple(self._active_render_workers):
            if int(getattr(worker, "request_id", -1)) != active_request_id:
                continue
            worker_path = getattr(worker, "path", None)
            if not isinstance(worker_path, Path):
                continue
            try:
                worker_key = str(worker_path.resolve())
            except Exception:
                worker_key = str(worker_path)
            if worker_key == path_key:
                return True
        return False

    def _preview_has_pending_plantuml_for_key(self, path_key: str | None) -> bool:
        """Return whether any PlantUML placeholder for one document is still pending."""
        if not path_key:
            return False
        placeholders = self._plantuml_placeholders_by_doc.get(path_key, {})
        if not isinstance(placeholders, dict) or not placeholders:
            return False
        for hash_key in placeholders:
            status, _payload = self._plantuml_results.get(hash_key, ("pending", ""))
            if status not in {"done", "error"}:
                return True
        return False

    def _stop_pdf_diagram_readiness_monitor(self) -> None:
        """Stop periodic JS readiness probing for PDF button gating."""
        self._pdf_diagram_probe_expected_key = None
        self._pdf_diagram_probe_inflight = False
        self._pdf_diagram_probe_started_at = 0.0
        if hasattr(self, "_pdf_diagram_probe_timer"):
            self._pdf_diagram_probe_timer.stop()

    def _start_pdf_diagram_readiness_monitor(self, expected_key: str) -> None:
        """Start periodic JS readiness probing for one active preview key."""
        if not expected_key:
            return
        if self._pdf_diagram_probe_expected_key != expected_key:
            self._pdf_diagram_probe_expected_key = expected_key
            self._pdf_diagram_probe_inflight = False
            self._pdf_diagram_probe_started_at = 0.0
        if not self._pdf_diagram_probe_timer.isActive():
            self._pdf_diagram_probe_timer.start()

    def _probe_pdf_diagram_readiness(self) -> None:
        """Probe in-page diagram readiness so PDF button reflects true export readiness."""
        expected_key = self._pdf_diagram_probe_expected_key
        if not expected_key:
            self._stop_pdf_diagram_readiness_monitor()
            return
        if self._current_preview_path_key() != expected_key:
            self._stop_pdf_diagram_readiness_monitor()
            return
        if self._preview_load_in_progress or self._pdf_export_in_progress:
            return

        _has_math, has_mermaid, has_plantuml = self._preview_feature_flags_for_key(
            expected_key
        )
        if not (has_mermaid or has_plantuml):
            self._pdf_diagram_ready_by_key[expected_key] = True
            self._stop_pdf_diagram_readiness_monitor()
            self._update_pdf_button_state()
            return
        if self._preview_has_pending_plantuml_for_key(expected_key):
            self._pdf_diagram_ready_by_key[expected_key] = False
            return
        if has_plantuml and not has_mermaid:
            self._pdf_diagram_ready_by_key[expected_key] = True
            self._stop_pdf_diagram_readiness_monitor()
            self._update_pdf_button_state()
            return

        if self._pdf_diagram_probe_inflight:
            if (time.monotonic() - self._pdf_diagram_probe_started_at) < 1.8:
                return
            self._pdf_diagram_probe_inflight = False
            self._pdf_diagram_probe_started_at = 0.0

        self._pdf_diagram_probe_inflight = True
        self._pdf_diagram_probe_started_at = time.monotonic()
        js = _render_js_asset("preview/probe_pdf_diagram_readiness.js")
        self.preview.page().runJavaScript(
            js,
            lambda result, key=expected_key: self._on_pdf_diagram_readiness_probe_result(
                key, result
            ),
        )

    def _on_pdf_diagram_readiness_probe_result(self, expected_key: str, result) -> None:
        """Handle JS readiness probe response for diagram-aware PDF button gating."""
        self._pdf_diagram_probe_inflight = False
        self._pdf_diagram_probe_started_at = 0.0
        if self._pdf_diagram_probe_expected_key != expected_key:
            return
        if self._current_preview_path_key() != expected_key:
            self._stop_pdf_diagram_readiness_monitor()
            self._update_pdf_button_state()
            return

        parsed_result = result
        if isinstance(result, str):
            result_text = result.strip()
            if result_text.startswith("{") and result_text.endswith("}"):
                try:
                    parsed_result = json.loads(result_text)
                except Exception:
                    parsed_result = result

        _has_math, has_mermaid, has_plantuml = self._preview_feature_flags_for_key(
            expected_key
        )
        parsed_is_dict = isinstance(parsed_result, dict)
        settle_deadline = float(
            self._pdf_diagram_settle_deadline_by_key.get(expected_key, 0.0)
        )
        settle_window_elapsed = settle_deadline <= 0.0 or (
            time.monotonic() >= settle_deadline
        )
        mermaid_ready = True
        plantuml_dom_ready = True
        if has_mermaid:
            if parsed_is_dict:
                mermaid_ready = bool(parsed_result.get("mermaidReady"))
            else:
                mermaid_ready = settle_window_elapsed
        if has_plantuml:
            if parsed_is_dict:
                plantuml_dom_ready = bool(parsed_result.get("plantumlDomReady"))
            else:
                plantuml_dom_ready = True
        if self._preview_has_pending_plantuml_for_key(expected_key):
            plantuml_dom_ready = False

        diagrams_ready = mermaid_ready and plantuml_dom_ready
        self._pdf_diagram_ready_by_key[expected_key] = diagrams_ready
        if diagrams_ready:
            self._stop_pdf_diagram_readiness_monitor()
        self._update_pdf_button_state()

    def _update_pdf_button_state(self) -> None:
        """Enable PDF action only when export can run against a stable preview."""
        if not hasattr(self, "pdf_btn"):
            return

        default_tooltip = "Export the currently previewed markdown rendering to PDF"
        enabled = not self._pdf_export_in_progress
        tooltip = default_tooltip
        current_key = self._current_preview_path_key()

        if self._pdf_export_in_progress:
            tooltip = "PDF export unavailable while another PDF export is in progress"
        elif current_key is not None:
            _has_math, has_mermaid, has_plantuml = self._preview_feature_flags_for_key(
                current_key
            )
            has_diagrams = bool(has_mermaid or has_plantuml)
            if self._preview_load_in_progress:
                enabled = False
                tooltip = "PDF export unavailable while preview content is loading"
                if has_diagrams:
                    self._pdf_diagram_ready_by_key[current_key] = False
            elif self._has_preview_render_inflight_for_key(current_key):
                enabled = False
                tooltip = "PDF export unavailable while preview rendering is still in progress"
                if has_diagrams:
                    self._pdf_diagram_ready_by_key[current_key] = False
            elif has_diagrams:
                has_pending_plantuml = self._preview_has_pending_plantuml_for_key(
                    current_key
                )
                if has_pending_plantuml:
                    self._pdf_diagram_ready_by_key[current_key] = False
                diagrams_ready = bool(self._pdf_diagram_ready_by_key.get(current_key, False))
                if not diagrams_ready:
                    enabled = False
                    tooltip = (
                        "PDF export unavailable while diagrams are rendering or "
                        "cached diagram output is being restored"
                    )
                    self._start_pdf_diagram_readiness_monitor(current_key)
                elif self._pdf_diagram_probe_expected_key == current_key:
                    self._stop_pdf_diagram_readiness_monitor()
            else:
                self._pdf_diagram_ready_by_key[current_key] = True
                if self._pdf_diagram_probe_expected_key == current_key:
                    self._stop_pdf_diagram_readiness_monitor()
        elif self._pdf_diagram_probe_expected_key is not None:
            self._stop_pdf_diagram_readiness_monitor()

        self.pdf_btn.setEnabled(enabled)
        self.pdf_btn.setToolTip(tooltip)

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
            self.statusBar().showMessage(
                f"Preview ready: {self.current_file.name}", 3500
            )
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
        self._stop_restore_overlay_monitor()
        self._stop_pdf_diagram_readiness_monitor()
        self._capture_current_preview_scroll(force=True)
        self._persist_document_view_session()
        self._capture_splitter_sizes_for_session()
        self.root = new_root.resolve()
        self.statusBar().showMessage(f"Root changed to {self.root}", 3000)
        self.last_directory_selection = self.root
        self.current_file = None
        self._preview_load_in_progress = False
        self._pdf_diagram_settle_deadline_by_key.clear()
        self._current_document_total_lines = 1
        self._reset_document_views()
        self._clear_current_preview_signature()
        self._preview_capture_enabled = False
        self._scroll_restore_block_until = 0.0
        self._pending_preview_search_terms = []
        self._pending_preview_search_close_groups = []
        self._current_preview_text_highlights = []
        root_index = self.model.setRootPath(str(self.root))
        self.tree.setRootIndex(root_index)
        self.tree.clearSelection()
        self.path_label.setText("Select a markdown file")
        self._set_preview_html(
            self._placeholder_html("Select a markdown file to preview"),
            QUrl.fromLocalFile(f"{self.root}/"),
        )
        self._initial_split_applied = False
        self._update_up_button_state()
        self._update_window_title()
        self._cancel_pending_preview_render()
        self._rerun_active_search_for_scope()
        self._update_add_view_button_state()
        self._update_pdf_button_state()
        self._refresh_tree_multi_view_markers(full_scan=True)
        QTimer.singleShot(0, self._maybe_apply_initial_split)

    def _on_preview_load_finished(self, ok: bool) -> None:
        """Apply deferred in-preview highlighting after a page finishes loading."""
        self._preview_load_in_progress = False
        if not ok:
            self._stop_restore_overlay_monitor()
            self._stop_pdf_diagram_readiness_monitor()
            self.statusBar().showMessage("Preview load failed", 5000)
            self._update_pdf_button_state()
            return
        current_key = self._current_preview_path_key()
        if current_key is None:
            self._stop_restore_overlay_monitor()
            self._stop_pdf_diagram_readiness_monitor()
            self._update_pdf_button_state()
            return
        has_math, has_mermaid, has_plantuml = self._preview_feature_flags_for_key(
            current_key
        )
        has_client_renderers = has_math or has_mermaid
        has_ready_plantuml = self._has_ready_plantuml_for_key(current_key)
        has_text_highlights = self._has_persistent_preview_highlights_for_key(
            current_key
        )
        has_named_view_markers = self._has_named_view_markers_for_key(current_key)
        has_saved_diagram_state = self._has_saved_diagram_view_state_for_key(
            current_key
        )
        # Kick client-side renderer startup now and a bit later to tolerate
        # delayed external script availability (MathJax/Mermaid).
        if has_client_renderers:
            self._trigger_client_renderers_for(current_key)
            QTimer.singleShot(
                450, lambda key=current_key: self._trigger_client_renderers_for(key)
            )
            QTimer.singleShot(
                1500, lambda key=current_key: self._trigger_client_renderers_for(key)
            )
        # PlantUML completions are patched in-place, but a full page load can
        # still happen from cache refreshes; re-apply any ready results.
        if has_ready_plantuml:
            self._apply_all_ready_plantuml_to_current_preview()
        if has_text_highlights:
            self._apply_persistent_preview_highlights(current_key)
        if has_named_view_markers:
            self._refresh_named_view_markers_in_preview(current_key, force=True)
        if has_mermaid:
            self._schedule_mermaid_cache_harvest_for(current_key)
        if has_saved_diagram_state:
            self._reapply_diagram_view_state_for(current_key)
            QTimer.singleShot(
                120, lambda key=current_key: self._reapply_diagram_view_state_for(key)
            )
            QTimer.singleShot(
                420, lambda key=current_key: self._reapply_diagram_view_state_for(key)
            )
            QTimer.singleShot(
                980, lambda key=current_key: self._reapply_diagram_view_state_for(key)
            )
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
            self._begin_preview_restore_block(1.2)
            QTimer.singleShot(
                90, lambda key=current_key: self._restore_current_preview_scroll(key)
            )
            QTimer.singleShot(
                320, lambda key=current_key: self._restore_current_preview_scroll(key)
            )
            QTimer.singleShot(
                900, lambda key=current_key: self._restore_current_preview_scroll(key)
            )
            QTimer.singleShot(
                1250,
                lambda key=current_key: self._enable_preview_scroll_capture_for(key),
            )
        else:
            self._preview_capture_enabled = True
            self._scroll_restore_block_until = 0.0
            self._view_line_probe_block_until = 0.0
        self._request_active_view_top_line_update(force=True)
        self._show_preview_progress_status()
        self._check_restore_overlay_progress()
        has_diagrams = bool(has_mermaid or has_plantuml)
        if has_diagrams:
            settle_deadline = 0.0
            if has_mermaid:
                # Cached Mermaid restore and post-load renderer retries are
                # intentionally staggered; keep PDF disabled during this window.
                settle_deadline = time.monotonic() + 2.8
            self._pdf_diagram_settle_deadline_by_key[current_key] = settle_deadline
            self._pdf_diagram_ready_by_key[current_key] = False
            self._start_pdf_diagram_readiness_monitor(current_key)
        else:
            self._pdf_diagram_settle_deadline_by_key[current_key] = 0.0
            self._pdf_diagram_ready_by_key[current_key] = True
            if self._pdf_diagram_probe_expected_key == current_key:
                self._stop_pdf_diagram_readiness_monitor()
        self._update_pdf_button_state()

    def _trigger_client_renderers_for(self, expected_key: str) -> None:
        """Run in-page renderer helpers only if the same preview is still active."""
        if self._current_preview_path_key() != expected_key:
            return
        js = _render_js_asset("preview/trigger_client_renderers.js")
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
        self._stop_restore_overlay_monitor()
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
        self._refresh_tree_multi_view_markers(full_scan=True)

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
                self._stop_restore_overlay_monitor()
                self._reset_document_views()
                self._clear_current_preview_signature()
                self.path_label.setText("Select a markdown file")
                self._set_preview_html(
                    self._placeholder_html("Select a markdown file to preview"),
                    QUrl.fromLocalFile(f"{self.root}/"),
                )
                if not restored_selection:
                    self.tree.clearSelection()
                self.statusBar().showMessage(
                    "Directory view refreshed; preview file no longer exists", 4500
                )
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
        self._position_preview_zoom_overlay()

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

    def _cancel_pending_tree_marker_scan(self) -> None:
        """Drop queued tree-sidecar scans and invalidate stale worker results."""
        self._tree_marker_scan_request_id += 1
        self._tree_marker_scan_pool.clear()
        self._tree_marker_scan_dirty_paths.clear()

    def _merge_live_tree_marker_state(
        self,
        multi_view_paths: set[str] | None = None,
        highlighted_paths: set[str] | None = None,
        *,
        root_key: str | None = None,
    ) -> tuple[set[str], set[str]]:
        """Merge worker/disk marker sets with current in-memory document state."""
        merged_multi_view = set(multi_view_paths or ())
        merged_highlighted = set(highlighted_paths or ())
        if root_key is None:
            root = getattr(self, "root", None)
            if isinstance(root, Path):
                root_key = self._path_key(root)

        for raw_path_key, session in self._document_view_sessions.items():
            if self._session_has_multiple_views(session) and self._is_path_key_under_root(
                raw_path_key, root_key
            ):
                merged_multi_view.add(raw_path_key)

        current_path_key = self._current_preview_path_key()
        if (
            current_path_key
            and self.view_tabs.count() > 1
            and self._is_path_key_under_root(current_path_key, root_key)
        ):
            merged_multi_view.add(current_path_key)

        return merged_multi_view, merged_highlighted

    def _start_tree_marker_scan(self) -> None:
        """Scan the current root for tree badge sidecars in the background."""
        root = getattr(self, "root", None)
        if not isinstance(root, Path) or not root.exists():
            self._tree_multi_view_marker_paths.clear()
            self._tree_highlight_marker_paths.clear()
            self._tree_marker_cache_root_key = None
            self._sync_tree_multi_view_markers_to_model()
            return

        self._cancel_pending_tree_marker_scan()
        request_id = self._tree_marker_scan_request_id
        self._tree_marker_scan_dirty_paths.clear()
        worker = TreeMarkerScanWorker(
            root,
            request_id,
            self.VIEWS_FILE_NAME,
            self.HIGHLIGHTING_FILE_NAME,
        )
        self._active_tree_marker_scan_workers.add(worker)
        worker.signals.finished.connect(self._on_tree_marker_scan_finished)
        self._tree_marker_scan_pool.start(worker)

    def _on_tree_marker_scan_finished(
        self,
        request_id: int,
        root_key: str,
        multi_view_paths,
        highlighted_paths,
        error_text: str,
    ) -> None:
        """Apply finished tree-sidecar scan if it still matches the current root."""
        worker_to_remove = None
        for worker in self._active_tree_marker_scan_workers:
            if worker.request_id == request_id:
                worker_to_remove = worker
                break
        if worker_to_remove is not None:
            self._active_tree_marker_scan_workers.remove(worker_to_remove)

        if request_id != self._tree_marker_scan_request_id:
            return

        current_root = getattr(self, "root", None)
        if not isinstance(current_root, Path):
            return
        current_root_key = self._path_key(current_root)
        if root_key != current_root_key:
            return

        if error_text:
            self.statusBar().showMessage(
                f"Tree badge scan failed: {self._truncate_error_text(error_text, 220)}",
                4000,
            )
            return

        next_multi_view_paths = {
            str(path_key)
            for path_key in (multi_view_paths or ())
            if isinstance(path_key, str)
        }
        next_highlighted_paths = {
            str(path_key)
            for path_key in (highlighted_paths or ())
            if isinstance(path_key, str)
        }
        next_multi_view_paths, next_highlighted_paths = self._merge_live_tree_marker_state(
            next_multi_view_paths,
            next_highlighted_paths,
            root_key=current_root_key,
        )
        self._tree_multi_view_marker_paths = next_multi_view_paths
        self._tree_highlight_marker_paths = next_highlighted_paths
        self._tree_marker_cache_root_key = current_root_key
        dirty_paths = [
            path_key
            for path_key in self._tree_marker_scan_dirty_paths
            if self._is_path_key_under_root(path_key, current_root_key)
        ]
        self._tree_marker_scan_dirty_paths.clear()
        for path_key in dirty_paths:
            self._update_tree_multi_view_markers_for_path_key(
                path_key, root_key=current_root_key
            )
        self._sync_tree_multi_view_markers_to_model()

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
        hit_counter = self._compile_match_hit_counter(query)
        candidates = self._list_markdown_files_non_recursive(scope)
        self.statusBar().showMessage(
            f"Searching {len(candidates)} markdown file(s) in {scope}...",
        )
        matches: list[Path] = []
        match_counts: dict[Path, int] = {}

        for path in candidates:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                content = ""
            # Search over file name + body to support quick discovery.
            if predicate(path.name, content):
                matches.append(path)
                count = hit_counter(path.name, content)
                match_counts[path] = count if count > 0 else 1

        self.current_match_files = matches
        self.model.set_search_match_counts(match_counts)
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

    @staticmethod
    def _session_has_multiple_views(session: dict | None) -> bool:
        """Return whether a persisted/in-memory session represents more than one view."""
        if not isinstance(session, dict):
            return False
        tabs = session.get("tabs")
        return isinstance(tabs, list) and len(tabs) > 1

    def _is_path_key_under_root(
        self, path_key: str | None, root_key: str | None = None
    ) -> bool:
        """Return whether a file path key belongs to the current tree root."""
        if not path_key:
            return False
        if root_key is None:
            root = getattr(self, "root", None)
            if not isinstance(root, Path):
                return False
            root_key = self._path_key(root)
        return path_key == root_key or path_key.startswith(root_key + os.sep)

    def _sync_tree_multi_view_markers_to_model(self) -> None:
        """Push cached tree badge sets to the model and repaint only on change."""
        if not hasattr(self, "model"):
            return
        should_update = (
            self._tree_multi_view_marker_paths != self.model._multi_view_paths
            or self._tree_highlight_marker_paths
            != self.model._highlighted_preview_paths
        )
        self.model.set_multi_view_path_keys(self._tree_multi_view_marker_paths)
        self.model.set_persistent_highlight_path_keys(
            self._tree_highlight_marker_paths
        )
        if should_update and hasattr(self, "tree"):
            self.tree.viewport().update()

    def _update_tree_multi_view_markers_for_path_key(
        self, path_key: str | None, *, root_key: str | None = None
    ) -> None:
        """Incrementally refresh tree badges for one markdown file path."""
        if not path_key:
            return
        if root_key is None:
            root = getattr(self, "root", None)
            if not isinstance(root, Path):
                return
            root_key = self._path_key(root)
        if not self._is_path_key_under_root(path_key, root_key):
            self._tree_multi_view_marker_paths.discard(path_key)
            self._tree_highlight_marker_paths.discard(path_key)
            return

        has_multi_view = self._session_has_multiple_views(
            self._document_view_sessions.get(path_key)
        )
        current_path_key = self._current_preview_path_key()
        if (
            not has_multi_view
            and current_path_key == path_key
            and self.view_tabs.count() > 1
        ):
            has_multi_view = True

        if has_multi_view:
            self._tree_multi_view_marker_paths.add(path_key)
        else:
            self._tree_multi_view_marker_paths.discard(path_key)

        if self._load_text_highlights_for_path_key(path_key):
            self._tree_highlight_marker_paths.add(path_key)
        else:
            self._tree_highlight_marker_paths.discard(path_key)

    def _rebuild_tree_multi_view_marker_cache(self) -> None:
        """Rebuild root-scoped tree badges with one recursive scan per root."""
        root = getattr(self, "root", None)
        if not isinstance(root, Path) or not root.exists():
            self._tree_multi_view_marker_paths.clear()
            self._tree_highlight_marker_paths.clear()
            self._tree_marker_cache_root_key = None
            self._sync_tree_multi_view_markers_to_model()
            return

        root_key = self._path_key(root)
        marked_paths: set[str] = set()
        highlighted_paths: set[str] = set()

        for raw_path_key, session in self._document_view_sessions.items():
            if self._session_has_multiple_views(session) and self._is_path_key_under_root(
                raw_path_key, root_key
            ):
                marked_paths.add(raw_path_key)

        current_path_key = self._current_preview_path_key()
        if (
            current_path_key
            and self.view_tabs.count() > 1
            and self._is_path_key_under_root(current_path_key, root_key)
        ):
            marked_paths.add(current_path_key)

        def on_walk_error(_err) -> None:
            return

        for dirpath, _dirnames, filenames in os.walk(
            root, onerror=on_walk_error, followlinks=False
        ):
            directory = Path(dirpath)
            if self.VIEWS_FILE_NAME in filenames:
                sessions = self._directory_view_sessions(directory)
                for file_name, session in sessions.items():
                    if self._session_has_multiple_views(session):
                        marked_paths.add(self._path_key(directory / file_name))

            if self.HIGHLIGHTING_FILE_NAME in filenames:
                highlights_by_file = self._directory_text_highlights(directory)
                for file_name, entries in highlights_by_file.items():
                    if self._normalize_text_highlight_entries(entries):
                        highlighted_paths.add(self._path_key(directory / file_name))

        self._tree_multi_view_marker_paths = marked_paths
        self._tree_highlight_marker_paths = highlighted_paths
        self._tree_marker_cache_root_key = root_key
        self._sync_tree_multi_view_markers_to_model()

    def _refresh_tree_multi_view_markers(
        self,
        *,
        full_scan: bool = False,
        changed_path_key: str | None = None,
    ) -> None:
        """Update tree badges cheaply and reserve recursive scans for root refreshes."""
        root = getattr(self, "root", None)
        if not isinstance(root, Path) or not root.exists():
            self._rebuild_tree_multi_view_marker_cache()
            return

        root_key = self._path_key(root)
        if full_scan or self._tree_marker_cache_root_key != root_key:
            self._start_tree_marker_scan()
            return

        if changed_path_key:
            if self._active_tree_marker_scan_workers:
                self._tree_marker_scan_dirty_paths.add(changed_path_key)
            self._update_tree_multi_view_markers_for_path_key(
                changed_path_key, root_key=root_key
            )
            self._sync_tree_multi_view_markers_to_model()

    def _current_search_terms(self) -> list[tuple[str, bool]]:
        """Extract searchable terms with case mode from the current query."""
        query = self.match_input.text().strip()
        if not query:
            return []
        return _search_query.extract_search_terms(query)

    def _current_near_term_groups(self) -> list[list[tuple[str, bool]]]:
        """Extract NEAR(...) argument groups from the current query."""
        query = self.match_input.text().strip()
        if not query:
            return []

        return self._extract_near_term_groups(query)

    def _current_close_term_groups(self) -> list[list[tuple[str, bool]]]:
        """Backward-compatible alias for NEAR(...) term-group extraction."""
        return self._current_near_term_groups()

    def _extract_near_term_groups(self, query: str) -> list[list[tuple[str, bool]]]:
        """Extract NEAR(...) argument groups from an arbitrary search query."""
        return _search_query.extract_near_term_groups(query)

    def _extract_close_term_groups(self, query: str) -> list[list[tuple[str, bool]]]:
        """Backward-compatible alias for NEAR(...) term-group extraction."""
        return self._extract_near_term_groups(query)

    def _collect_near_focus_windows(
        self, content: str, groups: list[list[tuple[str, bool]]]
    ) -> list[dict[str, object]]:
        """Return qualifying non-overlapping NEAR() windows in document order."""
        return _search_query.collect_near_focus_windows(content, groups)

    def _collect_close_focus_windows(
        self, content: str, groups: list[list[tuple[str, bool]]]
    ) -> list[dict[str, object]]:
        """Backward-compatible alias for NEAR() focus-window collection."""
        return self._collect_near_focus_windows(content, groups)

    def _best_near_focus_window(
        self, content: str, groups: list[list[tuple[str, bool]]]
    ) -> dict[str, object] | None:
        """Return the first qualifying NEAR() window used for preview focus."""
        return _search_query.best_near_focus_window(content, groups)

    def _best_close_focus_window(
        self, content: str, groups: list[list[tuple[str, bool]]]
    ) -> dict[str, object] | None:
        """Backward-compatible alias for NEAR() preview focus selection."""
        return self._best_near_focus_window(content, groups)

    def _count_highlighted_term_ranges(
        self,
        content: str,
        terms: list[tuple[str, bool]],
        *,
        close_focus_range: tuple[int, int] | None = None,
        enforce_close_boundaries: bool = False,
    ) -> int:
        """Count non-overlapping term highlight ranges in raw file content."""
        return _search_query.count_highlighted_term_ranges(
            content,
            terms,
            near_focus_range=close_focus_range,
            enforce_near_boundaries=enforce_close_boundaries,
        )

    def _remove_preview_search_highlights(self) -> None:
        """Remove in-preview search highlight spans from the current page."""
        js = _render_js_asset("preview/clear_search_highlights.js")
        # Mutates preview DOM to strip search marks; return value is not needed.
        self.preview.page().runJavaScript(js)

    def _highlight_preview_search_terms(
        self,
        terms: list[tuple[str, bool]],
        scroll_to_first: bool,
        close_term_groups: list[list[tuple[str, bool]]] | None = None,
    ) -> None:
        """Highlight term matches in preview and optionally scroll to first one."""
        cleaned_terms = [
            {"text": term_text, "caseSensitive": bool(is_case_sensitive)}
            for term_text, is_case_sensitive in terms
            if term_text.strip()
        ]
        if not cleaned_terms:
            self._remove_preview_search_highlights()
            return

        close_groups_payload: list[list[dict[str, object]]] = []
        for group in close_term_groups or []:
            payload_group: list[dict[str, object]] = []
            for term_text, is_case_sensitive in group:
                if term_text.strip():
                    payload_group.append(
                        {
                            "text": term_text,
                            "caseSensitive": bool(is_case_sensitive),
                        }
                    )
            if len(payload_group) >= 2:
                close_groups_payload.append(payload_group)

        js = _render_js_asset(
            "preview/highlight_search_terms.js",
            {
                "__TERMS_JSON__": json.dumps(cleaned_terms),
                "__SCROLL_BOOL__": "true" if scroll_to_first else "false",
                "__NEAR_WORD_GAP__": str(int(SEARCH_CLOSE_WORD_GAP)),
                "__NEAR_GROUPS_JSON__": json.dumps(close_groups_payload),
            },
        )
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

    def _compile_match_hit_counter(self, query: str):
        """Compile a lightweight per-file hit counter from query terms.

        The counter mirrors search term semantics:
        - Single-quoted terms are case-sensitive.
        - Double-quoted and unquoted terms are case-insensitive.
        - Boolean operators are ignored for counting; counts are additive across
          unique TERM tokens and are only used for files that already matched
          the full boolean predicate.
        - NEAR(...) counts mirror preview highlighting: qualifying NEAR
          windows are counted once each, with distinct qualifying occurrences
          per term.
        """
        return _search_query.compile_match_hit_counter(query)

    def _compile_match_predicate(self, query: str):
        """Compile boolean query with implicit AND, quotes, and NEAR(...)."""
        return _search_query.compile_match_predicate(query)

    def _tokenize_match_query(self, query: str) -> list[tuple[str, str, bool]]:
        """Tokenize query supporting operators plus single/double-quoted terms."""
        return _search_query.tokenize_match_query(query)

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
        clear_in_directory_action: QAction | None = None

        if path.is_file() and path.suffix.lower() == ".md":
            for idx, (color_name, color_value) in enumerate(self.HIGHLIGHT_COLORS):
                label = f"Highlight {color_name}" if idx == 0 else f"... {color_name}"
                action = menu.addAction(label)
                action.setData(color_value)
                color_actions[action] = color_value

            menu.addSeparator()
            clear_action = menu.addAction("Clear Highlight")

        clear_scope = path if path.is_dir() else path.parent
        clear_in_directory_action = menu.addAction("Clear in Directory")
        clear_all_action = menu.addAction("Clear All")
        chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if clear_in_directory_action is not None and chosen == clear_in_directory_action:
            self._confirm_and_clear_directory_highlighting(clear_scope)
            self.tree.viewport().update()
            return
        if chosen == clear_all_action:
            self._confirm_and_clear_all_highlighting(clear_scope)
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
        self._request_preview_context_menu_selection_info(
            click_x,
            click_y,
            selected_text_hint,
            lambda result: self._show_preview_context_menu_with_cached_selection(
                pos, result, selected_text_hint
            ),
        )

    def _request_preview_context_menu_selection_info(
        self, click_x: int, click_y: int, selected_text_hint: str, callback
    ) -> None:
        """Read preview selection/click metadata used to build the context menu."""
        js_expr = _render_js_asset(
            "preview/context_menu_selection_probe.js",
            {
                "__CLICK_X__": str(int(click_x)),
                "__CLICK_Y__": str(int(click_y)),
                "__SELECTED_HINT__": json.dumps(
                    selected_text_hint or "", ensure_ascii=True
                ),
            },
        )
        js = f"""
(() => {{
  try {{
    const __result = {js_expr};
    return JSON.stringify(__result || {{}});
  }} catch (err) {{
    return JSON.stringify({{
      __error__: String(err),
      __stack__: err && err.stack ? String(err.stack) : "",
    }});
  }}
}})();
"""

        def _on_result(result) -> None:
            normalized: dict = {}
            if isinstance(result, dict):
                normalized = result
            elif isinstance(result, str):
                try:
                    parsed = json.loads(result)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    normalized = parsed
            self._debug_log(
                "preview-menu-probe result "
                f"has_selection={bool(normalized.get('hasSelection'))} "
                f"offsets={self._selection_offsets_from_info(normalized)} "
                f"clicked_highlight={'yes' if bool(str(normalized.get('clickedHighlightId') or '').strip()) else 'no'} "
                f"clicked_offset={normalized.get('clickedOffset')}"
            )
            callback(normalized)

        self.preview.page().runJavaScript(js, _on_result)

    def _show_preview_context_menu_with_cached_selection(
        self, pos, selection_info, selected_text_hint: str
    ) -> None:
        """Build preview menu and use cached selection metadata for copy action."""
        standard_menu = self.preview.createStandardContextMenu()
        menu = QMenu(self.preview)
        has_selection = bool(selected_text_hint.strip() or self.preview.selectedText().strip())
        if isinstance(selection_info, dict):
            if selection_info.get("hasSelection"):
                has_selection = True
            selected_raw = selection_info.get("selectedText")
            if isinstance(selected_raw, str) and selected_raw.strip():
                has_selection = True
            # Offsets are the strongest signal; trust them even if hasSelection
            # was lost by a timing edge between selection and context-menu probe.
            if self._selection_offsets_from_info(selection_info) is not None:
                has_selection = True

        has_highlighted_part = bool(
            isinstance(selection_info, dict)
            and selection_info.get("selectionHasHighlightedPart")
        )
        has_unhighlighted_part = bool(
            isinstance(selection_info, dict)
            and selection_info.get("selectionHasUnhighlightedPart")
        )
        clicked_highlight_id = ""
        if isinstance(selection_info, dict):
            raw_clicked = selection_info.get("clickedHighlightId")
            if isinstance(raw_clicked, str):
                clicked_highlight_id = raw_clicked.strip()
        has_existing_persistent_highlights = bool(
            self._normalize_text_highlight_entries(
                self._current_preview_text_highlights
            )
        )
        self._debug_log(
            "preview-menu "
            f"has_selection={has_selection} "
            f"has_highlighted_part={has_highlighted_part} "
            f"has_unhighlighted_part={has_unhighlighted_part} "
            f"clicked_highlight_id={'yes' if bool(clicked_highlight_id) else 'no'} "
            f"clicked_offset={selection_info.get('clickedOffset') if isinstance(selection_info, dict) else None} "
            f"offsets={self._selection_offsets_from_info(selection_info)}"
        )

        highlight_action: QAction | None = None
        highlight_important_action: QAction | None = None
        remove_highlight_action: QAction | None = None
        # Always allow creating/extending highlight from any non-empty selection,
        # even if metadata probing fails on a specific right-click event.
        if has_selection:
            highlight_action = menu.addAction("Highlight")
            highlight_important_action = menu.addAction("Highlight Important")
        if (
            has_selection
            or clicked_highlight_id
            or (has_selection and has_highlighted_part)
            or has_existing_persistent_highlights
        ):
            remove_highlight_action = menu.addAction("Remove Highlight")
        if (
            highlight_action is not None
            or highlight_important_action is not None
            or remove_highlight_action is not None
        ):
            menu.addSeparator()

        copy_source_action: QAction | None = None
        copy_rendered_action: QAction | None = None
        if has_selection:
            copy_rendered_action = menu.addAction("Copy Rendered Text")
            copy_source_action = menu.addAction("Copy Source Markdown")
            menu.addSeparator()

        for action in standard_menu.actions():
            menu.addAction(action)

        chosen = menu.exec(self.preview.mapToGlobal(pos))
        if highlight_action is not None and chosen == highlight_action:
            self._debug_log("preview-menu action=Highlight")
            self._add_persistent_preview_highlight(
                selection_info,
                selected_text_hint,
                kind=PREVIEW_HIGHLIGHT_KIND_NORMAL,
            )
            standard_menu.deleteLater()
            menu.deleteLater()
            return
        if (
            highlight_important_action is not None
            and chosen == highlight_important_action
        ):
            self._debug_log("preview-menu action=Highlight Important")
            self._add_persistent_preview_highlight(
                selection_info,
                selected_text_hint,
                kind=PREVIEW_HIGHLIGHT_KIND_IMPORTANT,
            )
            standard_menu.deleteLater()
            menu.deleteLater()
            return
        if remove_highlight_action is not None and chosen == remove_highlight_action:
            self._debug_log("preview-menu action=Remove Highlight")
            self._remove_persistent_preview_highlight(
                selection_info, selected_text_hint
            )
            standard_menu.deleteLater()
            menu.deleteLater()
            return
        if copy_rendered_action is not None and chosen == copy_rendered_action:
            self._copy_preview_selection_as_rendered_text(
                selection_info, selected_text_hint
            )
            standard_menu.deleteLater()
            menu.deleteLater()
            return
        if copy_source_action is not None and chosen == copy_source_action:
            self._copy_preview_selection_as_source_markdown(
                selection_info, selected_text_hint
            )
        standard_menu.deleteLater()
        menu.deleteLater()

    @staticmethod
    def _selection_offsets_from_info(selection_info) -> tuple[int, int] | None:
        """Extract normalized selection text offsets from cached JS metadata."""
        if not isinstance(selection_info, dict):
            return None
        start_raw = selection_info.get("selectionOffsetStart")
        end_raw = selection_info.get("selectionOffsetEnd")
        if not isinstance(start_raw, (int, float)) or not isinstance(
            end_raw, (int, float)
        ):
            return None
        start = int(start_raw)
        end = int(end_raw)
        if start < 0 or end <= start:
            return None
        return (start, end)

    def _request_live_preview_selection_offsets(
        self, selected_text_hint: str, callback
    ) -> None:
        """Read live selection offsets from preview DOM, with text fallback."""
        self._debug_log(
            "highlight-live-probe request "
            f"hint_len={len(selected_text_hint or '')}"
        )
        # Keep embedding JS-safe even if selection contains U+2028/U+2029 or
        # other non-ASCII characters that can break inline script parsing.
        hint_json = json.dumps(selected_text_hint or "", ensure_ascii=True)
        js_expr = _render_js_asset(
            "preview/live_selection_offsets.js",
            {"__SELECTED_HINT__": hint_json},
        )
        # Force a stable string return type from QtWebEngine marshalling.
        # If evaluation fails, include the JS error payload in the JSON.
        js = f"""
(() => {{
  try {{
    const __result = {js_expr};
    return JSON.stringify(__result || {{}});
  }} catch (err) {{
    return JSON.stringify({{
      __error__: String(err),
    }});
  }}
}})();
"""
        def _on_result(result) -> None:
            normalized: dict = {}
            if isinstance(result, dict):
                normalized = result
            elif isinstance(result, str):
                try:
                    parsed = json.loads(result)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    normalized = parsed
            self._debug_log(
                "highlight-live-probe result "
                f"raw_type={type(result).__name__} "
                f"raw_is_dict={isinstance(result, dict)} "
                f"raw_preview={repr(str(result)[:120])} "
                f"js_error={repr(str(normalized.get('__error__', ''))[:120])} "
                f"hasSelection={bool(normalized.get('hasSelection'))} "
                f"offsets={self._selection_offsets_from_info(normalized)} "
                f"selected_len={len(str(normalized.get('selectedText') or ''))}"
            )
            callback(normalized)

        self.preview.page().runJavaScript(js, _on_result)

    def _request_live_preview_highlight_target(self, callback) -> None:
        """Read the latest in-page clicked highlight metadata from preview DOM."""
        js_expr = _render_js_asset("preview/live_highlight_target.js")
        js = f"""
(() => {{
  try {{
    const __result = {js_expr};
    return JSON.stringify(__result || {{}});
  }} catch (err) {{
    return JSON.stringify({{
      __error__: String(err),
    }});
  }}
}})();
"""

        def _on_result(result) -> None:
            normalized: dict = {}
            if isinstance(result, dict):
                normalized = result
            elif isinstance(result, str):
                try:
                    parsed = json.loads(result)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    normalized = parsed
            self._debug_log(
                "highlight-click-target result "
                f"id={'yes' if bool(str(normalized.get('clickedHighlightId') or '').strip()) else 'no'} "
                f"offset={normalized.get('clickedOffset')}"
            )
            callback(normalized)

        self.preview.page().runJavaScript(js, _on_result)

    def _replace_persistent_preview_highlight_range(
        self, path_key: str, start: int, end: int, kind: str
    ) -> None:
        """Replace overlap with one highlight kind, persist, and reapply."""
        if end <= start:
            self.statusBar().showMessage("Select text to highlight", 3000)
            return
        normalized_kind = self._normalize_preview_highlight_kind(kind)
        self._debug_log(
            "highlight-replace "
            f"path={path_key} start={start} end={end} "
            f"len={max(0, end-start)} kind={normalized_kind}"
        )
        entries = self._normalize_text_highlight_entries(
            self._clone_json_compatible_list(self._current_preview_text_highlights)
        )
        updated: list[dict[str, int | str]] = []
        for entry in entries:
            entry_start = int(entry.get("start", -1))
            entry_end = int(entry.get("end", -1))
            entry_kind = self._normalize_preview_highlight_kind(entry.get("kind"))
            if entry_end <= start or entry_start >= end:
                updated.append(
                    {
                        "id": str(entry.get("id", "")).strip()
                        or self._new_text_highlight_id(),
                        "start": entry_start,
                        "end": entry_end,
                        "kind": entry_kind,
                    }
                )
                continue
            if entry_start < start:
                updated.append(
                    {
                        "id": self._new_text_highlight_id(),
                        "start": entry_start,
                        "end": start,
                        "kind": entry_kind,
                    }
                )
            if entry_end > end:
                updated.append(
                    {
                        "id": self._new_text_highlight_id(),
                        "start": end,
                        "end": entry_end,
                        "kind": entry_kind,
                    }
                )
        updated.append(
            {
                "id": self._new_text_highlight_id(),
                "start": start,
                "end": end,
                "kind": normalized_kind,
            }
        )
        entries = self._normalize_text_highlight_entries(updated)
        self._current_preview_text_highlights = entries
        self._persist_text_highlights_for_path_key(path_key, entries)

        def _on_applied(result: dict) -> None:
            applied_count = (
                int(result.get("applied", 0)) if isinstance(result, dict) else 0
            )
            self._debug_log(
                "highlight-apply-result "
                f"applied={applied_count} entries={len(entries)} "
                f"result={result if isinstance(result, dict) else {}}"
            )
            if applied_count <= 0:
                self.statusBar().showMessage(
                    "Highlight saved, but no visible span was applied",
                    3500,
                )
                return
            message = (
                "Important highlight added"
                if normalized_kind == PREVIEW_HIGHLIGHT_KIND_IMPORTANT
                else "Highlight added"
            )
            self.statusBar().showMessage(message, 2500)

        self._apply_persistent_preview_highlights(path_key, completion=_on_applied)

    def _apply_persistent_preview_highlights(
        self,
        expected_key: str | None = None,
        completion: Callable[[dict], None] | None = None,
    ) -> None:
        """Apply persisted preview text highlights to the active document DOM."""
        current_key = self._current_preview_path_key()
        if not current_key:
            return
        if expected_key is not None and expected_key != current_key:
            return
        entries = self._normalize_text_highlight_entries(
            self._current_preview_text_highlights
        )
        self._current_preview_text_highlights = entries
        self._debug_log(
            "highlight-apply-request "
            f"current_key={current_key} expected_key={expected_key} "
            f"entries={len(entries)} completion={bool(completion)}"
        )
        payload_json = json.dumps(entries, separators=(",", ":"), ensure_ascii=False)
        color_json = json.dumps(self.PREVIEW_HIGHLIGHT_COLOR)
        important_color_json = json.dumps(self.PREVIEW_HIGHLIGHT_IMPORTANT_COLOR)
        important_text_color_json = json.dumps(
            PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_TEXT_COLOR
        )
        marker_color_json = json.dumps(PREVIEW_PERSISTENT_HIGHLIGHT_MARKER_COLOR)
        important_marker_color_json = json.dumps(
            PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR
        )
        js_expr = _render_js_asset(
            "preview/apply_persistent_highlights.js",
            {
                "__PAYLOAD__": payload_json,
                "__COLOR__": color_json,
                "__IMPORTANT_COLOR__": important_color_json,
                "__IMPORTANT_TEXT_COLOR__": important_text_color_json,
                "__MARKER_COLOR__": marker_color_json,
                "__IMPORTANT_MARKER_COLOR__": important_marker_color_json,
                "__NORMAL_KIND__": PREVIEW_HIGHLIGHT_KIND_NORMAL,
                "__IMPORTANT_KIND__": PREVIEW_HIGHLIGHT_KIND_IMPORTANT,
            },
        )
        js = f"""
(() => {{
  try {{
    const __result = {js_expr};
    return JSON.stringify(__result || {{}});
  }} catch (err) {{
    return JSON.stringify({{
      __error__: String(err),
      __stack__: err && err.stack ? String(err.stack) : "",
    }});
  }}
}})();
"""

        def _normalize_apply_result(result) -> dict:
            if isinstance(result, dict):
                return result
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    return parsed
            return {}

        if completion is None:
            self.preview.page().runJavaScript(
                js,
                lambda result: self._debug_log(
                    "highlight-apply-no-completion "
                    f"result={_normalize_apply_result(result)}"
                ),
            )
            return
        self.preview.page().runJavaScript(
            js, lambda result: completion(_normalize_apply_result(result))
        )

    def _add_persistent_preview_highlight(
        self,
        selection_info,
        selected_text_hint: str = "",
        kind: str = PREVIEW_HIGHLIGHT_KIND_NORMAL,
    ) -> None:
        """Persist a new preview text-highlight range for the current file."""
        path_key = self._current_preview_path_key()
        if not path_key:
            self._debug_log("highlight-add aborted reason=no-current-path")
            self.statusBar().showMessage("No markdown file selected", 3000)
            return
        normalized_kind = self._normalize_preview_highlight_kind(kind)
        offsets = self._selection_offsets_from_info(selection_info)
        self._debug_log(
            "highlight-add start "
            f"path={path_key} cached_offsets={offsets} "
            f"hint_len={len(selected_text_hint or '')} "
            f"kind={normalized_kind}"
        )
        if offsets is not None:
            start, end = offsets
            self._replace_persistent_preview_highlight_range(
                path_key, start, end, normalized_kind
            )
            return

        selected_text = ""
        if isinstance(selection_info, dict):
            raw_selected = selection_info.get("selectedText")
            if isinstance(raw_selected, str):
                selected_text = raw_selected
        if not selected_text.strip():
            selected_text = selected_text_hint
        self._debug_log(
            "highlight-add fallback-live-probe "
            f"selected_len={len(selected_text)}"
        )

        def _apply_live_selection(live_info: dict) -> None:
            if self._current_preview_path_key() != path_key:
                self._debug_log(
                    "highlight-add live-probe ignored reason=path-changed "
                    f"expected={path_key} actual={self._current_preview_path_key()}"
                )
                return
            live_offsets = self._selection_offsets_from_info(live_info)
            if live_offsets is None:
                self._debug_log(
                    "highlight-add live-probe failed offsets=None "
                    f"live_info={live_info}"
                )
                self.statusBar().showMessage("Select text to highlight", 3000)
                return
            start_live, end_live = live_offsets
            self._debug_log(
                "highlight-add live-probe success "
                f"start={start_live} end={end_live}"
            )
            self._replace_persistent_preview_highlight_range(
                path_key, start_live, end_live, normalized_kind
            )

        self._request_live_preview_selection_offsets(selected_text, _apply_live_selection)

    def _remove_persistent_preview_highlight(
        self, selection_info, selected_text_hint: str = ""
    ) -> None:
        """Remove persisted highlight blocks by click target and/or selected range."""
        path_key = self._current_preview_path_key()
        if not path_key:
            self.statusBar().showMessage("No markdown file selected", 3000)
            return
        entries = self._normalize_text_highlight_entries(
            self._clone_json_compatible_list(self._current_preview_text_highlights)
        )
        if not entries:
            self.statusBar().showMessage("No persistent highlights to remove", 2500)
            return

        def _ids_overlapping(offsets: tuple[int, int]) -> set[str]:
            sel_start, sel_end = offsets
            found: set[str] = set()
            for entry in entries:
                start = int(entry.get("start", -1))
                end = int(entry.get("end", -1))
                if max(sel_start, start) < min(sel_end, end):
                    entry_id = str(entry.get("id", "")).strip()
                    if entry_id:
                        found.add(entry_id)
            return found

        def _ids_containing_offset(offset: int) -> set[str]:
            found: set[str] = set()
            for entry in entries:
                start = int(entry.get("start", -1))
                end = int(entry.get("end", -1))
                # Accept exact-end clicks as part of the same block for usability.
                if start <= offset <= end:
                    entry_id = str(entry.get("id", "")).strip()
                    if entry_id:
                        found.add(entry_id)
            return found

        def _apply_ids(ids_to_remove: set[str]) -> None:
            updated = [
                entry for entry in entries if str(entry.get("id", "")) not in ids_to_remove
            ]
            updated = self._normalize_text_highlight_entries(updated)
            if len(updated) == len(entries):
                self.statusBar().showMessage("No highlighted block selected", 2500)
                return
            self._current_preview_text_highlights = updated
            self._persist_text_highlights_for_path_key(path_key, updated)
            self._apply_persistent_preview_highlights(path_key)
            self.statusBar().showMessage("Highlight removed", 2500)

        ids_to_remove: set[str] = set()
        if isinstance(selection_info, dict):
            raw_clicked = selection_info.get("clickedHighlightId")
            if isinstance(raw_clicked, str) and raw_clicked.strip():
                ids_to_remove.add(raw_clicked.strip())

            offsets = self._selection_offsets_from_info(selection_info)
            if offsets is not None:
                ids_to_remove.update(_ids_overlapping(offsets))

            raw_ids = selection_info.get("selectedHighlightIds")
            if isinstance(raw_ids, list):
                for item in raw_ids:
                    if isinstance(item, str) and item.strip():
                        ids_to_remove.add(item.strip())
            clicked_offset_raw = selection_info.get("clickedOffset")
            if isinstance(clicked_offset_raw, (int, float)):
                ids_to_remove.update(_ids_containing_offset(int(clicked_offset_raw)))

        if ids_to_remove:
            _apply_ids(ids_to_remove)
            return

        selected_text = ""
        if isinstance(selection_info, dict):
            raw_selected = selection_info.get("selectedText")
            if isinstance(raw_selected, str):
                selected_text = raw_selected
        if not selected_text.strip():
            selected_text = selected_text_hint
        if not selected_text.strip():
            def _apply_live_click_removal(target_info: dict) -> None:
                if self._current_preview_path_key() != path_key:
                    return
                live_ids: set[str] = set()
                if isinstance(target_info, dict):
                    live_clicked = target_info.get("clickedHighlightId")
                    if isinstance(live_clicked, str) and live_clicked.strip():
                        live_ids.add(live_clicked.strip())
                    live_offset_raw = target_info.get("clickedOffset")
                    if isinstance(live_offset_raw, (int, float)):
                        live_ids.update(_ids_containing_offset(int(live_offset_raw)))
                if not live_ids:
                    self.statusBar().showMessage("No highlighted block selected", 2500)
                    return
                _apply_ids(live_ids)

            self._request_live_preview_highlight_target(_apply_live_click_removal)
            return

        def _apply_live_removal(live_info: dict) -> None:
            if self._current_preview_path_key() != path_key:
                return
            live_offsets = self._selection_offsets_from_info(live_info)
            if live_offsets is None:
                self.statusBar().showMessage("No highlighted block selected", 2500)
                return
            live_ids = _ids_overlapping(live_offsets)
            if not live_ids:
                self.statusBar().showMessage("No highlighted block selected", 2500)
                return
            _apply_ids(live_ids)

        self._request_live_preview_selection_offsets(
            selected_text, _apply_live_removal
        )

    def _copy_preview_selection_as_rendered_text(
        self, selection_info, selected_text_hint: str
    ) -> None:
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

    def _copy_preview_selection_as_source_markdown(
        self, selection_info, selected_text_hint: str
    ) -> None:
        """Copy source markdown lines that correspond to selected preview content."""
        if self.current_file is None:
            self.statusBar().showMessage("No markdown file selected", 3000)
            return
        source_path = self.current_file
        try:
            lines = source_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines(keepends=True)
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
            if isinstance(start_raw, (int, float)) and isinstance(
                end_raw, (int, float)
            ):
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

        query = (
            js_selected_text
            or selected_text_hint.strip()
            or self.preview.selectedText().strip()
        )
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

            normalized_match = self._find_source_span_by_normalized_document_match(
                lines, query
            )
            if normalized_match is not None:
                start_idx, end_idx, label = normalized_match
                snippet = "".join(lines[start_idx:end_idx])
                self._set_plain_text_clipboard(snippet)
                self.statusBar().showMessage(
                    f"Copied source markdown lines {start_idx + 1}-{end_idx} ({label})",
                    4000,
                )
                return

            fuzzy_match = self._find_source_span_by_fuzzy_lines(lines, query)
            if fuzzy_match is not None:
                start_idx, end_idx = fuzzy_match
                snippet = "".join(lines[start_idx:end_idx])
                self._set_plain_text_clipboard(snippet)
                self.statusBar().showMessage(
                    f"Copied source markdown lines {start_idx + 1}-{end_idx} (fuzzy)",
                    4000,
                )
                return

        self._set_plain_text_clipboard("".join(lines))
        self.statusBar().showMessage(
            "Could not map selection exactly; copied full source markdown",
            4500,
        )

    @staticmethod
    def _normalize_for_fuzzy_match(text: str) -> str:
        lowered = (
            text.casefold()
            .replace("\\", " ")
            .replace("—", " ")
            .replace("–", " ")
            .replace("\u00a0", " ")
        )
        stripped = re.sub(r"[`*_~>#\[\](){}|!+\-:;,.?/]", " ", lowered)
        stripped = re.sub(r"^\s*[-*+]\s+", "", stripped, flags=re.MULTILINE)
        stripped = re.sub(r"^\s*\d+[.)]\s+", "", stripped, flags=re.MULTILINE)
        return re.sub(r"\s+", " ", stripped).strip()

    @classmethod
    def _meaningful_normalized_query_lines(cls, query_text: str) -> list[str]:
        """Return non-empty normalized query lines for boundary refinement."""
        normalized: list[str] = []
        for raw_line in query_text.splitlines():
            value = cls._normalize_for_fuzzy_match(raw_line)
            if value:
                normalized.append(value)
        return normalized

    @staticmethod
    def _line_match_score(query_norm: str, candidate: str) -> float:
        """Score how well a normalized query line matches a normalized source line."""
        if not query_norm or not candidate:
            return 0.0
        if query_norm in candidate:
            return 0.92 + min(0.08, len(query_norm) / max(len(candidate), 1))
        if candidate in query_norm and len(candidate) >= 8:
            return 0.65 + min(0.20, len(candidate) / max(len(query_norm), 1))
        return SequenceMatcher(None, query_norm, candidate).ratio()

    @classmethod
    def _align_query_lines_to_source(
        cls,
        query_lines: list[str],
        normalized_lines: list[str],
        approx_start: int,
        approx_end: int,
    ) -> tuple[int, int] | None:
        """Refine an approximate source span by matching query lines in order."""
        if not query_lines:
            return None

        total_query_chars = sum(len(line) for line in query_lines)
        if total_query_chars < 32:
            return None

        search_start = max(0, approx_start - 18)
        search_end = min(
            len(normalized_lines),
            max(approx_end + max(48, len(query_lines) * 3), search_start + 1),
        )
        current_index = search_start
        matched_indexes: list[int] = []
        matched_chars = 0

        for query_norm in query_lines:
            if len(query_norm) < 3:
                continue

            best_idx = -1
            best_score = 0.0
            line_span_end = min(
                search_end, current_index + max(42, len(query_lines) * 2)
            )
            for idx in range(current_index, line_span_end):
                candidate = normalized_lines[idx]
                if not candidate:
                    continue
                score = cls._line_match_score(query_norm, candidate)
                if score > best_score:
                    best_idx = idx
                    best_score = score
                    if score >= 0.995:
                        break

            min_score = 0.52 if len(query_norm) >= 18 else 0.68
            if best_idx >= 0 and best_score >= min_score:
                matched_indexes.append(best_idx)
                matched_chars += len(query_norm)
                current_index = best_idx + 1
            elif matched_indexes:
                current_index = min(search_end, current_index + 4)

        min_required_matches = max(
            3, min(len(query_lines), max(1, len(query_lines) // 3))
        )
        if len(matched_indexes) < min_required_matches:
            return None
        if matched_chars / max(total_query_chars, 1) < 0.58:
            return None

        start_idx = matched_indexes[0]
        end_idx = matched_indexes[-1] + 1
        if end_idx <= start_idx:
            return None
        return max(0, start_idx), min(end_idx, len(normalized_lines))

    @classmethod
    def _best_span_from_line_start_candidates(
        cls,
        query_lines: list[str],
        normalized_lines: list[str],
        candidate_starts: list[int],
        preferred_span_len: int,
    ) -> tuple[int, int, float] | None:
        """Score full spans from plausible line starts for long rendered selections."""
        if not query_lines or not candidate_starts:
            return None

        normalized_query = " ".join(query_lines)
        if len(normalized_query) < 48:
            return None

        target_len = max(len(query_lines), preferred_span_len, 1)
        candidate_lengths = {
            max(1, int(target_len * 0.85)),
            target_len,
            int(target_len * 1.05),
            int(target_len * 1.15),
            int(target_len * 1.30),
            int(target_len * 1.50),
            int(target_len * 1.75),
        }

        best_span: tuple[int, int, float] | None = None
        best_score = 0.0
        for start_idx in candidate_starts:
            for span_len in sorted(candidate_lengths):
                end_idx = min(len(normalized_lines), start_idx + span_len)
                if end_idx <= start_idx:
                    continue
                candidate_text = " ".join(
                    line for line in normalized_lines[start_idx:end_idx] if line
                )
                if not candidate_text:
                    continue
                score = SequenceMatcher(None, normalized_query, candidate_text).ratio()
                relative_len = span_len / max(target_len, 1)
                if relative_len > 1.80:
                    score -= 0.14
                elif relative_len > 1.45:
                    score -= 0.07
                elif relative_len < 0.75:
                    score -= 0.10
                elif relative_len < 0.90:
                    score -= 0.04
                if score > best_score:
                    best_score = score
                    best_span = (start_idx, end_idx, score)

        return best_span

    @classmethod
    def _find_source_span_by_normalized_document_match(
        cls, lines: list[str], query_text: str
    ) -> tuple[int, int, str] | None:
        """Match rendered selection against normalized source text across the whole document."""
        normalized_query = cls._normalize_for_fuzzy_match(query_text)
        if len(normalized_query) < 24:
            return None

        normalized_lines = [cls._normalize_for_fuzzy_match(line) for line in lines]
        source_parts: list[str] = []
        char_to_line: list[int] = []
        for idx, line in enumerate(lines):
            normalized_line = normalized_lines[idx]
            if not normalized_line:
                continue
            if source_parts:
                source_parts.append(" ")
                char_to_line.append(idx)
            source_parts.append(normalized_line)
            char_to_line.extend([idx] * len(normalized_line))

        if not source_parts or not char_to_line:
            return None

        normalized_source = "".join(source_parts)
        query_lines = cls._meaningful_normalized_query_lines(query_text)

        def map_char_span(start_char: int, end_char: int) -> tuple[int, int] | None:
            if start_char < 0 or end_char <= start_char:
                return None
            if start_char >= len(char_to_line):
                return None
            end_char = min(end_char, len(char_to_line))
            start_idx = char_to_line[start_char]
            end_idx = char_to_line[max(start_char, end_char - 1)] + 1
            if start_idx < 0 or end_idx <= start_idx:
                return None
            return start_idx, min(end_idx, len(lines))

        def refine_span(start_idx: int, end_idx: int) -> tuple[int, int]:
            if not query_lines:
                return start_idx, end_idx

            refined_start = start_idx
            refined_end = end_idx
            prefix_candidates = query_lines[: min(8, len(query_lines))]
            suffix_candidates = list(
                reversed(query_lines[max(0, len(query_lines) - 8) :])
            )

            def best_line_match(
                candidates: list[str],
                range_start: int,
                range_end: int,
                min_score: float,
            ) -> tuple[int, float]:
                best_idx = -1
                best_score = 0.0
                for query_norm in candidates:
                    for idx in range(
                        max(0, range_start), min(len(normalized_lines), range_end)
                    ):
                        candidate = normalized_lines[idx]
                        if not candidate:
                            continue
                        score = cls._line_match_score(query_norm, candidate)
                        if score > best_score:
                            best_idx = idx
                            best_score = score
                    if best_idx >= 0 and best_score >= min_score:
                        return best_idx, best_score
                return best_idx, best_score

            start_match_idx, start_score = best_line_match(
                prefix_candidates,
                max(0, start_idx - 10),
                min(len(lines), start_idx + 30),
                0.48,
            )
            if start_match_idx >= 0 and start_score >= 0.48:
                refined_start = start_match_idx

            end_match_idx, end_score = best_line_match(
                suffix_candidates,
                max(refined_start, end_idx - 30),
                min(len(lines), end_idx + 10),
                0.45,
            )
            if end_match_idx >= refined_start and end_score >= 0.45:
                refined_end = end_match_idx + 1

            return refined_start, max(refined_start + 1, min(refined_end, len(lines)))

        found_at = normalized_source.find(normalized_query)
        if found_at != -1:
            mapped = map_char_span(found_at, found_at + len(normalized_query))
            if mapped is not None:
                start_idx, end_idx = refine_span(*mapped)
                aligned = cls._align_query_lines_to_source(
                    query_lines, normalized_lines, start_idx, end_idx
                )
                if aligned is not None:
                    start_idx, end_idx = aligned
                snippet = "".join(lines[start_idx:end_idx])
                if snippet.strip():
                    return start_idx, end_idx, "normalized"

        # Rendered selections often differ slightly from source due to markdown
        # punctuation stripping. Use multiple anchors from across the normalized
        # selection, project candidate spans back into source space, and score
        # the resulting slices. This is much more resilient than relying on a
        # single prefix anchor.
        anchor_lengths = (320, 240, 180, 120, 80, 48)
        anchor_offsets = [
            0,
            max(0, len(normalized_query) // 4),
            max(0, len(normalized_query) // 2),
            max(0, (len(normalized_query) * 3) // 4),
            max(0, len(normalized_query) - 320),
        ]
        candidate_positions: list[tuple[int, int, int]] = []
        seen_positions: set[tuple[int, int]] = set()
        for offset in anchor_offsets:
            for size in anchor_lengths:
                if len(normalized_query) < size or offset + size > len(
                    normalized_query
                ):
                    continue
                fragment = normalized_query[offset : offset + size]
                search_at = normalized_source.find(fragment)
                hits = 0
                while search_at != -1 and hits < 8:
                    key = (search_at, offset)
                    if key not in seen_positions:
                        seen_positions.add(key)
                        candidate_positions.append((search_at, offset, size))
                    hits += 1
                    search_at = normalized_source.find(fragment, search_at + 1)

        line_start_candidates: list[int] = []
        seen_line_start_buckets: set[int] = set()
        for query_offset, query_norm in enumerate(
            query_lines[: min(12, len(query_lines))]
        ):
            if len(query_norm) < 4:
                continue
            min_score = 0.52 if len(query_norm) >= 18 else 0.68
            per_query_candidates: list[tuple[float, int]] = []
            for idx, candidate in enumerate(normalized_lines):
                if not candidate:
                    continue
                score = cls._line_match_score(query_norm, candidate)
                if score >= min_score:
                    estimated_start = max(0, idx - query_offset)
                    per_query_candidates.append((score, estimated_start))
            per_query_candidates.sort(key=lambda item: item[0], reverse=True)
            for score, estimated_start in per_query_candidates[:6]:
                bucket = estimated_start // 3
                if bucket in seen_line_start_buckets:
                    continue
                seen_line_start_buckets.add(bucket)
                line_start_candidates.append(estimated_start)

        best_span: tuple[int, int, str] | None = None
        best_score = 0.0
        for position, offset, size in candidate_positions:
            estimated_start = max(0, position - offset)
            estimated_end = min(
                len(normalized_source), estimated_start + len(normalized_query)
            )
            candidate_text = normalized_source[estimated_start:estimated_end]
            score = SequenceMatcher(None, normalized_query, candidate_text).ratio()
            if estimated_end - estimated_start < len(normalized_query) * 0.72:
                score *= 0.82
            score += min(0.05, size / 6400.0)
            if score <= best_score:
                continue
            mapped = map_char_span(estimated_start, estimated_end)
            if mapped is None:
                continue
            start_idx, end_idx = refine_span(*mapped)
            aligned = cls._align_query_lines_to_source(
                query_lines, normalized_lines, start_idx, end_idx
            )
            if aligned is not None:
                start_idx, end_idx = aligned
            snippet = "".join(lines[start_idx:end_idx])
            if not snippet.strip():
                continue
            best_score = score
            best_span = (start_idx, end_idx, "normalized anchor")

        preferred_span_len = (
            (best_span[1] - best_span[0])
            if best_span is not None
            else max(1, len(query_lines))
        )
        candidate_span = cls._best_span_from_line_start_candidates(
            query_lines,
            normalized_lines,
            line_start_candidates[:18],
            preferred_span_len,
        )
        if candidate_span is not None:
            start_idx, end_idx, score = candidate_span
            if score >= 0.86 and score > best_score + 0.015:
                best_score = score
                best_span = (start_idx, end_idx, "normalized lines")

        if best_span is not None and best_score >= 0.74:
            return best_span

        return None

    @classmethod
    def _find_source_span_by_fuzzy_lines(
        cls, lines: list[str], query_text: str
    ) -> tuple[int, int] | None:
        """Fuzzy-match selected first/last lines against markdown source lines."""
        raw_query_lines = [
            line.strip() for line in query_text.splitlines() if line.strip()
        ]
        if not raw_query_lines:
            return None

        normalized_lines = [cls._normalize_for_fuzzy_match(line) for line in lines]
        meaningful_query_lines: list[str] = []
        for line in raw_query_lines:
            normalized = cls._normalize_for_fuzzy_match(line)
            if normalized:
                meaningful_query_lines.append(normalized)
        if not meaningful_query_lines:
            return None

        normalized_query = " ".join(meaningful_query_lines)

        def best_line_match(query_norm: str, start_index: int = 0) -> tuple[int, float]:
            best_idx = -1
            best_score = 0.0
            for idx in range(start_index, len(normalized_lines)):
                candidate = normalized_lines[idx]
                if not candidate:
                    continue
                score = cls._line_match_score(query_norm, candidate)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            return best_idx, best_score

        def find_anchor(
            candidates: list[str], start_index: int, min_score: float
        ) -> tuple[int, float]:
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

        candidate_starts: list[tuple[float, int]] = []
        seen_candidate_starts: set[int] = set()
        anchor_queries = meaningful_query_lines[: min(8, len(meaningful_query_lines))]
        for query_offset, query_norm in enumerate(anchor_queries):
            if len(query_norm) < 4:
                continue
            min_score = 0.52 if len(query_norm) >= 18 else 0.68
            per_query_candidates: list[tuple[float, int]] = []
            for idx, candidate in enumerate(normalized_lines):
                if not candidate:
                    continue
                score = cls._line_match_score(query_norm, candidate)
                if score >= min_score:
                    estimated_start = max(0, idx - query_offset)
                    per_query_candidates.append((score, estimated_start))
            per_query_candidates.sort(key=lambda item: item[0], reverse=True)
            for score, estimated_start in per_query_candidates[:6]:
                bucket = estimated_start // 3
                if bucket in seen_candidate_starts:
                    continue
                seen_candidate_starts.add(bucket)
                candidate_starts.append((score, estimated_start))

        best_span: tuple[int, int] | None = None
        best_span_score = 0.0
        for anchor_score, estimated_start in sorted(
            candidate_starts, key=lambda item: item[0], reverse=True
        )[:18]:
            approx_end = min(
                len(lines), estimated_start + len(meaningful_query_lines) + 24
            )
            aligned = cls._align_query_lines_to_source(
                meaningful_query_lines,
                normalized_lines,
                estimated_start,
                approx_end,
            )
            if aligned is None:
                continue
            start_idx, end_idx = aligned
            candidate_text = " ".join(
                line for line in normalized_lines[start_idx:end_idx] if line
            )
            if not candidate_text:
                continue
            span_score = SequenceMatcher(None, normalized_query, candidate_text).ratio()
            span_len = max(1, end_idx - start_idx)
            target_len = max(1, len(meaningful_query_lines))
            if span_len > target_len * 1.6:
                span_score -= 0.10
            elif span_len > target_len * 1.25:
                span_score -= 0.04
            elif span_len < target_len * 0.60:
                span_score -= 0.12
            span_score += min(0.04, anchor_score * 0.04)
            if span_score > best_span_score:
                best_span_score = span_score
                best_span = (start_idx, end_idx)

        if best_span is not None and best_span_score >= 0.62:
            return best_span

        start_idx, start_score = find_anchor(meaningful_query_lines, 0, 0.45)
        if start_idx < 0 or start_score < 0.45:
            return None

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
        aligned = cls._align_query_lines_to_source(
            meaningful_query_lines, normalized_lines, start_idx, end_idx
        )
        if aligned is not None:
            start_idx, end_idx = aligned
        snippet = "".join(lines[start_idx:end_idx])
        if not snippet.strip():
            return None

        return start_idx, end_idx

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
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=text,
                    text=True,
                    check=False,
                )
                return
            if os.environ.get("DISPLAY") and shutil.which("xsel"):
                subprocess.run(
                    ["xsel", "--clipboard", "--input"],
                    input=text,
                    text=True,
                    check=False,
                )
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
        if (
            self.last_directory_selection is not None
            and self.last_directory_selection.is_dir()
        ):
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
        if (
            self.last_directory_selection is not None
            and self.last_directory_selection.is_dir()
        ):
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
        self._stop_restore_overlay_monitor()
        self._persist_document_view_session()
        self._persist_effective_root()
        self._cleanup_preview_temp_files()
        super().closeEvent(event)

    def _confirm_and_clear_directory_highlighting(
        self, scope: Path | None = None
    ) -> None:
        """Prompt and clear file highlights in one directory (non-recursive)."""
        target_scope = (
            scope if isinstance(scope, Path) else self._highlight_scope_directory()
        )
        if not target_scope.is_dir():
            return
        try:
            display_scope = target_scope.resolve()
        except Exception:
            display_scope = target_scope
        reply = QMessageBox.question(
            self,
            "Clear Directory Highlights",
            f"Clear all file highlights in this directory only:\n{display_scope}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        cleared = self.model.clear_directory_highlights(target_scope)
        self.statusBar().showMessage(
            f"Cleared {cleared} highlight assignment(s) in {display_scope}",
            4500,
        )

    def _confirm_and_clear_all_highlighting(self, scope: Path | None = None) -> None:
        """Prompt and clear all highlight metadata recursively under current scope."""
        target_scope = (
            scope if isinstance(scope, Path) else self._highlight_scope_directory()
        )
        if not target_scope.is_dir():
            return
        try:
            display_scope = target_scope.resolve()
        except Exception:
            display_scope = target_scope
        reply = QMessageBox.question(
            self,
            "Clear All Highlights",
            f"Clear all file highlights recursively under:\n{display_scope}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        cleared = self.model.clear_all_highlights(target_scope)
        self.statusBar().showMessage(
            f"Cleared {cleared} highlight assignment(s) under {display_scope}",
            4500,
        )

    def _copy_destination_is_directory(self) -> bool:
        """Return whether copy actions target a folder instead of the clipboard."""
        radio = getattr(self, "copy_directory_radio", None)
        return bool(radio is not None and radio.isChecked())

    def _default_copy_destination_directory(self) -> Path:
        """Return default folder used by the copy destination chooser."""
        if (
            isinstance(self._copy_destination_directory, Path)
            and self._copy_destination_directory.is_dir()
        ):
            return self._copy_destination_directory
        return self._effective_root_for_persistence()

    def _prompt_copy_destination_directory(self) -> Path | None:
        """Prompt for destination folder and remember the selection."""
        default_directory = self._default_copy_destination_directory()
        selected_path = QFileDialog.getExistingDirectory(
            self,
            "Select Target Directory",
            str(default_directory),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected_path:
            return None

        selected = Path(selected_path).expanduser()
        try:
            selected = selected.resolve()
        except Exception:
            pass
        if not selected.is_dir():
            self.statusBar().showMessage("Selected directory is unavailable", 3500)
            return None
        self._copy_destination_directory = selected
        return selected

    def _normalize_unique_file_paths(self, files: list[Path]) -> list[Path]:
        """Resolve and de-duplicate a list of filesystem paths."""
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
        return normalized

    def _copy_files_to_clipboard(self, files: list[Path]) -> int:
        """Copy file paths to clipboard with file-manager compatible MIME payloads."""
        normalized = self._normalize_unique_file_paths(files)

        clipboard = QApplication.clipboard()
        mime_data = QMimeData()
        urls = [QUrl.fromLocalFile(str(path)) for path in normalized]
        mime_data.setUrls(urls)

        # Nemo/Nautilus paste support: this custom format marks clipboard data
        # as file copy operations rather than plain text.
        if urls:
            gnome_payload = "copy\n" + "\n".join(url.toString() for url in urls)
            mime_data.setData(
                "x-special/gnome-copied-files", gnome_payload.encode("utf-8")
            )

        # Keep plain text for editors/terminals.
        mime_data.setText("\n".join(str(path) for path in normalized))
        clipboard.setMimeData(mime_data)
        return len(normalized)

    def _view_session_for_path_key(self, path_key: str | None) -> dict | None:
        """Return best-available persisted/in-memory view session for one markdown path."""
        if not path_key:
            return None

        if path_key == self._current_preview_path_key():
            self._save_document_view_session(path_key, capture_current=True)
        self._load_persisted_document_view_session(path_key)
        session = self._document_view_sessions.get(path_key)
        if isinstance(session, dict):
            cloned_session = self._clone_json_compatible_dict(session)
            if cloned_session:
                return cloned_session

        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return None
        directory, file_name = resolved
        persisted = self._directory_view_sessions(directory).get(file_name)
        if not isinstance(persisted, dict):
            return None
        cloned_session = self._clone_json_compatible_dict(persisted)
        return cloned_session if cloned_session else None

    def _merge_copied_file_metadata(
        self, copied_pairs: list[tuple[Path, Path]], target_directory: Path
    ) -> tuple[int, int, int]:
        """Merge copied markdown metadata into destination .mdexplore sidecars."""
        if not copied_pairs:
            return 0, 0, 0

        color_updates = 0
        view_updates = 0
        highlight_updates = 0
        target_sessions = self._directory_view_sessions(target_directory)
        target_highlights = self._directory_text_highlights(target_directory)
        sessions_changed = False
        highlights_changed = False

        for source_path, destination_path in copied_pairs:
            source_color = self.model.color_for_file(source_path)
            normalized_color = (
                source_color.strip()
                if isinstance(source_color, str) and source_color.strip()
                else None
            )
            self.model.set_color_for_file(destination_path, normalized_color)
            if normalized_color is not None:
                color_updates += 1

            source_key = self._path_key(source_path)
            source_session = self._view_session_for_path_key(source_key)
            if self._should_persist_document_view_session(source_session):
                target_sessions[destination_path.name] = self._clone_json_compatible_dict(
                    source_session if isinstance(source_session, dict) else {}
                )
                sessions_changed = True
                view_updates += 1
            elif destination_path.name in target_sessions:
                target_sessions.pop(destination_path.name, None)
                sessions_changed = True

            source_highlights = self._load_text_highlights_for_path_key(source_key)
            if source_highlights:
                target_highlights[destination_path.name] = self._clone_json_compatible_list(
                    source_highlights
                )
                highlights_changed = True
                highlight_updates += 1
            elif destination_path.name in target_highlights:
                target_highlights.pop(destination_path.name, None)
                highlights_changed = True

        if sessions_changed:
            self._save_directory_view_sessions(target_directory)
        if highlights_changed:
            self._save_directory_text_highlights(target_directory)

        for _source_path, destination_path in copied_pairs:
            destination_key = self._path_key(destination_path)
            if self._is_path_key_under_root(destination_key):
                self._refresh_tree_multi_view_markers(changed_path_key=destination_key)

        return color_updates, view_updates, highlight_updates

    def _copy_files_to_directory_with_metadata(
        self, files: list[Path]
    ) -> tuple[int, int, int, int, int, Path] | None:
        """Copy files into a chosen folder and merge their .mdexplore metadata."""
        normalized = self._normalize_unique_file_paths(files)
        if not normalized:
            return None

        target_directory = self._prompt_copy_destination_directory()
        if target_directory is None:
            return None

        copied_pairs: list[tuple[Path, Path]] = []
        copy_failures = 0
        for source_path in normalized:
            if not source_path.is_file():
                copy_failures += 1
                continue

            destination_path = target_directory / source_path.name
            if self._path_key(source_path) == self._path_key(destination_path):
                copied_pairs.append((source_path, destination_path))
                continue
            try:
                shutil.copy2(source_path, destination_path)
                copied_pairs.append((source_path, destination_path))
            except Exception:
                copy_failures += 1

        color_updates, view_updates, highlight_updates = self._merge_copied_file_metadata(
            copied_pairs, target_directory
        )
        return (
            len(copied_pairs),
            copy_failures,
            color_updates,
            view_updates,
            highlight_updates,
            target_directory,
        )

    def _copy_current_preview_file_to_clipboard(self) -> None:
        """Copy the current markdown file to clipboard or selected folder."""
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

        if self._copy_destination_is_directory():
            result = self._copy_files_to_directory_with_metadata([target])
            if result is None:
                return
            copied, failed, colors, views, highlights, directory = result
            self.statusBar().showMessage(
                f"Copied {copied} file(s) to {directory} "
                f"(metadata: colors {colors}, views {views}, highlights {highlights}; "
                f"failures {failed})",
                5000,
            )
            return

        copied = self._copy_files_to_clipboard([target])
        if copied:
            self.statusBar().showMessage(
                f"Copied previewed file to clipboard: {target.name}", 4000
            )

    def _copy_highlighted_files_to_clipboard(
        self, color_value: str, color_name: str
    ) -> None:
        """Copy highlighted markdown files to clipboard or selected folder."""
        scope = self._highlight_scope_directory()
        matches = self.model.collect_files_with_color(scope, color_value)
        if self._copy_destination_is_directory():
            if not matches:
                self.statusBar().showMessage(
                    f"No {color_name.lower()} highlighted file(s) to copy",
                    3500,
                )
                return
            result = self._copy_files_to_directory_with_metadata(matches)
            if result is None:
                return
            copied, failed, colors, views, highlights, directory = result
            self.statusBar().showMessage(
                f"Copied {copied} {color_name.lower()} highlighted file(s) to {directory} "
                f"(metadata: colors {colors}, views {views}, highlights {highlights}; "
                f"failures {failed})",
                5500,
            )
            return

        copied = self._copy_files_to_clipboard(matches)
        self.statusBar().showMessage(
            f"Copied {copied} {color_name.lower()} highlighted file(s) from {scope}",
            4000,
        )

    def _on_tree_selection_changed(self, current, _previous) -> None:
        # Any selection transition means the user is leaving the previous
        # document/scope, so immediately clear restore UI state.
        self._stop_restore_overlay_monitor()
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
        render_metadata: dict[str, object] = {}
        if worker_to_remove is not None:
            render_metadata = dict(getattr(worker_to_remove, "render_metadata", {}) or {})
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
            html_doc = self._placeholder_html(
                f"Could not render preview for {self.current_file.name}: {error_text}"
            )
            self._set_preview_feature_flags(
                path_key,
                has_math=False,
                has_mermaid=False,
                has_plantuml=False,
            )
        else:
            total_lines = max(1, int(render_metadata.get("total_lines", 1)))
            self._document_line_counts[path_key] = total_lines
            self._current_document_total_lines = total_lines
            self._sync_all_view_tab_progress()
            self._set_preview_feature_flags(
                path_key,
                has_math=bool(render_metadata.get("has_math")),
                has_mermaid=bool(render_metadata.get("has_mermaid")),
                has_plantuml=bool(render_metadata.get("has_plantuml")),
            )

            previous_placeholders = self._plantuml_placeholders_by_doc.get(path_key, {})
            for hash_key in previous_placeholders:
                docs = self._plantuml_docs_by_hash.get(hash_key)
                if docs is not None:
                    docs.discard(path_key)
                    if not docs:
                        self._plantuml_docs_by_hash.pop(hash_key, None)

            placeholders_by_hash = render_metadata.get("placeholders_by_hash", {})
            if not isinstance(placeholders_by_hash, dict):
                placeholders_by_hash = {}
            self._plantuml_placeholders_by_doc[path_key] = placeholders_by_hash
            for hash_key in placeholders_by_hash:
                self._plantuml_docs_by_hash.setdefault(hash_key, set()).add(path_key)

            prepared_sources = render_metadata.get(
                "prepared_plantuml_sources_by_hash", {}
            )
            if not isinstance(prepared_sources, dict):
                prepared_sources = {}
            for hash_key, prepared_code in prepared_sources.items():
                if not isinstance(hash_key, str) or not isinstance(prepared_code, str):
                    continue
                status, _payload = self._plantuml_results.get(hash_key, ("pending", ""))
                if status not in {"done", "error"}:
                    self._ensure_plantuml_render_started(hash_key, prepared_code)

            self._merge_renderer_pdf_mermaid_cache_seed(
                render_metadata.get("pdf_mermaid_by_hash")
                if isinstance(render_metadata.get("pdf_mermaid_by_hash"), dict)
                else None
            )
            self.cache[path_key] = (mtime_ns, size, html_doc)
            self._set_current_preview_signature(path_key, int(mtime_ns), int(size))
            self.statusBar().showMessage(f"Preview rendered: {self.current_file.name}")

        try:
            base_url = QUrl.fromLocalFile(f"{self.current_file.parent.resolve()}/")
        except Exception:
            base_url = QUrl.fromLocalFile(f"{self.root}/")
        self._set_preview_html(
            self._inject_mermaid_cache_seed(html_doc, path_key), base_url
        )
        self._update_pdf_button_state()

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

    def _set_current_preview_signature(
        self, path_key: str, mtime_ns: int, size: int
    ) -> None:
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
        if (
            self._current_preview_signature_key != path_key
            or self._current_preview_signature is None
        ):
            self._set_current_preview_signature(
                path_key, current_sig[0], current_sig[1]
            )
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
        state = self._current_view_state()
        if isinstance(state, dict):
            try:
                scroll_y = float(state.get("scroll_y", 0.0))
            except Exception:
                scroll_y = 0.0
            if math.isfinite(scroll_y) and scroll_y > 0.5:
                return True
            try:
                top_line = int(state.get("top_line", 1))
            except Exception:
                top_line = 1
            if top_line > 1:
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
        self._view_line_probe_block_until = 0.0
        self._capture_current_preview_scroll(force=True)
        self._capture_current_diagram_view_state(expected_key)
        self._request_active_view_top_line_update(force=True)

    def _restore_current_preview_scroll(
        self, expected_key: str | None = None, *, stabilize: bool = True
    ) -> None:
        """Restore previously captured scroll position for the selected file."""
        path_key = self._current_preview_path_key()
        if path_key is None:
            return
        if expected_key is not None and path_key != expected_key:
            return
        scroll_key = self._current_preview_scroll_key()
        scroll_y = (
            self._preview_scroll_positions.get(scroll_key)
            if scroll_key is not None
            else None
        )
        top_line_fallback = 1
        if scroll_y is None:
            state = self._current_view_state()
            if state is not None:
                try:
                    scroll_y = float(state.get("scroll_y", 0.0))
                except Exception:
                    scroll_y = 0.0
                try:
                    top_line_fallback = max(1, int(state.get("top_line", 1)))
                except Exception:
                    top_line_fallback = 1
        else:
            state = self._current_view_state()
            if state is not None:
                try:
                    top_line_fallback = max(1, int(state.get("top_line", 1)))
                except Exception:
                    top_line_fallback = 1
        if scroll_y is None:
            # Backward compatibility with pre-view-tab scroll cache entries.
            scroll_y = self._preview_scroll_positions.get(path_key)
        if scroll_y is None:
            return
        try:
            scroll_value = float(scroll_y)
        except Exception:
            scroll_value = 0.0
        if not math.isfinite(scroll_value):
            scroll_value = 0.0

        scroll_json = json.dumps(scroll_value)
        top_line_json = json.dumps(int(max(1, top_line_fallback)))
        stabilize_hook = "setTimeout(maybeCorrectLineDrift, 140);" if stabilize else ""
        js = f"""
(() => {{
  const y = {scroll_json};
  const targetLine = {top_line_json};
  const probeTopLine = () => {{
    const topBandY = 12;
    const viewportHeight = Math.max(1, Number(window.innerHeight) || 0);
    const taggedNodes = Array.from(document.querySelectorAll("[data-md-line-start]"));
    let crossingLine = null;
    let crossingTop = -Infinity;
    let aboveLine = null;
    let aboveBottom = -Infinity;
    let belowLine = null;
    let belowTop = Infinity;
    for (const node of taggedNodes) {{
      const rawValue = parseInt(node.getAttribute("data-md-line-start") || "", 10);
      if (Number.isNaN(rawValue)) {{
        continue;
      }}
      const lineValue = rawValue + 1;
      const rect = node.getBoundingClientRect();
      if (!rect) {{
        continue;
      }}
      if (!Number.isFinite(rect.top) || !Number.isFinite(rect.bottom)) {{
        continue;
      }}
      if (rect.height <= 0) {{
        continue;
      }}
      if (rect.bottom <= 0 || rect.top >= viewportHeight) {{
        continue;
      }}
      if (rect.top <= topBandY && rect.bottom > topBandY) {{
        if (
          rect.top > crossingTop
          || (rect.top === crossingTop && (crossingLine === null || lineValue < crossingLine))
        ) {{
          crossingTop = rect.top;
          crossingLine = lineValue;
        }}
        continue;
      }}
      if (rect.bottom <= topBandY) {{
        if (
          rect.bottom > aboveBottom
          || (rect.bottom === aboveBottom && (aboveLine === null || lineValue > aboveLine))
        ) {{
          aboveBottom = rect.bottom;
          aboveLine = lineValue;
        }}
        continue;
      }}
      if (
        rect.top < belowTop
        || (rect.top === belowTop && (belowLine === null || lineValue < belowLine))
      ) {{
        belowTop = rect.top;
        belowLine = lineValue;
      }}
    }}
    if (crossingLine !== null) {{
      return crossingLine;
    }}
    if (aboveLine !== null) {{
      return aboveLine;
    }}
    if (belowLine !== null) {{
      return belowLine;
    }}
    return 1;
  }};
  // Apply twice (RAF + timeout) because late layout work can override scroll.
  const applyScrollY = () => {{
    requestAnimationFrame(() => window.scrollTo(0, y));
    setTimeout(() => window.scrollTo(0, y), 60);
  }};
  const applyLineFallback = () => {{
    if (targetLine <= 1) {{
      applyScrollY();
      return;
    }}
    const targetLineIndex = Math.max(0, targetLine - 1);
    let best = null;
    for (const node of Array.from(document.querySelectorAll("[data-md-line-start][data-md-line-end]"))) {{
      const start = Number(node.getAttribute("data-md-line-start"));
      const end = Number(node.getAttribute("data-md-line-end"));
      if (!Number.isFinite(start) || !Number.isFinite(end)) {{
        continue;
      }}
      if (targetLineIndex < start || targetLineIndex >= end) {{
        continue;
      }}
      best = node;
      break;
    }}
    if (!best) {{
      for (const node of Array.from(document.querySelectorAll("[data-md-line-start]"))) {{
        const start = Number(node.getAttribute("data-md-line-start"));
        if (!Number.isFinite(start)) {{
          continue;
        }}
        if (start >= targetLineIndex) {{
          best = node;
          break;
        }}
      }}
    }}
    if (!best) {{
      applyScrollY();
      return;
    }}
    const rect = best.getBoundingClientRect();
    const targetY = Math.max(0, window.scrollY + rect.top - 14);
    requestAnimationFrame(() => window.scrollTo(0, targetY));
    setTimeout(() => window.scrollTo(0, targetY), 60);
  }};
  const maybeCorrectLineDrift = () => {{
    if (targetLine <= 1) {{
      return;
    }}
    const currentTopLine = probeTopLine();
    if (!Number.isFinite(currentTopLine)) {{
      applyLineFallback();
      return;
    }}
    if (Math.abs(currentTopLine - targetLine) >= 3) {{
      applyLineFallback();
    }}
  }};
  if (targetLine > 1 && (!Number.isFinite(y) || y <= 1)) {{
    applyLineFallback();
    return;
  }}
  applyScrollY();
  __MDEXPLORE_LINE_STABILIZE_HOOK__
}})();
"""
        js = js.replace("__MDEXPLORE_LINE_STABILIZE_HOOK__", stabilize_hook)
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

    def _plantuml_block_html(
        self,
        placeholder_id: str,
        line_attrs: str,
        status: str,
        payload: str,
        hash_key: str | None = None,
    ) -> str:
        inner = (
            self._plantuml_inner_html(status, payload)
            if status in {"done", "error"}
            else "PlantUML rendering..."
        )
        classes = ["mdexplore-fence", "plantuml-async"]
        if status == "pending":
            classes.append("plantuml-pending")
        elif status == "error":
            classes.append("plantuml-error")
        else:
            classes.append("plantuml-ready")
        class_attr = " ".join(classes)
        hash_attr = ""
        if hash_key:
            safe_hash = html.escape(hash_key, quote=True)
            hash_attr = f' data-mdexplore-plantuml-hash="{safe_hash}"'
        return (
            f'<div class="{class_attr}" id="{placeholder_id}"{hash_attr}{line_attrs}>'
            f"{inner}"
            "</div>\n"
        )

    def _ensure_plantuml_render_started(
        self, hash_key: str, prepared_code: str
    ) -> None:
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

    def _on_plantuml_render_finished(
        self, hash_key: str, status: str, payload: str
    ) -> None:
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
            current_placeholders = self._plantuml_placeholders_by_doc.get(
                current_key, {}
            )
            if hash_key in current_placeholders:
                self._show_preview_progress_status()
                self._check_restore_overlay_progress()
                self._start_pdf_diagram_readiness_monitor(current_key)
        self._update_pdf_button_state()

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
  if (window.__mdexploreApplyPlantUmlZoomControls) {{
    window.__mdexploreApplyPlantUmlZoomControls("auto");
  }}
  for (const shell of Array.from(document.querySelectorAll(".mdexplore-mermaid-shell"))) {{
    const fn = shell && shell.__mdexploreReapplySavedState;
    if (typeof fn !== "function") {{
      continue;
    }}
    try {{
      fn();
    }} catch (_error) {{
      // Ignore per-shell restore failures.
    }}
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
        ready_hashes: list[str] = []
        for hash_key in placeholders_by_hash:
            status, _payload = self._plantuml_results.get(hash_key, ("pending", ""))
            if status in {"done", "error"}:
                ready_hashes.append(hash_key)

        if not ready_hashes:
            return

        batch_size = max(1, int(PLANTUML_RESTORE_BATCH_SIZE))

        def apply_batch(start_index: int) -> None:
            if self._current_preview_path_key() != current_key:
                return
            end_index = min(len(ready_hashes), start_index + batch_size)
            for hash_key in ready_hashes[start_index:end_index]:
                self._apply_plantuml_result_to_current_preview(hash_key)
            if end_index < len(ready_hashes):
                QTimer.singleShot(
                    0, lambda next_index=end_index: apply_batch(next_index)
                )

        apply_batch(0)

    def _load_preview(self, path: Path) -> None:
        # Render markdown quickly with async PlantUML placeholders so the
        # document appears immediately while diagrams render in background.
        self._stop_restore_overlay_monitor()
        previous_path_key = self._current_preview_path_key()
        next_path_key = self._path_key(path)
        self._capture_current_preview_scroll(force=True)
        if previous_path_key is not None and previous_path_key != next_path_key:
            # Best-effort capture only: file switching should stay responsive
            # even if the embedded page is busy finishing diagram work.
            if self._has_diagram_features_for_key(previous_path_key):
                self._capture_current_diagram_view_state_blocking(
                    previous_path_key, timeout_ms=90
                )
            self._persist_document_view_session(previous_path_key)
        self._cancel_pending_preview_render()
        self._preview_capture_enabled = False
        self._scroll_restore_block_until = 0.0
        self._view_line_probe_block_until = 0.0
        if previous_path_key != next_path_key:
            self._load_persisted_document_view_session(next_path_key)
            restored = self._restore_document_view_session(next_path_key)
            if not restored:
                self._reset_document_views(initial_scroll=0.0, initial_line=1)
        should_highlight_search = bool(
            self.match_input.text().strip()
        ) and self._is_path_in_current_matches(path)
        self._pending_preview_search_terms = (
            self._current_search_terms() if should_highlight_search else []
        )
        self._pending_preview_search_close_groups = (
            self._current_close_term_groups() if should_highlight_search else []
        )
        self.statusBar().showMessage(f"Loading preview content: {path.name}...")
        self._preview_load_in_progress = True
        self._pdf_diagram_ready_by_key[next_path_key] = False
        self._pdf_diagram_settle_deadline_by_key[next_path_key] = 0.0
        self._stop_pdf_diagram_readiness_monitor()

        self.current_file = path
        self._update_pdf_button_state()
        self._current_preview_text_highlights = self._load_text_highlights_for_path_key(
            next_path_key
        )
        # Explicitly clear any stale overlay at document entry before
        # considering whether the new document needs one.
        self._stop_restore_overlay_monitor()
        self._current_document_total_lines = max(
            1, int(self._document_line_counts.get(next_path_key, 1))
        )
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
            self._set_current_preview_signature(
                cache_key, int(stat.st_mtime_ns), int(stat.st_size)
            )
            cached = self.cache.get(cache_key)
            if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
                has_math, has_mermaid, has_plantuml = (
                    self._detect_special_features_for_path(
                        resolved,
                        cached_html=cached[2],
                    )
                )
                self._set_preview_feature_flags(
                    cache_key,
                    has_math=has_math,
                    has_mermaid=has_mermaid,
                    has_plantuml=has_plantuml,
                )
                self._begin_restore_overlay_monitor(
                    cache_key,
                    needs_math=has_math,
                    needs_mermaid=has_mermaid,
                    needs_plantuml=has_plantuml,
                    phase="restoring",
                )
                self.statusBar().showMessage(
                    f"Using cached preview: {resolved.name}..."
                )
                self._set_preview_html(
                    self._inject_mermaid_cache_seed(cached[2], cache_key), base_url
                )
                return

            self.statusBar().showMessage(
                f"Rendering markdown in background: {resolved.name}..."
            )
            request_id = self._render_request_id
            worker = PreviewRenderWorker(
                resolved,
                request_id,
                self._build_preview_render_payload,
            )
            self._active_render_workers.add(worker)
            worker.signals.finished.connect(self._on_preview_render_finished)
            self._render_pool.start(worker)
        except Exception as exc:
            self._stop_restore_overlay_monitor()
            self.statusBar().showMessage(f"Preview render failed: {exc}", 5000)
            self._set_preview_html(
                self._placeholder_html(
                    f"Could not render preview for {path.name}: {exc}"
                ),
                base_url,
            )

    def _refresh_current_preview(
        self, _checked: bool = False, *, reason: str | None = None
    ) -> None:
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
        self._update_pdf_button_state()

    def _prepare_preview_zoom_for_pdf_export(self) -> None:
        """Temporarily reset preview-wide zoom so PDF snapshots use 100% scale."""
        if self._pdf_export_saved_preview_zoom is not None:
            return
        current_zoom = float(self.preview.zoomFactor())
        self._pdf_export_saved_preview_zoom = current_zoom
        if abs(current_zoom - PREVIEW_ZOOM_RESET) > 1e-6:
            # Use the raw setter so export does not show the user-facing zoom
            # overlay or status message while preparing the PDF snapshot.
            self.preview.setZoomFactor(PREVIEW_ZOOM_RESET)

    def _restore_preview_zoom_after_pdf_export(self) -> None:
        """Restore the interactive preview zoom after PDF export completes."""
        saved_zoom = self._pdf_export_saved_preview_zoom
        self._pdf_export_saved_preview_zoom = None
        if saved_zoom is None:
            return
        if abs(float(self.preview.zoomFactor()) - float(saved_zoom)) <= 1e-6:
            return
        # Restore silently so PDF completion does not look like a user zoom action.
        self.preview.setZoomFactor(float(saved_zoom))

    def _export_current_preview_pdf(self) -> None:
        """Export the currently previewed markdown rendering to a numbered PDF."""
        if self.current_file is None:
            QMessageBox.information(
                self,
                "No file selected",
                "Select a markdown file before exporting to PDF.",
            )
            return
        if self._pdf_export_in_progress:
            self.statusBar().showMessage("PDF export already in progress", 3000)
            return

        try:
            source_path = self.current_file.resolve()
        except Exception:
            source_path = self.current_file
        source_key = str(source_path)
        output_path = source_path.with_suffix(".pdf")
        self._pdf_export_source_key = source_key
        self._pending_pdf_layout_hints = {}
        # Preserve current diagram zoom/pan before forcing PDF-safe rendering mode.
        self._capture_current_diagram_view_state_blocking(source_key, timeout_ms=500)
        self._capture_current_diagram_view_state(source_key)
        self._prepare_preview_zoom_for_pdf_export()

        self._set_pdf_export_busy(True)
        self.statusBar().showMessage(f"Preparing PDF for {source_path.name}...")
        self._prepare_preview_for_pdf_export(
            output_path, attempt=0, source_key=source_key
        )

    def _prepare_preview_for_pdf_export(
        self, output_path: Path, attempt: int, source_key: str
    ) -> None:
        """Wait for math/Mermaid/fonts readiness and inject print style before export."""
        # Keep PDF preflight policy sourced from Python constants so the
        # external JS remains a consumer of print-layout rules, not the owner.
        js = _render_js_asset(
            "pdf/precheck_layout.js",
            {
                "__MDEXPLORE_FORCE_MERMAID__": "true" if attempt == 0 else "false",
                "__MDEXPLORE_RESET_MERMAID__": "true" if attempt == 0 else "false",
                "__MDEXPLORE_PRINT_LAYOUT_KNOBS__": json.dumps(_pdf_print_layout_knobs()),
                "__MDEXPLORE_LANDSCAPE_PAGE_TOKEN_JSON__": json.dumps(
                    PDF_LANDSCAPE_PAGE_TOKEN
                ),
            },
        )
        self.preview.page().runJavaScript(
            js,
            lambda result, target=output_path, tries=attempt, key=source_key: self._on_pdf_precheck_result(
                target, tries, key, result
            ),
        )

    def _on_pdf_precheck_result(
        self, output_path: Path, attempt: int, source_key: str, result
    ) -> None:
        """Continue waiting until print assets are ready, then trigger print."""
        parsed_result = result
        if isinstance(result, str):
            result_text = result.strip()
            if result_text.startswith("{") and result_text.endswith("}"):
                try:
                    parsed_result = json.loads(result_text)
                except Exception:
                    parsed_result = result

        math_ready = False
        mermaid_ready = False
        plantuml_ready = False
        fonts_ready = False
        if isinstance(parsed_result, dict):
            math_ready = bool(parsed_result.get("mathReady"))
            mermaid_ready = bool(parsed_result.get("mermaidReady"))
            plantuml_ready = bool(parsed_result.get("plantumlReady"))
            fonts_ready = bool(parsed_result.get("fontsReady"))
            diagram_layout = parsed_result.get("diagramLayout")
            if isinstance(diagram_layout, dict):
                self._pending_pdf_layout_hints = dict(diagram_layout)

        if math_ready and mermaid_ready and plantuml_ready and fonts_ready:
            self._trigger_pdf_print(output_path, source_key)
            return

        if attempt < PDF_EXPORT_PRECHECK_MAX_ATTEMPTS:
            if attempt == 0:
                self.statusBar().showMessage(
                    "Waiting for math/Mermaid/PlantUML/fonts before PDF export..."
                )
            QTimer.singleShot(
                PDF_EXPORT_PRECHECK_INTERVAL_MS,
                lambda target=output_path, tries=attempt + 1, key=source_key: self._prepare_preview_for_pdf_export(
                    target, tries, key
                ),
            )
            return

        # Fall back instead of blocking export indefinitely if some assets never settle.
        self.statusBar().showMessage(
            "Proceeding with PDF export before all preview assets reported ready",
            3500,
        )
        self._trigger_pdf_print(output_path, source_key)

    def _trigger_pdf_print(self, output_path: Path, source_key: str) -> None:
        """Start Qt WebEngine PDF generation for the active preview page."""
        self.statusBar().showMessage(f"Rendering PDF snapshot: {output_path.name}...")
        preprint_js = _render_js_asset("pdf/preprint_normalize.js")

        def _print_after_dom_normalized(_result) -> None:
            try:
                # Give layout a brief turn after wrapper flattening before snapshot.
                QTimer.singleShot(
                    70,
                    lambda: self.preview.page().printToPdf(
                        lambda pdf_data, target=output_path, key=source_key: self._on_pdf_render_ready(
                            target, key, pdf_data
                        ),
                    ),
                )
            except Exception as exc:
                self._set_pdf_export_busy(False)
                self._restore_preview_mermaid_palette(source_key)
                self._restore_preview_zoom_after_pdf_export()
                error_text = self._truncate_error_text(str(exc), 500)
                QMessageBox.critical(
                    self,
                    "PDF export failed",
                    f"Could not start PDF rendering:\n{error_text}",
                )
                self.statusBar().showMessage(f"PDF export failed: {error_text}", 5000)

        self.preview.page().runJavaScript(preprint_js, _print_after_dom_normalized)

    def _restore_preview_mermaid_palette(self, source_key: str | None = None) -> None:
        """Switch Mermaid back to preview palette after PDF export attempts."""
        js = _render_js_asset("pdf/restore_preview_palette.js")
        self.preview.page().runJavaScript(js)
        if source_key:
            self._reapply_diagram_view_state_for(source_key)
            QTimer.singleShot(
                120, lambda key=source_key: self._reapply_diagram_view_state_for(key)
            )
            QTimer.singleShot(
                420, lambda key=source_key: self._reapply_diagram_view_state_for(key)
            )
            QTimer.singleShot(
                980, lambda key=source_key: self._reapply_diagram_view_state_for(key)
            )

    def _on_pdf_render_ready(
        self, output_path: Path, source_key: str, pdf_data
    ) -> None:
        """Receive raw PDF bytes from WebEngine and start footer stamping."""
        try:
            raw_pdf = bytes(pdf_data)
        except Exception:
            raw_pdf = b""

        if not raw_pdf:
            self._set_pdf_export_busy(False)
            self._restore_preview_mermaid_palette(source_key)
            self._restore_preview_zoom_after_pdf_export()
            message = "Qt WebEngine returned an empty PDF payload"
            QMessageBox.critical(self, "PDF export failed", message)
            self.statusBar().showMessage(f"PDF export failed: {message}", 5000)
            return

        layout_hints = dict(self._pending_pdf_layout_hints)
        self._pending_pdf_layout_hints = {}
        worker = PdfExportWorker(output_path, raw_pdf, layout_hints)
        self._active_pdf_workers.add(worker)
        worker.signals.finished.connect(
            lambda path_text, error_text, current_worker=worker, key=source_key: self._on_pdf_export_finished(
                current_worker,
                path_text,
                error_text,
                key,
            )
        )
        self._pdf_pool.start(worker)
        self.statusBar().showMessage(f"Writing numbered PDF: {output_path.name}...")

    def _on_pdf_export_finished(
        self,
        worker: PdfExportWorker,
        output_path_text: str,
        error_text: str,
        source_key: str,
    ) -> None:
        """Finalize async PDF export and report result."""
        self._active_pdf_workers.discard(worker)
        self._set_pdf_export_busy(False)
        self._restore_preview_mermaid_palette(source_key)
        self._restore_preview_zoom_after_pdf_export()
        self._pdf_export_source_key = None

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
            QMessageBox.information(
                self, "No file selected", "Select a markdown file before using Edit."
            )
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
    parser.add_argument(
        "--mermaid-backend",
        choices=[MERMAID_BACKEND_JS, MERMAID_BACKEND_RUST],
        default=MERMAID_BACKEND_JS,
        help="Mermaid render backend: 'js' (default) or 'rust' (requires mmdr).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging to mdexplore.log in the project directory.",
    )
    args = parser.parse_args()

    root = (
        Path(args.path).expanduser()
        if args.path is not None
        else _load_default_root_from_config()
    )
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
    # Probe after QApplication is initialized so the result reflects the active
    # Qt runtime/platform setup used by this process.
    gpu_context_available = _gpu_context_available()
    window = MdExploreWindow(
        root,
        app_icon,
        _config_file_path(),
        mermaid_backend=str(args.mermaid_backend or MERMAID_BACKEND_JS),
        gpu_context_available=gpu_context_available,
        debug_mode=bool(args.debug),
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
