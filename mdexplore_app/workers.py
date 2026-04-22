"""Background worker objects used by mdexplore."""

from __future__ import annotations

import base64
from collections import OrderedDict
import json
import os
import subprocess
from pathlib import Path
from threading import Lock
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, Signal

from .pdf import extract_plantuml_error_details, stamp_pdf_page_numbers
from . import search as search_query


_SEARCH_CONTENT_CACHE_MAX_ENTRIES = 512
_SEARCH_CONTENT_CACHE_MAX_CHARS = 64 * 1024 * 1024
_SEARCH_CONTENT_CACHE_LOCK = Lock()
_SEARCH_CONTENT_CACHE: OrderedDict[str, tuple[int, int, str]] = OrderedDict()
_SEARCH_CONTENT_CACHE_TOTAL_CHARS = 0


def _lookup_cached_search_content(
    path_key: str, mtime_ns: int, size: int
) -> str | None:
    with _SEARCH_CONTENT_CACHE_LOCK:
        cached = _SEARCH_CONTENT_CACHE.get(path_key)
        if cached is None:
            return None
        cached_mtime_ns, cached_size, cached_text = cached
        if cached_mtime_ns != mtime_ns or cached_size != size:
            return None
        _SEARCH_CONTENT_CACHE.move_to_end(path_key)
        return cached_text


def _store_cached_search_content(
    path_key: str, mtime_ns: int, size: int, searchable_text: str
) -> None:
    global _SEARCH_CONTENT_CACHE_TOTAL_CHARS
    with _SEARCH_CONTENT_CACHE_LOCK:
        previous = _SEARCH_CONTENT_CACHE.pop(path_key, None)
        if previous is not None:
            _SEARCH_CONTENT_CACHE_TOTAL_CHARS -= len(previous[2])
        _SEARCH_CONTENT_CACHE[path_key] = (mtime_ns, size, searchable_text)
        _SEARCH_CONTENT_CACHE_TOTAL_CHARS += len(searchable_text)
        while (
            len(_SEARCH_CONTENT_CACHE) > _SEARCH_CONTENT_CACHE_MAX_ENTRIES
            or _SEARCH_CONTENT_CACHE_TOTAL_CHARS > _SEARCH_CONTENT_CACHE_MAX_CHARS
        ):
            _evicted_key, evicted = _SEARCH_CONTENT_CACHE.popitem(last=False)
            _SEARCH_CONTENT_CACHE_TOTAL_CHARS -= len(evicted[2])


def _load_searchable_markdown_content(path: Path) -> tuple[str, str]:
    """Return `(path_key, searchable_content)` for one markdown file."""
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    path_key = str(resolved)

    mtime_ns: int | None = None
    size: int | None = None
    try:
        stat = resolved.stat()
        mtime_ns = int(stat.st_mtime_ns)
        size = int(stat.st_size)
    except Exception:
        stat = None

    if mtime_ns is not None and size is not None:
        cached = _lookup_cached_search_content(path_key, mtime_ns, size)
        if cached is not None:
            return path_key, cached

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except Exception:
        content = ""
    searchable_content = search_query.strip_inline_image_data_uris(
        content,
        preserve_line_structure=False,
    )

    if mtime_ns is not None and size is not None:
        _store_cached_search_content(path_key, mtime_ns, size, searchable_content)
    return path_key, searchable_content


class PreviewRenderWorkerSignals(QObject):
    """Signals emitted by background preview rendering workers."""

    finished = Signal(int, str, str, object, object, str)


class PreviewRenderWorker(QRunnable):
    """Render markdown HTML in a worker thread to keep UI responsive."""

    def __init__(
        self,
        path: Path,
        request_id: int,
        render_callback: Callable[[Path], tuple[str, int, int]],
    ):
        super().__init__()
        self.path = path
        self.request_id = request_id
        self.render_callback = render_callback
        self.signals = PreviewRenderWorkerSignals()
        self.render_metadata: dict[str, object] = {}

    def run(self) -> None:
        try:
            resolved = self.path.resolve()
            result = self.render_callback(resolved)
            if isinstance(result, tuple) and len(result) == 4:
                html_doc, mtime_ns, size, metadata = result
                self.render_metadata = (
                    metadata if isinstance(metadata, dict) else {}
                )
            else:
                html_doc, mtime_ns, size = result
                self.render_metadata = {}
            self.signals.finished.emit(
                self.request_id,
                str(resolved),
                html_doc,
                mtime_ns,
                size,
                "",
            )
        except Exception as exc:
            self.signals.finished.emit(self.request_id, str(self.path), "", 0, 0, str(exc))


class PlantUmlRenderWorkerSignals(QObject):
    """Signals emitted by background PlantUML render workers."""

    finished = Signal(str, str, str)


class PlantUmlRenderWorker(QRunnable):
    """Render one PlantUML source block to SVG data URI in background."""

    def __init__(
        self,
        hash_key: str,
        prepared_code: str,
        jar_path: Path | None,
        setup_issue: str | None,
    ):
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
            self.signals.finished.emit(
                self.hash_key, "error", "plantuml.jar not available"
            )
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
            result = subprocess.run(
                command,
                input=self.prepared_code,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            self.signals.finished.emit(
                self.hash_key, "error", "Local PlantUML render timed out"
            )
            return
        except Exception as exc:
            self.signals.finished.emit(
                self.hash_key, "error", f"Local PlantUML render failed: {exc}"
            )
            return

        if result.returncode != 0:
            details = extract_plantuml_error_details(result.stderr or "")
            self.signals.finished.emit(
                self.hash_key, "error", f"Local PlantUML render failed: {details}"
            )
            return

        svg_text = (result.stdout or "").strip()
        if "<svg" not in svg_text.casefold():
            self.signals.finished.emit(
                self.hash_key, "error", "Local PlantUML did not return SVG output"
            )
            return

        encoded_svg = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
        data_uri = f"data:image/svg+xml;base64,{encoded_svg}"
        self.signals.finished.emit(self.hash_key, "done", data_uri)


class PdfExportWorkerSignals(QObject):
    """Signals emitted by background PDF export workers."""

    finished = Signal(str, str)


class PdfExportWorker(QRunnable):
    """Apply footer page numbers and write exported PDF in background."""

    def __init__(
        self,
        output_path: Path,
        pdf_bytes: bytes,
        layout_hints: dict[str, object] | None = None,
    ):
        super().__init__()
        self.output_path = output_path
        self.pdf_bytes = pdf_bytes
        self.layout_hints = layout_hints if isinstance(layout_hints, dict) else {}
        self.signals = PdfExportWorkerSignals()

    def run(self) -> None:
        try:
            stamped_pdf = stamp_pdf_page_numbers(self.pdf_bytes, self.layout_hints)
            self.output_path.write_bytes(stamped_pdf)
            self.signals.finished.emit(str(self.output_path), "")
        except Exception as exc:
            self.signals.finished.emit(str(self.output_path), str(exc))


def _session_has_multiple_views(session: object) -> bool:
    """Return whether a persisted view session represents more than one tab."""
    if not isinstance(session, dict):
        return False
    tabs = session.get("tabs")
    return isinstance(tabs, list) and len(tabs) > 1


def _normalize_text_highlight_entries(raw_entries: object) -> list[dict[str, int | str]]:
    """Sanitize persisted text-highlight ranges enough to detect live entries."""
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
        if start < 0 or end <= start:
            continue
        sanitized.append({"id": raw_id, "start": start, "end": end})
    return sanitized


class TreeMarkerScanWorkerSignals(QObject):
    """Signals emitted by background tree-sidecar scan workers."""

    finished = Signal(int, str, object, object, str)


class TreeMarkerScanWorker(QRunnable):
    """Scan root sidecar files for tree badges without blocking the UI thread."""

    def __init__(
        self,
        root: Path,
        request_id: int,
        views_file_name: str,
        highlighting_file_name: str,
    ) -> None:
        super().__init__()
        self.root = root
        self.request_id = request_id
        self.views_file_name = views_file_name
        self.highlighting_file_name = highlighting_file_name
        self.signals = TreeMarkerScanWorkerSignals()

    def run(self) -> None:
        try:
            resolved_root = self.root.resolve()
            root_key = str(resolved_root)
            multi_view_paths: set[str] = set()
            highlighted_paths: set[str] = set()

            def on_walk_error(_err) -> None:
                return

            for dirpath, _dirnames, filenames in os.walk(
                resolved_root, onerror=on_walk_error, followlinks=False
            ):
                directory = Path(dirpath)

                if self.views_file_name in filenames:
                    sessions = self._load_directory_view_sessions(
                        directory / self.views_file_name
                    )
                    for file_name, session in sessions.items():
                        if _session_has_multiple_views(session):
                            multi_view_paths.add(str((directory / file_name).resolve()))

                if self.highlighting_file_name in filenames:
                    highlights_by_file = self._load_directory_text_highlights(
                        directory / self.highlighting_file_name
                    )
                    for file_name, entries in highlights_by_file.items():
                        if _normalize_text_highlight_entries(entries):
                            highlighted_paths.add(
                                str((directory / file_name).resolve())
                            )

            self.signals.finished.emit(
                self.request_id,
                root_key,
                multi_view_paths,
                highlighted_paths,
                "",
            )
        except Exception as exc:
            self.signals.finished.emit(
                self.request_id,
                str(self.root),
                set(),
                set(),
                str(exc),
            )

    @staticmethod
    def _load_directory_view_sessions(file_path: Path) -> dict[str, dict]:
        """Load one directory's persisted view-session sidecar."""
        sessions: dict[str, dict] = {}
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
                if file_name.lower().endswith(".md"):
                    sessions[file_name] = raw_session
        return sessions

    @staticmethod
    def _load_directory_text_highlights(file_path: Path) -> dict[str, list[dict]]:
        """Load one directory's persisted preview-highlight sidecar."""
        highlights_by_file: dict[str, list[dict]] = {}
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
                if file_name.lower().endswith(".md") and isinstance(raw_entries, list):
                    highlights_by_file[file_name] = raw_entries
        return highlights_by_file


class SearchScanWorkerSignals(QObject):
    """Signals emitted by background markdown search workers."""

    finished = Signal(int, object, object, object, str)


class SearchScanWorker(QRunnable):
    """Scan a markdown-file chunk for query matches in background."""

    def __init__(self, request_id: int, query: str, paths: list[Path]) -> None:
        super().__init__()
        self.request_id = request_id
        self.query = query
        self.paths = list(paths)
        self.signals = SearchScanWorkerSignals()

    def run(self) -> None:
        try:
            predicate = search_query.compile_match_predicate(
                self.query,
                strip_inline_image_data=False,
            )
            hit_counter = search_query.compile_match_hit_counter(
                self.query,
                strip_inline_image_data=False,
            )
            filename_terms = search_query.extract_search_terms(self.query)
            filename_patterns = [
                search_query.compile_term_pattern(term_text, bool(is_case_sensitive))
                for term_text, is_case_sensitive in filename_terms
                if term_text
            ]

            matched_paths: list[str] = []
            match_counts: dict[str, int] = {}
            filename_match_paths: list[str] = []

            for path in self.paths:
                try:
                    path_key, searchable_content = _load_searchable_markdown_content(path)
                    if not predicate(path.name, searchable_content):
                        continue
                except Exception:
                    continue

                matched_paths.append(path_key)
                if any(pattern.search(path.name) for pattern in filename_patterns):
                    filename_match_paths.append(path_key)
                try:
                    count = hit_counter(path.name, searchable_content)
                except Exception:
                    count = 1
                match_counts[path_key] = count if count > 0 else 1

            self.signals.finished.emit(
                self.request_id,
                matched_paths,
                match_counts,
                filename_match_paths,
                "",
            )
        except Exception as exc:
            self.signals.finished.emit(self.request_id, [], {}, [], str(exc))
