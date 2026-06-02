"""Background workers used by pdfexplore."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal


class PdfSearchWorkerSignals(QObject):
    """Signals emitted by background PDF search workers."""

    finished = Signal(int, object, object, object, str)


class PdfSearchWorker(QRunnable):
    """Scan a visible-PDF chunk for query matches without blocking the UI thread."""

    def __init__(
        self,
        request_id: int,
        paths: list[Path],
        predicate,
        hit_counter,
        filename_patterns: list,
        content_loader,
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.paths = list(paths)
        self.predicate = predicate
        self.hit_counter = hit_counter
        self.filename_patterns = list(filename_patterns)
        self.content_loader = content_loader
        self.signals = PdfSearchWorkerSignals()

    def run(self) -> None:
        try:
            matched_paths: list[str] = []
            match_counts: dict[str, int] = {}
            filename_match_paths: list[str] = []

            for path in self.paths:
                try:
                    resolved = path.resolve()
                except Exception:
                    resolved = path
                path_key = str(resolved)
                filename_search_text = path.stem
                try:
                    searchable_content = self.content_loader(path)
                except Exception:
                    searchable_content = ""
                try:
                    if not self.predicate(filename_search_text, searchable_content):
                        continue
                except Exception:
                    continue

                matched_paths.append(path_key)
                if any(
                    pattern.search(filename_search_text)
                    for pattern in self.filename_patterns
                ):
                    filename_match_paths.append(path_key)
                try:
                    count = self.hit_counter(filename_search_text, searchable_content)
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


class PdfTreeMarkerScanWorkerSignals(QObject):
    """Signals emitted by background tree-sidecar scan workers."""

    finished = Signal(int, str, object, object, str)


class PdfTreeMarkerScanWorker(QRunnable):
    """Scan root sidecar files for PDF tree badges without blocking the UI thread."""

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
        self.signals = PdfTreeMarkerScanWorkerSignals()

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
                        if self._session_has_multiple_views(session):
                            multi_view_paths.add(str((directory / file_name).resolve()))

                if self.highlighting_file_name in filenames:
                    highlights_by_file = self._load_directory_text_highlights(
                        directory / self.highlighting_file_name
                    )
                    for file_name, entries in highlights_by_file.items():
                        if self._normalize_text_highlight_entries(entries):
                            highlighted_paths.add(str((directory / file_name).resolve()))

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
                if file_name.lower().endswith(".pdf"):
                    sessions[file_name] = raw_session
        return sessions

    @staticmethod
    def _load_directory_text_highlights(file_path: Path) -> dict[str, list[dict]]:
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
                if file_name.lower().endswith(".pdf") and isinstance(raw_entries, list):
                    highlights_by_file[file_name] = raw_entries
        return highlights_by_file

    @staticmethod
    def _session_has_multiple_views(session: dict | None) -> bool:
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

    @staticmethod
    def _normalize_text_highlight_entries(raw_entries) -> list[dict[str, int | str]]:
        normalized: list[dict[str, int | str]] = []
        if not isinstance(raw_entries, list):
            return normalized
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            try:
                page = int(item.get("page", 0))
                start = int(item.get("start", -1))
                end = int(item.get("end", -1))
            except Exception:
                continue
            raw_id = str(item.get("id", "")).strip()
            if page <= 0 or start < 0 or end <= start or not raw_id:
                continue
            kind = str(item.get("kind", "normal")).strip().lower()
            if kind not in {"normal", "important"}:
                kind = "normal"
            text = str(item.get("text", ""))
            normalized.append(
                {
                    "id": raw_id,
                    "page": page,
                    "start": start,
                    "end": end,
                    "kind": kind,
                    "text": text,
                }
            )
        normalized.sort(key=lambda item: (int(item["page"]), int(item["start"]), int(item["end"])))
        return normalized
