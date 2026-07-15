"""Background worker jobs used by pdfexplore.

This module intentionally keeps worker classes small and explicit because they
run on thread-pool threads and communicate back to the GUI thread only through
Qt signals.

Design notes:
- Workers never touch widgets directly.
- Inputs are copied into worker-owned fields at construction time.
- All worker `run` methods are defensive: failures are reported through
    `finished` signals instead of raising into Qt internals.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat as stat_module

from PySide6.QtCore import QObject, QRunnable, Signal

from mdexplore_app.file_coordination import advisory_file_lock


BACKGROUND_LEASE_BUSY = "background-lease-busy"


class GlobalActivityHeartbeatWorkerSignals(QObject):
    """Signals emitted after one shared activity-stamp write attempt."""

    finished = Signal(int)


class GlobalActivityHeartbeatWorker(QRunnable):
    """Touch the multi-instance activity stamp without blocking the GUI."""

    def __init__(self, request_id: int, stamp_path: Path) -> None:
        """Capture one immutable heartbeat write request."""
        super().__init__()
        self.request_id = int(request_id)
        self.stamp_path = Path(stamp_path)
        self.signals = GlobalActivityHeartbeatWorkerSignals()
        self.setAutoDelete(False)

    def run(self) -> None:
        """Create/touch the stamp best-effort, then always report completion."""
        try:
            self.stamp_path.parent.mkdir(parents=True, exist_ok=True)
            self.stamp_path.touch(exist_ok=True)
        except OSError:
            pass
        finally:
            try:
                self.signals.finished.emit(self.request_id)
            except RuntimeError:
                pass


class GlobalActivityProbeWorkerSignals(QObject):
    """Signals emitted after one shared activity-stamp read attempt."""

    finished = Signal(int, object)


class GlobalActivityProbeWorker(QRunnable):
    """Read the multi-instance activity stamp without blocking the GUI."""

    def __init__(self, request_id: int, stamp_path: Path) -> None:
        """Capture one immutable heartbeat read request."""
        super().__init__()
        self.request_id = int(request_id)
        self.stamp_path = Path(stamp_path)
        self.signals = GlobalActivityProbeWorkerSignals()
        self.setAutoDelete(False)

    def run(self) -> None:
        """Read the stamp mtime best-effort, then always report completion."""
        observed_wall_time: float | None = None
        try:
            observed_wall_time = float(self.stamp_path.stat().st_mtime)
        except OSError:
            pass
        finally:
            try:
                self.signals.finished.emit(
                    self.request_id,
                    observed_wall_time,
                )
            except RuntimeError:
                pass


class PdfSearchWorkerSignals(QObject):
    """Signals emitted by PDF content-search workers.

    Signal payload:
    - request id
    - list of matched path keys
    - map of path key -> hit count
    - list of path keys with filename-term hits
    - error text (empty on success)
    """

    finished = Signal(int, object, object, object, str)


class PdfSearchWorker(QRunnable):
    """Scan one chunk of visible PDFs for query matches.

    The main window creates one or more instances of this worker per active
    search request. The worker uses precompiled predicate/counter callables to
    keep the per-file loop lightweight.
    """

    def __init__(
        self,
        request_id: int,
        paths: list[Path],
        predicate,
        hit_counter,
        filename_patterns: list,
        content_loader,
        should_abort=None,
    ) -> None:
        """Store immutable worker inputs used by `run`.

        `should_abort` is an optional callback checked between expensive steps
        so stale search requests can stop quickly.
        """
        super().__init__()
        self.request_id = request_id
        self.paths = list(paths)
        self.predicate = predicate
        self.hit_counter = hit_counter
        self.filename_patterns = list(filename_patterns)
        self.content_loader = content_loader
        self.should_abort = should_abort
        self.signals = PdfSearchWorkerSignals()

    def run(self) -> None:
        """Execute chunk search and emit aggregated results.

        The worker resolves file keys as strings so cross-thread payloads remain
        JSON-like and easy to merge on the GUI side.
        """
        try:
            matched_paths: list[str] = []
            match_counts: dict[str, int] = {}
            filename_match_paths: list[str] = []

            for path in self.paths:
                # Respect cancellation before doing any file work.
                if callable(self.should_abort) and self.should_abort():
                    break
                try:
                    resolved = path.resolve()
                except Exception:
                    resolved = path
                path_key = str(resolved)
                filename_search_text = path.stem
                try:
                    # content_loader owns cache strategy (memory/disk/extract).
                    searchable_content = self.content_loader(path)
                except Exception:
                    searchable_content = ""
                # Finish the already-loaded file before honoring cancellation.
                # Predicate/count work is lightweight, and publishing this
                # completed result preserves progressive-search continuity.
                abort_after_current = bool(
                    callable(self.should_abort) and self.should_abort()
                )
                try:
                    if not self.predicate(filename_search_text, searchable_content):
                        if abort_after_current:
                            break
                        continue
                except Exception:
                    if abort_after_current:
                        break
                    continue

                matched_paths.append(path_key)
                if any(
                    pattern.search(filename_search_text)
                    for pattern in self.filename_patterns
                ):
                    filename_match_paths.append(path_key)
                try:
                    # Count computation is isolated so a single failure does not
                    # invalidate the overall search chunk.
                    count = self.hit_counter(filename_search_text, searchable_content)
                except Exception:
                    count = 1
                match_counts[path_key] = count if count > 0 else 1
                if abort_after_current:
                    break

            self.signals.finished.emit(
                self.request_id,
                matched_paths,
                match_counts,
                filename_match_paths,
                "",
            )
        except Exception as exc:
            self.signals.finished.emit(self.request_id, [], {}, [], str(exc))


class PdfTextPrefetchWorkerSignals(QObject):
    """Signals emitted by low-priority text-prefetch workers.

    Signal payload:
    - request id
    - successfully prefetched file count
    - skipped/error file count
    - error text (empty on success)
    """

    finished = Signal(int, int, int, str)


class PdfTextPrefetchWorker(QRunnable):
    """Warm shared PDF text caches for current scope in the background."""

    def __init__(
        self,
        request_id: int,
        paths: list[Path],
        content_loader,
        should_abort=None,
        lease_lock_path: Path | None = None,
    ) -> None:
        """Store immutable prefetch inputs used by `run`."""
        super().__init__()
        self.request_id = request_id
        self.paths = list(paths)
        self.content_loader = content_loader
        self.should_abort = should_abort
        self.lease_lock_path = (
            Path(lease_lock_path) if lease_lock_path is not None else None
        )
        self.signals = PdfTextPrefetchWorkerSignals()

    def run(self) -> None:
        """Execute best-effort cache warming for each candidate path."""
        result: tuple[int, int, str] = (0, 0, "")
        try:
            if self.lease_lock_path is not None:
                with advisory_file_lock(
                    self.lease_lock_path,
                    exclusive=True,
                    blocking=False,
                ) as acquired:
                    if not acquired:
                        result = (0, 0, BACKGROUND_LEASE_BUSY)
                    else:
                        result = self._run_paths()
            else:
                result = self._run_paths()
        except Exception as exc:
            # A lease-path mkdir/open/flock failure must not strand the owning
            # window's active-worker bookkeeping.  Report it like any other
            # worker failure and let the GUI decide when to retry.
            result = (0, 0, str(exc))
        self._emit_finished(*result)

    def _emit_finished(
        self,
        prefetched_count: int,
        skipped_count: int,
        error_text: str,
    ) -> None:
        """Emit only after any cross-process background lease is released."""
        try:
            self.signals.finished.emit(
                self.request_id,
                prefetched_count,
                skipped_count,
                error_text,
            )
        except RuntimeError:
            pass

    def _run_paths(self) -> tuple[int, int, str]:
        """Warm candidates while the optional cross-process lease is held."""
        try:
            prefetched_count = 0
            skipped_count = 0
            for path in self.paths:
                # Abort checks are intentionally frequent to keep UI response
                # predictable when user interaction pauses prefetch.
                if callable(self.should_abort) and self.should_abort():
                    break
                try:
                    self.content_loader(path)
                    prefetched_count += 1
                except Exception:
                    skipped_count += 1
                if callable(self.should_abort) and self.should_abort():
                    break
            return prefetched_count, skipped_count, ""
        except Exception as exc:
            return 0, 0, str(exc)


class PdfTextCacheGcWorkerSignals(QObject):
    """Signals emitted by idle extracted-text cache garbage collection."""

    finished = Signal(int, object, str)


class PdfTextCacheGcWorker(QRunnable):
    """Find and optionally apply bounded missing-source cache cleanup.

    Filesystem checks always run here. The optional payload applier lets the
    owning window perform guarded cache deletion under its locks on this same
    background thread, leaving only badge publication for the GUI thread.
    """

    METADATA_SUFFIX = ".txt.gz.meta.json"

    def __init__(
        self,
        request_id: int,
        memory_path_keys: list[str],
        disk_cache_dir: Path,
        *,
        memory_cursor: str = "",
        disk_cursor: str = "",
        batch_size: int = 128,
        should_abort=None,
        payload_applier=None,
        lease_lock_path: Path | None = None,
    ) -> None:
        """Store one bounded garbage-collection pass configuration."""
        super().__init__()
        self.request_id = int(request_id)
        self.memory_path_keys = [str(value) for value in memory_path_keys if value]
        self.disk_cache_dir = Path(disk_cache_dir)
        self.memory_cursor = str(memory_cursor or "")
        self.disk_cursor = str(disk_cursor or "")
        self.batch_size = max(1, int(batch_size))
        self.should_abort = should_abort
        self.payload_applier = payload_applier
        self.lease_lock_path = (
            Path(lease_lock_path) if lease_lock_path is not None else None
        )
        self.signals = PdfTextCacheGcWorkerSignals()
        self.setAutoDelete(False)

    @staticmethod
    def _rotating_batch(
        values: list[str],
        cursor: str,
        batch_size: int,
    ) -> tuple[list[str], str]:
        """Return a sorted batch beginning immediately after `cursor`."""
        ordered = sorted(set(values))
        if not ordered:
            return [], ""
        start = 0
        if cursor:
            for index, value in enumerate(ordered):
                if value > cursor:
                    start = index
                    break
            else:
                start = 0
        selected = ordered[start : start + max(1, int(batch_size))]
        if not selected:
            selected = ordered[: max(1, int(batch_size))]
        return selected, selected[-1] if selected else ""

    @staticmethod
    def _source_is_missing(path_text: str) -> bool:
        """Return true only for a definite missing-source result."""
        try:
            source_stat = Path(path_text).stat()
        except FileNotFoundError:
            return True
        except OSError:
            # Permission and transient filesystem failures must not evict data.
            return False
        return not stat_module.S_ISREG(source_stat.st_mode)

    def _aborted(self) -> bool:
        return bool(callable(self.should_abort) and self.should_abort())

    def _emit_finished(self, payload: object, error_text: str) -> None:
        """Emit unless the owning Qt objects disappeared during shutdown."""
        try:
            self.signals.finished.emit(self.request_id, payload, error_text)
        except RuntimeError:
            pass

    def run(self) -> None:
        """Inspect a bounded memory/disk batch and emit deletion candidates."""
        result: tuple[object, str] = ({}, "")
        try:
            if self.lease_lock_path is not None:
                with advisory_file_lock(
                    self.lease_lock_path,
                    exclusive=True,
                    blocking=False,
                ) as acquired:
                    if not acquired:
                        result = ({}, BACKGROUND_LEASE_BUSY)
                    else:
                        result = self._run_with_lease()
            else:
                result = self._run_with_lease()
        except Exception as exc:
            # Always publish completion even when entering or leaving the
            # optional cross-process lease fails.
            result = ({}, str(exc))
        self._emit_finished(*result)

    def _run_with_lease(self) -> tuple[object, str]:
        """Inspect and apply one batch while owning the maintenance lease."""
        payload = {
            "missing_memory_path_keys": [],
            "missing_disk_entries": [],
            "stale_metadata_paths": [],
            "memory_cursor": self.memory_cursor,
            "disk_cursor": self.disk_cursor,
        }
        try:
            if self._aborted():
                return payload, ""
            memory_batch, _memory_batch_end = self._rotating_batch(
                self.memory_path_keys,
                self.memory_cursor,
                self.batch_size,
            )
            for path_key in memory_batch:
                if self._aborted():
                    break
                if self._source_is_missing(path_key):
                    payload["missing_memory_path_keys"].append(path_key)
                payload["memory_cursor"] = path_key

            if self._aborted():
                return payload, ""
            metadata_names: list[str] = []
            try:
                with os.scandir(self.disk_cache_dir) as entries:
                    for entry in entries:
                        if self._aborted():
                            return payload, ""
                        if entry.name.endswith(self.METADATA_SUFFIX):
                            metadata_names.append(entry.name)
            except FileNotFoundError:
                metadata_names = []
            except OSError:
                metadata_names = []
            if self._aborted():
                return payload, ""
            disk_batch, _disk_batch_end = self._rotating_batch(
                metadata_names,
                self.disk_cursor,
                self.batch_size,
            )
            if self._aborted():
                return payload, ""

            for metadata_name in disk_batch:
                if self._aborted():
                    break
                # Once this entry starts, it is handled synchronously without
                # another cancellation boundary, so advancing here records
                # only work actually attempted rather than the batch tail.
                payload["disk_cursor"] = metadata_name
                metadata_path = self.disk_cache_dir / metadata_name
                cache_name = metadata_name[: -len(".meta.json")]
                cache_path = self.disk_cache_dir / cache_name
                try:
                    cache_stat = cache_path.stat()
                except FileNotFoundError:
                    if metadata_path.is_file():
                        payload["stale_metadata_paths"].append(str(metadata_path))
                    continue
                except OSError:
                    continue
                if not stat_module.S_ISREG(cache_stat.st_mode):
                    payload["stale_metadata_paths"].append(str(metadata_path))
                    continue
                try:
                    raw_text = metadata_path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    continue
                except OSError:
                    # Permission/transient I/O errors are not evidence that a
                    # valid companion is stale.
                    continue
                except UnicodeError:
                    payload["stale_metadata_paths"].append(str(metadata_path))
                    continue
                try:
                    raw_payload = json.loads(raw_text)
                except (json.JSONDecodeError, TypeError, ValueError):
                    payload["stale_metadata_paths"].append(str(metadata_path))
                    continue
                source_path = str(
                    raw_payload.get("source_path", "")
                    if isinstance(raw_payload, dict)
                    else ""
                ).strip()
                if not source_path or not Path(source_path).is_absolute():
                    payload["stale_metadata_paths"].append(str(metadata_path))
                    continue
                if self._source_is_missing(source_path):
                    payload["missing_disk_entries"].append(
                        {
                            "source_path": source_path,
                            "cache_path": str(cache_path),
                            "metadata_path": str(metadata_path),
                            "cache_identity": [
                                int(cache_stat.st_dev),
                                int(cache_stat.st_ino),
                                int(cache_stat.st_mtime_ns),
                                int(cache_stat.st_size),
                            ],
                        }
                    )
            if not self._aborted() and callable(self.payload_applier):
                removed_memory, removed_disk = self.payload_applier(
                    payload,
                    self._aborted,
                )
                payload["removed_memory_entries"] = int(removed_memory)
                payload["removed_disk_entries"] = int(removed_disk)
            return payload, ""
        except Exception as exc:
            return payload, str(exc)


class PdfTreeMarkerScanWorkerSignals(QObject):
    """Signals emitted by background sidecar marker scans.

    Signal payload:
    - request id
    - resolved root key
    - set of file path keys with multi-view marker
    - set of file path keys with persistent-highlight marker
    - error text (empty on success)
    """

    finished = Signal(int, str, object, object, str)


class PdfTreeMarkerScanWorker(QRunnable):
    """Scan root sidecars for tree badges without blocking the GUI thread."""

    def __init__(
        self,
        root: Path,
        request_id: int,
        views_file_name: str,
        highlighting_file_name: str,
    ) -> None:
        """Capture scan configuration for one request id."""
        super().__init__()
        self.root = root
        self.request_id = request_id
        self.views_file_name = views_file_name
        self.highlighting_file_name = highlighting_file_name
        self.signals = PdfTreeMarkerScanWorkerSignals()

    def run(self) -> None:
        """Walk the root tree and collect marker path sets from sidecars."""
        try:
            resolved_root = self.root.resolve()
            root_key = str(resolved_root)
            multi_view_paths: set[str] = set()
            highlighted_paths: set[str] = set()

            def on_walk_error(_err) -> None:
                """Handle walk error."""
                return

            for dirpath, _dirnames, filenames in os.walk(
                resolved_root, onerror=on_walk_error, followlinks=False
            ):
                directory = Path(dirpath)

                if self.views_file_name in filenames:
                    # View sidecar contributes multi-view badge state.
                    sessions = self._load_directory_view_sessions(
                        directory / self.views_file_name
                    )
                    for file_name, session in sessions.items():
                        if self._session_has_multiple_views(session):
                            multi_view_paths.add(str((directory / file_name).resolve()))

                if self.highlighting_file_name in filenames:
                    # Highlighting sidecar contributes persistent-highlight badge state.
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
        """Load one views sidecar into `{pdf_file_name: session}` form."""
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
        """Load one highlighting sidecar into `{pdf_file_name: entries}` form."""
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
        """Return whether a session should display the multi-view marker."""
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
        """Normalize raw highlight entries and drop malformed records.

        This mirrors app-side validation so worker-side marker detection remains
        resilient to partial/corrupt sidecars.
        """
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
