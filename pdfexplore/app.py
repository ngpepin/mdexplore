#!/usr/bin/env python3
"""pdfexplore: read-only PDF browser with search and highlights."""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
import fcntl
import gzip
import hashlib
import json
import os
import re
import stat as stat_module
import subprocess
import sys
from threading import Lock
import time
import uuid
from pathlib import Path

from pypdf import PdfReader
from PySide6.QtCore import (
    QDir,
    QEvent,
    QEventLoop,
    QMimeData,
    QPoint,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
)
from PySide6.QtGui import QAction, QClipboard, QIcon, QKeyEvent, QKeySequence, QShortcut
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
    QStackedWidget,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from mdexplore_app.icons import build_clear_x_icon, ui_asset_path
from mdexplore_app.file_coordination import (
    advisory_file_lock,
    atomic_write_text,
    load_files_payload,
    transform_files_sidecar_entry,
    update_files_sidecar,
)
from mdexplore_app.constants import (
    PREVIEW_PERSISTENT_HIGHLIGHT_COLOR,
    PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_COLOR,
    PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR,
    PREVIEW_PERSISTENT_HIGHLIGHT_MARKER_COLOR,
    PREVIEW_ZOOM_MAX,
    PREVIEW_ZOOM_MIN,
    PREVIEW_ZOOM_OVERLAY_TIMEOUT_MS,
    PREVIEW_ZOOM_RESET,
    PREVIEW_ZOOM_STEP,
    SEARCH_CLOSE_WORD_GAP,
)
from mdexplore_app.runtime import (
    config_file_path as _shared_config_file_path,
    gpu_context_available,
)
from mdexplore_app.search import (
    compile_term_pattern,
    compile_match_hit_counter,
    compile_match_predicate,
    extract_near_term_groups,
    extract_search_terms,
)
from mdexplore_app.tabs import ViewTabBar

from .settings import APP_SETTINGS, VIEWER_BRIDGE_SETTINGS
from .tree import ColorizedPdfModel, PdfTreeItemDelegate
from .workers import (
    BACKGROUND_LEASE_BUSY,
    GlobalActivityHeartbeatWorker,
    GlobalActivityProbeWorker,
    PdfSearchWorker,
    PdfTextCacheGcWorker,
    PdfTextPrefetchWorker,
    PdfTreeMarkerScanWorker,
)


def _app_setting(name: str, default):
    return APP_SETTINGS.get(name, default)


def _normalize_highlight_colors(raw_value) -> list[tuple[str, str]]:
    if not isinstance(raw_value, list):
        return []
    normalized: list[tuple[str, str]] = []
    for entry in raw_value:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        value = str(entry.get("value") or "").strip()
        if not name or not value:
            continue
        normalized.append((name, value))
    return normalized


_DEFAULT_HIGHLIGHT_COLORS = [
    ("Yellow", "#f5d34f"),
    ("Green", "#78d389"),
    ("Blue", "#7bb9ff"),
    ("Orange", "#f6a05f"),
    ("Purple", "#bb9df5"),
    ("Light Gray", "#d1d5db"),
    ("Medium Gray", "#9ca3af"),
    ("Red", "#ef7d7d"),
]

_PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_FILE_NAME = str(_app_setting("config_file_name", ".pdfexplore.cfg"))
VIEWS_FILE_NAME = str(_app_setting("views_file_name", ".pdfexplore-views.json"))
HIGHLIGHTING_FILE_NAME = str(
    _app_setting("highlighting_file_name", ".pdfexplore-highlighting.json")
)
VIEWER_HTML = _PROJECT_DIR / str(
    _app_setting("viewer_html_relative", "vendor/pdfjs/web/viewer.html")
)
VIEWER_BRIDGE_JS = _PROJECT_DIR / str(
    _app_setting("viewer_bridge_js_relative", "assets/viewer_bridge.js")
)
OKULAR_EDIT_LAUNCHER = Path(
    str(
        _app_setting(
            "okular_edit_launcher",
            "/home/npepin/.local/share/applications-scripts/run-okular.sh",
        )
    )
)
VIEWER_ACTIVITY_CONSOLE_MESSAGE = "__pdfexplore_user_activity__"


class PdfPreviewPage(QWebEnginePage):
    """WebEngine page that forwards throttled viewer activity to the window."""

    def __init__(self, activity_handler, parent=None) -> None:
        """Initialize the page-level activity bridge."""
        super().__init__(parent)
        self._activity_handler = activity_handler

    def javaScriptConsoleMessage(  # noqa: N802
        self,
        level,
        message: str,
        line_number: int,
        source_id: str,
    ) -> None:
        """Consume the private activity sentinel emitted by the viewer bridge."""
        if str(message).strip() == VIEWER_ACTIVITY_CONSOLE_MESSAGE:
            try:
                if callable(self._activity_handler):
                    self._activity_handler()
            except Exception:
                pass
            return
        super().javaScriptConsoleMessage(level, message, line_number, source_id)


class PdfPreviewWebView(QWebEngineView):
    """WebEngine view that lets the app intercept hotkeys before pdf.js consumes them."""

    def __init__(self, key_handler, activity_handler, parent=None) -> None:
        """Initialize instance state."""
        super().__init__(parent)
        self._key_handler = key_handler
        self.setPage(PdfPreviewPage(activity_handler, self))

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Handle keyPressEvent."""
        try:
            if callable(self._key_handler) and self._key_handler(event):
                event.accept()
                return
        except Exception:
            pass
        super().keyPressEvent(event)


class PdfExploreWindow(QMainWindow):
    """Main window for browsing and highlighting PDFs."""

    MAX_DOCUMENT_VIEWS = int(_app_setting("max_document_views", 8))
    SPLITTER_TREE_DEFAULT_FRACTION = max(
        0.05,
        min(0.8, float(_app_setting("splitter_tree_default_fraction", 0.307))),
    )
    MAX_RECENT_ROOT_DIRECTORIES = int(_app_setting("max_recent_root_directories", 35))
    RECENT_ROOT_MENU_MRU_COUNT = int(_app_setting("recent_root_menu_mru_count", 10))
    MIN_RECENT_ROOT_DWELL_SECONDS = float(_app_setting("min_recent_root_dwell_seconds", 30.0))
    CONFIG_DEFAULT_ROOT_KEY = str(_app_setting("config_default_root_key", "default_root"))
    CONFIG_RECENT_ROOTS_KEY = str(_app_setting("config_recent_roots_key", "recent_roots"))
    PREVIEW_HIGHLIGHT_COLOR = PREVIEW_PERSISTENT_HIGHLIGHT_COLOR
    PREVIEW_HIGHLIGHT_IMPORTANT_COLOR = PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_COLOR
    PDF_TEXT_CACHE_MAX_ENTRIES = int(_app_setting("pdf_text_cache_max_entries", 384))
    PDF_TEXT_CACHE_MAX_CHARS = int(_app_setting("pdf_text_cache_max_chars", 96 * 1024 * 1024))
    PDF_TEXT_DISK_CACHE_MAX_FILES = int(_app_setting("pdf_text_disk_cache_max_files", 1200))
    PDF_TEXT_DISK_CACHE_MAX_BYTES = int(_app_setting("pdf_text_disk_cache_max_bytes", 384 * 1024 * 1024))
    PDF_TEXT_DISK_CACHE_TRIM_INTERVAL = int(_app_setting("pdf_text_disk_cache_trim_interval", 24))
    PDF_TEXT_DISK_CACHE_TOUCH_INTERVAL_SECONDS = float(
        _app_setting("pdf_text_disk_cache_touch_interval_seconds", 60.0)
    )
    PDF_TEXT_CACHE_GC_INTERVAL_SECONDS = float(
        _app_setting("pdf_text_cache_gc_interval_seconds", 30.0)
    )
    PDF_TEXT_CACHE_GC_IDLE_SECONDS = float(
        _app_setting("pdf_text_cache_gc_idle_seconds", 10.0)
    )
    PDF_TEXT_CACHE_GC_BATCH_SIZE = int(
        _app_setting("pdf_text_cache_gc_batch_size", 128)
    )
    PREFETCH_ENABLED = bool(_app_setting("prefetch_enabled", True))
    PREFETCH_BATCH_SIZE = int(_app_setting("prefetch_batch_size", 1))
    PREFETCH_IDLE_SECONDS = float(_app_setting("prefetch_idle_seconds", 10.0))
    PREFETCH_BATCH_COOLDOWN_SECONDS = float(
        _app_setting("prefetch_batch_cooldown_seconds", 1.0)
    )
    MULTI_INSTANCE_ACTIVITY_TOUCH_INTERVAL_SECONDS = float(
        _app_setting("multi_instance_activity_touch_interval_seconds", 0.25)
    )
    MULTI_INSTANCE_ACTIVITY_PROBE_INTERVAL_SECONDS = float(
        _app_setting("multi_instance_activity_probe_interval_seconds", 0.5)
    )
    PREFETCH_HEAVY_USE_WINDOW_SECONDS = float(_app_setting("prefetch_heavy_use_window_seconds", 1.2))
    PREFETCH_HEAVY_USE_EVENT_THRESHOLD = int(_app_setting("prefetch_heavy_use_event_threshold", 14))
    PREFETCH_HEAVY_USE_PAUSE_SECONDS = float(_app_setting("prefetch_heavy_use_pause_seconds", 1.2))
    PREFETCH_VIEWER_ACTIVITY_PAUSE_SECONDS = float(_app_setting("prefetch_viewer_activity_pause_seconds", 1.6))
    PREFETCH_TREE_MUTATION_PAUSE_SECONDS = float(_app_setting("prefetch_tree_mutation_pause_seconds", 0.85))
    TREE_SEARCH_REFRESH_DEBOUNCE_MS = int(_app_setting("tree_search_refresh_debounce_ms", 300))
    TREE_INTERACTION_VISUAL_RELAX_MS = int(_app_setting("tree_interaction_visual_relax_ms", 320))
    CACHED_BADGE_SYNC_INTERVAL_MS = int(_app_setting("cached_badge_sync_interval_ms", 200))
    CACHE_BADGE_PROBE_TIMER_INTERVAL_MS = int(
        _app_setting("cache_badge_probe_timer_interval_ms", 75)
    )
    MATCH_TIMER_INTERVAL_MS = int(_app_setting("match_timer_interval_ms", 320))
    SCOPE_PREFETCH_TIMER_INTERVAL_MS = int(_app_setting("scope_prefetch_timer_interval_ms", 550))
    VIEWER_READY_TIMER_INTERVAL_MS = int(_app_setting("viewer_ready_timer_interval_ms", 160))
    VIEW_STATE_POLL_TIMER_INTERVAL_MS = int(_app_setting("view_state_poll_timer_interval_ms", 900))
    FILE_CHANGE_WATCH_INTERVAL_MS = int(_app_setting("file_change_watch_interval_ms", 1200))
    SEARCH_THREAD_POOL_MAX_THREADS = int(_app_setting("search_thread_pool_max_threads", 1))
    SEARCH_WORKER_CHUNK_SIZE = int(_app_setting("search_worker_chunk_size", 8))
    SEARCH_PROGRESS_PUBLISH_INTERVAL_MS = int(
        _app_setting("search_progress_publish_interval_ms", 100)
    )
    PREFETCH_THREAD_POOL_MAX_THREADS = int(_app_setting("prefetch_thread_pool_max_threads", 1))
    PREVIEW_WIDGET_CACHE_MAX_ENTRIES = int(
        _app_setting("preview_widget_cache_max_entries", 2)
    )
    TREE_MARKER_SCAN_THREAD_POOL_MAX_THREADS = int(
        _app_setting("tree_marker_scan_thread_pool_max_threads", 1)
    )
    HIGHLIGHT_COLORS = _normalize_highlight_colors(
        _app_setting("highlight_colors", [])
    ) or list(_DEFAULT_HIGHLIGHT_COLORS)

    def __init__(
        self,
        root: Path,
        app_icon: QIcon,
        config_path: Path,
        *,
        gpu_context_available: bool = False,
        debug_mode: bool = False,
    ) -> None:
        """Initialize instance state."""
        super().__init__()
        self.root = root.resolve()
        self.config_path = config_path
        self.debug_mode = bool(debug_mode)
        self._recent_root_directories = self._load_recent_root_directories_from_config()
        self._recent_root_events: list[Path] = []
        self._recent_active_root = self.root
        self._recent_root_entered_at = time.monotonic()
        self.current_file: Path | None = None
        self.last_directory_selection: Path | None = self.root
        self.current_match_files: list[Path] = []
        self._current_match_counts: dict[Path, int] = {}
        self._pdf_text_cache: OrderedDict[str, tuple[int, int, str]] = OrderedDict()
        self._pdf_text_cache_total_chars = 0
        self._pdf_text_cache_lock = Lock()
        self._cached_pdf_path_keys: set[str] = set()
        self._cached_pdf_path_keys_lock = Lock()
        self._cached_pdf_path_keys_revision = 0
        self._cached_pdf_path_keys_synced_revision = -1
        self._pdf_text_disk_cache_lock = Lock()
        self._pdf_text_disk_cache_store_count = 0
        self._pdf_text_disk_cache_trim_requested_revision = 1
        self._pdf_text_disk_cache_trim_completed_revision = 0
        self._pdf_text_disk_cache_dir = self._build_pdf_text_disk_cache_dir()
        self._last_pdf_text_cache_gc_at = time.monotonic()
        self._pdf_text_cache_gc_memory_cursor = ""
        self._pdf_text_cache_gc_disk_cursor = ""
        self._active_pdf_text_cache_gc_worker: PdfTextCacheGcWorker | None = None
        self._prefetch_scope_key = ""
        self._prefetch_cursor = 0
        self._prefetch_last_prioritized_current_path = ""
        self._next_scope_prefetch_at = 0.0
        self._last_user_interaction_at = time.monotonic()
        self._last_global_activity_touch_at = 0.0
        self._global_activity_touch_pending = False
        self._global_activity_touch_force_pending = False
        self._global_activity_touch_request_id = 0
        self._active_global_activity_touch_worker: (
            GlobalActivityHeartbeatWorker | None
        ) = None
        self._last_observed_global_activity_wall_time = time.time()
        self._global_activity_probe_request_id = 0
        self._active_global_activity_probe_worker: (
            GlobalActivityProbeWorker | None
        ) = None
        self._recent_interaction_timestamps: deque[float] = deque(maxlen=96)
        self._prefetch_paused_until = 0.0
        self._persisted_view_sessions_by_dir: dict[str, dict[str, dict]] = {}
        self._document_view_sessions: dict[str, dict] = {}
        self._persisted_text_highlights_by_dir: dict[str, dict[str, list[dict]]] = {}
        self._current_text_highlights: list[dict[str, int | str]] = []
        self._copy_destination_directory: Path | None = None
        self._tree_multi_view_marker_paths: set[str] = set()
        self._tree_highlight_marker_paths: set[str] = set()
        self._active_view_tab_index = -1
        self._next_view_id = 1
        self._next_view_sequence = 1
        self._next_tab_color_index = 0
        self._pending_search_terms: list[tuple[str, bool]] = []
        self._pending_search_scroll_to_first_path_keys: set[str] = set()
        self._visible_tree_pdf_cache: list[Path] = []
        self._visible_tree_pdf_cache_dirty = True
        self._visible_tree_pdf_cache_fetch_more_complete = False
        viewer_bridge_payload = VIEWER_BRIDGE_JS.read_text(encoding="utf-8")
        viewer_bridge_settings = dict(VIEWER_BRIDGE_SETTINGS)
        viewer_bridge_settings.setdefault("near_word_gap", SEARCH_CLOSE_WORD_GAP)
        viewer_bridge_settings.setdefault(
            "persistent_highlight_marker_color",
            PREVIEW_PERSISTENT_HIGHLIGHT_MARKER_COLOR,
        )
        viewer_bridge_settings.setdefault(
            "persistent_highlight_important_marker_color",
            PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR,
        )
        self._viewer_bridge_source = (
            f"window.__pdfexploreBridgeConfig = {json.dumps(viewer_bridge_settings, ensure_ascii=False)};\n"
            + viewer_bridge_payload
        )
        self._preview_widgets_by_path: dict[str, QWebEngineView] = {}
        self._preview_widget_lru: OrderedDict[str, None] = OrderedDict()
        self._viewer_bridge_ready_by_path: dict[str, bool] = {}
        self._viewer_pending_restore_state_by_path: dict[str, dict | None] = {}
        self._preview_signatures_by_path: dict[str, tuple[int, int]] = {}
        self._preview_dark_mode = False
        self._gpu_context_available = bool(gpu_context_available)
        self._global_shortcuts: list[QShortcut] = []
        self._closing = False
        self._view_state_capture_in_flight = False
        self._view_state_capture_serial = 0

        self.thread_pool = QThreadPool(self)
        # pypdf text extraction is Python-heavy; keeping search concurrency low
        # avoids GIL thrash that can make the GUI feel blocked during searches.
        self.thread_pool.setMaxThreadCount(
            max(1, min(self.SEARCH_THREAD_POOL_MAX_THREADS, self.thread_pool.maxThreadCount()))
        )
        self._search_request_id = 0
        self._active_search_workers: set[PdfSearchWorker] = set()
        self._search_scan_expected_workers = 0
        self._search_scan_completed_workers = 0
        self._search_scan_total_candidates = 0
        self._search_scan_scope: Path | None = None
        self._search_scan_candidate_order: dict[str, int] = {}
        self._search_scan_match_counts: dict[str, int] = {}
        self._search_scan_filename_match_paths: set[str] = set()
        self._search_scan_error_count = 0
        self._prefetch_pool = QThreadPool(self)
        self._prefetch_pool.setMaxThreadCount(max(1, self.PREFETCH_THREAD_POOL_MAX_THREADS))
        self._prefetch_request_id = 0
        self._active_prefetch_workers: set[PdfTextPrefetchWorker] = set()
        self._cache_badge_probe_pool = QThreadPool(self)
        self._cache_badge_probe_pool.setMaxThreadCount(1)
        self._cache_badge_probe_request_id = 0
        self._active_cache_badge_probe_workers: set[PdfTextPrefetchWorker] = set()
        self._global_activity_touch_pool = QThreadPool(self)
        self._global_activity_touch_pool.setMaxThreadCount(1)
        self._global_activity_probe_pool = QThreadPool(self)
        self._global_activity_probe_pool.setMaxThreadCount(1)
        self._tree_marker_scan_pool = QThreadPool(self)
        self._tree_marker_scan_pool.setMaxThreadCount(
            max(1, self.TREE_MARKER_SCAN_THREAD_POOL_MAX_THREADS)
        )
        self._tree_marker_scan_request_id = 0
        self._active_tree_marker_scan_workers: set[PdfTreeMarkerScanWorker] = set()
        self._tree_marker_cache_root_key: str | None = None

        self.match_timer = QTimer(self)
        self.match_timer.setInterval(int(self.MATCH_TIMER_INTERVAL_MS))
        self.match_timer.setSingleShot(True)
        self.match_timer.timeout.connect(self._run_match_search)

        self._tree_search_refresh_timer = QTimer(self)
        self._tree_search_refresh_timer.setSingleShot(True)
        self._tree_search_refresh_timer.setInterval(
            int(self.TREE_SEARCH_REFRESH_DEBOUNCE_MS)
        )
        self._tree_search_refresh_timer.timeout.connect(self._run_match_search)

        self._search_progress_publish_timer = QTimer(self)
        self._search_progress_publish_timer.setSingleShot(True)
        self._search_progress_publish_timer.setInterval(
            max(16, int(self.SEARCH_PROGRESS_PUBLISH_INTERVAL_MS))
        )
        self._search_progress_publish_timer.timeout.connect(
            self._publish_search_scan_progress
        )

        self._scope_prefetch_timer = QTimer(self)
        self._scope_prefetch_timer.setSingleShot(True)
        self._scope_prefetch_timer.setInterval(int(self.SCOPE_PREFETCH_TIMER_INTERVAL_MS))
        self._scope_prefetch_timer.timeout.connect(self._start_scope_prefetch)

        self._pdf_text_cache_gc_timer = QTimer(self)
        self._pdf_text_cache_gc_timer.setInterval(
            max(100, int(float(self.PDF_TEXT_CACHE_GC_INTERVAL_SECONDS) * 1000))
        )
        self._pdf_text_cache_gc_timer.timeout.connect(
            self._on_pdf_text_cache_gc_timer
        )
        self._pdf_text_cache_gc_timer.start()

        self._cached_badge_sync_timer = QTimer(self)
        self._cached_badge_sync_timer.setSingleShot(True)
        self._cached_badge_sync_timer.setInterval(int(self.CACHED_BADGE_SYNC_INTERVAL_MS))
        self._cached_badge_sync_timer.timeout.connect(self._sync_cached_tree_badges)

        self._cache_badge_probe_timer = QTimer(self)
        self._cache_badge_probe_timer.setSingleShot(True)
        self._cache_badge_probe_timer.setInterval(
            max(0, int(self.CACHE_BADGE_PROBE_TIMER_INTERVAL_MS))
        )
        self._cache_badge_probe_timer.timeout.connect(self._start_cached_badge_probe)

        self._global_activity_touch_timer = QTimer(self)
        self._global_activity_touch_timer.setSingleShot(True)
        self._global_activity_touch_timer.timeout.connect(
            self._start_global_activity_touch_worker
        )

        self._global_activity_probe_timer = QTimer(self)
        self._global_activity_probe_timer.setInterval(
            max(
                100,
                int(
                    float(self.MULTI_INSTANCE_ACTIVITY_PROBE_INTERVAL_SECONDS)
                    * 1000.0
                ),
            )
        )
        self._global_activity_probe_timer.timeout.connect(
            self._start_global_activity_probe_worker
        )
        self._global_activity_probe_timer.start()

        self._tree_visual_relax_timer = QTimer(self)
        self._tree_visual_relax_timer.setSingleShot(True)
        self._tree_visual_relax_timer.setInterval(
            int(self.TREE_INTERACTION_VISUAL_RELAX_MS)
        )
        self._tree_visual_relax_timer.timeout.connect(
            self._on_tree_interaction_visual_relax_timeout
        )

        self._viewer_ready_timer = QTimer(self)
        self._viewer_ready_timer.setInterval(int(self.VIEWER_READY_TIMER_INTERVAL_MS))
        self._viewer_ready_timer.timeout.connect(self._ensure_viewer_bridge_ready)

        self._view_state_poll_timer = QTimer(self)
        self._view_state_poll_timer.setInterval(int(self.VIEW_STATE_POLL_TIMER_INTERVAL_MS))
        self._view_state_poll_timer.timeout.connect(self._poll_current_view_state)
        self._view_state_poll_timer.start()

        self._file_change_watch_timer = QTimer(self)
        self._file_change_watch_timer.setInterval(int(self.FILE_CHANGE_WATCH_INTERVAL_MS))
        self._file_change_watch_timer.timeout.connect(self._on_file_change_watch_tick)
        self._file_change_watch_timer.start()

        self.setWindowTitle("pdfexplore")
        self.setWindowIcon(app_icon)
        self.resize(1848, 980)

        self.model = ColorizedPdfModel(self)
        self.model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)
        self.model.setNameFilters(["*.pdf"])
        self.model.setNameFilterDisables(False)
        self.model.directoryLoaded.connect(self._on_model_directory_loaded)

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setItemDelegate(PdfTreeItemDelegate(self.tree))
        self.tree.setIconSize(ColorizedPdfModel.decorated_icon_size())
        self.tree.setIndentation(14)
        self.tree.setUniformRowHeights(True)
        self.tree.setAnimated(False)
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
        self.tree.collapsed.connect(self._on_tree_directory_collapsed)

        self.preview_stack = QStackedWidget()
        self._empty_preview = QWidget()
        self.preview_stack.addWidget(self._empty_preview)
        self.preview_stack.setCurrentWidget(self._empty_preview)

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

        self.recent_btn = QPushButton("Recent")
        self.recent_btn.setToolTip(
            "Open one of the 35 retained recent root directories"
        )
        self.recent_menu = QMenu(self.recent_btn)
        self.recent_menu.aboutToShow.connect(
            self._reload_recent_root_directories_before_menu_open
        )
        self.recent_btn.setMenu(self.recent_menu)
        self._apply_compact_toolbar_button_width(
            self.recent_btn, horizontal_padding_px=12, include_menu_indicator=True
        )
        self._refresh_recent_root_menu()

        self.up_btn = QPushButton("^")
        self._apply_compact_toolbar_button_width(self.up_btn, horizontal_padding_px=10)
        self.up_btn.clicked.connect(self._go_up_directory)

        refresh_btn = QPushButton("Refresh")
        self._apply_compact_toolbar_button_width(refresh_btn, horizontal_padding_px=12)
        refresh_btn.clicked.connect(self._refresh_directory_view)

        self.expand_btn = QPushButton("Expand")
        self._apply_compact_toolbar_button_width(
            self.expand_btn, horizontal_padding_px=12
        )
        self.expand_btn.setToolTip(
            "Expand every directory under the current root that contains PDFs"
        )
        self.expand_btn.clicked.connect(self._expand_pdf_directories)

        self.collapse_btn = QPushButton("Collapse")
        self._apply_compact_toolbar_button_width(
            self.collapse_btn, horizontal_padding_px=12
        )
        self.collapse_btn.setToolTip("Collapse all directories under the current root")
        self.collapse_btn.clicked.connect(self._collapse_all_directories)

        self.dark_mode_btn = QPushButton("Dark")
        self._apply_compact_toolbar_button_width(
            self.dark_mode_btn,
            horizontal_padding_px=12,
            sizing_text="Light",
        )
        self.dark_mode_btn.setToolTip("Show PDFs with dark pages and light text")
        self.dark_mode_btn.clicked.connect(self._toggle_preview_dark_mode)
        self.dark_mode_btn.setEnabled(False)

        self.add_view_btn = QPushButton("Add View")
        self._apply_compact_toolbar_button_width(
            self.add_view_btn, horizontal_padding_px=12
        )
        self.add_view_btn.clicked.connect(self._add_document_view)

        self.edit_btn = QPushButton("Edit")
        self._apply_compact_toolbar_button_width(self.edit_btn, horizontal_padding_px=12)
        self.edit_btn.clicked.connect(self._edit_current_file)
        self._update_edit_button_state()

        self.path_label = QLabel("")
        self.path_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.path_label.setMinimumWidth(0)
        self.path_label.setToolTip("")

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
            "Copy currently previewed PDF file to selected destination"
        )
        copy_current_btn.setStyleSheet(
            "border: 1px solid #4b5563; border-radius: 3px; padding: 0px;"
        )
        pin_icon_path = ui_asset_path("pin.png")
        if pin_icon_path.is_file():
            pin_icon = QIcon(str(pin_icon_path))
            copy_current_btn.setIcon(pin_icon)
            copy_current_btn.setIconSize(QSize(16, 16))
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

        match_label = QLabel("Search and highlight: ")
        self.match_input = QLineEdit()
        self.match_input.setClearButtonEnabled(False)
        self.match_input.setPlaceholderText(
            'words, "phrases", \'case-sensitive\', AND/OR/NOT, NEAR(...)'
        )
        self.match_input.setMinimumWidth(220)
        self.match_clear_action = self.match_input.addAction(
            build_clear_x_icon(),
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
        top_bar.addWidget(self.recent_btn)
        top_bar.addWidget(self.up_btn)
        top_bar.addWidget(refresh_btn)
        top_bar.addWidget(self.expand_btn)
        top_bar.addWidget(self.collapse_btn)
        top_bar.addWidget(self.dark_mode_btn)
        top_bar.addWidget(self.add_view_btn)
        top_bar.addWidget(self.edit_btn)
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
        preview_layout.addWidget(self.preview_stack, 1)

        self._preview_zoom_overlay = QLabel("", self.preview_container)
        self._preview_zoom_overlay.setObjectName("pdfexplore-preview-zoom-overlay")
        self._preview_zoom_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_zoom_overlay.setStyleSheet(
            """
            QLabel#pdfexplore-preview-zoom-overlay {
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
        splitter_units = 1000
        tree_units = round(
            splitter_units * float(self.SPLITTER_TREE_DEFAULT_FRACTION)
        )
        self.splitter.setSizes([tree_units, splitter_units - tree_units])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(top_bar_widget)
        layout.addWidget(self.splitter, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
        self._gpu_status_label = QLabel("GPU")
        self._gpu_status_label.setStyleSheet("color: rgba(156, 163, 175, 0.45);")
        self._gpu_status_label.setVisible(self._gpu_context_available)
        self.statusBar().addPermanentWidget(self._gpu_status_label)

        refresh_action = QAction("Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self._refresh_directory_view)
        self.addAction(refresh_action)

        preview_zoom_in_action = QAction("Preview Zoom In", self)
        preview_zoom_in_action.setShortcuts(["Ctrl++", "Ctrl+=", "Ctrl+Shift+="])
        preview_zoom_in_action.triggered.connect(self._zoom_preview_in)
        self.addAction(preview_zoom_in_action)

        preview_zoom_out_action = QAction("Preview Zoom Out", self)
        preview_zoom_out_action.setShortcuts(["Ctrl+-", "Ctrl+_"])
        preview_zoom_out_action.triggered.connect(self._zoom_preview_out)
        self.addAction(preview_zoom_out_action)

        preview_zoom_reset_action = QAction("Preview Zoom Reset", self)
        preview_zoom_reset_action.setShortcuts(["Ctrl+0"])
        preview_zoom_reset_action.triggered.connect(self._reset_preview_zoom)
        self.addAction(preview_zoom_reset_action)

        self.preview_toggle_three_up_action = QAction("Preview Toggle 3-Up", self)
        self.preview_toggle_three_up_action.setShortcuts(
            [
                "Ctrl+\\",
                "Ctrl+|",
                "Ctrl+Shift+\\",
                "Ctrl+(",
                "Ctrl+Shift+9",
                "Ctrl+9",
            ]
        )
        self.preview_toggle_three_up_action.triggered.connect(self._toggle_preview_three_up)
        self.addAction(self.preview_toggle_three_up_action)

        self.preview_zoom_one_hundred_action = QAction("Preview Zoom 100%", self)
        self.preview_zoom_one_hundred_action.setShortcuts(["Ctrl+)", "Ctrl+Shift+0"])
        self.preview_zoom_one_hundred_action.triggered.connect(
            self._set_preview_zoom_one_hundred
        )
        self.addAction(self.preview_zoom_one_hundred_action)

        self._register_global_shortcut(
            QKeySequence("Ctrl+\\"),
            self._toggle_preview_three_up,
        )
        self._register_global_shortcut(
            QKeySequence("Ctrl+|"),
            self._toggle_preview_three_up,
        )
        self._register_global_shortcut(
            QKeySequence("Ctrl+Shift+\\"),
            self._toggle_preview_three_up,
        )
        self._register_global_shortcut(
            QKeySequence(
                Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Backslash
            ),
            self._toggle_preview_three_up,
        )
        self._register_global_shortcut(
            QKeySequence(
                Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_Bar
            ),
            self._toggle_preview_three_up,
        )
        self._register_global_shortcut(
            QKeySequence("Ctrl+Shift+9"),
            self._toggle_preview_three_up,
        )
        self._register_global_shortcut(
            QKeySequence("Ctrl+9"),
            self._toggle_preview_three_up,
        )
        self._register_global_shortcut(
            QKeySequence("Ctrl+("),
            self._toggle_preview_three_up,
        )
        self._register_global_shortcut(
            QKeySequence(
                Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_ParenLeft
            ),
            self._toggle_preview_three_up,
        )
        self._register_global_shortcut(
            QKeySequence("Ctrl+Shift+0"),
            self._set_preview_zoom_one_hundred,
        )
        self._register_global_shortcut(
            QKeySequence("Ctrl+)"),
            self._set_preview_zoom_one_hundred,
        )
        self._register_global_shortcut(
            QKeySequence(
                Qt.KeyboardModifier.ControlModifier | Qt.Key.Key_ParenRight
            ),
            self._set_preview_zoom_one_hundred,
        )

        self._set_root_directory(self.root)
        self._update_window_title()
        self._update_up_button_state()
        # Track input across this window's controls without installing a
        # QApplication-wide filter. Global Qt filters can intercept private
        # QtWebEngine renderer widgets and destabilize Chromium event delivery.
        self.installEventFilter(self)
        for widget in self.findChildren(QWidget):
            widget.installEventFilter(self)
        self._touch_global_activity_stamp(force=True)
        self._start_global_activity_probe_worker()

    def _apply_compact_toolbar_button_width(
        self,
        button: QPushButton,
        *,
        horizontal_padding_px: int = 12,
        include_menu_indicator: bool = False,
        sizing_text: str | None = None,
    ) -> None:
        """Size a top-bar button to its label plus slim horizontal padding."""
        width_text = button.text() if sizing_text is None else sizing_text
        text_width = button.fontMetrics().horizontalAdvance(width_text or "")
        menu_indicator_width = 0
        if include_menu_indicator or button.menu() is not None:
            menu_indicator_width = max(
                8,
                int(
                    button.style().pixelMetric(
                        QStyle.PixelMetric.PM_MenuButtonIndicator
                    )
                ),
            )
        width = text_width + menu_indicator_width + max(8, int(horizontal_padding_px))
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        button.setFixedWidth(max(20, int(width)))

    def _lookup_cached_pdf_text(
        self,
        path_key: str,
        mtime_ns: int,
        size: int,
    ) -> str | None:
        """Handle lookup cached pdf text."""
        with self._pdf_text_cache_lock:
            cached = self._pdf_text_cache.get(path_key)
            if cached is None:
                return None
            cached_mtime_ns, cached_size, cached_text = cached
            if cached_mtime_ns != mtime_ns or cached_size != size:
                return None
            self._pdf_text_cache.move_to_end(path_key)
            return cached_text

    def _store_cached_pdf_text(
        self,
        path_key: str,
        mtime_ns: int,
        size: int,
        searchable_text: str,
    ) -> None:
        """Handle store cached pdf text."""
        with self._pdf_text_cache_lock:
            previous = self._pdf_text_cache.pop(path_key, None)
            if previous is not None:
                self._pdf_text_cache_total_chars -= len(previous[2])

            self._pdf_text_cache[path_key] = (mtime_ns, size, searchable_text)
            self._pdf_text_cache_total_chars += len(searchable_text)

            while (
                len(self._pdf_text_cache) > int(self.PDF_TEXT_CACHE_MAX_ENTRIES)
                or self._pdf_text_cache_total_chars > int(self.PDF_TEXT_CACHE_MAX_CHARS)
            ):
                _evicted_key, evicted = self._pdf_text_cache.popitem(last=False)
                self._pdf_text_cache_total_chars -= len(evicted[2])

    def _invalidate_cached_pdf_text(self, path_key: str) -> None:
        """Handle invalidate cached pdf text."""
        with self._pdf_text_cache_lock:
            removed = self._pdf_text_cache.pop(path_key, None)
            if removed is not None:
                self._pdf_text_cache_total_chars -= len(removed[2])
        with self._cached_pdf_path_keys_lock:
            if path_key in self._cached_pdf_path_keys:
                self._cached_pdf_path_keys.discard(path_key)
                self._cached_pdf_path_keys_revision += 1
        self._prefetch_last_prioritized_current_path = ""
        self._request_cached_badge_sync()
        self._request_scope_prefetch()

    def _record_cached_pdf_path_key(self, path_key: str) -> None:
        """Handle record cached pdf path key."""
        if not path_key:
            return
        normalized = str(path_key)
        with self._cached_pdf_path_keys_lock:
            if normalized in self._cached_pdf_path_keys:
                return
            self._cached_pdf_path_keys.add(normalized)
            self._cached_pdf_path_keys_revision += 1

    def _snapshot_cached_pdf_path_keys(self) -> tuple[int, set[str]]:
        """Handle snapshot cached pdf path keys."""
        with self._cached_pdf_path_keys_lock:
            return self._cached_pdf_path_keys_revision, set(self._cached_pdf_path_keys)

    def _sync_cached_tree_badges(self) -> None:
        """Handle sync cached tree badges."""
        if self._prefetch_temporarily_paused():
            self._cached_badge_sync_timer.start()
            return
        model = getattr(self, "model", None)
        if model is None:
            return
        revision, path_keys = self._snapshot_cached_pdf_path_keys()
        if revision <= self._cached_pdf_path_keys_synced_revision:
            return
        changed = model.set_cached_path_keys(path_keys)
        self._cached_pdf_path_keys_synced_revision = revision
        if changed:
            self.tree.viewport().update()

    def _request_cached_badge_sync(self) -> None:
        """Request cached badge sync."""
        with self._cached_pdf_path_keys_lock:
            needs_sync = (
                self._cached_pdf_path_keys_revision
                > self._cached_pdf_path_keys_synced_revision
            )
        if not needs_sync:
            return
        self._cached_badge_sync_timer.start()

    def _cancel_cached_badge_probe(self) -> None:
        """Cancel queued persisted-cache discovery and invalidate running work."""
        self._cache_badge_probe_request_id += 1
        for worker in tuple(self._active_cache_badge_probe_workers):
            try:
                if self._cache_badge_probe_pool.tryTake(worker):
                    self._active_cache_badge_probe_workers.discard(worker)
            except Exception:
                pass
        self._cache_badge_probe_pool.clear()

    def _request_cached_badge_probe(self) -> None:
        """Coalesce a prompt background scan for persisted cache entries."""
        if self._closing:
            return
        self._cache_badge_probe_timer.start()

    def _start_cached_badge_probe(self) -> None:
        """Restore cached icons without waiting for idle extraction prefetch.

        The probe checks current PDF signatures and matching cache-file
        existence only. It does not decompress text or extract PDF content.
        """
        if self._closing:
            return
        candidates = self._list_visible_pdf_files_in_tree(fetch_more=False)
        if not candidates:
            return

        self._cancel_cached_badge_probe()
        request_id = self._cache_badge_probe_request_id

        def _should_abort(req: int = request_id) -> bool:
            return bool(self._closing or req != self._cache_badge_probe_request_id)

        worker = PdfTextPrefetchWorker(
            request_id,
            candidates,
            lambda path: self._is_pdf_text_cached_for_path(
                path,
                mark_cached_badge=True,
                should_abort=_should_abort,
            ),
            should_abort=_should_abort,
        )
        worker.signals.finished.connect(self._on_cached_badge_probe_finished)
        self._active_cache_badge_probe_workers.add(worker)
        try:
            self._cache_badge_probe_pool.start(worker)
        except Exception:
            self._active_cache_badge_probe_workers.discard(worker)

    def _on_cached_badge_probe_finished(
        self,
        request_id: int,
        _probed_count: int,
        _skipped_count: int,
        _error_text: str,
    ) -> None:
        """Publish icons found by the prompt persisted-cache probe."""
        worker_to_remove = None
        for worker in self._active_cache_badge_probe_workers:
            if worker.request_id == request_id:
                worker_to_remove = worker
                break
        if worker_to_remove is not None:
            self._active_cache_badge_probe_workers.discard(worker_to_remove)
        if request_id != self._cache_badge_probe_request_id or self._closing:
            return
        self._request_cached_badge_sync()

    def _mark_user_interaction(self) -> None:
        """Handle mark user interaction."""
        now = time.monotonic()
        self._last_user_interaction_at = now
        self._recent_interaction_timestamps.append(now)

        current_request_id = self._prefetch_request_id
        has_current_prefetch = any(
            worker.request_id == current_request_id
            for worker in self._active_prefetch_workers
        )
        gc_worker = self._active_pdf_text_cache_gc_worker
        has_current_gc = bool(
            gc_worker is not None and gc_worker.request_id == current_request_id
        )
        if has_current_prefetch or has_current_gc:
            # A queued/running current-document warmup may not have reached its
            # loader yet. Let the next idle period prioritize it again.
            self._prefetch_last_prioritized_current_path = ""
            self._cancel_scope_prefetch()
            self._request_scope_prefetch()

        window_seconds = float(self.PREFETCH_HEAVY_USE_WINDOW_SECONDS)
        cutoff = now - window_seconds
        while self._recent_interaction_timestamps and self._recent_interaction_timestamps[0] < cutoff:
            self._recent_interaction_timestamps.popleft()

        threshold = max(1, int(self.PREFETCH_HEAVY_USE_EVENT_THRESHOLD))
        if len(self._recent_interaction_timestamps) >= threshold:
            self._prefetch_paused_until = max(
                self._prefetch_paused_until,
                now + float(self.PREFETCH_HEAVY_USE_PAUSE_SECONDS),
            )

        # Publish activity only after local idle/cancellation state is current.
        # The method merely schedules a dedicated worker; it performs no cache
        # filesystem I/O on this GUI-thread input path.
        self._touch_global_activity_stamp()

    def _set_tree_interaction_visual_mode(self, enabled: bool) -> None:
        """Set tree interaction visual mode."""
        model = getattr(self, "model", None)
        tree = getattr(self, "tree", None)
        if model is None or tree is None:
            return
        setter = getattr(model, "set_reduce_paint_cost", None)
        if not callable(setter):
            return
        try:
            changed = bool(setter(enabled))
        except Exception:
            changed = False
        if changed:
            tree.viewport().update()

    def _on_tree_interaction_visual_relax_timeout(self) -> None:
        """Handle tree interaction visual relax timeout."""
        self._set_tree_interaction_visual_mode(False)

    def _prefetch_temporarily_paused(self) -> bool:
        """Handle prefetch temporarily paused."""
        return time.monotonic() < float(self._prefetch_paused_until)

    def _global_activity_stamp_path(self) -> Path:
        """Return the heartbeat shared by all pdfexplore instances."""
        return self._pdf_text_disk_cache_dir / ".pdfexplore-activity"

    def _pdf_text_background_lease_lock_path(self) -> Path:
        """Return the single idle-maintenance lease shared across processes."""
        return self._pdf_text_disk_cache_dir / ".pdfexplore-background.lock"

    def _touch_global_activity_stamp(self, *, force: bool = False) -> None:
        """Coalesce one asynchronous shared activity-stamp write."""
        if self._closing:
            return
        self._global_activity_touch_pending = True
        if force:
            self._global_activity_touch_force_pending = True
        self._start_global_activity_touch_worker()

    def _start_global_activity_touch_worker(self) -> None:
        """Dispatch a due heartbeat write on its dedicated one-thread pool."""
        if self._closing:
            self._global_activity_touch_pending = False
            self._global_activity_touch_force_pending = False
            self._global_activity_touch_timer.stop()
            return
        if self._active_global_activity_touch_worker is not None:
            return
        if not self._global_activity_touch_pending:
            return

        now = time.monotonic()
        force = bool(self._global_activity_touch_force_pending)
        interval = max(
            0.0,
            float(self.MULTI_INSTANCE_ACTIVITY_TOUCH_INTERVAL_SECONDS),
        )
        remaining = interval - (now - self._last_global_activity_touch_at)
        if not force and remaining > 0.0:
            self._global_activity_touch_timer.start(
                max(1, int(remaining * 1000.0) + 1)
            )
            return

        self._global_activity_touch_pending = False
        self._global_activity_touch_force_pending = False
        self._global_activity_touch_timer.stop()
        self._global_activity_touch_request_id += 1
        request_id = self._global_activity_touch_request_id
        worker = GlobalActivityHeartbeatWorker(
            request_id,
            self._global_activity_stamp_path(),
        )
        self._active_global_activity_touch_worker = worker
        worker.signals.finished.connect(self._on_global_activity_touch_finished)
        try:
            self._global_activity_touch_pool.start(worker)
        except Exception:
            self._active_global_activity_touch_worker = None
            self._global_activity_touch_pending = True
            self._global_activity_touch_force_pending = force
            self._global_activity_touch_timer.start(max(10, int(interval * 1000.0)))

    def _on_global_activity_touch_finished(self, request_id: int) -> None:
        """Release one writer and schedule any activity coalesced behind it."""
        worker = self._active_global_activity_touch_worker
        if worker is None or worker.request_id != int(request_id):
            return
        self._active_global_activity_touch_worker = None
        self._last_global_activity_touch_at = time.monotonic()
        if worker.stamp_path == self._global_activity_stamp_path():
            self._last_observed_global_activity_wall_time = max(
                self._last_observed_global_activity_wall_time,
                time.time(),
            )
        if self._closing:
            self._global_activity_touch_pending = False
            self._global_activity_touch_force_pending = False
            return
        self._start_global_activity_touch_worker()

    def _start_global_activity_probe_worker(self) -> None:
        """Read the shared idle heartbeat on a dedicated worker thread."""
        if self._closing or self._active_global_activity_probe_worker is not None:
            return
        self._global_activity_probe_request_id += 1
        request_id = self._global_activity_probe_request_id
        worker = GlobalActivityProbeWorker(
            request_id,
            self._global_activity_stamp_path(),
        )
        self._active_global_activity_probe_worker = worker
        worker.signals.finished.connect(self._on_global_activity_probe_finished)
        try:
            self._global_activity_probe_pool.start(worker)
        except Exception:
            self._active_global_activity_probe_worker = None

    def _on_global_activity_probe_finished(
        self,
        request_id: int,
        observed_wall_time,
    ) -> None:
        """Publish one off-thread observation of sibling-instance activity."""
        worker = self._active_global_activity_probe_worker
        if worker is None or worker.request_id != int(request_id):
            return
        self._active_global_activity_probe_worker = None
        if worker.stamp_path != self._global_activity_stamp_path():
            return
        try:
            observed = float(observed_wall_time)
        except (TypeError, ValueError):
            return
        self._last_observed_global_activity_wall_time = max(
            self._last_observed_global_activity_wall_time,
            observed,
        )

    def _stop_global_activity_heartbeat(self) -> None:
        """Cancel queued heartbeat reads/writes without waiting on cache I/O."""
        self._global_activity_touch_timer.stop()
        self._global_activity_probe_timer.stop()
        self._global_activity_touch_pending = False
        self._global_activity_touch_force_pending = False
        worker = self._active_global_activity_touch_worker
        if worker is not None:
            try:
                if self._global_activity_touch_pool.tryTake(worker):
                    self._active_global_activity_touch_worker = None
            except Exception:
                pass
        self._global_activity_touch_pool.clear()
        probe_worker = self._active_global_activity_probe_worker
        if probe_worker is not None:
            try:
                if self._global_activity_probe_pool.tryTake(probe_worker):
                    self._active_global_activity_probe_worker = None
            except Exception:
                pass
        self._global_activity_probe_pool.clear()

    def _global_idle_elapsed_seconds(self) -> float:
        """Return idle time constrained by activity in every live instance."""
        local_elapsed = max(0.0, time.monotonic() - self._last_user_interaction_at)
        shared_elapsed = max(
            0.0,
            time.time() - self._last_observed_global_activity_wall_time,
        )
        return min(local_elapsed, shared_elapsed)

    def _is_object_within_current_preview(self, watched) -> bool:
        """Return whether object within current preview."""
        try:
            preview = self._current_preview_widget()
        except Exception:
            return False
        if preview is None or watched is None:
            return False
        return watched is preview

    def _pause_prefetch_for_user_input(self, watched, event_type) -> None:
        """Pause prefetch while direct user input is active.

        This method intentionally treats wheel input as a high-priority signal to
        protect preview scroll smoothness, and treats preview-local pointer/key
        input as interaction that should temporarily suppress background parsing.
        """
        interactive_types = {
            QEvent.Type.Wheel,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.MouseMove,
            QEvent.Type.KeyPress,
            QEvent.Type.KeyRelease,
        }
        if event_type not in interactive_types:
            return

        within_preview = False
        if event_type != QEvent.Type.Wheel:
            try:
                within_preview = self._is_object_within_current_preview(watched)
            except Exception:
                within_preview = False

        if event_type == QEvent.Type.Wheel or within_preview:
            self._pause_prefetch_temporarily(
                self.PREFETCH_VIEWER_ACTIVITY_PAUSE_SECONDS,
                cancel_inflight=True,
            )

    def _on_viewer_user_activity(self, path_key: str) -> None:
        """Reset idle work from activity reported inside the pdf.js document."""
        if self._closing or self._current_preview_path_key() != str(path_key):
            return
        self._mark_user_interaction()

    def _is_pdf_text_cached_for_path(
        self,
        path: Path,
        *,
        mark_cached_badge: bool = False,
        should_abort=None,
    ) -> bool:
        """Return whether current text cache contains a valid entry for `path`.

        Lookup order:
        1) bounded in-memory cache,
        2) on-disk compressed cache.

        When `mark_cached_badge` is true, cache hits also update tree badge
        state so the UI reflects already-cached files even when no extraction is
        needed.
        """
        try:
            resolved_path = path.resolve(strict=True)
            path_key = str(resolved_path)
            stat = resolved_path.stat()
        except (OSError, RuntimeError):
            return False
        cached = self._lookup_cached_pdf_text(path_key, stat.st_mtime_ns, stat.st_size)
        if cached is not None:
            try:
                if str(path.resolve(strict=True)) != path_key:
                    return False
            except (OSError, RuntimeError):
                return False
            if mark_cached_badge:
                self._record_cached_pdf_path_key(path_key)
            return True
        cache_path = self._pdf_text_disk_cache_file_path(
            path_key,
            stat.st_mtime_ns,
            stat.st_size,
        )
        with advisory_file_lock(
            self._pdf_text_disk_cache_process_lock_path(),
            exclusive=False,
            blocking=True,
            should_abort=should_abort,
        ) as acquired:
            if not acquired:
                return False
            try:
                exists = cache_path.is_file()
            except Exception:
                return False
        try:
            if str(path.resolve(strict=True)) != path_key:
                return False
        except (OSError, RuntimeError):
            return False
        if exists and mark_cached_badge:
            # Metadata repair is useful for legacy cache entries but must never
            # make an idle probe wait behind another process's writer.
            with advisory_file_lock(
                self._pdf_text_disk_cache_process_lock_path(),
                exclusive=True,
                blocking=False,
            ) as acquired:
                if acquired and cache_path.is_file():
                    self._ensure_pdf_text_disk_cache_metadata_locked(
                        cache_path,
                        path_key,
                    )
        if exists and mark_cached_badge:
            self._record_cached_pdf_path_key(path_key)
        return exists

    @staticmethod
    def _build_pdf_text_disk_cache_dir() -> Path:
        """Build pdf text disk cache dir."""
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
        if xdg_cache_home:
            cache_root = Path(xdg_cache_home).expanduser()
        else:
            cache_root = Path.home() / ".cache"
        return cache_root / "mdexplore" / "pdfexplore-text-cache"

    @staticmethod
    def _pdf_text_disk_cache_file_name(path_key: str, mtime_ns: int, size: int) -> str:
        """Handle pdf text disk cache file name."""
        material = f"{path_key}|{int(mtime_ns)}|{int(size)}"
        digest = hashlib.sha256(material.encode("utf-8", errors="surrogatepass")).hexdigest()
        return f"{digest}.txt.gz"

    def _pdf_text_disk_cache_file_path(
        self,
        path_key: str,
        mtime_ns: int,
        size: int,
    ) -> Path:
        """Handle pdf text disk cache file path."""
        file_name = self._pdf_text_disk_cache_file_name(path_key, mtime_ns, size)
        return self._pdf_text_disk_cache_dir / file_name

    def _pdf_text_disk_cache_process_lock_path(self) -> Path:
        """Return the stable lock shared by every pdfexplore process."""
        return self._pdf_text_disk_cache_dir / ".pdfexplore-text-cache.lock"

    def _pdf_text_disk_cache_trim_dirty_path(self) -> Path:
        """Return the cross-process marker requesting an idle quota trim."""
        return self._pdf_text_disk_cache_dir / ".pdfexplore-trim-dirty"

    @staticmethod
    def _pdf_text_disk_cache_metadata_path(cache_path: Path) -> Path:
        """Return the source-path metadata companion for a compressed entry."""
        return cache_path.with_name(f"{cache_path.name}.meta.json")

    def _ensure_pdf_text_disk_cache_metadata_locked(
        self,
        cache_path: Path,
        path_key: str,
    ) -> None:
        """Atomically record which source PDF owns a disk-cache entry."""
        metadata_path = self._pdf_text_disk_cache_metadata_path(cache_path)
        try:
            raw_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                isinstance(raw_payload, dict)
                and str(raw_payload.get("source_path", "")) == str(path_key)
            ):
                return
        except Exception:
            pass

        try:
            atomic_write_text(
                metadata_path,
                json.dumps(
                    {"version": 1, "source_path": str(path_key)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        except Exception:
            pass

    def _load_pdf_text_from_disk_cache(
        self,
        path_key: str,
        mtime_ns: int,
        size: int,
        *,
        should_abort=None,
    ) -> str | None:
        """Load pdf text from disk cache."""
        cache_path = self._pdf_text_disk_cache_file_path(path_key, mtime_ns, size)
        opened_handle = None
        opened_identity: tuple[int, int, int, int] | None = None
        with advisory_file_lock(
            self._pdf_text_disk_cache_process_lock_path(),
            exclusive=False,
            blocking=True,
            should_abort=should_abort,
        ) as acquired:
            if not acquired:
                return None
            try:
                opened_handle = cache_path.open("rb")
                cache_stat = os.fstat(opened_handle.fileno())
                opened_identity = (
                    int(cache_stat.st_dev),
                    int(cache_stat.st_ino),
                    int(cache_stat.st_mtime_ns),
                    int(cache_stat.st_size),
                )
            except Exception:
                if opened_handle is not None:
                    opened_handle.close()
                return None

        if callable(should_abort) and should_abort():
            opened_handle.close()
            return None
        cancelled = False
        try:
            with opened_handle:
                with gzip.open(opened_handle, mode="rt", encoding="utf-8") as handle:
                    chunks: list[str] = []
                    while True:
                        if callable(should_abort) and should_abort():
                            cancelled = True
                            break
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    cached_text = None if cancelled else "".join(chunks)
        except Exception:
            cached_text = None

        # Cancellation is not corruption. Leave the current cache entry and
        # its metadata untouched so the next idle/search request can reuse it.
        if cancelled or (callable(should_abort) and should_abort()):
            return None

        # Repair metadata/LRU state or remove corruption only when the lock is
        # immediately available. Decompression used the already-open inode, so
        # no writer is held up by a potentially large gzip read.
        with advisory_file_lock(
            self._pdf_text_disk_cache_process_lock_path(),
            exclusive=True,
            blocking=False,
        ) as acquired:
            if acquired:
                try:
                    current_stat = cache_path.stat()
                    current_identity = (
                        int(current_stat.st_dev),
                        int(current_stat.st_ino),
                        int(current_stat.st_mtime_ns),
                        int(current_stat.st_size),
                    )
                except Exception:
                    current_stat = None
                    current_identity = None
                if opened_identity is not None and current_identity == opened_identity:
                    if cached_text is not None:
                        self._ensure_pdf_text_disk_cache_metadata_locked(
                            cache_path,
                            path_key,
                        )
                        try:
                            age_seconds = time.time() - float(current_stat.st_mtime)
                            if age_seconds >= float(
                                self.PDF_TEXT_DISK_CACHE_TOUCH_INTERVAL_SECONDS
                            ):
                                os.utime(cache_path, None)
                        except Exception:
                            pass
                    else:
                        try:
                            cache_path.unlink(missing_ok=True)
                            self._pdf_text_disk_cache_metadata_path(cache_path).unlink(
                                missing_ok=True
                            )
                        except Exception:
                            pass
        return cached_text

    def _trim_pdf_text_disk_cache(self, *, should_abort=None) -> bool:
        """Run an abortable quota trim using only short per-entry locks."""
        cache_dir = self._pdf_text_disk_cache_dir
        files_with_stats: list[
            tuple[Path, int, int, tuple[int, int, int, int]]
        ] = []
        try:
            with os.scandir(cache_dir) as entries:
                for entry in entries:
                    if callable(should_abort) and should_abort():
                        return False
                    if not entry.name.endswith(".txt.gz"):
                        continue
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    except OSError:
                        return False
                    if not stat_module.S_ISREG(entry_stat.st_mode):
                        continue
                    identity = (
                        int(entry_stat.st_dev),
                        int(entry_stat.st_ino),
                        int(entry_stat.st_mtime_ns),
                        int(entry_stat.st_size),
                    )
                    files_with_stats.append(
                        (
                            Path(entry.path),
                            int(entry_stat.st_mtime_ns),
                            int(entry_stat.st_size),
                            identity,
                        )
                    )
        except FileNotFoundError:
            return True
        except OSError:
            return False

        files_with_stats.sort(key=lambda payload: payload[1], reverse=True)
        max_files = int(self.PDF_TEXT_DISK_CACHE_MAX_FILES)
        max_bytes = int(self.PDF_TEXT_DISK_CACHE_MAX_BYTES)
        kept_count = 0
        kept_bytes = 0
        deletion_candidates: list[tuple[Path, tuple[int, int, int, int]]] = []
        for path, _mtime_ns, size_bytes, identity in files_with_stats:
            keep_for_count = kept_count < max_files
            keep_for_size = (kept_bytes + size_bytes) <= max_bytes
            if keep_for_count and keep_for_size:
                kept_count += 1
                kept_bytes += size_bytes
                continue
            deletion_candidates.append((path, identity))

        for path, identity in deletion_candidates:
            if callable(should_abort) and should_abort():
                return False
            with advisory_file_lock(
                self._pdf_text_disk_cache_process_lock_path(),
                exclusive=True,
                blocking=False,
            ) as acquired:
                if not acquired:
                    return False
                try:
                    current_stat = path.stat()
                    current_identity = (
                        int(current_stat.st_dev),
                        int(current_stat.st_ino),
                        int(current_stat.st_mtime_ns),
                        int(current_stat.st_size),
                    )
                except FileNotFoundError:
                    continue
                except OSError:
                    return False
                if current_identity != identity:
                    return False
                try:
                    path.unlink(missing_ok=True)
                    self._pdf_text_disk_cache_metadata_path(path).unlink(
                        missing_ok=True
                    )
                except OSError:
                    return False
        return True

    def _store_pdf_text_to_disk_cache(
        self,
        path_key: str,
        mtime_ns: int,
        size: int,
        searchable_text: str,
        *,
        should_abort=None,
    ) -> bool:
        """Handle store pdf text to disk cache."""
        cache_dir = self._pdf_text_disk_cache_dir
        cache_path = self._pdf_text_disk_cache_file_path(path_key, mtime_ns, size)
        if callable(should_abort) and should_abort():
            return False
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return False
        temp_path = cache_path.with_name(
            f".{cache_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
        )
        try:
            with gzip.open(temp_path, mode="wt", encoding="utf-8") as handle:
                chunk_chars = 1024 * 1024
                for offset in range(0, len(searchable_text), chunk_chars):
                    if callable(should_abort) and should_abort():
                        raise InterruptedError("cache store cancelled")
                    handle.write(searchable_text[offset : offset + chunk_chars])
        except Exception:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False

        with advisory_file_lock(
            self._pdf_text_disk_cache_process_lock_path(),
            exclusive=True,
            blocking=True,
            should_abort=should_abort,
        ) as acquired:
            if not acquired or (callable(should_abort) and should_abort()):
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                return False
            try:
                temp_path.replace(cache_path)
            except Exception:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return False

            self._ensure_pdf_text_disk_cache_metadata_locked(cache_path, path_key)
            try:
                atomic_write_text(
                    self._pdf_text_disk_cache_trim_dirty_path(),
                    f"{os.getpid()}:{time.time_ns()}\n",
                )
            except OSError:
                pass
            with self._pdf_text_disk_cache_lock:
                self._pdf_text_disk_cache_store_count += 1
                if (
                    self._pdf_text_disk_cache_store_count
                    % max(1, int(self.PDF_TEXT_DISK_CACHE_TRIM_INTERVAL))
                ) == 0:
                    self._pdf_text_disk_cache_trim_requested_revision += 1
        return True

    @staticmethod
    def _pdf_source_definitely_missing(path_key: str) -> bool:
        """Return true only when the filesystem definitively reports absence."""
        try:
            source_stat = Path(path_key).stat()
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return not stat_module.S_ISREG(source_stat.st_mode)

    def _snapshot_pdf_text_cache_gc_path_keys(self) -> list[str]:
        """Snapshot source paths represented by memory entries or cache badges."""
        with self._pdf_text_cache_lock:
            path_keys = set(self._pdf_text_cache.keys())
        with self._cached_pdf_path_keys_lock:
            path_keys.update(self._cached_pdf_path_keys)
        # Sorting is intentionally deferred to the worker thread.
        return list(path_keys)

    def _apply_pdf_text_cache_gc_payload(
        self,
        payload: object,
        should_abort=None,
    ) -> tuple[int, int]:
        """Delete confirmed stale cache data under the appropriate locks.

        This method is widget-free and may run on the low-priority GC thread.
        It revalidates every deletion candidate and yields between entries when
        user/search activity invalidates the idle request.
        """
        if not isinstance(payload, dict):
            return 0, 0

        def _aborted() -> bool:
            return bool(callable(should_abort) and should_abort())

        missing_sources: set[str] = set()
        for value in payload.get("missing_memory_path_keys", []):
            if _aborted():
                break
            path_key = str(value or "")
            if not path_key or not self._pdf_source_definitely_missing(path_key):
                continue
            if _aborted():
                break
            missing_sources.add(path_key)

        disk_entries = payload.get("missing_disk_entries", [])
        if not isinstance(disk_entries, list):
            disk_entries = []

        removed_disk_entries = 0
        cache_dir = self._pdf_text_disk_cache_dir
        for entry in disk_entries:
            if _aborted():
                break
            if not isinstance(entry, dict):
                continue
            source_path = str(entry.get("source_path", "")).strip()
            cache_path = Path(str(entry.get("cache_path", "")))
            metadata_path = Path(str(entry.get("metadata_path", "")))
            raw_identity = entry.get("cache_identity")
            scanned_identity = None
            if isinstance(raw_identity, (list, tuple)) and len(raw_identity) == 4:
                try:
                    scanned_identity = tuple(int(value) for value in raw_identity)
                except (TypeError, ValueError):
                    scanned_identity = None
            if (
                not source_path
                or cache_path.parent != cache_dir
                or metadata_path != self._pdf_text_disk_cache_metadata_path(cache_path)
            ):
                continue
            if not self._pdf_source_definitely_missing(source_path) or _aborted():
                continue
            removed = False
            with advisory_file_lock(
                self._pdf_text_disk_cache_process_lock_path(),
                exclusive=True,
                blocking=False,
            ) as cache_lock_acquired:
                if not cache_lock_acquired:
                    break
                if _aborted():
                    break
                try:
                    current_stat = cache_path.stat()
                except (FileNotFoundError, OSError):
                    continue
                current_identity = (
                    int(current_stat.st_dev),
                    int(current_stat.st_ino),
                    int(current_stat.st_mtime_ns),
                    int(current_stat.st_size),
                )
                if (
                    scanned_identity is not None
                    and current_identity != scanned_identity
                ):
                    continue

                # The gzip inode alone does not identify its owner. Another
                # process may have repaired the metadata after the worker scan,
                # so re-read and compare ownership while holding the same lock
                # used by writers before deleting either companion.
                try:
                    current_metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    continue
                current_source = str(
                    current_metadata.get("source_path", "")
                    if isinstance(current_metadata, dict)
                    else ""
                ).strip()
                if current_source != source_path:
                    continue
                try:
                    cache_path.unlink()
                    removed = True
                except FileNotFoundError:
                    continue
                except OSError:
                    continue
                try:
                    metadata_path.unlink(missing_ok=True)
                except OSError:
                    # The gzip deletion is authoritative. A later idle pass
                    # will clean up the now-orphaned metadata companion.
                    pass
            if removed:
                removed_disk_entries += 1
                missing_sources.add(source_path)

        stale_metadata_paths = payload.get("stale_metadata_paths", [])
        if isinstance(stale_metadata_paths, list):
            for raw_path in stale_metadata_paths:
                if _aborted():
                    break
                metadata_path = Path(str(raw_path))
                if (
                    metadata_path.parent != cache_dir
                    or not metadata_path.name.endswith(".txt.gz.meta.json")
                ):
                    continue
                cache_path = metadata_path.with_name(
                    metadata_path.name[: -len(".meta.json")]
                )
                with advisory_file_lock(
                    self._pdf_text_disk_cache_process_lock_path(),
                    exclusive=True,
                    blocking=False,
                ) as cache_lock_acquired:
                    if not cache_lock_acquired:
                        break
                    if _aborted():
                        break
                    try:
                        cache_stat = cache_path.stat()
                    except FileNotFoundError:
                        cache_is_regular = False
                    except OSError:
                        # A transient/permission failure is not proof that the
                        # companion is orphaned. Defer to a later idle pass.
                        continue
                    else:
                        cache_is_regular = stat_module.S_ISREG(cache_stat.st_mode)

                    if cache_is_regular:
                        try:
                            current_payload = json.loads(
                                metadata_path.read_text(encoding="utf-8")
                            )
                        except OSError:
                            continue
                        except (
                            UnicodeError,
                            json.JSONDecodeError,
                            TypeError,
                            ValueError,
                        ):
                            current_payload = None
                        current_source = str(
                            current_payload.get("source_path", "")
                            if isinstance(current_payload, dict)
                            else ""
                        ).strip()
                        if current_source and Path(current_source).is_absolute():
                            # A writer repaired/recreated this companion after
                            # the scan. Preserve the now-valid metadata.
                            continue
                    try:
                        metadata_path.unlink(missing_ok=True)
                    except OSError:
                        # Malformed/orphan metadata can be retried later.
                        pass

        removed_memory_entries = 0
        committed_missing_sources: set[str] = set()
        if missing_sources and not _aborted():
            for path_key in missing_sources:
                if _aborted():
                    break
                if not self._pdf_source_definitely_missing(path_key):
                    continue
                with self._pdf_text_cache_lock:
                    removed = self._pdf_text_cache.pop(path_key, None)
                    committed_missing_sources.add(path_key)
                    if removed is not None:
                        self._pdf_text_cache_total_chars -= len(removed[2])
                        removed_memory_entries += 1
                    self._pdf_text_cache_total_chars = max(
                        0,
                        self._pdf_text_cache_total_chars,
                    )

        if committed_missing_sources:
            with self._cached_pdf_path_keys_lock:
                previous_count = len(self._cached_pdf_path_keys)
                self._cached_pdf_path_keys.difference_update(
                    committed_missing_sources
                )
                if len(self._cached_pdf_path_keys) != previous_count:
                    self._cached_pdf_path_keys_revision += 1

        if not _aborted():
            with self._pdf_text_disk_cache_lock:
                trim_revision = self._pdf_text_disk_cache_trim_requested_revision
                trim_completed = self._pdf_text_disk_cache_trim_completed_revision
            dirty_path = self._pdf_text_disk_cache_trim_dirty_path()
            dirty_identity = None
            dirty_state_unknown = False
            try:
                dirty_stat = dirty_path.stat()
                dirty_identity = (
                    int(dirty_stat.st_dev),
                    int(dirty_stat.st_ino),
                    int(dirty_stat.st_mtime_ns),
                    int(dirty_stat.st_size),
                )
            except FileNotFoundError:
                pass
            except OSError:
                dirty_state_unknown = True
            trim_needed = bool(
                trim_revision > trim_completed
                or dirty_identity is not None
                or dirty_state_unknown
            )
            if trim_needed and self._trim_pdf_text_disk_cache(
                should_abort=should_abort,
            ):
                with self._pdf_text_disk_cache_lock:
                    self._pdf_text_disk_cache_trim_completed_revision = max(
                        self._pdf_text_disk_cache_trim_completed_revision,
                        trim_revision,
                    )
                if dirty_identity is not None:
                    with advisory_file_lock(
                        self._pdf_text_disk_cache_process_lock_path(),
                        exclusive=True,
                        blocking=False,
                    ) as cache_lock_acquired:
                        if cache_lock_acquired:
                            try:
                                current_stat = dirty_path.stat()
                                current_identity = (
                                    int(current_stat.st_dev),
                                    int(current_stat.st_ino),
                                    int(current_stat.st_mtime_ns),
                                    int(current_stat.st_size),
                                )
                                if current_identity == dirty_identity:
                                    dirty_path.unlink(missing_ok=True)
                            except OSError:
                                pass

        return removed_memory_entries, removed_disk_entries

    def _maybe_start_pdf_text_cache_gc(self) -> bool:
        """Start one bounded cache-GC pass when the app remains idle."""
        if self._closing:
            return False
        if self._active_pdf_text_cache_gc_worker is not None:
            return True
        if self._active_prefetch_workers:
            return False
        now = time.monotonic()
        if self._prefetch_temporarily_paused():
            return False
        if self._global_idle_elapsed_seconds() < float(
            self.PDF_TEXT_CACHE_GC_IDLE_SECONDS
        ):
            return False
        if (
            self._search_scan_expected_workers > 0
            and self._search_scan_completed_workers < self._search_scan_expected_workers
        ):
            return False
        if (now - self._last_pdf_text_cache_gc_at) < max(
            0.1,
            float(self.PDF_TEXT_CACHE_GC_INTERVAL_SECONDS),
        ):
            return False

        request_id = self._prefetch_request_id

        def _should_abort_gc(req: int = request_id) -> bool:
            """Yield promptly once this no longer remains an idle request."""
            return bool(
                self._closing
                or req != self._prefetch_request_id
                or self._prefetch_temporarily_paused()
                or self._global_idle_elapsed_seconds()
                < float(self.PDF_TEXT_CACHE_GC_IDLE_SECONDS)
                or (
                    self._search_scan_expected_workers > 0
                    and self._search_scan_completed_workers
                    < self._search_scan_expected_workers
                )
            )

        worker = PdfTextCacheGcWorker(
            request_id,
            self._snapshot_pdf_text_cache_gc_path_keys(),
            self._pdf_text_disk_cache_dir,
            memory_cursor=self._pdf_text_cache_gc_memory_cursor,
            disk_cursor=self._pdf_text_cache_gc_disk_cursor,
            batch_size=max(1, int(self.PDF_TEXT_CACHE_GC_BATCH_SIZE)),
            should_abort=_should_abort_gc,
            payload_applier=self._apply_pdf_text_cache_gc_payload,
            lease_lock_path=self._pdf_text_background_lease_lock_path(),
        )
        worker._previous_gc_cadence_at = self._last_pdf_text_cache_gc_at
        self._last_pdf_text_cache_gc_at = now
        self._active_pdf_text_cache_gc_worker = worker
        worker.signals.finished.connect(self._on_pdf_text_cache_gc_finished)
        self._prefetch_pool.start(worker, -2)
        return True

    def _on_pdf_text_cache_gc_finished(
        self,
        request_id: int,
        payload: object,
        error_text: str,
    ) -> None:
        """Publish one worker result and return the idle pool to prefetching."""
        worker = self._active_pdf_text_cache_gc_worker
        if worker is not None and worker.request_id == request_id:
            self._active_pdf_text_cache_gc_worker = None
        lease_busy = error_text == BACKGROUND_LEASE_BUSY
        if lease_busy and worker is not None:
            self._last_pdf_text_cache_gc_at = float(
                getattr(
                    worker,
                    "_previous_gc_cadence_at",
                    self._last_pdf_text_cache_gc_at,
                )
            )
        if request_id != self._prefetch_request_id:
            if not lease_busy:
                # A now-stale GC request may already have completed safe
                # deletions before activity cancelled its remaining batch.
                self._request_cached_badge_sync()
            return
        if lease_busy:
            self._request_pdf_text_cache_gc_retry()
            return
        self._request_cached_badge_sync()
        if isinstance(payload, dict):
            self._pdf_text_cache_gc_memory_cursor = str(
                payload.get("memory_cursor", "") or ""
            )
            self._pdf_text_cache_gc_disk_cursor = str(
                payload.get("disk_cursor", "") or ""
            )
        self._request_scope_prefetch()

    def _request_pdf_text_cache_gc_retry(self) -> None:
        """Retry a lease-busy GC pass soon without changing its cadence."""
        retry_ms = max(100, int(self.SCOPE_PREFETCH_TIMER_INTERVAL_MS))
        QTimer.singleShot(retry_ms, self._maybe_start_pdf_text_cache_gc)

    def _on_pdf_text_cache_gc_timer(self) -> None:
        """Offer one idle worker slot to GC without waking scope prefetch."""
        self._maybe_start_pdf_text_cache_gc()

    def _cancel_scope_prefetch(self) -> None:
        """Handle cancel scope prefetch."""
        self._prefetch_request_id += 1
        for worker in tuple(self._active_prefetch_workers):
            try:
                if self._prefetch_pool.tryTake(worker):
                    self._active_prefetch_workers.discard(worker)
            except Exception:
                pass
        gc_worker = self._active_pdf_text_cache_gc_worker
        if gc_worker is not None:
            try:
                if self._prefetch_pool.tryTake(gc_worker):
                    self._active_pdf_text_cache_gc_worker = None
            except Exception:
                pass
        self._prefetch_pool.clear()

    def _request_scope_prefetch(self) -> None:
        """Schedule one prefetch pass through the scope timer."""
        if self._closing or not self.PREFETCH_ENABLED:
            return
        self._scope_prefetch_timer.start()

    def _start_scope_prefetch(self) -> None:
        """Start one low-priority cache-warming batch for visible PDFs.

        The scheduler is intentionally conservative:
        - pauses during interaction/search pressure,
        - prioritizes the currently open document first,
        - advances with a cursor so warmup progresses across the visible set.
        """
        if self._closing or not self.PREFETCH_ENABLED:
            return

        if self._prefetch_temporarily_paused():
            self._scope_prefetch_timer.start()
            return

        idle_elapsed = self._global_idle_elapsed_seconds()
        idle_required = float(self.PREFETCH_IDLE_SECONDS)
        if idle_elapsed < idle_required:
            remaining_ms = int(max(0.0, idle_required - idle_elapsed) * 1000) + 1
            self._scope_prefetch_timer.start(
                max(int(self.SCOPE_PREFETCH_TIMER_INTERVAL_MS), remaining_ms)
            )
            return

        now = time.monotonic()
        if now < self._next_scope_prefetch_at:
            remaining_ms = int((self._next_scope_prefetch_at - now) * 1000)
            self._scope_prefetch_timer.start(
                max(int(self.SCOPE_PREFETCH_TIMER_INTERVAL_MS), remaining_ms)
            )
            return

        if (
            self._search_scan_expected_workers > 0
            and self._search_scan_completed_workers < self._search_scan_expected_workers
        ):
            self._scope_prefetch_timer.start()
            return

        if self._active_prefetch_workers:
            self._scope_prefetch_timer.start()
            return

        if self._active_pdf_text_cache_gc_worker is not None:
            self._scope_prefetch_timer.start()
            return

        # QFileSystemModel loads directories asynchronously. Prefetch consumes
        # only rows already present in the model so an idle pass never forces a
        # directory fetch or performs a fallback filesystem walk on the GUI
        # thread.
        candidates = self._list_visible_pdf_files_in_tree(fetch_more=False)
        if not candidates:
            return

        current_scope_key = self._path_key(self._highlight_scope_directory())
        if current_scope_key != self._prefetch_scope_key:
            self._prefetch_scope_key = current_scope_key
            self._prefetch_cursor = 0
            self._prefetch_last_prioritized_current_path = ""

        total_candidates = len(candidates)
        if total_candidates <= 0:
            return

        cursor = self._prefetch_cursor % total_candidates
        batch: list[Path] = []
        visited = 0
        target_batch_size = max(1, int(self.PREFETCH_BATCH_SIZE))

        prioritized_current_identity: str | None = None
        current = self.current_file
        if isinstance(current, Path) and current.suffix.lower() == ".pdf":
            current_identity = self._path_identity_without_io(current)
            if current_identity != self._prefetch_last_prioritized_current_path:
                batch.append(current)
                prioritized_current_identity = current_identity
                self._prefetch_last_prioritized_current_path = current_identity

        while visited < total_candidates and len(batch) < target_batch_size:
            candidate = candidates[cursor]
            candidate_identity = self._path_identity_without_io(candidate)
            if (
                prioritized_current_identity
                and candidate_identity == prioritized_current_identity
            ):
                # Skip duplicate insertion when the current file is also in the
                # regular rotating candidate stream.
                cursor = (cursor + 1) % total_candidates
                visited += 1
                continue
            # Every visible candidate is worker-probed in rotation. Cache badge
            # state is intentionally not used as a validity index: signatures
            # can change and disk entries can be trimmed outside this pass.
            batch.append(candidate)
            cursor = (cursor + 1) % total_candidates
            visited += 1
        self._prefetch_cursor = cursor
        if not batch:
            self._request_cached_badge_sync()
            return

        self._cancel_scope_prefetch()
        request_id = self._prefetch_request_id
        worker = PdfTextPrefetchWorker(
            request_id,
            batch,
            lambda path, req=request_id: self._prefetch_pdf_text(
                path,
                should_abort=lambda: (
                    self._closing
                    or req != self._prefetch_request_id
                    or self._global_idle_elapsed_seconds()
                    < float(self.PREFETCH_IDLE_SECONDS)
                ),
            ),
            should_abort=lambda req=request_id: (
                self._closing
                or req != self._prefetch_request_id
                or self._global_idle_elapsed_seconds()
                < float(self.PREFETCH_IDLE_SECONDS)
                or (
                    self._search_scan_expected_workers > 0
                    and self._search_scan_completed_workers
                    < self._search_scan_expected_workers
                )
            ),
            lease_lock_path=self._pdf_text_background_lease_lock_path(),
        )
        self._active_prefetch_workers.add(worker)
        worker.signals.finished.connect(self._on_scope_prefetch_finished)
        self._prefetch_pool.start(worker, -1)

    def _on_scope_prefetch_finished(
        self,
        request_id: int,
        _prefetched_count: int,
        _skipped_count: int,
        error_text: str,
    ) -> None:
        """Handle scope prefetch finished."""
        worker_to_remove = None
        for worker in self._active_prefetch_workers:
            if worker.request_id == request_id:
                worker_to_remove = worker
                break
        if worker_to_remove is not None:
            self._active_prefetch_workers.remove(worker_to_remove)
        if error_text == BACKGROUND_LEASE_BUSY:
            self._next_scope_prefetch_at = 0.0
            self._scope_prefetch_timer.start(
                max(100, int(self.SCOPE_PREFETCH_TIMER_INTERVAL_MS))
            )
            return
        self._next_scope_prefetch_at = (
            time.monotonic() + float(self.PREFETCH_BATCH_COOLDOWN_SECONDS)
        )
        self._request_cached_badge_sync()
        # Give overdue cleanup first refusal on the shared low-priority pool so
        # a continuous sequence of one-file warmup batches cannot starve GC.
        if not self._maybe_start_pdf_text_cache_gc():
            self._request_scope_prefetch()

    def _register_global_shortcut(self, sequence: QKeySequence, callback) -> None:
        """Handle register global shortcut."""
        shortcut = QShortcut(sequence, self)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut.activated.connect(callback)
        self._global_shortcuts.append(shortcut)

    @staticmethod
    def _has_ctrl_without_alt_meta(modifiers: Qt.KeyboardModifiers) -> bool:
        """Return whether ctrl without alt meta."""
        if not (modifiers & Qt.KeyboardModifier.ControlModifier):
            return False
        if modifiers & Qt.KeyboardModifier.AltModifier:
            return False
        if modifiers & Qt.KeyboardModifier.MetaModifier:
            return False
        return True

    @classmethod
    def _is_ctrl_left_paren_key_event(cls, event: QKeyEvent) -> bool:
        """Return whether ctrl left paren key event."""
        if event.type() not in {QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride}:
            return False
        modifiers = event.modifiers()
        if not cls._has_ctrl_without_alt_meta(modifiers):
            return False
        key = event.key()
        has_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if key == Qt.Key.Key_ParenLeft:
            return True
        if key == Qt.Key.Key_9:
            return True
        if has_shift and key in {Qt.Key.Key_9, Qt.Key.Key_ParenLeft}:
            return True
        key_text = str(event.text() or "")
        if key_text in {"(", "9"}:
            return True
        return False

    @classmethod
    def _is_ctrl_bar_key_event(cls, event: QKeyEvent) -> bool:
        """Return whether ctrl bar key event."""
        if event.type() not in {QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride}:
            return False
        modifiers = event.modifiers()
        if not cls._has_ctrl_without_alt_meta(modifiers):
            return False
        key = event.key()
        if key in {Qt.Key.Key_Bar, Qt.Key.Key_Backslash}:
            return True
        key_text = str(event.text() or "")
        if key_text in {"|", "\\", "¦"}:
            return True
        return False

    @classmethod
    def _is_ctrl_right_paren_key_event(cls, event: QKeyEvent) -> bool:
        """Return whether ctrl right paren key event."""
        if event.type() not in {QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride}:
            return False
        modifiers = event.modifiers()
        if not cls._has_ctrl_without_alt_meta(modifiers):
            return False
        key = event.key()
        has_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if key == Qt.Key.Key_ParenRight:
            return True
        if has_shift and key in {Qt.Key.Key_0, Qt.Key.Key_ParenRight}:
            return True
        key_text = str(event.text() or "")
        if key_text == ")":
            return True
        return False

    def _handle_custom_shortcut_key_event(self, event: QKeyEvent) -> bool:
        """Handle handle custom shortcut key event."""
        is_toggle = self._is_ctrl_bar_key_event(event) or self._is_ctrl_left_paren_key_event(event)
        is_zoom_100 = self._is_ctrl_right_paren_key_event(event)
        if not (is_toggle or is_zoom_100):
            return False
        if event.type() == QEvent.Type.ShortcutOverride:
            return True
        if event.type() == QEvent.Type.KeyPress:
            if is_toggle:
                self.preview_toggle_three_up_action.trigger()
                return True
            if is_zoom_100:
                self.preview_zoom_one_hundred_action.trigger()
                return True
        return False

    def eventFilter(self, watched, event) -> bool:
        """Global event hook used for interaction tracking and hotkeys.

        The filter is intentionally lightweight: it records user activity,
        updates prefetch throttling, and handles custom preview shortcuts.
        """
        event_type = event.type() if event is not None else None
        if event_type in {
            QEvent.Type.KeyPress,
            QEvent.Type.KeyRelease,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.Wheel,
        }:
            self._mark_user_interaction()
        try:
            self._pause_prefetch_for_user_input(watched, event_type)
        except Exception:
            pass

        if isinstance(event, QKeyEvent):
            if self._handle_custom_shortcut_key_event(event):
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _debug_log(self, message: str) -> None:
        """Handle debug log."""
        if self.debug_mode:
            print(f"[pdfexplore] {message}", file=sys.stderr)

    @staticmethod
    def _path_key(path: Path) -> str:
        """Handle path key."""
        try:
            return str(path.resolve())
        except Exception:
            return str(path)

    @staticmethod
    def _path_identity_without_io(path: Path) -> str:
        """Return a stable lexical identity without touching the filesystem."""
        try:
            return os.path.normcase(os.path.abspath(os.fspath(path)))
        except Exception:
            return str(path)

    @staticmethod
    def _config_file_path() -> Path:
        """Handle config file path."""
        shared = _shared_config_file_path()
        return shared.with_name(CONFIG_FILE_NAME)

    def _config_lock_file_path(self) -> Path:
        """Handle config lock file path."""
        return self.config_path.with_name(self.config_path.name + ".lock")

    @classmethod
    def _parse_config_payload_text(cls, text: str) -> tuple[str | None, list[str]]:
        """Parse config payload text."""
        raw = text.strip()
        if not raw:
            return None, []
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            default_root = parsed.get(cls.CONFIG_DEFAULT_ROOT_KEY)
            recent_roots = parsed.get(cls.CONFIG_RECENT_ROOTS_KEY)
            normalized_default = (
                default_root.strip()
                if isinstance(default_root, str) and default_root.strip()
                else None
            )
            normalized_recent: list[str] = []
            if isinstance(recent_roots, list):
                for entry in recent_roots:
                    if isinstance(entry, str) and entry.strip():
                        normalized_recent.append(entry.strip())
            return normalized_default, normalized_recent
        if isinstance(parsed, str) and parsed.strip():
            return parsed.strip(), []
        return raw, []

    @classmethod
    def _normalize_recent_root_directories(
        cls, directories: list[Path | str]
    ) -> list[Path]:
        """Normalize recent root directories."""
        normalized: list[Path] = []
        seen: set[str] = set()
        for raw_entry in directories:
            if isinstance(raw_entry, Path):
                candidate = raw_entry.expanduser()
            else:
                raw_text = str(raw_entry or "").strip()
                if not raw_text:
                    continue
                candidate = Path(raw_text).expanduser()
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            try:
                if not resolved.is_dir():
                    continue
            except Exception:
                continue
            key = cls._path_key(resolved)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(resolved)
            if len(normalized) >= cls.MAX_RECENT_ROOT_DIRECTORIES:
                break
        return normalized

    def _read_config_payload(self) -> tuple[Path | None, list[Path]]:
        """Handle read config payload."""
        raw = ""
        lock_path = self._config_lock_file_path()
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+", encoding="utf-8") as lock_handle:
                lock_acquired = False
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    lock_acquired = True
                    try:
                        os.utime(lock_path, None)
                    except OSError:
                        pass
                except (BlockingIOError, OSError):
                    lock_acquired = False
                try:
                    if self.config_path.exists():
                        raw = self.config_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                finally:
                    if lock_acquired:
                        try:
                            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                        except Exception:
                            pass
        except Exception:
            try:
                if self.config_path.exists():
                    raw = self.config_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
            except Exception:
                raw = ""

        default_root_raw, recent_roots_raw = self._parse_config_payload_text(raw)
        default_root: Path | None = None
        if default_root_raw:
            normalized_default = self._normalize_recent_root_directories(
                [default_root_raw]
            )
            if normalized_default:
                default_root = normalized_default[0]
        recent_roots = self._normalize_recent_root_directories(recent_roots_raw)
        return default_root, recent_roots

    def _load_recent_root_directories_from_config(self) -> list[Path]:
        """Load recent root directories from config."""
        default_root, recent_roots = self._read_config_payload()
        merged: list[Path | str] = []
        if default_root is not None:
            merged.append(default_root)
        merged.extend(recent_roots)
        return self._normalize_recent_root_directories(merged)

    def _commit_recent_root_from_departure(
        self, departed_root: Path, elapsed_seconds: float
    ) -> None:
        """Handle commit recent root from departure."""
        if float(elapsed_seconds) < float(self.MIN_RECENT_ROOT_DWELL_SECONDS):
            return
        self._record_recent_root_directory(departed_root)

    def _on_recent_root_navigation(self, next_root: Path) -> Path:
        """Handle recent root navigation."""
        try:
            target_root = next_root.resolve()
        except Exception:
            target_root = next_root
        now = time.monotonic()
        previous_root = getattr(self, "_recent_active_root", None)
        previous_entered_at = float(
            getattr(self, "_recent_root_entered_at", now) or now
        )
        if isinstance(previous_root, Path):
            if self._path_key(previous_root) != self._path_key(target_root):
                elapsed = max(0.0, now - previous_entered_at)
                self._commit_recent_root_from_departure(previous_root, elapsed)
        self._recent_active_root = target_root
        self._recent_root_entered_at = now
        return target_root

    def _refresh_recent_root_menu(self) -> None:
        """Refresh recent root menu."""
        menu = getattr(self, "recent_menu", None)
        if menu is None:
            return
        menu.clear()
        if not self._recent_root_directories:
            empty_action = menu.addAction("(No recent directories)")
            empty_action.setEnabled(False)
            return

        current_key = self._path_key(self.root) if isinstance(self.root, Path) else ""
        mru_count = max(
            0,
            min(
                int(self.RECENT_ROOT_MENU_MRU_COUNT),
                len(self._recent_root_directories),
            ),
        )
        recent_directories = self._recent_root_directories[:mru_count]
        remaining_directories = sorted(
            self._recent_root_directories[mru_count:],
            key=lambda path: str(path).casefold(),
        )

        def _add_recent_action(directory: Path) -> None:
            """Handle add recent action."""
            label = str(directory)
            if current_key and self._path_key(directory) == current_key:
                label = f"{label} (current)"
            action = menu.addAction(label)
            action.triggered.connect(
                lambda _checked=False, target=directory: self._open_recent_root_directory(
                    target
                )
            )

        for directory in recent_directories:
            _add_recent_action(directory)

        if recent_directories and remaining_directories:
            menu.addSeparator()

        for directory in remaining_directories:
            _add_recent_action(directory)

    def _reload_recent_root_directories_before_menu_open(self) -> None:
        """Handle reload recent root directories before menu open."""
        on_disk = self._load_recent_root_directories_from_config()
        merged: list[Path | str] = []
        merged.extend(on_disk)
        merged.extend(self._recent_root_directories)
        self._recent_root_directories = self._normalize_recent_root_directories(merged)
        self._refresh_recent_root_menu()

    def _record_recent_root_directory(self, directory: Path) -> None:
        """Handle record recent root directory."""
        directory_key = self._path_key(directory)
        self._recent_root_events = [
            path
            for path in self._recent_root_events
            if self._path_key(path) != directory_key
        ]
        self._recent_root_events.append(directory)
        merged: list[Path | str] = [directory]
        merged.extend(self._recent_root_directories)
        self._recent_root_directories = self._normalize_recent_root_directories(merged)
        self._refresh_recent_root_menu()

    def _open_recent_root_directory(self, directory: Path) -> None:
        """Open recent root directory."""
        normalized = self._normalize_recent_root_directories([directory])
        if not normalized:
            self._recent_root_directories = self._normalize_recent_root_directories(
                self._recent_root_directories
            )
            self._refresh_recent_root_menu()
            self.statusBar().showMessage("Recent directory is unavailable", 3500)
            return
        target = normalized[0]
        if self._path_key(target) == self._path_key(self.root):
            self._refresh_recent_root_menu()
            return
        self._set_root_directory(target)

    def _update_up_button_state(self) -> None:
        """Handle update up button state."""
        self.up_btn.setEnabled(self.root.parent != self.root)

    def _invalidate_visible_tree_pdf_cache(self) -> None:
        """Handle invalidate visible tree pdf cache."""
        self._visible_tree_pdf_cache_dirty = True
        self._visible_tree_pdf_cache_fetch_more_complete = False

    def _on_model_directory_loaded(self, _path_text: str) -> None:
        """Refresh async QFileSystemModel visibility after a directory loads."""
        self._invalidate_visible_tree_pdf_cache()
        self._request_cached_badge_probe()
        match_input = getattr(self, "match_input", None)
        if match_input is not None and match_input.text().strip():
            self._rerun_active_search_for_scope()
        else:
            self._request_scope_prefetch()

    def _pause_prefetch_temporarily(
        self,
        seconds: float,
        *,
        cancel_inflight: bool = False,
    ) -> None:
        """Handle pause prefetch temporarily."""
        pause_until = time.monotonic() + max(0.0, float(seconds))
        self._prefetch_paused_until = max(self._prefetch_paused_until, pause_until)
        current_request_id = self._prefetch_request_id
        has_current_worker = any(
            worker.request_id == current_request_id
            for worker in self._active_prefetch_workers
        )
        gc_worker = self._active_pdf_text_cache_gc_worker
        has_current_gc = bool(
            gc_worker is not None and gc_worker.request_id == current_request_id
        )
        if cancel_inflight and (has_current_worker or has_current_gc):
            self._cancel_scope_prefetch()

    def _reset_search_scan_state(self) -> None:
        """Handle reset search scan state."""
        self._search_scan_expected_workers = 0
        self._search_scan_completed_workers = 0
        self._search_scan_total_candidates = 0
        self._search_scan_scope = None
        self._search_scan_candidate_order.clear()
        self._search_scan_match_counts.clear()
        self._search_scan_filename_match_paths.clear()
        self._search_scan_error_count = 0

    def _cancel_pending_search_scan(self) -> None:
        """Handle cancel pending search scan."""
        self._search_progress_publish_timer.stop()
        self._search_request_id += 1
        for worker in list(self._active_search_workers):
            try:
                if self.thread_pool.tryTake(worker):
                    self._active_search_workers.discard(worker)
            except Exception:
                pass
        self.thread_pool.clear()
        self._reset_search_scan_state()

    def _cancel_pending_tree_marker_scan(self) -> None:
        """Handle cancel pending tree marker scan."""
        self._tree_marker_scan_request_id += 1
        for worker in list(self._active_tree_marker_scan_workers):
            try:
                if self._tree_marker_scan_pool.tryTake(worker):
                    self._active_tree_marker_scan_workers.discard(worker)
            except Exception:
                pass
        self._tree_marker_scan_pool.clear()

    def _rerun_active_search_for_scope(self) -> None:
        """Handle rerun active search for scope."""
        if not self.match_input.text().strip():
            self._request_scope_prefetch()
            return
        # Halt stale extraction immediately when scope visibility changes.
        self._cancel_pending_search_scan()
        self.match_timer.stop()
        self._tree_search_refresh_timer.stop()
        delay_ms = int(self.TREE_SEARCH_REFRESH_DEBOUNCE_MS)
        remaining_pause_seconds = max(
            0.0,
            float(self._prefetch_paused_until) - time.monotonic(),
        )
        if remaining_pause_seconds > 0.0:
            delay_ms = max(delay_ms, int((remaining_pause_seconds * 1000.0) + 80.0))
        self._tree_search_refresh_timer.start(max(1, delay_ms))

    def _merge_live_tree_marker_state(
        self,
        multi_view_paths: set[str] | None = None,
        highlighted_paths: set[str] | None = None,
        *,
        root_key: str | None = None,
    ) -> tuple[set[str], set[str]]:
        """Merge marker state from scans with current live/in-memory state.

        This method protects marker continuity while background scans race with
        active user actions (tab switches, fresh highlight edits).
        """
        merged_multi_view = set(multi_view_paths or ())
        merged_highlighted = set(highlighted_paths or ())
        if root_key is None:
            root = getattr(self, "root", None)
            if isinstance(root, Path):
                root_key = self._path_key(root)

        open_counts: dict[str, int] = {}
        for index in range(self.view_tabs.count()):
            data = self._tab_data(index)
            if not isinstance(data, dict):
                continue
            path_key = str(data.get("path_key") or "").strip()
            if not path_key:
                continue
            if root_key and not (
                path_key == root_key or path_key.startswith(root_key + os.sep)
            ):
                continue
            open_counts[path_key] = open_counts.get(path_key, 0) + 1
        for path_key, count in open_counts.items():
            if count > 1:
                merged_multi_view.add(path_key)

        # Keep marker badges for any files with known persisted highlights that
        # are already loaded in-memory. This prevents stale background scan
        # payloads from temporarily dropping highlight markers on tab switches.
        for directory_key, by_file in self._persisted_text_highlights_by_dir.items():
            if not isinstance(directory_key, str) or not isinstance(by_file, dict):
                continue
            if root_key and not self._path_key_is_under_root(directory_key, root_key):
                continue
            for file_name, entries in by_file.items():
                if not isinstance(file_name, str) or not entries:
                    continue
                candidate_key = self._path_key_from_parts(directory_key, file_name)
                merged_highlighted.add(candidate_key)

        current_path_key = self._current_preview_path_key()
        if (
            current_path_key
            and self._load_text_highlights_for_path_key(current_path_key)
            and (not root_key or self._path_key_is_under_root(current_path_key, root_key))
        ):
            merged_highlighted.add(current_path_key)

        return merged_multi_view, merged_highlighted

    def _start_tree_marker_scan(self) -> None:
        """Handle start tree marker scan."""
        root = getattr(self, "root", None)
        if not isinstance(root, Path) or not root.exists():
            self._tree_multi_view_marker_paths.clear()
            self._tree_highlight_marker_paths.clear()
            self._tree_marker_cache_root_key = None
            self._sync_tree_markers_to_model()
            return

        self._cancel_pending_tree_marker_scan()
        request_id = self._tree_marker_scan_request_id
        worker = PdfTreeMarkerScanWorker(
            root,
            request_id,
            VIEWS_FILE_NAME,
            HIGHLIGHTING_FILE_NAME,
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
        """Handle tree marker scan finished."""
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
                f"Tree badge scan failed: {error_text}",
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
        self._sync_tree_markers_to_model()

    def _clear_preview_for_missing_file(self) -> None:
        """Handle clear preview for missing file."""
        self.current_file = None
        self.path_label.setText("")
        self.path_label.setToolTip("")
        self.preview_stack.setCurrentWidget(self._empty_preview)
        self._update_dark_mode_button_state()
        blocked = self.view_tabs.blockSignals(True)
        while self.view_tabs.count() > 0:
            self.view_tabs.removeTab(0)
        self.view_tabs.blockSignals(blocked)
        self._active_view_tab_index = -1
        self._refresh_view_tabs_visibility()

    def _update_dark_mode_button_state(self) -> None:
        """Enable dark mode only while a PDF preview is visible."""
        self.dark_mode_btn.setEnabled(self._current_preview_widget() is not None)

    def _update_edit_button_state(self) -> None:
        """Handle update edit button state."""
        enabled = False
        current = getattr(self, "current_file", None)
        if isinstance(current, Path):
            try:
                enabled = current.is_file()
            except Exception:
                enabled = False
        self.edit_btn.setEnabled(enabled)

    def _pdf_directory_paths_for_expansion(self) -> list[Path]:
        """Return PDF-bearing directories and the ancestors needed to reveal them."""
        try:
            root = self.root.resolve()
        except Exception:
            root = self.root
        if not root.is_dir():
            return []

        paths: set[Path] = set()
        for current_text, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            current = Path(current_text)
            directory_names[:] = sorted(
                (
                    name
                    for name in directory_names
                    if not (current / name).is_symlink()
                ),
                key=str.casefold,
            )
            if not any(name.lower().endswith(".pdf") for name in file_names):
                continue

            candidate = current
            while candidate != root:
                try:
                    candidate.relative_to(root)
                except ValueError:
                    break
                paths.add(candidate)
                candidate = candidate.parent

        return sorted(
            paths,
            key=lambda path: (len(path.relative_to(root).parts), str(path).casefold()),
        )

    def _expand_pdf_directories(self, _checked: bool = False) -> None:
        """Expand all PDF-bearing directories beneath the current root."""
        self._set_tree_interaction_visual_mode(True)
        self._tree_visual_relax_timer.start()
        self._pause_prefetch_temporarily(
            self.PREFETCH_TREE_MUTATION_PAUSE_SECONDS,
            cancel_inflight=True,
        )

        paths = self._pdf_directory_paths_for_expansion()
        expanded_count = 0
        for path in paths:
            parent_index = (
                self.tree.rootIndex()
                if path.parent == self.root
                else self.model.index(str(path.parent))
            )
            if parent_index.isValid() and self.model.canFetchMore(parent_index):
                self.model.fetchMore(parent_index)
                QApplication.processEvents(
                    QEventLoop.ProcessEventsFlag.AllEvents,
                    10,
                )
            index = self.model.index(str(path))
            if not index.isValid():
                QApplication.processEvents(
                    QEventLoop.ProcessEventsFlag.AllEvents,
                    10,
                )
                index = self.model.index(str(path))
            if index.isValid() and not self.tree.isExpanded(index):
                self.tree.expand(index)
                expanded_count += 1

        self._invalidate_visible_tree_pdf_cache()
        if self.match_input.text().strip():
            self._rerun_active_search_for_scope()
        else:
            self._request_scope_prefetch()
        self.statusBar().showMessage(
            f"Expanded {expanded_count} PDF directory branch(es)",
            3000,
        )

    def _collapse_all_directories(self, _checked: bool = False) -> None:
        """Collapse every directory branch beneath the current root."""
        self._set_tree_interaction_visual_mode(True)
        self._tree_visual_relax_timer.start()
        self._pause_prefetch_temporarily(
            self.PREFETCH_TREE_MUTATION_PAUSE_SECONDS,
            cancel_inflight=True,
        )
        self.tree.collapseAll()
        self._invalidate_visible_tree_pdf_cache()
        if self.match_input.text().strip():
            self._rerun_active_search_for_scope()
        else:
            self._request_scope_prefetch()
        self.statusBar().showMessage("Collapsed all directory branches", 3000)

    def _expanded_directory_paths(self) -> list[str]:
        """Handle expanded directory paths."""
        expanded: list[str] = []
        model = self.tree.model()
        if model is None:
            return expanded
        root_index = self.tree.rootIndex()
        for row in range(model.rowCount(root_index)):
            child_index = model.index(row, 0, root_index)
            self._collect_expanded_paths(child_index, expanded)
        return expanded

    def _collect_expanded_paths(self, index, expanded: list[str]) -> None:
        """Handle collect expanded paths."""
        if not index.isValid():
            return
        if self.tree.isExpanded(index):
            path_text = str(self.model.filePath(index))
            if path_text:
                expanded.append(path_text)
            for row in range(self.model.rowCount(index)):
                child_index = self.model.index(row, 0, index)
                self._collect_expanded_paths(child_index, expanded)

    def _restore_expanded_directory_paths(self, paths: list[str]) -> None:
        """Handle restore expanded directory paths."""
        for path_text in paths:
            index = self.model.index(path_text)
            if index.isValid():
                self.tree.expand(index)

    def _set_preview_signature_for_path(self, path: Path) -> None:
        """Set preview signature for path."""
        path_key = self._path_key(path)
        try:
            stat = path.stat()
        except Exception:
            return
        self._preview_signatures_by_path[path_key] = (int(stat.st_mtime_ns), int(stat.st_size))

    def _reload_current_preview(self, reason: str) -> None:
        """Handle reload current preview."""
        if self.current_file is None:
            return
        path = self.current_file
        path_key = self._path_key(path)
        self._invalidate_cached_pdf_text(path_key)
        current_index = self.view_tabs.currentIndex()
        if current_index >= 0:
            self._capture_tab_state(current_index, blocking=True)
            data = self._tab_data(current_index)
            if isinstance(data, dict) and isinstance(data.get("state"), dict):
                self._viewer_pending_restore_state_by_path[path_key] = self._clone_json_compatible_dict(
                    data.get("state")
                )
        preview = self._preview_widget_for_path(path)
        self.preview_stack.setCurrentWidget(preview)
        self._update_dark_mode_button_state()
        self._trim_preview_widget_cache(protected_path_keys={path_key})
        self._viewer_bridge_ready_by_path[path_key] = False
        preview.setUrl(self._viewer_url_for_pdf(path))
        self.statusBar().showMessage(
            f"Auto-refreshed preview: {path.name} ({reason})",
            3500,
        )

    def _on_file_change_watch_tick(self) -> None:
        """Handle file change watch tick."""
        if self.current_file is None:
            return
        path = self.current_file
        path_key = self._path_key(path)
        try:
            stat = path.stat()
        except Exception:
            return
        current_sig = (int(stat.st_mtime_ns), int(stat.st_size))
        previous_sig = self._preview_signatures_by_path.get(path_key)
        if previous_sig is None:
            self._preview_signatures_by_path[path_key] = current_sig
            return
        if previous_sig == current_sig:
            return
        self._preview_signatures_by_path[path_key] = current_sig
        self._reload_current_preview("file changed on disk")

    def _directory_key(self, directory: Path) -> str:
        """Handle directory key."""
        return self._path_key(directory)

    def _path_directory_and_name(self, path_key: str | None) -> tuple[Path, str] | None:
        """Handle path directory and name."""
        if not path_key:
            return None
        path = Path(path_key)
        return path.parent, path.name

    def _read_pdf_text(self, path: Path, *, should_abort=None) -> str:
        """Return searchable text with cooperative, cross-process caching."""

        def _aborted() -> bool:
            return bool(callable(should_abort) and should_abort())

        def _signature(source_stat) -> tuple[int, int]:
            return int(source_stat.st_mtime_ns), int(source_stat.st_size)

        def _extract(source_path: Path) -> str | None:
            """Extract text, returning ``None`` only for cancellation."""
            parts: list[str] = []
            try:
                reader = PdfReader(str(source_path))
                for page in reader.pages:
                    if _aborted():
                        return None
                    try:
                        extracted = page.extract_text() or ""
                    except Exception:
                        extracted = ""
                    if extracted:
                        parts.append(extracted)
            except Exception:
                return ""
            return None if _aborted() else "\n\n".join(parts)

        if _aborted():
            return ""

        while not _aborted():
            try:
                resolved_path = path.resolve(strict=True)
                path_key = str(resolved_path)
                initial_stat = resolved_path.stat()
            except (OSError, RuntimeError):
                return ""
            initial_signature = _signature(initial_stat)
            mtime_ns, size = initial_signature

            # Use a canonical-path-stable stripe, rather than a
            # version-specific cache-key stripe, so a PDF replaced in place
            # cannot be extracted under two locks. Canonical ownership is
            # recomputed on every retry so retargeted symlinks never file one
            # target's text under another target's key.
            producer_digest = hashlib.sha256(
                path_key.encode("utf-8", errors="surrogatepass")
            ).hexdigest()
            producer_lock_path = self._pdf_text_disk_cache_dir / (
                f".pdfexplore-producer-{producer_digest[:2]}.lock"
            )

            cached = self._lookup_cached_pdf_text(path_key, mtime_ns, size)
            if cached is not None:
                try:
                    if str(path.resolve(strict=True)) != path_key:
                        continue
                    if _signature(resolved_path.stat()) != initial_signature:
                        continue
                except (OSError, RuntimeError):
                    return ""
                self._record_cached_pdf_path_key(path_key)
                return cached

            disk_cached = self._load_pdf_text_from_disk_cache(
                path_key,
                mtime_ns,
                size,
                should_abort=should_abort,
            )
            if _aborted():
                return ""
            if disk_cached is not None:
                try:
                    if str(path.resolve(strict=True)) != path_key:
                        continue
                    if _signature(resolved_path.stat()) != initial_signature:
                        continue
                except (OSError, RuntimeError):
                    return ""
                self._store_cached_pdf_text(
                    path_key,
                    mtime_ns,
                    size,
                    disk_cached,
                )
                self._record_cached_pdf_path_key(path_key)
                return disk_cached

            with advisory_file_lock(
                producer_lock_path,
                exclusive=True,
                blocking=True,
                should_abort=should_abort,
            ) as producer_acquired:
                if _aborted():
                    return ""

                # Lock setup can fail on a damaged/read-only cache location.
                # Search must remain functional in that case, so extraction
                # continues with the process-local memory cache only.
                coordinated = bool(producer_acquired)
                try:
                    if str(path.resolve(strict=True)) != path_key:
                        continue
                    locked_stat = resolved_path.stat()
                except (OSError, RuntimeError):
                    return ""
                locked_signature = _signature(locked_stat)
                if coordinated and locked_signature != initial_signature:
                    continue
                mtime_ns, size = locked_signature

                cached = self._lookup_cached_pdf_text(path_key, mtime_ns, size)
                if cached is not None:
                    try:
                        if str(path.resolve(strict=True)) != path_key:
                            continue
                        if _signature(resolved_path.stat()) != locked_signature:
                            continue
                    except (OSError, RuntimeError):
                        return ""
                    self._record_cached_pdf_path_key(path_key)
                    return cached

                if coordinated:
                    disk_cached = self._load_pdf_text_from_disk_cache(
                        path_key,
                        mtime_ns,
                        size,
                        should_abort=should_abort,
                    )
                    if _aborted():
                        return ""
                    if disk_cached is not None:
                        try:
                            if str(path.resolve(strict=True)) != path_key:
                                continue
                            if _signature(resolved_path.stat()) != locked_signature:
                                continue
                        except (OSError, RuntimeError):
                            return ""
                        self._store_cached_pdf_text(
                            path_key,
                            mtime_ns,
                            size,
                            disk_cached,
                        )
                        self._record_cached_pdf_path_key(path_key)
                        return disk_cached

                text = _extract(resolved_path)
                if text is None or _aborted():
                    return ""
                try:
                    if str(path.resolve(strict=True)) != path_key:
                        continue
                    if _signature(resolved_path.stat()) != locked_signature:
                        continue
                except (OSError, RuntimeError):
                    return ""

                self._store_cached_pdf_text(path_key, mtime_ns, size, text)
                if coordinated:
                    self._store_pdf_text_to_disk_cache(
                        path_key,
                        mtime_ns,
                        size,
                        text,
                        should_abort=should_abort,
                    )
                    if _aborted():
                        return ""
                self._record_cached_pdf_path_key(path_key)
                return text

        return ""

    def _prefetch_pdf_text(self, path: Path, *, should_abort=None) -> str:
        """Probe or warm one text-cache entry entirely off the GUI thread."""
        if callable(should_abort) and should_abort():
            return ""
        if self._is_pdf_text_cached_for_path(
            path,
            mark_cached_badge=True,
            should_abort=should_abort,
        ):
            return ""
        return self._read_pdf_text(path, should_abort=should_abort)

    def _list_pdf_files_non_recursive(self, directory: Path) -> list[Path]:
        """Handle list pdf files non recursive."""
        if not directory.is_dir():
            return []
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
        except Exception:
            return []
        files: list[Path] = []
        for entry in entries:
            try:
                if entry.is_file() and entry.suffix.lower() == ".pdf":
                    files.append(entry.resolve())
            except Exception:
                pass
        return files

    def _list_visible_pdf_files_in_tree(self, *, fetch_more: bool = True) -> list[Path]:
        """Handle list visible pdf files in tree."""
        if not self._visible_tree_pdf_cache_dirty and (
            (not fetch_more) or self._visible_tree_pdf_cache_fetch_more_complete
        ):
            return list(self._visible_tree_pdf_cache)

        root_index = self.tree.rootIndex()
        if not root_index.isValid():
            if not fetch_more:
                return []
            files = self._list_pdf_files_non_recursive(self._highlight_scope_directory())
            self._visible_tree_pdf_cache = list(files)
            self._visible_tree_pdf_cache_dirty = False
            self._visible_tree_pdf_cache_fetch_more_complete = bool(fetch_more)
            return list(files)

        files: list[Path] = []
        seen: set[str] = set()

        def _append_file(path: Path) -> None:
            """Handle append file."""
            key = self._path_identity_without_io(path)
            if key in seen:
                return
            seen.add(key)
            files.append(path)

        def _walk_visible_children(parent_index) -> None:
            """Handle walk visible children."""
            if fetch_more and self.model.canFetchMore(parent_index):
                self.model.fetchMore(parent_index)
            row_count = self.model.rowCount(parent_index)
            for row in range(row_count):
                if self.tree.isRowHidden(row, parent_index):
                    continue
                index = self.model.index(row, 0, parent_index)
                if not index.isValid():
                    continue
                path = Path(self.model.filePath(index))
                is_dir = bool(self.model.isDir(index))
                if is_dir and self.tree.isExpanded(index):
                    _walk_visible_children(index)
                elif not is_dir and path.suffix.lower() == ".pdf":
                    _append_file(path)

        _walk_visible_children(root_index)
        self._visible_tree_pdf_cache = list(files)
        self._visible_tree_pdf_cache_dirty = False
        self._visible_tree_pdf_cache_fetch_more_complete = bool(fetch_more)
        return list(files)

    def _highlight_scope_directory(self) -> Path:
        """Handle highlight scope directory."""
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

    def _effective_root_for_persistence(self) -> Path:
        """Handle effective root for persistence."""
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
        """Handle persist effective root."""
        scope = self._effective_root_for_persistence()
        try:
            resolved_scope = scope.resolve()
        except Exception:
            resolved_scope = scope

        lock_path = self._config_lock_file_path()
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        def _serialize_payload(recent_roots: list[Path]) -> str:
            """Handle serialize payload."""
            payload = {
                self.CONFIG_DEFAULT_ROOT_KEY: str(resolved_scope),
                self.CONFIG_RECENT_ROOTS_KEY: [str(path) for path in recent_roots],
            }
            return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

        try:
            with lock_path.open("a+", encoding="utf-8") as lock_handle:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (BlockingIOError, OSError):
                    return
                try:
                    os.utime(lock_path, None)
                except OSError:
                    pass

                raw = ""
                try:
                    if self.config_path.exists():
                        raw = self.config_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                except Exception:
                    raw = ""

                _on_disk_default, on_disk_recent = self._parse_config_payload_text(raw)
                merged_recent = self._normalize_recent_root_directories(
                    list(on_disk_recent)
                )
                for event_path in self._recent_root_events:
                    event_key = self._path_key(event_path)
                    merged_recent = [
                        path
                        for path in merged_recent
                        if self._path_key(path) != event_key
                    ]
                    merged_recent.insert(0, event_path)
                    merged_recent = self._normalize_recent_root_directories(
                        merged_recent
                    )

                serialized = _serialize_payload(merged_recent)
                atomic_write_text(self.config_path, serialized)

                self._recent_root_directories = merged_recent
                self._recent_root_events.clear()
                self._refresh_recent_root_menu()

                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
        except Exception:
            pass

    def _set_root_directory(self, directory: Path) -> None:
        """Set root directory."""
        self._cancel_pending_search_scan()
        self._cancel_scope_prefetch()
        self._cancel_cached_badge_probe()
        target_root = self._on_recent_root_navigation(directory)
        if self.current_file is not None:
            self._persist_document_view_session(
                self._path_key(self.current_file), capture_current=True
            )
        self._invalidate_visible_tree_pdf_cache()
        self.root = target_root
        self.model.setRootPath("")
        root_index = self.model.setRootPath(str(self.root))
        self.tree.setRootIndex(root_index)
        self.tree.expand(root_index)
        self.last_directory_selection = self.root
        self.tree.clearSelection()
        self._clear_preview_for_missing_file()
        self._rebuild_tree_marker_cache()
        self._update_window_title()
        self._update_up_button_state()
        self._persist_effective_root()
        self.statusBar().showMessage(f"Root changed to {self.root}", 3000)
        self._request_cached_badge_probe()
        if self.match_input.text().strip():
            self._run_match_search()
        else:
            self._request_scope_prefetch()

    def _go_up_directory(self) -> None:
        """Handle go up directory."""
        parent = self.root.parent
        if parent == self.root:
            return
        self._set_tree_interaction_visual_mode(True)
        self._tree_visual_relax_timer.start()
        self._set_root_directory(parent)

    def _refresh_directory_view(self, _checked: bool = False) -> None:
        """Refresh directory view."""
        self.statusBar().showMessage("Refreshing directory view...")
        self._cancel_cached_badge_probe()
        self._invalidate_visible_tree_pdf_cache()
        self._persisted_view_sessions_by_dir.clear()
        self._persisted_text_highlights_by_dir.clear()
        self.model.invalidate_persisted_color_cache()
        active_path_key = (
            self._path_key(self.current_file)
            if self.current_file is not None
            else ""
        )
        active_live_session = self._document_view_sessions.get(active_path_key)
        self._document_view_sessions.clear()
        if active_path_key and isinstance(active_live_session, dict):
            # Preserve only the document the user is actively editing. Other
            # cached sessions must be reloaded from disk after Refresh so a
            # sibling process's committed changes are not overwritten later.
            self._document_view_sessions[active_path_key] = active_live_session
        if self.current_file is not None:
            self._current_text_highlights = self._load_text_highlights_for_path_key(
                active_path_key
            )
            self._apply_persistent_text_highlights()
        selected_path: Path | None = None
        current_index = self.tree.currentIndex()
        if current_index.isValid():
            selected = Path(self.model.filePath(current_index))
            try:
                selected_path = selected.resolve()
            except Exception:
                selected_path = selected

        expanded_paths = self._expanded_directory_paths()

        self.model.setRootPath("")
        root_index = self.model.setRootPath(str(self.root))
        self.tree.setRootIndex(root_index)
        self._rebuild_tree_marker_cache()
        self._start_tree_marker_scan()

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
                file_exists = self.current_file.is_file()
            except Exception:
                file_exists = False
            if not file_exists:
                self._clear_preview_for_missing_file()
                if not restored_selection:
                    self.tree.clearSelection()
                self.statusBar().showMessage(
                    "Directory view refreshed; preview file no longer exists",
                    4500,
                )
            else:
                self.statusBar().showMessage("Directory view refreshed", 2500)
        else:
            self.statusBar().showMessage("Directory view refreshed", 2500)

        self._update_window_title()
        self._update_up_button_state()
        self._request_cached_badge_probe()
        if self.match_input.text().strip():
            self._run_match_search()
        else:
            self._request_scope_prefetch()

    def _on_match_text_changed(self, text: str) -> None:
        """Handle match text changed."""
        self.match_clear_action.setVisible(bool(text.strip()))
        self._cancel_pending_search_scan()
        if not text.strip():
            self.match_timer.stop()
            self._tree_search_refresh_timer.stop()
            self._clear_match_results()
            return
        self.match_timer.start()

    def _clear_match_input(self) -> None:
        """Handle clear match input."""
        self.match_timer.stop()
        self._tree_search_refresh_timer.stop()
        self.match_input.clear()
        self._clear_match_results()

    def _run_match_search_now(self) -> None:
        """Run match search now."""
        self.match_timer.stop()
        self._tree_search_refresh_timer.stop()
        self._run_match_search()

    def _run_match_search(self) -> None:
        """Run match search."""
        self._search_progress_publish_timer.stop()
        query = self.match_input.text().strip()
        if not query:
            self._clear_match_results()
            return
        scope = self._highlight_scope_directory()
        candidates = self._list_visible_pdf_files_in_tree(fetch_more=False)
        prioritized_candidates = list(candidates)
        self._cancel_scope_prefetch()
        self._cancel_pending_search_scan()
        request_id = self._search_request_id
        self._search_scan_total_candidates = len(candidates)
        self._search_scan_scope = scope
        self._search_scan_candidate_order = {
            self._path_key(path): index for index, path in enumerate(candidates)
        }
        self.current_match_files = []
        self._current_match_counts = {}
        self.model.clear_search_match_paths()
        self._sync_tree_markers_to_model()

        if not candidates:
            self.current_match_files = []
            self._current_match_counts = {}
            self.model.clear_search_match_paths()
            self.tree.viewport().update()
            self.statusBar().showMessage(
                f"Matched 0 of 0 visible PDF file(s) in tree under {scope}",
                2500,
            )
            self._remove_viewer_search_highlights()
            self._reset_search_scan_state()
            return

        try:
            predicate = compile_match_predicate(query)
            hit_counter = compile_match_hit_counter(query)
            filename_patterns = []
            for term_text, is_case_sensitive in extract_search_terms(query):
                if not term_text:
                    continue
                filename_patterns.append(
                    compile_term_pattern(term_text, bool(is_case_sensitive))
                )
        except Exception as exc:
            self.statusBar().showMessage(
                f"Search query preparation failed: {str(exc)}",
                4000,
            )
            self._clear_match_results()
            return

        # Small multi-file chunks preserve progressive results without creating
        # one QRunnable, queued signal, and tree repaint per PDF.
        chunk_size = max(1, int(self.SEARCH_WORKER_CHUNK_SIZE))
        started_workers = 0
        for start in range(0, len(prioritized_candidates), chunk_size):
            chunk = prioritized_candidates[start : start + chunk_size]
            if not chunk:
                continue
            worker = PdfSearchWorker(
                request_id,
                chunk,
                predicate,
                hit_counter,
                filename_patterns,
                lambda path, req=request_id: self._read_pdf_text(
                    path,
                    should_abort=lambda: req != self._search_request_id,
                ),
                should_abort=lambda req=request_id: req != self._search_request_id,
            )
            self._active_search_workers.add(worker)
            worker.signals.finished.connect(self._on_search_finished)
            self.thread_pool.start(worker)
            started_workers += 1

        self._search_scan_expected_workers = started_workers
        self.statusBar().showMessage(
            f"Searching {len(candidates)} visible PDF file(s) in tree under {scope} "
            f"in {started_workers} batch(es)..."
        )

    def _publish_search_scan_progress(self) -> None:
        """Publish the current aggregated search result set to tree/viewer state."""
        ordered_match_keys = sorted(
            self._search_scan_match_counts.keys(),
            key=lambda key: self._search_scan_candidate_order.get(key, 10**9),
        )
        self.current_match_files = [Path(path_key) for path_key in ordered_match_keys]
        self._current_match_counts = {
            Path(path_key): self._search_scan_match_counts[path_key]
            for path_key in ordered_match_keys
        }
        self.model.set_search_match_counts(
            self._current_match_counts,
            filename_match_path_keys=self._search_scan_filename_match_paths,
        )
        self.tree.viewport().update()

    def _on_search_finished(
        self,
        request_id: int,
        matched_paths,
        match_counts,
        filename_match_paths,
        error: str,
    ) -> None:
        """Handle search finished."""
        worker_to_remove = None
        for worker in self._active_search_workers:
            if worker.request_id == request_id:
                worker_to_remove = worker
                break
        if worker_to_remove is not None:
            self._active_search_workers.remove(worker_to_remove)

        # Cache publication is independent of whether this result still belongs
        # to the active query. A cancelled/stale worker may have completed valid
        # extraction before noticing cancellation, and its tree badge should not
        # remain hidden until some later search or prefetch cycle.
        self._request_cached_badge_sync()
        if request_id != self._search_request_id:
            return
        self._search_scan_completed_workers += 1
        if error:
            self._search_scan_error_count += 1

        current_preview_key = self._current_preview_path_key()
        current_preview_was_matched = bool(
            current_preview_key
            and current_preview_key in self._search_scan_match_counts
        )
        if isinstance(match_counts, dict):
            for raw_path_key, raw_count in match_counts.items():
                if not isinstance(raw_path_key, str):
                    continue
                try:
                    count = int(raw_count)
                except Exception:
                    count = 0
                self._search_scan_match_counts[raw_path_key] = count if count > 0 else 1
        if isinstance(matched_paths, (list, tuple, set)):
            for raw_path_key in matched_paths:
                if not isinstance(raw_path_key, str):
                    continue
                if raw_path_key not in self._search_scan_match_counts:
                    self._search_scan_match_counts[raw_path_key] = 1
        if isinstance(filename_match_paths, (list, tuple, set)):
            for raw_path_key in filename_match_paths:
                if isinstance(raw_path_key, str):
                    self._search_scan_filename_match_paths.add(raw_path_key)

        search_complete = (
            self._search_scan_completed_workers >= self._search_scan_expected_workers
        )
        if search_complete:
            self._search_progress_publish_timer.stop()
            self._publish_search_scan_progress()
        elif self._search_scan_completed_workers == 1:
            # Surface the first useful result promptly, then debounce later
            # batches so large searches do not repaint the tree continuously.
            self._publish_search_scan_progress()
        elif not self._search_progress_publish_timer.isActive():
            self._search_progress_publish_timer.start()

        if (
            current_preview_key
            and not current_preview_was_matched
            and current_preview_key in self._search_scan_match_counts
        ):
            self._apply_active_search_to_viewer()

        if not search_complete:
            scope = self._search_scan_scope or self._highlight_scope_directory()
            status_stride = max(1, self._search_scan_expected_workers // 10)
            if (
                self._search_scan_completed_workers == 1
                or self._search_scan_completed_workers % status_stride == 0
            ):
                self.statusBar().showMessage(
                    (
                        f"Searching {self._search_scan_total_candidates} visible PDF file(s) in tree under {scope}... "
                        f"matched {len(self._search_scan_match_counts)} so far "
                        f"({self._search_scan_completed_workers}/{self._search_scan_expected_workers} batches processed). "
                        "Full results will complete after all visible files are extracted/cached."
                    )
                )
            return

        # Re-apply marker sets once at completion so marker continuity stays
        # correct without forcing full marker-sync work on every file update.
        self._sync_tree_markers_to_model()
        self._apply_active_search_to_viewer()

        scope = self._search_scan_scope or self._highlight_scope_directory()
        status_message = (
            f"Matched {len(self.current_match_files)} of {self._search_scan_total_candidates} visible PDF file(s) in tree under {scope}"
        )
        if self._search_scan_error_count > 0:
            status_message = (
                f"{status_message} ({self._search_scan_error_count} worker error(s))"
            )
        self.statusBar().showMessage(status_message, 3500)
        self._request_cached_badge_sync()
        self._reset_search_scan_state()
        self._request_scope_prefetch()

    def _clear_match_results(self) -> None:
        """Handle clear match results."""
        self._search_progress_publish_timer.stop()
        self._tree_search_refresh_timer.stop()
        self._cancel_pending_search_scan()
        self.current_match_files = []
        self._current_match_counts = {}
        self._pending_search_scroll_to_first_path_keys.clear()
        self.model.clear_search_match_paths()
        self._sync_tree_markers_to_model()
        self.tree.viewport().update()
        self._remove_viewer_search_highlights()
        self._request_scope_prefetch()

    def _current_search_terms(self) -> list[tuple[str, bool]]:
        """Handle current search terms."""
        query = self.match_input.text().strip()
        return extract_search_terms(query) if query else []

    def _current_near_term_groups(self) -> list[list[tuple[str, bool]]]:
        """Extract variadic NEAR(...) groups for in-view PDF highlighting."""
        query = self.match_input.text().strip()
        return extract_near_term_groups(query) if query else []

    def _is_path_in_current_matches(self, path: Path) -> bool:
        """Return whether path in current matches."""
        target = self._path_key(path)
        return any(self._path_key(candidate) == target for candidate in self.current_match_files)

    def _apply_active_search_to_viewer(self) -> None:
        """Handle apply active search to viewer."""
        path_key = self._current_preview_path_key()
        if (
            self.current_file is None
            or not path_key
            or not self._viewer_bridge_ready_by_path.get(path_key, False)
        ):
            return
        if self._is_path_in_current_matches(self.current_file):
            scroll_to_first = path_key in self._pending_search_scroll_to_first_path_keys
            self._pending_search_scroll_to_first_path_keys.discard(path_key)
            self._highlight_viewer_search_terms(
                self._current_search_terms(),
                scroll_to_first=scroll_to_first,
            )
        else:
            self._pending_search_scroll_to_first_path_keys.discard(path_key)
            self._remove_viewer_search_highlights()

    def _highlight_viewer_search_terms(
        self,
        terms: list[tuple[str, bool]],
        *,
        scroll_to_first: bool = False,
    ) -> None:
        """Handle highlight viewer search terms."""
        payload = [
            {"text": text, "caseSensitive": bool(case_sensitive)}
            for text, case_sensitive in terms
            if text.strip()
        ]
        near_groups_payload = [
            [
                {"text": text, "caseSensitive": bool(case_sensitive)}
                for text, case_sensitive in group
                if text.strip()
            ]
            for group in self._current_near_term_groups()
        ]
        near_groups_payload = [
            group for group in near_groups_payload if len(group) >= 2
        ]
        self._pending_search_terms = terms
        path_key = self._current_preview_path_key()
        if not path_key or not self._viewer_bridge_ready_by_path.get(path_key, False):
            return
        js = (
            "window.__pdfexploreBridge && "
            "window.__pdfexploreBridge.setSearchTerms && "
            f"window.__pdfexploreBridge.setSearchTerms({json.dumps(payload)}, "
            f"{json.dumps(near_groups_payload)}, "
            f"{'true' if scroll_to_first else 'false'});"
        )
        self._run_viewer_js(js)

    def _remove_viewer_search_highlights(self) -> None:
        """Handle remove viewer search highlights."""
        self._pending_search_terms = []
        self._run_viewer_js(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.clearSearchTerms && window.__pdfexploreBridge.clearSearchTerms();"
        )

    def _apply_match_highlight_color(self, color_value: str, color_name: str) -> None:
        """Handle apply match highlight color."""
        self.match_timer.stop()
        if not self.current_match_files:
            self.statusBar().showMessage("No matched files to highlight", 3000)
            return
        updated = 0
        for path in self.current_match_files:
            try:
                if path.is_file() and path.suffix.lower() == ".pdf":
                    self.model.set_color_for_file(path, color_value)
                    updated += 1
            except Exception:
                pass
        self._clear_match_results()
        self.statusBar().showMessage(
            f"Applied {color_name.lower()} highlight to {updated} matched file(s)",
            4000,
        )

    def _show_tree_context_menu(self, pos) -> None:
        """Handle show tree context menu."""
        index = self.tree.indexAt(pos)
        if not index.isValid():
            return
        self.tree.setCurrentIndex(index)
        path = Path(self.model.filePath(index))

        menu = QMenu(self)
        color_actions: dict[QAction, str] = {}
        clear_action: QAction | None = None

        if path.is_file() and path.suffix.lower() == ".pdf":
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
        menu.addSeparator()
        copy_path_action = menu.addAction("Copy Path")
        chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == clear_in_directory_action:
            self._confirm_and_clear_directory_highlighting(clear_scope)
            self.tree.viewport().update()
            return
        if chosen == clear_all_action:
            self._confirm_and_clear_all_highlighting(clear_scope)
            self.tree.viewport().update()
            return
        if chosen == copy_path_action:
            self._copy_tree_path_to_clipboard(path)
            return
        if clear_action is not None and chosen == clear_action:
            self.model.set_color_for_file(path, None)
        elif chosen in color_actions:
            self.model.set_color_for_file(path, color_actions[chosen])
        self.tree.viewport().update()

    def _copy_tree_path_to_clipboard(self, path: Path) -> None:
        """Copy one filesystem path as shell-ready absolute text for bash paste."""
        try:
            display_path = str(path.resolve())
        except Exception:
            display_path = str(path)
        display_path = re.sub(
            r"([^A-Za-z0-9_@%+=:,./-])",
            r"\\\1",
            display_path,
        )
        QApplication.clipboard().setText(
            display_path,
            QClipboard.Mode.Clipboard,
        )
        self.statusBar().showMessage("Copied full path", 2500)

    def _confirm_and_clear_directory_highlighting(self, scope: Path | None = None) -> None:
        """Handle confirm and clear directory highlighting."""
        target_scope = scope if isinstance(scope, Path) else self._highlight_scope_directory()
        if not target_scope.is_dir():
            return
        reply = QMessageBox.question(
            self,
            "Clear Directory Highlights",
            f"Clear all file highlights in this directory only:\n{target_scope}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        cleared = self.model.clear_directory_highlights(target_scope)
        self.statusBar().showMessage(
            f"Cleared {cleared} highlight assignment(s) in {target_scope}",
            4500,
        )

    def _confirm_and_clear_all_highlighting(self, scope: Path | None = None) -> None:
        """Handle confirm and clear all highlighting."""
        target_scope = scope if isinstance(scope, Path) else self._highlight_scope_directory()
        if not target_scope.is_dir():
            return
        reply = QMessageBox.question(
            self,
            "Clear All Highlights",
            f"Clear all file highlights recursively under:\n{target_scope}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        cleared = self.model.clear_all_highlights(target_scope)
        self.statusBar().showMessage(
            f"Cleared {cleared} highlight assignment(s) under {target_scope}",
            4500,
        )

    def _on_tree_selection_changed(self, current, _previous) -> None:
        """Handle tree selection changed."""
        if not current.isValid():
            return
        path = Path(self.model.filePath(current))
        self._update_window_title()
        if path.is_dir():
            try:
                self.last_directory_selection = path.resolve()
            except Exception:
                self.last_directory_selection = path
            if self.match_input.text().strip():
                self._rerun_active_search_for_scope()
            else:
                self._request_scope_prefetch()
            return
        if path.is_file() and path.suffix.lower() == ".pdf":
            self._open_path_in_active_view(path)

    def _on_tree_directory_expanded(self, index) -> None:
        """Handle tree directory expanded."""
        self._set_tree_interaction_visual_mode(True)
        self._tree_visual_relax_timer.start()
        self._pause_prefetch_temporarily(
            self.PREFETCH_TREE_MUTATION_PAUSE_SECONDS,
            cancel_inflight=True,
        )
        self._invalidate_visible_tree_pdf_cache()
        self._request_cached_badge_probe()
        if not index.isValid():
            return
        path = Path(self.model.filePath(index))
        if not path.is_dir():
            return
        was_selected = self.tree.currentIndex() == index
        if was_selected:
            try:
                self.last_directory_selection = path.resolve()
            except Exception:
                self.last_directory_selection = path
            self._update_window_title()
        if self.match_input.text().strip():
            self._rerun_active_search_for_scope()
        else:
            self._request_scope_prefetch()

    def _on_tree_directory_collapsed(self, index) -> None:
        """Handle tree directory collapsed."""
        self._set_tree_interaction_visual_mode(True)
        self._tree_visual_relax_timer.start()
        self._pause_prefetch_temporarily(
            self.PREFETCH_TREE_MUTATION_PAUSE_SECONDS,
            cancel_inflight=True,
        )
        self._invalidate_visible_tree_pdf_cache()
        if not index.isValid():
            return
        collapsed_path = Path(self.model.filePath(index))
        if not collapsed_path.is_dir():
            return

        current_index = self.tree.currentIndex()
        if not current_index.isValid() or current_index != index:
            self._rerun_active_search_for_scope()
            return

        parent_index = index.parent()
        if not parent_index.isValid():
            parent_index = self.tree.rootIndex()

        next_scope = self.root
        if parent_index.isValid():
            parent_path = Path(self.model.filePath(parent_index))
            if parent_path.is_dir():
                try:
                    next_scope = parent_path.resolve()
                except Exception:
                    next_scope = parent_path
        self.last_directory_selection = next_scope

        if parent_index.isValid() and parent_index != current_index:
            self.tree.setCurrentIndex(parent_index)
            return

        self._update_window_title()
        self._rerun_active_search_for_scope()

    def _viewer_url_for_pdf(self, path: Path) -> QUrl:
        """Handle viewer url for pdf."""
        viewer_url = QUrl.fromLocalFile(str(VIEWER_HTML))
        pdf_url = QUrl.fromLocalFile(str(path))
        viewer_url.setQuery(f"file={pdf_url.toString(QUrl.ComponentFormattingOption.FullyEncoded)}")
        viewer_url.setFragment("zoom=page-fit")
        return viewer_url

    def _current_preview_widget(self) -> QWebEngineView | None:
        """Handle current preview widget."""
        widget = self.preview_stack.currentWidget()
        return widget if isinstance(widget, QWebEngineView) else None

    def _current_preview_path_key(self) -> str | None:
        """Handle current preview path key."""
        widget = self._current_preview_widget()
        if widget is None:
            return None
        raw = widget.property("pdfexplore_path_key")
        return str(raw).strip() if raw is not None else None

    def _create_preview_widget(self, path: Path) -> QWebEngineView:
        """Handle create preview widget."""
        path_key = self._path_key(path)
        preview = PdfPreviewWebView(
            self._handle_custom_shortcut_key_event,
            lambda key=path_key: self._on_viewer_user_activity(key),
        )
        preview.installEventFilter(self)
        preview.setProperty("pdfexplore_path_key", path_key)
        preview_settings = preview.settings()
        preview_settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False
        )
        preview_settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True
        )
        preview.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        preview.customContextMenuRequested.connect(self._show_preview_context_menu)
        preview.loadFinished.connect(
            lambda ok, key=path_key: self._on_preview_load_finished(key, ok)
        )
        self.preview_stack.addWidget(preview)
        self._preview_widgets_by_path[path_key] = preview
        self._touch_preview_widget_lru(path_key)
        self._viewer_bridge_ready_by_path[path_key] = False
        self._viewer_pending_restore_state_by_path[path_key] = None
        return preview

    def _touch_preview_widget_lru(self, path_key: str) -> None:
        """Record one full PDF viewer as the most recently used."""
        if not path_key:
            return
        self._preview_widget_lru.pop(path_key, None)
        self._preview_widget_lru[path_key] = None

    def _evict_preview_widget(self, path_key: str) -> bool:
        """Discard one WebEngine document and release its renderer process."""
        preview = self._preview_widgets_by_path.pop(path_key, None)
        self._preview_widget_lru.pop(path_key, None)
        self._viewer_bridge_ready_by_path.pop(path_key, None)
        self._viewer_pending_restore_state_by_path.pop(path_key, None)
        self._preview_signatures_by_path.pop(path_key, None)
        if preview is None:
            return False
        if self.preview_stack.currentWidget() is preview:
            self.preview_stack.setCurrentWidget(self._empty_preview)
            self._update_dark_mode_button_state()
        try:
            preview.loadFinished.disconnect()
        except Exception:
            pass
        try:
            preview.stop()
        except Exception:
            pass
        try:
            preview.page().setLifecycleState(
                QWebEnginePage.LifecycleState.Discarded
            )
        except Exception:
            pass
        self.preview_stack.removeWidget(preview)
        preview.deleteLater()
        return True

    def _trim_preview_widget_cache(
        self,
        *,
        protected_path_keys: set[str] | None = None,
    ) -> None:
        """Bound the number of heavyweight PDF.js/WebEngine documents."""
        limit = max(1, int(self.PREVIEW_WIDGET_CACHE_MAX_ENTRIES))
        protected = set(protected_path_keys or ())
        while len(self._preview_widgets_by_path) > limit:
            candidate_key = next(
                (
                    path_key
                    for path_key in self._preview_widget_lru
                    if path_key not in protected
                ),
                None,
            )
            if candidate_key is None:
                break
            self._evict_preview_widget(candidate_key)

    def _preview_widget_for_path(self, path: Path) -> QWebEngineView:
        """Handle preview widget for path."""
        path_key = self._path_key(path)
        existing = self._preview_widgets_by_path.get(path_key)
        if existing is not None:
            self._touch_preview_widget_lru(path_key)
            return existing
        return self._create_preview_widget(path)

    def _run_viewer_js(self, source: str, callback=None) -> None:
        """Run viewer js."""
        preview = self._current_preview_widget()
        if preview is None:
            return
        if callback is None:
            preview.page().runJavaScript(source)
            return
        preview.page().runJavaScript(source, callback)

    def _apply_preview_dark_mode(self) -> None:
        """Apply the current page-color mode to the visible PDF viewer."""
        enabled = "true" if self._preview_dark_mode else "false"
        self._run_viewer_js(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.setDarkMode && "
            f"window.__pdfexploreBridge.setDarkMode({enabled});"
        )

    def _toggle_preview_dark_mode(self) -> None:
        """Toggle the PDF preview between normal and dark page colors."""
        self._preview_dark_mode = not self._preview_dark_mode
        self.dark_mode_btn.setText("Light" if self._preview_dark_mode else "Dark")
        self.dark_mode_btn.setToolTip(
            "Show PDFs with normal page colors"
            if self._preview_dark_mode
            else "Show PDFs with dark pages and light text"
        )
        self._apply_preview_dark_mode()
        self.statusBar().showMessage(
            "Dark PDF mode enabled"
            if self._preview_dark_mode
            else "Normal PDF mode enabled",
            2500,
        )

    @staticmethod
    def _normalize_viewer_json_result(result) -> dict:
        """Normalize viewer json result."""
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

    def _run_viewer_js_json(self, expression: str, callback) -> None:
        """Run viewer js json."""
        js = f"""
(() => {{
  try {{
    const __result = {expression};
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
            """Handle result."""
            callback(self._normalize_viewer_json_result(result))

        self._run_viewer_js(js, _on_result)

    def _schedule_followup_view_restore(
        self,
        path_key: str | None,
        restore_state: dict | None,
        *,
        delays_ms: tuple[int, ...] = (180, 600),
    ) -> None:
        """Handle schedule followup view restore."""
        if not path_key or not isinstance(restore_state, dict) or not restore_state:
            return
        payload = self._clone_json_compatible_dict(restore_state)
        if not payload:
            return
        try:
            page = max(1, int(payload.get("page", 1) or 1))
        except Exception:
            page = 1
        try:
            scroll_top = float(payload.get("scrollTop", 0.0) or 0.0)
        except Exception:
            scroll_top = 0.0
        try:
            scroll_ratio = float(payload.get("scrollRatio", 0.0) or 0.0)
        except Exception:
            scroll_ratio = 0.0
        scale = str(payload.get("scale", "page-fit") or "page-fit").strip() or "page-fit"
        if (
            page <= 1
            and scroll_top <= 1.0
            and scroll_ratio <= 0.001
            and scale == "page-fit"
        ):
            return
        interaction_marker = float(self._last_user_interaction_at)

        def _reapply(
            expected_key: str = path_key,
            state: dict = payload,
            expected_interaction_marker: float = interaction_marker,
        ) -> None:
            """Handle reapply."""
            if (
                self._current_preview_path_key() != expected_key
                or not self._viewer_bridge_ready_by_path.get(expected_key, False)
                or self._last_user_interaction_at != expected_interaction_marker
            ):
                return
            self._run_viewer_js(
                "window.__pdfexploreBridge && window.__pdfexploreBridge.restoreViewState && "
                f"window.__pdfexploreBridge.restoreViewState({json.dumps(state)});"
            )
            self._apply_persistent_text_highlights()
            self._apply_active_search_to_viewer()

        for raw_delay in delays_ms:
            try:
                delay = max(1, int(raw_delay))
            except Exception:
                delay = 1
            QTimer.singleShot(delay, _reapply)

    def _request_preview_zoom_state(self, callback) -> None:
        """Request preview zoom state."""
        preview = self._current_preview_widget()
        path_key = self._current_preview_path_key()
        if preview is None or not path_key or not self._viewer_bridge_ready_by_path.get(
            path_key, False
        ):
            self.statusBar().showMessage("Open a PDF before changing zoom", 2000)
            return
        self._run_viewer_js_json(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.getZoomState && window.__pdfexploreBridge.getZoomState()",
            callback,
        )

    def _apply_preview_zoom_scale(self, scale_value: float) -> None:
        """Handle apply preview zoom scale."""
        preview = self._current_preview_widget()
        path_key = self._current_preview_path_key()
        if preview is None or not path_key or not self._viewer_bridge_ready_by_path.get(
            path_key, False
        ):
            self.statusBar().showMessage("Open a PDF before changing zoom", 2000)
            return
        clamped = max(PREVIEW_ZOOM_MIN, min(PREVIEW_ZOOM_MAX, float(scale_value)))
        self._run_viewer_js(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.setZoomScale && "
            f"window.__pdfexploreBridge.setZoomScale({json.dumps(clamped)});"
        )
        percent_text = f"{int(round(clamped * 100))}%"
        self.statusBar().showMessage(f"Preview zoom: {percent_text}", 1500)
        self._show_preview_zoom_overlay(percent_text)

    def _zoom_preview_in(self) -> None:
        """Handle zoom preview in."""
        def _on_zoom_state(payload: dict) -> None:
            """Handle zoom state."""
            try:
                current = float(payload.get("currentScale", PREVIEW_ZOOM_RESET) or PREVIEW_ZOOM_RESET)
            except Exception:
                current = PREVIEW_ZOOM_RESET
            self._apply_preview_zoom_scale(current + PREVIEW_ZOOM_STEP)

        self._request_preview_zoom_state(_on_zoom_state)

    def _zoom_preview_out(self) -> None:
        """Handle zoom preview out."""
        def _on_zoom_state(payload: dict) -> None:
            """Handle zoom state."""
            try:
                current = float(payload.get("currentScale", PREVIEW_ZOOM_RESET) or PREVIEW_ZOOM_RESET)
            except Exception:
                current = PREVIEW_ZOOM_RESET
            self._apply_preview_zoom_scale(current - PREVIEW_ZOOM_STEP)

        self._request_preview_zoom_state(_on_zoom_state)

    def _reset_preview_zoom(self) -> None:
        """Handle reset preview zoom."""
        preview = self._current_preview_widget()
        path_key = self._current_preview_path_key()
        if preview is None or not path_key or not self._viewer_bridge_ready_by_path.get(
            path_key, False
        ):
            self.statusBar().showMessage("Open a PDF before changing zoom", 2000)
            return
        self._run_viewer_js(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.resetZoom && "
            "window.__pdfexploreBridge.resetZoom();"
        )
        self.statusBar().showMessage("Preview zoom: page width", 1500)
        self._show_preview_zoom_overlay("Fit Width")

    def _toggle_preview_three_up(self) -> None:
        """Handle toggle preview three up."""
        preview = self._current_preview_widget()
        path_key = self._current_preview_path_key()
        if preview is None or not path_key or not self._viewer_bridge_ready_by_path.get(
            path_key, False
        ):
            self.statusBar().showMessage("Open a PDF before changing layout", 2000)
            return

        def _on_toggle(payload: dict) -> None:
            """Handle toggle."""
            active = bool(payload.get("threeUpActive") or payload.get("active"))
            try:
                one_page_scale = float(payload.get("onePageScale", PREVIEW_ZOOM_RESET) or PREVIEW_ZOOM_RESET)
            except Exception:
                one_page_scale = PREVIEW_ZOOM_RESET
            percent_text = f"{int(round(one_page_scale * 100))}%"
            if active:
                self.statusBar().showMessage(
                    f"Preview layout: 3-up rows (1-up zoom {percent_text})",
                    2200,
                )
                self._show_preview_zoom_overlay("3-Up")
            else:
                self.statusBar().showMessage(
                    f"Preview layout: single page ({percent_text})",
                    2200,
                )
                self._show_preview_zoom_overlay(percent_text)

        self._run_viewer_js_json(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.toggleThreeUpMode && "
            "window.__pdfexploreBridge.toggleThreeUpMode()",
            _on_toggle,
        )

    def _set_preview_zoom_one_hundred(self) -> None:
        """Set preview zoom one hundred."""
        preview = self._current_preview_widget()
        path_key = self._current_preview_path_key()
        if preview is None or not path_key or not self._viewer_bridge_ready_by_path.get(
            path_key, False
        ):
            self.statusBar().showMessage("Open a PDF before changing zoom", 2000)
            return

        def _on_result(payload: dict) -> None:
            """Handle result."""
            if payload.get("ok") is False:
                self.statusBar().showMessage("Unable to set preview zoom", 2200)
                return
            self.statusBar().showMessage("Preview zoom: 100%", 1800)
            self._show_preview_zoom_overlay("100%")

        self._run_viewer_js_json(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.setOnePageZoom100 && "
            "window.__pdfexploreBridge.setOnePageZoom100()",
            _on_result,
        )

    def _show_preview_zoom_overlay(self, percent_text: str) -> None:
        """Handle show preview zoom overlay."""
        self._preview_zoom_overlay.setText(percent_text)
        self._preview_zoom_overlay.adjustSize()
        self._position_preview_zoom_overlay()
        self._preview_zoom_overlay.raise_()
        self._preview_zoom_overlay.show()
        self._preview_zoom_overlay_timer.start()

    def _position_preview_zoom_overlay(self) -> None:
        """Handle position preview zoom overlay."""
        overlay = getattr(self, "_preview_zoom_overlay", None)
        target = self._current_preview_widget()
        if overlay is None or target is None:
            return
        top_left = target.mapTo(self.preview_container, QPoint(0, 0))
        target_width = max(1, target.width())
        x = top_left.x() + max(8, (target_width - overlay.width()) // 2)
        y = top_left.y() + 8
        overlay.move(x, y)

    @staticmethod
    def _default_view_state() -> dict[str, int | float | str]:
        """Return default view state."""
        return {
            "page": 1,
            "pagesCount": 1,
            "scale": "page-fit",
            "scrollTop": 0.0,
            "scrollRatio": 0.0,
        }

    @staticmethod
    def _view_tab_label_for_page(page_number: int) -> str:
        """Handle view tab label for page."""
        return str(max(1, int(page_number)))

    @staticmethod
    def _normalize_custom_view_label(raw_value) -> str | None:
        """Normalize custom view label."""
        if not isinstance(raw_value, str):
            return None
        if not raw_value.strip():
            return None
        cleaned = raw_value.replace("\r", " ").replace("\n", " ")
        if len(cleaned) > ViewTabBar.MAX_LABEL_CHARS:
            cleaned = cleaned[: ViewTabBar.MAX_LABEL_CHARS]
        return cleaned

    def _display_label_for_view(
        self, page_number: int, custom_label: str | None = None
    ) -> str:
        """Handle display label for view."""
        normalized = self._normalize_custom_view_label(custom_label)
        if normalized is not None:
            return normalized
        return self._view_tab_label_for_page(page_number)

    @staticmethod
    def _page_from_view_state(state: dict | None) -> int:
        """Handle page from view state."""
        if not isinstance(state, dict):
            return 1
        try:
            return max(1, int(state.get("page", 1)))
        except Exception:
            return 1

    @staticmethod
    def _progress_from_view_state(state: dict | None) -> float:
        """Handle progress from view state."""
        if not isinstance(state, dict):
            return 0.0
        try:
            pages = max(1, int(state.get("pagesCount", 0) or 1))
        except Exception:
            pages = 1
        try:
            page = max(1, int(state.get("page", 1) or 1))
        except Exception:
            page = 1
        return max(0.0, min(1.0, page / pages))

    @staticmethod
    def _clone_json_compatible_dict(raw) -> dict:
        """Handle clone json compatible dict."""
        if not isinstance(raw, dict):
            return {}
        try:
            return json.loads(json.dumps(raw))
        except Exception:
            return dict(raw)

    def _tab_custom_label(self, tab_index: int) -> str | None:
        """Handle tab custom label."""
        data = self._tab_data(tab_index)
        if data is None:
            return None
        return self._normalize_custom_view_label(data.get("custom_label"))

    def _tab_label_anchor(self, tab_index: int) -> tuple[float, int] | None:
        """Handle tab label anchor."""
        data = self._tab_data(tab_index)
        if data is None:
            return None
        if self._normalize_custom_view_label(data.get("custom_label")) is None:
            return None
        try:
            scroll_top = float(data.get("custom_label_anchor_scroll_y", 0.0))
        except Exception:
            scroll_top = 0.0
        if not isinstance(scroll_top, float) or not (scroll_top == scroll_top):
            scroll_top = 0.0
        try:
            page = max(1, int(data.get("custom_label_anchor_top_line", 1)))
        except Exception:
            page = 1
        return scroll_top, page

    def _used_tab_color_slots(self) -> set[int]:
        """Handle used tab color slots."""
        used: set[int] = set()
        palette_size = len(ViewTabBar.PASTEL_SEQUENCE)
        for index in range(self.view_tabs.count()):
            data = self._tab_data(index)
            if not isinstance(data, dict):
                continue
            try:
                slot = int(data.get("color_slot", -1))
            except Exception:
                continue
            if 0 <= slot < palette_size:
                used.add(slot)
        return used

    def _allocate_next_tab_color_slot(self) -> int:
        """Handle allocate next tab color slot."""
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
        slot = start
        self._next_tab_color_index = (slot + 1) % palette_size
        return slot

    def _ensure_current_tab(self) -> int:
        """Handle ensure current tab."""
        if self.view_tabs.count() > 0 and self.view_tabs.currentIndex() >= 0:
            return self.view_tabs.currentIndex()
        self._reset_document_views(self.current_file)
        tab_index = self.view_tabs.currentIndex()
        if tab_index < 0:
            tab_index = 0
        self._refresh_view_tabs_visibility()
        return tab_index

    def _new_tab_data(self, path: Path | None = None) -> dict:
        """Handle new tab data."""
        path_key = self._path_key(path) if isinstance(path, Path) else ""
        default_state = self._default_view_state()
        data = {
            "view_id": self._next_view_id,
            "sequence": self._next_view_sequence,
            "color_slot": self._allocate_next_tab_color_slot(),
            "progress": self._progress_from_view_state(default_state),
            "custom_label": None,
            "custom_label_anchor_scroll_y": 0.0,
            "custom_label_anchor_top_line": 1,
            "path": str(path) if isinstance(path, Path) else "",
            "path_key": path_key,
            "state": dict(default_state),
        }
        self._next_view_id += 1
        self._next_view_sequence += 1
        return data

    def _tab_data(self, index: int) -> dict | None:
        """Handle tab data."""
        if index < 0 or index >= self.view_tabs.count():
            return None
        data = self.view_tabs.tabData(index)
        return data if isinstance(data, dict) else None

    def _set_tab_data(self, index: int, data: dict) -> None:
        """Set tab data."""
        self.view_tabs.setTabData(index, data)

    def _capture_tab_state(
        self, index: int, *, blocking: bool = False, timeout_ms: int = 220
    ) -> dict | None:
        """Handle capture tab state."""
        data = self._tab_data(index)
        if data is None:
            return None
        path_key = str(data.get("path_key") or "").strip()
        current_path_key = self._current_preview_path_key()
        if (
            not path_key
            or not current_path_key
            or path_key != current_path_key
            or not self._viewer_bridge_ready_by_path.get(current_path_key, False)
        ):
            return None

        capture_serial = 0
        if not blocking:
            if self._view_state_capture_in_flight:
                return None
            self._view_state_capture_in_flight = True
            self._view_state_capture_serial += 1
            capture_serial = self._view_state_capture_serial

            def _expire_capture(serial: int = capture_serial) -> None:
                """Release a lost asynchronous WebEngine callback."""
                if serial == self._view_state_capture_serial:
                    self._view_state_capture_in_flight = False

            QTimer.singleShot(
                max(1000, int(self.VIEW_STATE_POLL_TIMER_INTERVAL_MS) * 2),
                _expire_capture,
            )

        loop: QEventLoop | None = None
        timeout_timer: QTimer | None = None
        done = False
        captured_state: dict | None = None
        if blocking:
            loop = QEventLoop(self)
            timeout_timer = QTimer(self)
            timeout_timer.setSingleShot(True)

            def _on_timeout() -> None:
                """Handle timeout."""
                nonlocal done
                done = True
                if loop is not None and loop.isRunning():
                    loop.quit()

            timeout_timer.timeout.connect(_on_timeout)

        def _on_state(result) -> None:
            """Handle state."""
            nonlocal done, captured_state
            if capture_serial == self._view_state_capture_serial:
                self._view_state_capture_in_flight = False
            if not isinstance(result, dict):
                done = True
                if loop is not None and loop.isRunning():
                    loop.quit()
                return
            latest_data = self._tab_data(index)
            if (
                not isinstance(latest_data, dict)
                or str(latest_data.get("path_key") or "").strip() != path_key
            ):
                done = True
                if loop is not None and loop.isRunning():
                    loop.quit()
                return
            previous_state = (
                latest_data.get("state")
                if isinstance(latest_data.get("state"), dict)
                else {}
            )
            try:
                previous_page = int(previous_state.get("page", 1) or 1)
            except Exception:
                previous_page = 1
            try:
                previous_scroll_top = float(previous_state.get("scrollTop", 0.0) or 0.0)
            except Exception:
                previous_scroll_top = 0.0
            try:
                previous_scroll_ratio = float(previous_state.get("scrollRatio", 0.0) or 0.0)
            except Exception:
                previous_scroll_ratio = 0.0

            try:
                next_page = int(result.get("page", 1) or 1)
            except Exception:
                next_page = previous_page
            try:
                next_scroll_top = float(result.get("scrollTop", 0.0) or 0.0)
            except Exception:
                next_scroll_top = previous_scroll_top
            try:
                next_scroll_ratio = float(result.get("scrollRatio", 0.0) or 0.0)
            except Exception:
                next_scroll_ratio = previous_scroll_ratio

            state_changed_by_scroll = (
                next_page != previous_page
                or abs(next_scroll_top - previous_scroll_top) > 32.0
                or abs(next_scroll_ratio - previous_scroll_ratio) > 0.015
            )
            if not blocking and state_changed_by_scroll:
                self._mark_user_interaction()
                self._pause_prefetch_temporarily(
                    self.PREFETCH_VIEWER_ACTIVITY_PAUSE_SECONDS,
                    cancel_inflight=True,
                )

            next_progress = self._progress_from_view_state(result)
            if result != previous_state or next_progress != latest_data.get("progress"):
                latest_data["state"] = result
                latest_data["progress"] = next_progress
                self._set_tab_data(index, latest_data)
            if self._normalize_custom_view_label(latest_data.get("custom_label")) is None:
                next_label = self._display_label_for_view(
                    self._page_from_view_state(result)
                )
                if self.view_tabs.tabText(index) != next_label:
                    self.view_tabs.setTabText(index, next_label)
            captured_state = self._clone_json_compatible_dict(result)
            done = True
            if loop is not None and loop.isRunning():
                loop.quit()

        self._run_viewer_js_json(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.getViewState && window.__pdfexploreBridge.getViewState()",
            _on_state,
        )
        if blocking and not done and loop is not None and timeout_timer is not None:
            timeout_timer.start(max(1, int(timeout_ms)))
            loop.exec()
            timeout_timer.stop()
            timeout_timer.deleteLater()
        return captured_state

    def _poll_current_view_state(self) -> None:
        """Handle poll current view state."""
        current_index = self.view_tabs.currentIndex()
        if current_index >= 0:
            self._capture_tab_state(current_index)

    @staticmethod
    def _should_persist_document_view_session(session: dict | None) -> bool:
        """Handle should persist document view session."""
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

    def _save_document_view_session(
        self, path_key: str | None = None, *, capture_current: bool = True
    ) -> None:
        """Save document view session."""
        if path_key is None:
            path_key = self._current_preview_path_key()
        if not path_key:
            return
        if capture_current:
            current_index = self.view_tabs.currentIndex()
            if current_index >= 0:
                self._capture_tab_state(current_index, blocking=True)

        tabs: list[dict] = []
        max_view_id = 0
        max_sequence = 0
        active_view_id = 0
        for index in range(self.view_tabs.count()):
            data = self._tab_data(index)
            if not isinstance(data, dict):
                continue
            if str(data.get("path_key") or "").strip() != path_key:
                continue
            try:
                view_id = int(data.get("view_id"))
            except Exception:
                continue
            try:
                sequence = max(1, int(data.get("sequence", len(tabs) + 1)))
            except Exception:
                sequence = len(tabs) + 1
            try:
                color_slot = int(data.get("color_slot", 0))
            except Exception:
                color_slot = 0
            state = data.get("state") if isinstance(data.get("state"), dict) else {}
            if index == self.view_tabs.currentIndex():
                active_view_id = view_id
            try:
                anchor_scroll = float(data.get("custom_label_anchor_scroll_y", 0.0) or 0.0)
            except Exception:
                anchor_scroll = 0.0
            try:
                anchor_page = max(1, int(data.get("custom_label_anchor_top_line", 1) or 1))
            except Exception:
                anchor_page = 1
            tabs.append(
                {
                    "view_id": view_id,
                    "sequence": sequence,
                    "color_slot": color_slot,
                    "custom_label": self._normalize_custom_view_label(
                        data.get("custom_label")
                    ),
                    "custom_label_anchor_scroll_y": anchor_scroll,
                    "custom_label_anchor_top_line": anchor_page,
                    "state": self._clone_json_compatible_dict(state),
                }
            )
            max_view_id = max(max_view_id, view_id)
            max_sequence = max(max_sequence, sequence)

        if not tabs:
            self._document_view_sessions.pop(path_key, None)
            return
        if active_view_id <= 0:
            active_view_id = int(tabs[0].get("view_id") or 1)
        self._document_view_sessions[path_key] = {
            "active_view_id": active_view_id,
            "next_view_id": max(int(self._next_view_id), max_view_id + 1),
            "next_view_sequence": max(int(self._next_view_sequence), max_sequence + 1),
            "next_tab_color_index": int(self._next_tab_color_index),
            "tabs": tabs,
        }

    def _load_persisted_document_view_session(self, path_key: str) -> None:
        """Load persisted document view session."""
        if not path_key or path_key in self._document_view_sessions:
            return
        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return
        directory, file_name = resolved
        session = self._directory_view_states(directory).get(file_name)
        if isinstance(session, dict):
            self._document_view_sessions[path_key] = self._clone_json_compatible_dict(
                session
            )

    def _restore_document_view_session(self, path: Path) -> bool:
        """Handle restore document view session."""
        path_key = self._path_key(path)
        session = self._document_view_sessions.get(path_key)
        if not isinstance(session, dict):
            return False
        raw_tabs = session.get("tabs")
        if not isinstance(raw_tabs, list) or not raw_tabs:
            return False

        blocked = self.view_tabs.blockSignals(True)
        while self.view_tabs.count() > 0:
            self.view_tabs.removeTab(0)

        max_view_id = 0
        max_sequence = 0
        for raw_tab in raw_tabs:
            if not isinstance(raw_tab, dict):
                continue
            try:
                view_id = int(raw_tab.get("view_id"))
            except Exception:
                continue
            try:
                sequence = max(1, int(raw_tab.get("sequence", self.view_tabs.count() + 1)))
            except Exception:
                sequence = self.view_tabs.count() + 1
            try:
                color_slot = int(raw_tab.get("color_slot", 0))
            except Exception:
                color_slot = 0
            state = (
                self._clone_json_compatible_dict(raw_tab.get("state"))
                if isinstance(raw_tab.get("state"), dict)
                else dict(self._default_view_state())
            )
            if not state:
                state = dict(self._default_view_state())
            custom_label = self._normalize_custom_view_label(raw_tab.get("custom_label"))
            data = {
                "view_id": view_id,
                "sequence": sequence,
                "color_slot": color_slot,
                "progress": self._progress_from_view_state(state),
                "custom_label": custom_label,
                "custom_label_anchor_scroll_y": float(
                    raw_tab.get("custom_label_anchor_scroll_y", 0.0) or 0.0
                ),
                "custom_label_anchor_top_line": max(
                    1, int(raw_tab.get("custom_label_anchor_top_line", 1) or 1)
                ),
                "path": str(path),
                "path_key": path_key,
                "state": state,
            }
            index = self.view_tabs.addTab(
                self._display_label_for_view(self._page_from_view_state(state), custom_label)
            )
            self._set_tab_data(index, data)
            max_view_id = max(max_view_id, view_id)
            max_sequence = max(max_sequence, sequence)

        if self.view_tabs.count() <= 0:
            self.view_tabs.blockSignals(blocked)
            return False

        wanted_view_id = 0
        try:
            wanted_view_id = int(session.get("active_view_id", 0))
        except Exception:
            wanted_view_id = 0
        active_index = 0
        for index in range(self.view_tabs.count()):
            data = self._tab_data(index)
            if not isinstance(data, dict):
                continue
            try:
                candidate_view_id = int(data.get("view_id", 0))
            except Exception:
                candidate_view_id = 0
            if candidate_view_id == wanted_view_id and candidate_view_id > 0:
                active_index = index
                break
        self.view_tabs.setCurrentIndex(active_index)
        self.view_tabs.blockSignals(blocked)

        self._active_view_tab_index = active_index
        try:
            next_view_id = int(session.get("next_view_id", max_view_id + 1) or max_view_id + 1)
        except Exception:
            next_view_id = max_view_id + 1
        try:
            next_view_sequence = int(
                session.get("next_view_sequence", max_sequence + 1) or max_sequence + 1
            )
        except Exception:
            next_view_sequence = max_sequence + 1
        try:
            next_color_index = int(session.get("next_tab_color_index", 0) or 0)
        except Exception:
            next_color_index = 0
        self._next_view_id = max(next_view_id, max_view_id + 1)
        self._next_view_sequence = max(next_view_sequence, max_sequence + 1)
        self._next_tab_color_index = next_color_index % max(
            1, len(ViewTabBar.PASTEL_SEQUENCE)
        )
        self._refresh_view_tabs_visibility()
        return True

    def _persist_document_view_session(
        self, path_key: str | None = None, *, capture_current: bool = True
    ) -> None:
        """Handle persist document view session."""
        if path_key is None:
            path_key = self._current_preview_path_key()
        if not path_key:
            return
        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return
        directory, file_name = resolved
        self._save_document_view_session(path_key, capture_current=capture_current)
        sessions = self._directory_view_states(directory)
        session = self._document_view_sessions.get(path_key)
        if self._should_persist_document_view_session(session):
            sessions[file_name] = self._clone_json_compatible_dict(session)
        else:
            sessions.pop(file_name, None)
        self._save_directory_view_states(directory, changed_file_names={file_name})
        self._rebuild_tree_marker_cache()

    def _reset_document_views(
        self,
        path: Path | None,
        *,
        initial_state: dict | None = None,
    ) -> None:
        """Handle reset document views."""
        blocked = self.view_tabs.blockSignals(True)
        while self.view_tabs.count() > 0:
            self.view_tabs.removeTab(0)
        data = self._new_tab_data(path)
        state = (
            self._clone_json_compatible_dict(initial_state)
            if isinstance(initial_state, dict)
            else dict(self._default_view_state())
        )
        if not state:
            state = dict(self._default_view_state())
        data["state"] = state
        data["progress"] = self._progress_from_view_state(state)
        page_label = self._display_label_for_view(self._page_from_view_state(state))
        index = self.view_tabs.addTab(page_label)
        self._set_tab_data(index, data)
        self.view_tabs.setCurrentIndex(index)
        self.view_tabs.blockSignals(blocked)
        self._active_view_tab_index = index
        self._refresh_view_tabs_visibility()

    def _open_path_in_active_view(self, path: Path) -> None:
        """Open path in active view."""
        if not path.is_file():
            return
        next_path_key = self._path_key(path)
        previous_path_key = self._current_preview_path_key()
        if previous_path_key and previous_path_key != next_path_key:
            self._persist_document_view_session(previous_path_key, capture_current=True)

        if previous_path_key != next_path_key:
            self._load_persisted_document_view_session(next_path_key)
            restored = self._restore_document_view_session(path)
            if not restored:
                legacy_state = self._view_state_for_path_key(next_path_key)
                self._reset_document_views(path, initial_state=legacy_state)

        if self.view_tabs.count() <= 0:
            self._reset_document_views(path)

        for index in range(self.view_tabs.count()):
            data = self._tab_data(index)
            if not isinstance(data, dict):
                continue
            data["path"] = str(path)
            data["path_key"] = next_path_key
            if not isinstance(data.get("state"), dict):
                data["state"] = dict(self._default_view_state())
            data["progress"] = self._progress_from_view_state(data.get("state"))
            if self._normalize_custom_view_label(data.get("custom_label")) is None:
                self.view_tabs.setTabText(
                    index,
                    self._display_label_for_view(
                        self._page_from_view_state(data.get("state"))
                    ),
                )
            self._set_tab_data(index, data)

        current_index = self.view_tabs.currentIndex()
        if current_index < 0:
            current_index = 0
            self.view_tabs.setCurrentIndex(current_index)
        self._active_view_tab_index = current_index
        self._rebuild_tree_marker_cache()
        self._load_tab_index(current_index)
        self._refresh_view_tabs_visibility()

    def _load_tab_index(self, index: int) -> None:
        """Load tab index."""
        data = self._tab_data(index)
        if data is None:
            return
        path_text = str(data.get("path") or "").strip()
        if not path_text:
            return
        path = Path(path_text)
        previous_preview_path_key = self._current_preview_path_key()
        self.current_file = path
        self._update_edit_button_state()
        self._cancel_scope_prefetch()
        self._request_scope_prefetch()
        self._set_preview_signature_for_path(path)
        try:
            label_text = str(path.relative_to(self.root))
        except Exception:
            label_text = str(path)
        self.path_label.setText(label_text)
        self.path_label.setToolTip(str(path))
        self._update_window_title()
        self._current_text_highlights = self._load_text_highlights_for_path_key(self._path_key(path))
        path_key = self._path_key(path)
        if (
            previous_preview_path_key != path_key
            and bool(self.match_input.text().strip())
        ):
            self._pending_search_scroll_to_first_path_keys.add(path_key)
        preview = self._preview_widget_for_path(path)
        self.preview_stack.setCurrentWidget(preview)
        self._update_dark_mode_button_state()
        self._trim_preview_widget_cache(protected_path_keys={path_key})
        wanted_state = data.get("state") if isinstance(data.get("state"), dict) else None
        if wanted_state is None:
            wanted_state = self._view_state_for_path_key(path_key) or dict(
                self._default_view_state()
            )
        self._viewer_pending_restore_state_by_path[path_key] = wanted_state
        current_url = preview.url().toString()
        wanted_url = self._viewer_url_for_pdf(path).toString()
        if current_url != wanted_url:
            self._viewer_bridge_ready_by_path[path_key] = False
            preview.setUrl(QUrl(wanted_url))
        elif self._viewer_bridge_ready_by_path.get(path_key, False):
            restore_state = wanted_state or {"scale": "page-fit"}
            self._run_viewer_js(
                "window.__pdfexploreBridge && window.__pdfexploreBridge.restoreViewState && "
                f"window.__pdfexploreBridge.restoreViewState({json.dumps(restore_state)});"
            )
            self._apply_preview_dark_mode()
            self._apply_persistent_text_highlights()
            self._apply_active_search_to_viewer()
            self._schedule_followup_view_restore(path_key, restore_state)
        else:
            self._viewer_ready_timer.start()
        self._rebuild_tree_marker_cache()

    def _add_document_view(self) -> None:
        """Handle add document view."""
        if self.current_file is None:
            self.statusBar().showMessage("Open a PDF before adding a view", 2500)
            return
        if self.view_tabs.count() >= self.MAX_DOCUMENT_VIEWS:
            self.statusBar().showMessage(
                f"Maximum of {self.MAX_DOCUMENT_VIEWS} views reached",
                3000,
            )
            return
        current_index = self._ensure_current_tab()
        self._capture_tab_state(current_index, blocking=True)
        current_data = self._tab_data(current_index) or {}
        duplicate_path = Path(str(current_data.get("path") or self.current_file))
        tab_index = self.view_tabs.addTab(duplicate_path.name)
        new_data = self._new_tab_data(duplicate_path)
        new_data["state"] = self._clone_json_compatible_dict(current_data.get("state"))
        if not new_data["state"]:
            new_data["state"] = dict(self._default_view_state())
        new_data["progress"] = self._progress_from_view_state(new_data.get("state"))
        new_data["custom_label"] = None
        new_data["custom_label_anchor_scroll_y"] = 0.0
        new_data["custom_label_anchor_top_line"] = self._page_from_view_state(
            new_data.get("state")
        )
        self._set_tab_data(tab_index, new_data)
        self.view_tabs.setTabText(
            tab_index,
            self._display_label_for_view(self._page_from_view_state(new_data.get("state"))),
        )
        self.view_tabs.setCurrentIndex(tab_index)
        self._refresh_view_tabs_visibility()
        self._persist_document_view_session(self._path_key(duplicate_path), capture_current=False)
        self.statusBar().showMessage(
            f"Added view {self.view_tabs.count()} of {self.MAX_DOCUMENT_VIEWS}",
            2500,
        )

    def _refresh_view_tabs_visibility(self) -> None:
        """Refresh view tabs visibility."""
        visible = False
        if self.view_tabs.count() > 1:
            visible = True
        elif self.view_tabs.count() == 1 and self._tab_custom_label(0) is not None:
            visible = True
        self.view_tabs.setVisible(visible)
        self._update_edit_button_state()
        self._rebuild_tree_marker_cache()

    def _on_view_tab_changed(self, index: int) -> None:
        """Handle view tab changed."""
        previous_index = self._active_view_tab_index
        if previous_index >= 0 and previous_index != index:
            self._capture_tab_state(previous_index, blocking=True)
        self._active_view_tab_index = index
        if index < 0:
            return
        self._load_tab_index(index)
        self._persist_document_view_session(capture_current=False)

    def _on_view_tab_close_requested(self, index: int) -> None:
        """Handle view tab close requested."""
        if self.view_tabs.count() <= 1:
            if self._tab_custom_label(index) is not None:
                data = self._tab_data(index) or {}
                data["custom_label"] = None
                data["custom_label_anchor_scroll_y"] = 0.0
                state = data.get("state") if isinstance(data.get("state"), dict) else {}
                data["custom_label_anchor_top_line"] = self._page_from_view_state(state)
                self._set_tab_data(index, data)
                self.view_tabs.setTabText(
                    index,
                    self._display_label_for_view(self._page_from_view_state(state)),
                )
                self._refresh_view_tabs_visibility()
                self._persist_document_view_session(capture_current=False)
                self.statusBar().showMessage(
                    "Cleared custom tab label and kept default view",
                    2600,
                )
                return
            self.statusBar().showMessage("At least one view must remain open", 2500)
            return
        data = self._tab_data(index)
        if data is not None:
            self._capture_tab_state(index, blocking=True)
        self.view_tabs.removeTab(index)
        self._active_view_tab_index = self.view_tabs.currentIndex()
        self._refresh_view_tabs_visibility()
        self._persist_document_view_session(capture_current=False)

    def _show_view_tab_context_menu(self, pos) -> None:
        """Handle show view tab context menu."""
        tab_index = self.view_tabs.tabAt(pos)
        if tab_index < 0:
            return
        menu = QMenu(self)
        edit_action = menu.addAction("Edit Tab Label...")
        return_action = None
        if self._tab_label_anchor(tab_index) is not None:
            return_action = menu.addAction("Return to beginning")
        chosen = menu.exec(self.view_tabs.mapToGlobal(pos))
        if chosen == edit_action:
            self._edit_view_tab_label(tab_index)
            return
        if return_action is not None and chosen == return_action:
            self._return_view_tab_to_beginning(tab_index)

    def _on_view_tab_home_requested(self, tab_index: int) -> None:
        """Handle view tab home requested."""
        if self._tab_label_anchor(tab_index) is None:
            return
        if self.view_tabs.currentIndex() != tab_index:
            self.view_tabs.setCurrentIndex(tab_index)
        self._return_view_tab_to_beginning(tab_index)

    def _on_view_tab_beginning_reset_requested(self, tab_index: int) -> None:
        """Handle view tab beginning reset requested."""
        if self._tab_label_anchor(tab_index) is None:
            return
        if self.view_tabs.currentIndex() != tab_index:
            self.view_tabs.setCurrentIndex(tab_index)
            # Tab switches restore view state asynchronously inside pdf.js.
            QTimer.singleShot(
                280, lambda idx=tab_index: self._reset_view_tab_beginning_to_current(idx)
            )
            return
        self._reset_view_tab_beginning_to_current(tab_index)

    def _reset_view_tab_beginning_to_current(self, tab_index: int) -> None:
        """Handle reset view tab beginning to current."""
        if self.view_tabs.currentIndex() == tab_index:
            self._capture_tab_state(tab_index, blocking=True)
        data = self._tab_data(tab_index)
        if data is None:
            return
        state = data.get("state") if isinstance(data.get("state"), dict) else {}
        try:
            anchor_scroll = float(state.get("scrollTop", 0.0))
        except Exception:
            anchor_scroll = 0.0
        data["custom_label_anchor_scroll_y"] = max(0.0, anchor_scroll)
        data["custom_label_anchor_top_line"] = self._page_from_view_state(state)
        self._set_tab_data(tab_index, data)
        self._persist_document_view_session(capture_current=False)
        self.statusBar().showMessage("Reset tab beginning to current page", 2200)

    def _edit_view_tab_label(self, tab_index: int) -> None:
        """Handle edit view tab label."""
        if self.view_tabs.currentIndex() == tab_index:
            self._capture_tab_state(tab_index, blocking=True)
        data = self._tab_data(tab_index)
        if data is None:
            return
        current_custom = self._tab_custom_label(tab_index) or ""
        label_text, accepted = QInputDialog.getText(
            self,
            "Edit Tab Label",
            "Enter a custom tab label (blank restores the page number):",
            text=current_custom,
        )
        if not accepted:
            return
        was_truncated = len(label_text) > ViewTabBar.MAX_LABEL_CHARS
        custom_label = self._normalize_custom_view_label(label_text)
        state = data.get("state") if isinstance(data.get("state"), dict) else {}
        page = self._page_from_view_state(state)
        previous_custom = self._normalize_custom_view_label(data.get("custom_label"))
        if custom_label is None:
            data["custom_label_anchor_scroll_y"] = 0.0
            data["custom_label_anchor_top_line"] = page
        elif previous_custom != custom_label:
            try:
                anchor_scroll = float(state.get("scrollTop", 0.0))
            except Exception:
                anchor_scroll = 0.0
            data["custom_label_anchor_scroll_y"] = max(0.0, anchor_scroll)
            data["custom_label_anchor_top_line"] = page
        data["custom_label"] = custom_label
        self._set_tab_data(tab_index, data)
        self.view_tabs.setTabText(tab_index, self._display_label_for_view(page, custom_label))
        self.view_tabs.updateGeometry()
        self.view_tabs.update()
        self._refresh_view_tabs_visibility()
        self._persist_document_view_session(capture_current=False)
        if custom_label is None:
            self.statusBar().showMessage("Restored dynamic page-number tab label", 2300)
        elif was_truncated:
            self.statusBar().showMessage(
                f"Tab label updated and truncated to {ViewTabBar.MAX_LABEL_CHARS} characters",
                2800,
            )
        else:
            self.statusBar().showMessage(f"Tab label updated to '{custom_label}'", 2200)

    def _return_view_tab_to_beginning(self, tab_index: int) -> None:
        """Handle return view tab to beginning."""
        data = self._tab_data(tab_index)
        if data is None:
            return
        anchor = self._tab_label_anchor(tab_index)
        if anchor is None:
            return
        anchor_scroll, anchor_page = anchor
        state = data.get("state") if isinstance(data.get("state"), dict) else {}
        if not state:
            state = dict(self._default_view_state())
        state["page"] = int(anchor_page)
        state["scrollTop"] = float(max(0.0, anchor_scroll))
        data["state"] = state
        data["progress"] = self._progress_from_view_state(state)
        self._set_tab_data(tab_index, data)
        self._persist_document_view_session(capture_current=False)
        if self.view_tabs.currentIndex() != tab_index:
            self.view_tabs.setCurrentIndex(tab_index)
            return
        self._load_tab_index(tab_index)
        self.statusBar().showMessage(
            f"Returned tab to labeled beginning at page {anchor_page}",
            2800,
        )

    def _refresh_tree_marker_cache_for_path(self, path_key: str | None) -> None:
        """Refresh tree marker cache for path."""
        if not path_key:
            return
        session = self._view_session_for_path_key(path_key)
        if self._load_text_highlights_for_path_key(path_key):
            self._tree_highlight_marker_paths.add(path_key)
        else:
            self._tree_highlight_marker_paths.discard(path_key)
        open_counts: dict[str, int] = {}
        for index in range(self.view_tabs.count()):
            data = self._tab_data(index)
            if not data:
                continue
            candidate = str(data.get("path_key") or "").strip()
            if candidate:
                open_counts[candidate] = open_counts.get(candidate, 0) + 1
        if open_counts.get(path_key, 0) > 1 or self._should_persist_document_view_session(
            session
        ):
            self._tree_multi_view_marker_paths.add(path_key)
        else:
            self._tree_multi_view_marker_paths.discard(path_key)
        self._sync_tree_markers_to_model()

    def _rebuild_tree_marker_cache(self) -> None:
        """Handle rebuild tree marker cache."""
        root_key = self._path_key(self.root)
        current_multi = {
            path_key
            for path_key in self._tree_multi_view_marker_paths
            if path_key == root_key or path_key.startswith(root_key + os.sep)
        }
        current_highlights = {
            path_key
            for path_key in self._tree_highlight_marker_paths
            if path_key == root_key or path_key.startswith(root_key + os.sep)
        }
        next_multi, next_highlights = self._merge_live_tree_marker_state(
            current_multi,
            current_highlights,
            root_key=root_key,
        )
        self._tree_multi_view_marker_paths = next_multi
        self._tree_highlight_marker_paths = next_highlights
        self._sync_tree_markers_to_model()
        if self._tree_marker_cache_root_key != root_key:
            self._start_tree_marker_scan()

    def _known_highlight_marker_paths_for_root(self, root_key: str) -> set[str]:
        """Return in-memory known highlight marker paths under the active root."""
        known: set[str] = set()
        for directory_key, by_file in self._persisted_text_highlights_by_dir.items():
            if not isinstance(directory_key, str) or not isinstance(by_file, dict):
                continue
            if root_key and not self._path_key_is_under_root(directory_key, root_key):
                continue
            for file_name, entries in by_file.items():
                if not isinstance(file_name, str) or not entries:
                    continue
                candidate_key = self._path_key_from_parts(directory_key, file_name)
                known.add(candidate_key)

        current_key = self._current_preview_path_key()
        if (
            current_key
            and self._current_text_highlights
            and self._path_key_is_under_root(current_key, root_key)
        ):
            known.add(current_key)
        return known

    @staticmethod
    def _path_key_is_under_root(path_key: str, root_key: str | None) -> bool:
        """Return whether one normalized path-key belongs to the active root."""
        if not root_key:
            return True
        return path_key == root_key or path_key.startswith(root_key + os.sep)

    @staticmethod
    def _path_key_from_parts(directory_key: str, file_name: str) -> str:
        """Build a normalized path-key without filesystem `resolve()` overhead."""
        return os.path.normpath(os.path.join(directory_key, file_name))

    def _sync_tree_markers_to_model(self) -> None:
        """Push merged marker sets to the tree model and repaint once.

        This method centralizes marker synchronization so all callers get the
        same ordering, root filtering, and repaint behavior.
        """
        # Marker badges must be visible immediately after highlight-state changes.
        # If temporary low-cost paint mode is still active, disable it now.
        self._set_tree_interaction_visual_mode(False)
        root_key = self._path_key(self.root)
        filtered_multi = {
            path_key
            for path_key in self._tree_multi_view_marker_paths
            if self._path_key_is_under_root(path_key, root_key)
        }
        filtered_highlights = {
            path_key
            for path_key in self._tree_highlight_marker_paths
            if self._path_key_is_under_root(path_key, root_key)
        }
        filtered_highlights.update(self._known_highlight_marker_paths_for_root(root_key))
        self._tree_multi_view_marker_paths = filtered_multi
        self._tree_highlight_marker_paths = filtered_highlights
        self.model.set_multi_view_path_keys(filtered_multi)
        self.model.set_persistent_highlight_path_keys(filtered_highlights)
        self.tree.viewport().update()

    def _directory_view_states(self, directory: Path) -> dict[str, dict]:
        """Handle directory view states."""
        key = self._directory_key(directory)
        cached = self._persisted_view_sessions_by_dir.get(key)
        if cached is not None:
            return cached
        sessions: dict[str, dict] = {}
        file_path = directory / VIEWS_FILE_NAME
        payload = {"files": load_files_payload(file_path)}
        if isinstance(payload, dict):
            files = payload.get("files", payload)
            if isinstance(files, dict):
                for file_name, raw_session in files.items():
                    if not isinstance(file_name, str) or not isinstance(raw_session, dict):
                        continue
                    raw_tabs = raw_session.get("tabs")
                    if isinstance(raw_tabs, list):
                        sessions[file_name] = self._clone_json_compatible_dict(raw_session)
                        continue
                    # Backward compatibility with legacy single-view payload.
                    legacy_state = self._clone_json_compatible_dict(raw_session)
                    if not legacy_state:
                        legacy_state = dict(self._default_view_state())
                    sessions[file_name] = {
                        "active_view_id": 1,
                        "next_view_id": 2,
                        "next_view_sequence": 2,
                        "next_tab_color_index": 1,
                        "tabs": [
                            {
                                "view_id": 1,
                                "sequence": 1,
                                "color_slot": 0,
                                "custom_label": None,
                                "custom_label_anchor_scroll_y": 0.0,
                                "custom_label_anchor_top_line": self._page_from_view_state(
                                    legacy_state
                                ),
                                "state": legacy_state,
                            }
                        ],
                    }
        self._persisted_view_sessions_by_dir[key] = sessions
        return sessions

    def _save_directory_view_states(
        self,
        directory: Path,
        *,
        changed_file_names: set[str] | None = None,
    ) -> None:
        """Save directory view states."""
        key = self._directory_key(directory)
        sessions = self._persisted_view_sessions_by_dir.get(key, {})
        file_path = directory / VIEWS_FILE_NAME
        try:
            replace_all = changed_file_names is None
            names = (
                set(sessions)
                if changed_file_names is None
                else set(changed_file_names)
            )
            updates: dict[str, object | None] = {}
            for file_name in names:
                session = sessions.get(file_name)
                if (
                    isinstance(file_name, str)
                    and isinstance(session, dict)
                    and self._should_persist_document_view_session(session)
                ):
                    updates[file_name] = self._clone_json_compatible_dict(session)
                else:
                    updates[file_name] = None
            update_files_sidecar(
                file_path,
                updates,
                replace_all=replace_all,
            )
            self._persisted_view_sessions_by_dir.pop(key, None)
            self._directory_view_states(directory)
        except Exception:
            pass

    def _view_state_for_path_key(self, path_key: str | None) -> dict | None:
        """Handle view state for path key."""
        if not path_key:
            return None
        current_session = self._document_view_sessions.get(path_key)
        if isinstance(current_session, dict):
            tabs = current_session.get("tabs")
            if isinstance(tabs, list):
                active_view_id = current_session.get("active_view_id")
                for entry in tabs:
                    if not isinstance(entry, dict):
                        continue
                    if active_view_id is not None and entry.get("view_id") != active_view_id:
                        continue
                    state = entry.get("state")
                    if isinstance(state, dict):
                        return self._clone_json_compatible_dict(state)
                for entry in tabs:
                    if isinstance(entry, dict) and isinstance(entry.get("state"), dict):
                        return self._clone_json_compatible_dict(entry.get("state"))
        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return None
        directory, file_name = resolved
        session = self._directory_view_states(directory).get(file_name)
        if not isinstance(session, dict):
            return None
        tabs = session.get("tabs")
        if not isinstance(tabs, list):
            return None
        active_view_id = session.get("active_view_id")
        for entry in tabs:
            if not isinstance(entry, dict):
                continue
            if active_view_id is not None and entry.get("view_id") != active_view_id:
                continue
            state = entry.get("state")
            if isinstance(state, dict):
                return self._clone_json_compatible_dict(state)
        for entry in tabs:
            if isinstance(entry, dict) and isinstance(entry.get("state"), dict):
                return self._clone_json_compatible_dict(entry.get("state"))
        return None

    def _view_session_for_path_key(self, path_key: str | None) -> dict | None:
        """Handle view session for path key."""
        if not path_key:
            return None
        session = self._document_view_sessions.get(path_key)
        if isinstance(session, dict):
            return self._clone_json_compatible_dict(session)
        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return None
        directory, file_name = resolved
        persisted = self._directory_view_states(directory).get(file_name)
        if isinstance(persisted, dict):
            return self._clone_json_compatible_dict(persisted)
        return None

    def _persist_view_state_for_path_key(self, path_key: str | None, state: dict) -> None:
        """Handle persist view state for path key."""
        if not path_key:
            return
        session = {
            "active_view_id": 1,
            "next_view_id": 2,
            "next_view_sequence": 2,
            "next_tab_color_index": 1,
            "tabs": [
                {
                    "view_id": 1,
                    "sequence": 1,
                    "color_slot": 0,
                    "custom_label": None,
                    "custom_label_anchor_scroll_y": 0.0,
                    "custom_label_anchor_top_line": self._page_from_view_state(state),
                    "state": self._clone_json_compatible_dict(state)
                    if isinstance(state, dict) and state
                    else dict(self._default_view_state()),
                }
            ],
        }
        self._document_view_sessions[path_key] = session
        self._persist_document_view_session(path_key, capture_current=False)

    def _highlighting_file_path(self, directory: Path) -> Path:
        """Handle highlighting file path."""
        return directory / HIGHLIGHTING_FILE_NAME

    def _directory_text_highlights(self, directory: Path) -> dict[str, list[dict]]:
        """Handle directory text highlights."""
        key = self._directory_key(directory)
        cached = self._persisted_text_highlights_by_dir.get(key)
        if cached is not None:
            return cached
        highlights_by_file: dict[str, list[dict]] = {}
        file_path = self._highlighting_file_path(directory)
        payload = {"files": load_files_payload(file_path)}
        if isinstance(payload, dict):
            files = payload.get("files", payload)
            if isinstance(files, dict):
                for file_name, raw_entries in files.items():
                    if isinstance(file_name, str):
                        highlights_by_file[file_name] = self._normalize_text_highlight_entries(raw_entries)
        self._persisted_text_highlights_by_dir[key] = highlights_by_file
        return highlights_by_file

    def _save_directory_text_highlights(
        self,
        directory: Path,
        *,
        changed_file_names: set[str] | None = None,
    ) -> None:
        """Save directory text highlights."""
        key = self._directory_key(directory)
        highlights = self._persisted_text_highlights_by_dir.get(key, {})
        file_path = self._highlighting_file_path(directory)
        try:
            replace_all = changed_file_names is None
            names = (
                set(highlights)
                if changed_file_names is None
                else set(changed_file_names)
            )
            updates: dict[str, object | None] = {}
            for file_name in names:
                entries = self._normalize_text_highlight_entries(
                    highlights.get(file_name, [])
                )
                updates[file_name] = entries if entries else None
            update_files_sidecar(
                file_path,
                updates,
                replace_all=replace_all,
            )
            self._persisted_text_highlights_by_dir.pop(key, None)
            self._directory_text_highlights(directory)
        except Exception:
            pass

    def _normalize_text_highlight_entries(self, raw_entries) -> list[dict[str, int | str]]:
        """Normalize text highlight entries."""
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

    def _new_text_highlight_id(self) -> str:
        """Handle new text highlight id."""
        return f"pdfhl-{uuid.uuid4().hex}"

    def _replace_persistent_preview_highlight_range(
        self,
        path_key: str,
        page: int,
        start: int,
        end: int,
        kind: str,
        text: str = "",
    ) -> None:
        """Handle replace persistent preview highlight range."""
        if page <= 0 or end <= start:
            self.statusBar().showMessage("Select text to highlight", 3000)
            return
        normalized_kind = "important" if kind == "important" else "normal"

        def _replace_latest(entries: list[dict]) -> list[dict]:
            """Apply this range edit to the latest lock-protected entry."""
            updated: list[dict[str, int | str]] = []
            for entry in entries:
                entry_page = int(entry.get("page", 0))
                entry_start = int(entry.get("start", -1))
                entry_end = int(entry.get("end", -1))
                entry_kind = (
                    "important"
                    if str(entry.get("kind", "normal")).strip().lower()
                    == "important"
                    else "normal"
                )
                entry_text = str(entry.get("text", ""))
                if entry_page != page or entry_end <= start or entry_start >= end:
                    updated.append(
                        {
                            "id": str(entry.get("id", "")).strip()
                            or self._new_text_highlight_id(),
                            "page": entry_page,
                            "start": entry_start,
                            "end": entry_end,
                            "kind": entry_kind,
                            "text": entry_text,
                        }
                    )
                    continue
                if entry_start < start:
                    updated.append(
                        {
                            "id": self._new_text_highlight_id(),
                            "page": entry_page,
                            "start": entry_start,
                            "end": start,
                            "kind": entry_kind,
                            "text": entry_text,
                        }
                    )
                if entry_end > end:
                    updated.append(
                        {
                            "id": self._new_text_highlight_id(),
                            "page": entry_page,
                            "start": end,
                            "end": entry_end,
                            "kind": entry_kind,
                            "text": entry_text,
                        }
                    )
            updated.append(
                {
                    "id": self._new_text_highlight_id(),
                    "page": page,
                    "start": start,
                    "end": end,
                    "kind": normalized_kind,
                    "text": str(text or ""),
                }
            )
            return updated

        normalized, committed = self._transform_text_highlights_for_path_key(
            path_key,
            _replace_latest,
        )
        self._current_text_highlights = normalized
        self._apply_persistent_text_highlights()
        if not committed:
            self.statusBar().showMessage("Highlight could not be saved", 4000)
            return
        self.statusBar().showMessage(
            "Important highlight added"
            if normalized_kind == "important"
            else "Highlight added",
            2500,
        )

    def _load_text_highlights_for_path_key(self, path_key: str | None) -> list[dict]:
        """Load text highlights for path key."""
        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return []
        directory, file_name = resolved
        return list(self._directory_text_highlights(directory).get(file_name, []))

    def _persist_text_highlights_for_path_key(self, path_key: str | None, entries: list[dict]) -> None:
        """Handle persist text highlights for path key."""
        normalized = self._normalize_text_highlight_entries(entries)
        committed, _saved = self._transform_text_highlights_for_path_key(
            path_key,
            lambda _latest: normalized,
        )
        if path_key and self._current_preview_path_key() == path_key:
            self._current_text_highlights = committed

    def _transform_text_highlights_for_path_key(
        self,
        path_key: str | None,
        transform,
    ) -> tuple[list[dict], bool]:
        """Commit one same-PDF highlight mutation against the latest sidecar."""
        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return [], False
        directory, file_name = resolved
        file_path = self._highlighting_file_path(directory)

        def _apply(raw_entries):
            latest = self._normalize_text_highlight_entries(raw_entries)
            updated = self._normalize_text_highlight_entries(transform(latest))
            return updated if updated else None

        try:
            committed_entry, committed_map = transform_files_sidecar_entry(
                file_path,
                file_name,
                _apply,
            )
        except Exception:
            committed_map = load_files_payload(file_path)
            committed_entry = committed_map.get(file_name)
            saved = False
        else:
            saved = True

        committed_by_file: dict[str, list[dict]] = {}
        for committed_name, raw_entries in committed_map.items():
            normalized_entries = self._normalize_text_highlight_entries(raw_entries)
            if normalized_entries:
                committed_by_file[committed_name] = normalized_entries
        self._persisted_text_highlights_by_dir[
            self._directory_key(directory)
        ] = committed_by_file
        self._refresh_tree_marker_cache_for_path(path_key)
        return self._normalize_text_highlight_entries(committed_entry), saved

    def _on_preview_load_finished(self, path_key: str, ok: bool) -> None:
        """Handle preview load finished."""
        if not ok:
            self.statusBar().showMessage("Failed to load PDF viewer", 4000)
            return
        self._viewer_bridge_ready_by_path[path_key] = False
        if self._current_preview_path_key() == path_key:
            self._viewer_ready_timer.start()

    def _ensure_viewer_bridge_ready(self) -> None:
        """Handle ensure viewer bridge ready."""
        path_key = self._current_preview_path_key()
        if not path_key:
            return
        js = f"""
(() => {{
  try {{
    const bridge = (function () {{
{self._viewer_bridge_source}
    }})();
    if (!window.__pdfexploreBridge || !window.__pdfexploreBridge.install()) {{
      return false;
    }}
    return !!(window.__pdfexploreBridge.isReady && window.__pdfexploreBridge.isReady());
  }} catch (err) {{
    return false;
  }}
}})();
"""

        def _on_ready(result) -> None:
            """Handle ready."""
            if result is not True:
                return
            self._viewer_ready_timer.stop()
            self._viewer_bridge_ready_by_path[path_key] = True
            restore_state = (
                self._viewer_pending_restore_state_by_path.get(path_key)
                or {"scale": "page-fit"}
            )
            self._viewer_pending_restore_state_by_path[path_key] = None
            self._run_viewer_js(
                "window.__pdfexploreBridge && window.__pdfexploreBridge.restoreViewState && "
                f"window.__pdfexploreBridge.restoreViewState({json.dumps(restore_state)});"
            )
            self._apply_preview_dark_mode()
            self._apply_persistent_text_highlights()
            self._apply_active_search_to_viewer()
            self._schedule_followup_view_restore(path_key, restore_state)

        self._run_viewer_js(js, _on_ready)

    def _apply_persistent_text_highlights(self) -> None:
        """Handle apply persistent text highlights."""
        payload = self._normalize_text_highlight_entries(self._current_text_highlights)
        self._run_viewer_js(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.setPersistentHighlights && "
            f"window.__pdfexploreBridge.setPersistentHighlights({json.dumps(payload)});"
        )

    def _show_preview_context_menu(self, pos: QPoint) -> None:
        """Handle show preview context menu."""
        preview = self._current_preview_widget()
        if preview is None:
            return
        request = getattr(preview, "lastContextMenuRequest", lambda: None)()
        selected_text_hint = ""
        if request is not None:
            try:
                selected_text_hint = str(request.selectedText() or "")
            except Exception:
                selected_text_hint = ""
        if not selected_text_hint.strip():
            try:
                selected_text_hint = str(preview.selectedText() or "")
            except Exception:
                selected_text_hint = ""
        click_x = int(pos.x())
        click_y = int(pos.y())
        # Capture page/range metadata before the modal menu can collapse the
        # browser selection or outlive the bridge's short selection cache.
        self._request_preview_context_menu_selection_info(
            click_x,
            click_y,
            selected_text_hint,
            lambda info: self._show_preview_context_menu_with_cached_selection(
                pos,
                info if isinstance(info, dict) else {},
                selected_text_hint,
                click_x=click_x,
                click_y=click_y,
            ),
        )

    def _request_preview_context_menu_selection_info(
        self,
        click_x: int,
        click_y: int,
        selected_text_hint: str,
        callback,
    ) -> None:
        """Request preview context menu selection info."""
        def _on_info(result) -> None:
            """Handle info."""
            info = self._normalize_viewer_json_result(result)
            if (
                isinstance(info, dict)
                and selected_text_hint.strip()
                and not str(info.get("selectedText", "") or "").strip()
            ):
                info["selectedText"] = selected_text_hint
                info["hasSelection"] = bool(
                    info.get("hasSelection")
                    or str(info.get("selectedText", "") or "").strip()
                )
            callback(info)

        self._run_viewer_js_json(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.getSelectionInfo && "
            f"window.__pdfexploreBridge.getSelectionInfo({int(click_x)}, {int(click_y)})",
            _on_info,
        )

    def _show_preview_context_menu_with_cached_selection(
        self,
        pos: QPoint,
        info: dict,
        selected_text_hint: str,
        *,
        click_x: int | None = None,
        click_y: int | None = None,
    ) -> None:
        """Handle show preview context menu with cached selection."""
        selection_snapshot = dict(info) if isinstance(info, dict) else {}

        def _has_valid_range(candidate: dict) -> bool:
            """Return whether context data identifies a persistable range."""
            try:
                page = int(candidate.get("page", 0))
                start = int(candidate.get("start", -1))
                end = int(candidate.get("end", -1))
            except Exception:
                return False
            return page > 0 and start >= 0 and end > start

        def _merge_with_selection_snapshot(live_info: dict) -> dict:
            """Keep pre-menu selection coordinates when a late probe lost them."""
            live = dict(live_info) if isinstance(live_info, dict) else {}
            merged = dict(selection_snapshot)

            for key in ("selectedText", "clickedHighlightId"):
                value = str(live.get(key, "") or "")
                if value.strip():
                    merged[key] = value
            for key in ("multiPageSelection", "threeUpActive"):
                if key in live and key not in merged:
                    merged[key] = bool(live.get(key))

            if _has_valid_range(live):
                merged["page"] = int(live["page"])
                merged["start"] = int(live["start"])
                merged["end"] = int(live["end"])
                merged["multiPageSelection"] = bool(
                    live.get("multiPageSelection")
                )
            if _has_valid_range(merged) or str(
                merged.get("selectedText", "") or ""
            ).strip():
                merged["hasSelection"] = True
            return merged

        menu = QMenu(self)
        selected_text = str(selection_snapshot.get("selectedText", "") or "")
        has_selection = bool(selected_text_hint.strip() or selected_text.strip())
        if selection_snapshot:
            if selection_snapshot.get("hasSelection"):
                has_selection = True
            try:
                page = int(selection_snapshot.get("page", 0))
                start = int(selection_snapshot.get("start", -1))
                end = int(selection_snapshot.get("end", -1))
            except Exception:
                page = 0
                start = -1
                end = -1
            if page > 0 and start >= 0 and end > start:
                has_selection = True
        clicked_highlight_id = str(
            selection_snapshot.get("clickedHighlightId", "") or ""
        ).strip()
        three_up_active = bool(selection_snapshot.get("threeUpActive"))
        has_existing_persistent_highlights = bool(
            self._normalize_text_highlight_entries(self._current_text_highlights)
        )

        highlight_action = None
        highlight_important_action = None
        if has_selection and not three_up_active:
            highlight_action = menu.addAction("Highlight")
            highlight_important_action = menu.addAction("Highlight Important")

        remove_action = None
        if (clicked_highlight_id or has_selection or has_existing_persistent_highlights) and not three_up_active:
            remove_action = menu.addAction("Remove Highlight")

        copy_action = None
        if has_selection:
            menu.addSeparator()
            copy_action = menu.addAction("Copy Selected Text")

        preview = self._current_preview_widget()
        if preview is None:
            return

        def _run_with_fresh_context_info(handler) -> None:
            """Run with fresh context info."""
            if click_x is None or click_y is None:
                handler(selection_snapshot)
                return
            self._request_preview_context_menu_selection_info(
                int(click_x),
                int(click_y),
                selected_text_hint,
                lambda live_info: handler(
                    _merge_with_selection_snapshot(
                        live_info if isinstance(live_info, dict) else {}
                    )
                ),
            )

        chosen = menu.exec(preview.mapToGlobal(pos))
        if chosen is None:
            return
        if highlight_action is not None and chosen == highlight_action:
            _run_with_fresh_context_info(
                lambda live_info: self._add_persistent_preview_highlight(
                    live_info,
                    kind="normal",
                    selected_text_hint=selected_text_hint,
                )
            )
            return
        if highlight_important_action is not None and chosen == highlight_important_action:
            _run_with_fresh_context_info(
                lambda live_info: self._add_persistent_preview_highlight(
                    live_info,
                    kind="important",
                    selected_text_hint=selected_text_hint,
                )
            )
            return
        if remove_action is not None and chosen == remove_action:
            _run_with_fresh_context_info(self._remove_persistent_preview_highlight)
            return
        if copy_action is not None and chosen == copy_action:
            def _copy_text(live_info: dict) -> None:
                """Copy text."""
                copied_text = str(live_info.get("selectedText", "") or "").strip() or selected_text_hint
                if not copied_text.strip():
                    self.statusBar().showMessage("No selected text to copy", 2500)
                    return
                QApplication.clipboard().setText(
                    copied_text,
                    QClipboard.Mode.Clipboard,
                )
                self.statusBar().showMessage("Copied selected text", 2500)

            _run_with_fresh_context_info(_copy_text)

    def _add_persistent_preview_highlight(
        self,
        info: dict,
        *,
        kind: str,
        selected_text_hint: str = "",
    ) -> None:
        """Handle add persistent preview highlight."""
        if self.current_file is None:
            return
        if info.get("multiPageSelection"):
            self.statusBar().showMessage("Highlights must stay within one PDF page", 3000)
            return
        try:
            page = int(info.get("page", 0))
            start = int(info.get("start", -1))
            end = int(info.get("end", -1))
        except Exception:
            self.statusBar().showMessage("Select text to highlight", 3000)
            return
        if page <= 0 or start < 0 or end <= start:
            self.statusBar().showMessage("Select text to highlight", 3000)
            return
        path_key = self._path_key(self.current_file)
        self._replace_persistent_preview_highlight_range(
            path_key,
            page,
            start,
            end,
            kind,
            str(info.get("selectedText", "") or "").strip() or selected_text_hint,
        )

    def _remove_persistent_preview_highlight(self, info: dict) -> None:
        """Handle remove persistent preview highlight."""
        if self.current_file is None:
            return
        clicked_highlight_id = str(info.get("clickedHighlightId", "") or "").strip()
        removed = False
        page = 0
        start = -1
        end = -1
        if not clicked_highlight_id:
            try:
                page = int(info.get("page", 0))
                start = int(info.get("start", -1))
                end = int(info.get("end", -1))
            except Exception:
                page = 0
                start = -1
                end = -1

        def _remove_from_latest(entries: list[dict]) -> list[dict]:
            """Remove only matching entries from the latest committed list."""
            nonlocal removed
            remaining: list[dict] = []
            for entry in entries:
                matches = str(entry.get("id")) == clicked_highlight_id
                if not clicked_highlight_id:
                    same_page = int(entry.get("page", 0)) == page
                    matches = (
                        same_page
                        and int(entry["start"]) < end
                        and int(entry["end"]) > start
                    )
                if matches:
                    removed = True
                    continue
                remaining.append(entry)
            return remaining

        path_key = self._path_key(self.current_file)
        remaining, committed = self._transform_text_highlights_for_path_key(
            path_key,
            _remove_from_latest,
        )
        self._current_text_highlights = remaining
        self._apply_persistent_text_highlights()
        if not committed:
            self.statusBar().showMessage(
                "Highlight removal could not be saved",
                4000,
            )
            return
        if not removed:
            self.statusBar().showMessage("No highlighted block selected", 2500)
            return
        self.statusBar().showMessage("Highlight removed", 2500)

    def _copy_destination_is_directory(self) -> bool:
        """Copy destination is directory."""
        return bool(self.copy_directory_radio.isChecked())

    def _default_copy_destination_directory(self) -> Path:
        """Return default copy destination directory."""
        if isinstance(self._copy_destination_directory, Path) and self._copy_destination_directory.is_dir():
            return self._copy_destination_directory
        return self._effective_root_for_persistence()

    def _prompt_copy_destination_directory(self) -> Path | None:
        """Handle prompt copy destination directory."""
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
        """Normalize unique file paths."""
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
        """Copy files to clipboard."""
        normalized = self._normalize_unique_file_paths(files)
        clipboard = QApplication.clipboard()
        mime_data = QMimeData()
        urls = [QUrl.fromLocalFile(str(path)) for path in normalized]
        mime_data.setUrls(urls)
        if urls:
            gnome_payload = "copy\n" + "\n".join(url.toString() for url in urls)
            mime_data.setData(
                "x-special/gnome-copied-files", gnome_payload.encode("utf-8")
            )
        mime_data.setText("\n".join(str(path) for path in normalized))
        clipboard.setMimeData(mime_data)
        return len(normalized)

    def _merge_copied_file_metadata(
        self, copied_pairs: list[tuple[Path, Path]], target_directory: Path
    ) -> tuple[int, int, int]:
        """Handle merge copied file metadata."""
        if not copied_pairs:
            return 0, 0, 0
        color_updates = 0
        view_updates = 0
        highlight_updates = 0
        target_states = self._directory_view_states(target_directory)
        target_highlights = self._directory_text_highlights(target_directory)
        for source_path, destination_path in copied_pairs:
            source_color = self.model.color_for_file(source_path)
            normalized_color = source_color.strip() if isinstance(source_color, str) and source_color.strip() else None
            self.model.set_color_for_file(destination_path, normalized_color)
            if normalized_color is not None:
                color_updates += 1

            source_key = self._path_key(source_path)
            source_session = self._view_session_for_path_key(source_key)
            if isinstance(source_session, dict) and self._should_persist_document_view_session(
                source_session
            ):
                target_states[destination_path.name] = self._clone_json_compatible_dict(
                    source_session
                )
                view_updates += 1
            elif destination_path.name in target_states:
                target_states.pop(destination_path.name, None)

            source_highlights = self._load_text_highlights_for_path_key(source_key)
            if source_highlights:
                target_highlights[destination_path.name] = list(source_highlights)
                highlight_updates += 1
            elif destination_path.name in target_highlights:
                target_highlights.pop(destination_path.name, None)

        changed_names = {destination.name for _source, destination in copied_pairs}
        self._save_directory_view_states(
            target_directory,
            changed_file_names=changed_names,
        )
        self._save_directory_text_highlights(
            target_directory,
            changed_file_names=changed_names,
        )
        return color_updates, view_updates, highlight_updates

    def _copy_files_to_directory_with_metadata(
        self, files: list[Path]
    ) -> tuple[int, int, int, int, int, Path] | None:
        """Copy files to directory with metadata."""
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
                import shutil

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
        """Copy current preview file to clipboard."""
        if self.current_file is None:
            self.statusBar().showMessage("No previewed PDF file to copy", 3000)
            return
        target = self.current_file.resolve()
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
                f"(metadata: colors {colors}, views {views}, highlights {highlights}; failures {failed})",
                5000,
            )
            return
        copied = self._copy_files_to_clipboard([target])
        if copied:
            self.statusBar().showMessage(
                f"Copied previewed file to clipboard: {target.name}",
                4000,
            )

    def _edit_current_file(self) -> None:
        """Handle edit current file."""
        if self.current_file is None:
            self.statusBar().showMessage("Open a PDF before using Edit", 2500)
            return
        target = self.current_file.resolve()
        if not target.is_file():
            self.statusBar().showMessage("Previewed PDF is unavailable", 3000)
            return
        if not OKULAR_EDIT_LAUNCHER.is_file():
            self.statusBar().showMessage(
                f"Edit launcher not found: {OKULAR_EDIT_LAUNCHER}",
                5000,
            )
            return
        try:
            subprocess.Popen([str(OKULAR_EDIT_LAUNCHER), str(target)])
        except PermissionError:
            self.statusBar().showMessage(
                f"Edit launcher is not executable: {OKULAR_EDIT_LAUNCHER}",
                5000,
            )
        except OSError as exc:
            self.statusBar().showMessage(f"Failed to open in Okular: {exc}", 5000)
        else:
            self.statusBar().showMessage(f"Opened in Okular: {target.name}", 2500)

    def _copy_highlighted_files_to_clipboard(self, color_value: str, color_name: str) -> None:
        """Copy highlighted files to clipboard."""
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
                f"(metadata: colors {colors}, views {views}, highlights {highlights}; failures {failed})",
                5500,
            )
            return
        copied = self._copy_files_to_clipboard(matches)
        self.statusBar().showMessage(
            f"Copied {copied} {color_name.lower()} highlighted file(s) from {scope}",
            4000,
        )

    def _update_window_title(self) -> None:
        """Handle update window title."""
        scope = self._highlight_scope_directory()
        if self.model.set_effective_scope_directory(scope):
            self.tree.viewport().update()
        self.setWindowTitle(f"pdfexplore - {scope}")

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Handle resizeEvent."""
        super().resizeEvent(event)
        self._position_preview_zoom_overlay()

    def closeEvent(self, event) -> None:  # noqa: N802
        """Handle closeEvent."""
        self._closing = True
        self._stop_global_activity_heartbeat()
        self._scope_prefetch_timer.stop()
        self._cache_badge_probe_timer.stop()
        self._pdf_text_cache_gc_timer.stop()
        self._cancel_scope_prefetch()
        self._cancel_cached_badge_probe()
        current_index = self.view_tabs.currentIndex()
        if current_index >= 0:
            self._capture_tab_state(current_index, blocking=True)
        self._persist_document_view_session(capture_current=False)
        self._persist_effective_root()
        for path_key in list(self._preview_widgets_by_path):
            self._evict_preview_widget(path_key)
        super().closeEvent(event)


def _default_root_from_config() -> Path:
    """Return default root from config."""
    fallback = Path.home()
    config_path = PdfExploreWindow._config_file_path()
    try:
        if not config_path.exists():
            return fallback
        raw = config_path.read_text(encoding="utf-8", errors="replace")
        default_root_raw, recent_roots_raw = PdfExploreWindow._parse_config_payload_text(
            raw
        )
        candidates: list[str] = []
        if default_root_raw:
            candidates.append(default_root_raw)
        candidates.extend(recent_roots_raw)
        normalized = PdfExploreWindow._normalize_recent_root_directories(candidates)
        if normalized:
            return normalized[0]
    except Exception:
        pass
    return fallback


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse args."""
    parser = argparse.ArgumentParser(description="Browse and highlight PDF files.")
    parser.add_argument("path", nargs="?", help="Root directory or PDF file to open")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Handle main."""
    args = parse_args(argv)
    if getattr(args, "path", None):
        candidate = Path(args.path).expanduser()
        if candidate.is_file():
            root = candidate.parent
            initial_file = candidate.resolve()
        elif candidate.is_dir():
            root = candidate.resolve()
            initial_file = None
        else:
            root = _default_root_from_config()
            initial_file = None
    else:
        if (sys.stdin.isatty() or sys.stdout.isatty()) and Path.cwd().is_dir():
            root = Path.cwd().resolve()
        else:
            root = _default_root_from_config()
        initial_file = None

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("pdfexplore")
    if hasattr(app, "setDesktopFileName"):
        app.setDesktopFileName("pdfexplore")
    icon_path = ui_asset_path("pdf.svg")
    app_icon = QIcon(str(icon_path)) if icon_path.is_file() else QIcon()
    window = PdfExploreWindow(
        root=root,
        app_icon=app_icon,
        config_path=PdfExploreWindow._config_file_path(),
        gpu_context_available=gpu_context_available(),
        debug_mode=bool(args.debug),
    )
    window.show()
    if initial_file is not None:
        window._open_path_in_active_view(initial_file)
    return app.exec()
