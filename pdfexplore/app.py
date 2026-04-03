#!/usr/bin/env python3
"""pdfexplore: read-only PDF browser with search and highlights."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pypdf import PdfReader
from PySide6.QtCore import (
    QDir,
    QEventLoop,
    QMimeData,
    QPoint,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
)
from PySide6.QtGui import QAction, QClipboard, QIcon
from PySide6.QtWebEngineCore import QWebEngineSettings
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
    QStackedWidget,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from mdexplore_app.icons import build_clear_x_icon, ui_asset_path
from mdexplore_app.runtime import (
    config_file_path as _shared_config_file_path,
    gpu_context_available,
)
from mdexplore_app.search import (
    compile_match_hit_counter,
    compile_match_predicate,
    extract_search_terms,
)
from mdexplore_app.tabs import ViewTabBar

from .tree import ColorizedPdfModel, PdfTreeItemDelegate
from .workers import PdfSearchWorker


CONFIG_FILE_NAME = ".pdfexplore.cfg"
VIEWS_FILE_NAME = ".pdfexplore-views.json"
HIGHLIGHTING_FILE_NAME = ".pdfexplore-highlighting.json"
VIEWER_HTML = Path(__file__).resolve().parent / "vendor" / "pdfjs" / "web" / "viewer.html"
VIEWER_BRIDGE_JS = Path(__file__).resolve().parent / "assets" / "viewer_bridge.js"


class PdfExploreWindow(QMainWindow):
    """Main window for browsing and highlighting PDFs."""

    MAX_DOCUMENT_VIEWS = 8
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
        *,
        gpu_context_available: bool = False,
        debug_mode: bool = False,
    ) -> None:
        super().__init__()
        self.root = root.resolve()
        self.config_path = config_path
        self.debug_mode = bool(debug_mode)
        self.current_file: Path | None = None
        self.last_directory_selection: Path | None = self.root
        self.current_match_files: list[Path] = []
        self._current_match_counts: dict[Path, int] = {}
        self._pdf_text_cache: dict[str, tuple[int, int, str]] = {}
        self._persisted_view_sessions_by_dir: dict[str, dict[str, dict]] = {}
        self._document_view_sessions: dict[str, dict] = {}
        self._persisted_text_highlights_by_dir: dict[str, dict[str, list[dict]]] = {}
        self._current_text_highlights: list[dict[str, int | str]] = []
        self._next_text_highlight_id = 1
        self._copy_destination_directory: Path | None = None
        self._tree_multi_view_marker_paths: set[str] = set()
        self._tree_highlight_marker_paths: set[str] = set()
        self._active_view_tab_index = -1
        self._next_view_id = 1
        self._next_view_sequence = 1
        self._next_tab_color_index = 0
        self._pending_search_terms: list[tuple[str, bool]] = []
        self._viewer_bridge_source = VIEWER_BRIDGE_JS.read_text(encoding="utf-8")
        self._preview_widgets_by_path: dict[str, QWebEngineView] = {}
        self._viewer_bridge_ready_by_path: dict[str, bool] = {}
        self._viewer_pending_restore_state_by_path: dict[str, dict | None] = {}
        self._preview_signatures_by_path: dict[str, tuple[int, int]] = {}
        self._gpu_context_available = bool(gpu_context_available)

        self.thread_pool = QThreadPool(self)
        self._search_request_id = 0

        self.match_timer = QTimer(self)
        self.match_timer.setInterval(220)
        self.match_timer.setSingleShot(True)
        self.match_timer.timeout.connect(self._run_match_search)

        self._viewer_ready_timer = QTimer(self)
        self._viewer_ready_timer.setInterval(160)
        self._viewer_ready_timer.timeout.connect(self._ensure_viewer_bridge_ready)

        self._view_state_poll_timer = QTimer(self)
        self._view_state_poll_timer.setInterval(900)
        self._view_state_poll_timer.timeout.connect(self._poll_current_view_state)
        self._view_state_poll_timer.start()

        self._file_change_watch_timer = QTimer(self)
        self._file_change_watch_timer.setInterval(1200)
        self._file_change_watch_timer.timeout.connect(self._on_file_change_watch_tick)
        self._file_change_watch_timer.start()

        self.setWindowTitle("pdfexplore")
        self.setWindowIcon(app_icon)
        self.resize(1848, 980)

        self.model = ColorizedPdfModel(self)
        self.model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)
        self.model.setNameFilters(["*.pdf"])
        self.model.setNameFilterDisables(False)

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setItemDelegate(PdfTreeItemDelegate(self.tree))
        self.tree.setIconSize(ColorizedPdfModel.decorated_icon_size())
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
        self.tree.selectionModel().currentChanged.connect(self._on_tree_selection_changed)

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

        self.up_btn = QPushButton("^")
        self.up_btn.clicked.connect(self._go_up_directory)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_directory_view)

        self.add_view_btn = QPushButton("Add View")
        self.add_view_btn.clicked.connect(self._add_document_view)

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
        top_bar.addWidget(self.up_btn)
        top_bar.addWidget(refresh_btn)
        top_bar.addWidget(self.add_view_btn)
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
        preview_layout.addWidget(self.preview_stack, 1)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.tree)
        self.splitter.addWidget(preview_container)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)

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

        self._set_root_directory(self.root)
        self._update_window_title()
        self._update_up_button_state()

    def _debug_log(self, message: str) -> None:
        if self.debug_mode:
            print(f"[pdfexplore] {message}", file=sys.stderr)

    @staticmethod
    def _path_key(path: Path) -> str:
        try:
            return str(path.resolve())
        except Exception:
            return str(path)

    @staticmethod
    def _config_file_path() -> Path:
        shared = _shared_config_file_path()
        return shared.with_name(CONFIG_FILE_NAME)

    def _update_up_button_state(self) -> None:
        self.up_btn.setEnabled(self.root.parent != self.root)

    def _clear_preview_for_missing_file(self) -> None:
        self.current_file = None
        self.path_label.setText("")
        self.path_label.setToolTip("")
        self.preview_stack.setCurrentWidget(self._empty_preview)
        blocked = self.view_tabs.blockSignals(True)
        while self.view_tabs.count() > 0:
            self.view_tabs.removeTab(0)
        self.view_tabs.blockSignals(blocked)
        self._active_view_tab_index = -1
        self._refresh_view_tabs_visibility()

    def _expanded_directory_paths(self) -> list[str]:
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
        for path_text in paths:
            index = self.model.index(path_text)
            if index.isValid():
                self.tree.expand(index)

    def _set_preview_signature_for_path(self, path: Path) -> None:
        path_key = self._path_key(path)
        try:
            stat = path.stat()
        except Exception:
            return
        self._preview_signatures_by_path[path_key] = (int(stat.st_mtime_ns), int(stat.st_size))

    def _reload_current_preview(self, reason: str) -> None:
        if self.current_file is None:
            return
        path = self.current_file
        path_key = self._path_key(path)
        self._pdf_text_cache.pop(path_key, None)
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
        self._viewer_bridge_ready_by_path[path_key] = False
        preview.setUrl(self._viewer_url_for_pdf(path))
        self.statusBar().showMessage(
            f"Auto-refreshed preview: {path.name} ({reason})",
            3500,
        )

    def _on_file_change_watch_tick(self) -> None:
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
        return self._path_key(directory)

    def _path_directory_and_name(self, path_key: str | None) -> tuple[Path, str] | None:
        if not path_key:
            return None
        path = Path(path_key)
        return path.parent, path.name

    def _read_pdf_text(self, path: Path) -> str:
        try:
            stat = path.stat()
        except Exception:
            return ""
        path_key = self._path_key(path)
        cached = self._pdf_text_cache.get(path_key)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2]

        parts: list[str] = []
        try:
            reader = PdfReader(str(path))
            for page in reader.pages:
                try:
                    extracted = page.extract_text() or ""
                except Exception:
                    extracted = ""
                if extracted:
                    parts.append(extracted)
        except Exception:
            text = ""
        else:
            text = "\n\n".join(parts)
        self._pdf_text_cache[path_key] = (stat.st_mtime_ns, stat.st_size, text)
        return text

    def _list_pdf_files_non_recursive(self, directory: Path) -> list[Path]:
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

    def _highlight_scope_directory(self) -> Path:
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
        scope = self._effective_root_for_persistence()
        try:
            self.config_path.write_text(str(scope.resolve()) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _set_root_directory(self, directory: Path) -> None:
        try:
            self.root = directory.resolve()
        except Exception:
            self.root = directory
        self.model.setRootPath("")
        root_index = self.model.setRootPath(str(self.root))
        self.tree.setRootIndex(root_index)
        self.tree.expand(root_index)
        self._rebuild_tree_marker_cache()
        self._update_window_title()
        self._update_up_button_state()

    def _go_up_directory(self) -> None:
        parent = self.root.parent
        if parent == self.root:
            return
        current_key = self._current_preview_path_key()
        if current_key:
            self._persist_document_view_session(current_key, capture_current=True)
        self._set_root_directory(parent)
        if self.match_input.text().strip():
            self._run_match_search()

    def _refresh_directory_view(self, _checked: bool = False) -> None:
        self.statusBar().showMessage("Refreshing directory view...")
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
        if self.match_input.text().strip():
            self._run_match_search()

    def _on_match_text_changed(self, text: str) -> None:
        self.match_clear_action.setVisible(bool(text))
        self.match_timer.start()

    def _clear_match_input(self) -> None:
        self.match_input.clear()
        self._clear_match_results()

    def _run_match_search_now(self) -> None:
        self.match_timer.stop()
        self._run_match_search()

    def _search_scope_files(
        self, scope: Path, query: str
    ) -> tuple[list[Path], dict[Path, int]]:
        predicate = compile_match_predicate(query)
        hit_counter = compile_match_hit_counter(query)
        candidates = self._list_pdf_files_non_recursive(scope)
        matches: list[Path] = []
        match_counts: dict[Path, int] = {}
        for path in candidates:
            content = self._read_pdf_text(path)
            if predicate(path.name, content):
                matches.append(path)
                count = hit_counter(path.name, content)
                match_counts[path] = count if count > 0 else 1
        return matches, match_counts

    def _run_match_search(self) -> None:
        query = self.match_input.text().strip()
        if not query:
            self._clear_match_results()
            return
        scope = self._highlight_scope_directory()
        candidates = self._list_pdf_files_non_recursive(scope)
        self.statusBar().showMessage(
            f"Searching {len(candidates)} PDF file(s) in {scope}..."
        )
        self._search_request_id += 1
        request_id = self._search_request_id
        worker = PdfSearchWorker(scope, request_id, query, self._search_scope_files)
        worker.signals.finished.connect(self._on_search_finished)
        self.thread_pool.start(worker)

    def _on_search_finished(
        self,
        request_id: int,
        matches: list[Path],
        match_counts: dict[Path, int],
        error: str,
    ) -> None:
        if request_id != self._search_request_id:
            return
        if error:
            self.statusBar().showMessage(f"Search failed: {error}", 4000)
            return
        self.current_match_files = list(matches or [])
        self._current_match_counts = dict(match_counts or {})
        self.model.set_search_match_counts(self._current_match_counts)
        self.tree.viewport().update()
        scope = self._highlight_scope_directory()
        self.statusBar().showMessage(
            f"Matched {len(self.current_match_files)} PDF file(s) in {scope}",
            3500,
        )
        self._apply_active_search_to_viewer()

    def _clear_match_results(self) -> None:
        self.current_match_files = []
        self._current_match_counts = {}
        self.model.clear_search_match_paths()
        self.tree.viewport().update()
        self._remove_viewer_search_highlights()

    def _current_search_terms(self) -> list[tuple[str, bool]]:
        query = self.match_input.text().strip()
        return extract_search_terms(query) if query else []

    def _is_path_in_current_matches(self, path: Path) -> bool:
        target = self._path_key(path)
        return any(self._path_key(candidate) == target for candidate in self.current_match_files)

    def _apply_active_search_to_viewer(self) -> None:
        path_key = self._current_preview_path_key()
        if (
            self.current_file is None
            or not path_key
            or not self._viewer_bridge_ready_by_path.get(path_key, False)
        ):
            return
        if self._is_path_in_current_matches(self.current_file):
            self._highlight_viewer_search_terms(self._current_search_terms())
        else:
            self._remove_viewer_search_highlights()

    def _highlight_viewer_search_terms(self, terms: list[tuple[str, bool]]) -> None:
        payload = [
            {"text": text, "caseSensitive": bool(case_sensitive)}
            for text, case_sensitive in terms
            if text.strip()
        ]
        self._pending_search_terms = terms
        path_key = self._current_preview_path_key()
        if not path_key or not self._viewer_bridge_ready_by_path.get(path_key, False):
            return
        js = (
            "window.__pdfexploreBridge && "
            f"window.__pdfexploreBridge.setSearchTerms({json.dumps(payload)});"
        )
        self._run_viewer_js(js)

    def _remove_viewer_search_highlights(self) -> None:
        self._pending_search_terms = []
        self._run_viewer_js(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.clearSearchTerms && window.__pdfexploreBridge.clearSearchTerms();"
        )

    def _apply_match_highlight_color(self, color_value: str, color_name: str) -> None:
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
        if clear_action is not None and chosen == clear_action:
            self.model.set_color_for_file(path, None)
        elif chosen in color_actions:
            self.model.set_color_for_file(path, color_actions[chosen])
        self.tree.viewport().update()

    def _confirm_and_clear_directory_highlighting(self, scope: Path | None = None) -> None:
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
                self._run_match_search()
            return
        if path.is_file() and path.suffix.lower() == ".pdf":
            self._open_path_in_active_view(path)

    def _viewer_url_for_pdf(self, path: Path) -> QUrl:
        viewer_url = QUrl.fromLocalFile(str(VIEWER_HTML))
        pdf_url = QUrl.fromLocalFile(str(path))
        viewer_url.setQuery(f"file={pdf_url.toString(QUrl.ComponentFormattingOption.FullyEncoded)}")
        viewer_url.setFragment("zoom=page-width")
        return viewer_url

    def _current_preview_widget(self) -> QWebEngineView | None:
        widget = self.preview_stack.currentWidget()
        return widget if isinstance(widget, QWebEngineView) else None

    def _current_preview_path_key(self) -> str | None:
        widget = self._current_preview_widget()
        if widget is None:
            return None
        raw = widget.property("pdfexplore_path_key")
        return str(raw).strip() if raw is not None else None

    def _create_preview_widget(self, path: Path) -> QWebEngineView:
        path_key = self._path_key(path)
        preview = QWebEngineView()
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
        self._viewer_bridge_ready_by_path[path_key] = False
        self._viewer_pending_restore_state_by_path[path_key] = None
        return preview

    def _preview_widget_for_path(self, path: Path) -> QWebEngineView:
        path_key = self._path_key(path)
        existing = self._preview_widgets_by_path.get(path_key)
        if existing is not None:
            return existing
        return self._create_preview_widget(path)

    def _run_viewer_js(self, source: str, callback=None) -> None:
        preview = self._current_preview_widget()
        if preview is None:
            return
        if callback is None:
            preview.page().runJavaScript(source)
            return
        preview.page().runJavaScript(source, callback)

    @staticmethod
    def _default_view_state() -> dict[str, int | float | str]:
        return {
            "page": 1,
            "pagesCount": 1,
            "scale": "page-width",
            "scrollTop": 0.0,
            "scrollRatio": 0.0,
        }

    @staticmethod
    def _view_tab_label_for_page(page_number: int) -> str:
        return str(max(1, int(page_number)))

    @staticmethod
    def _normalize_custom_view_label(raw_value) -> str | None:
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
        normalized = self._normalize_custom_view_label(custom_label)
        if normalized is not None:
            return normalized
        return self._view_tab_label_for_page(page_number)

    @staticmethod
    def _page_from_view_state(state: dict | None) -> int:
        if not isinstance(state, dict):
            return 1
        try:
            return max(1, int(state.get("page", 1)))
        except Exception:
            return 1

    @staticmethod
    def _progress_from_view_state(state: dict | None) -> float:
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
        if not isinstance(raw, dict):
            return {}
        try:
            return json.loads(json.dumps(raw))
        except Exception:
            return dict(raw)

    def _tab_custom_label(self, tab_index: int) -> str | None:
        data = self._tab_data(tab_index)
        if data is None:
            return None
        return self._normalize_custom_view_label(data.get("custom_label"))

    def _tab_label_anchor(self, tab_index: int) -> tuple[float, int] | None:
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
        if self.view_tabs.count() > 0 and self.view_tabs.currentIndex() >= 0:
            return self.view_tabs.currentIndex()
        self._reset_document_views(self.current_file)
        tab_index = self.view_tabs.currentIndex()
        if tab_index < 0:
            tab_index = 0
        self._refresh_view_tabs_visibility()
        return tab_index

    def _new_tab_data(self, path: Path | None = None) -> dict:
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
        if index < 0 or index >= self.view_tabs.count():
            return None
        data = self.view_tabs.tabData(index)
        return data if isinstance(data, dict) else None

    def _set_tab_data(self, index: int, data: dict) -> None:
        self.view_tabs.setTabData(index, data)

    def _capture_tab_state(
        self, index: int, *, blocking: bool = False, timeout_ms: int = 220
    ) -> dict | None:
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

        loop: QEventLoop | None = None
        timeout_timer: QTimer | None = None
        done = False
        captured_state: dict | None = None
        if blocking:
            loop = QEventLoop(self)
            timeout_timer = QTimer(self)
            timeout_timer.setSingleShot(True)

            def _on_timeout() -> None:
                nonlocal done
                done = True
                if loop is not None and loop.isRunning():
                    loop.quit()

            timeout_timer.timeout.connect(_on_timeout)

        def _on_state(result) -> None:
            nonlocal done, captured_state
            if not isinstance(result, dict):
                done = True
                if loop is not None and loop.isRunning():
                    loop.quit()
                return
            data["state"] = result
            data["progress"] = self._progress_from_view_state(result)
            self._set_tab_data(index, data)
            if self._normalize_custom_view_label(data.get("custom_label")) is None:
                self.view_tabs.setTabText(
                    index, self._display_label_for_view(self._page_from_view_state(result))
                )
            captured_state = self._clone_json_compatible_dict(result)
            done = True
            if loop is not None and loop.isRunning():
                loop.quit()

        self._run_viewer_js(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.getViewState && window.__pdfexploreBridge.getViewState();",
            _on_state,
        )
        if blocking and not done and loop is not None and timeout_timer is not None:
            timeout_timer.start(max(1, int(timeout_ms)))
            loop.exec()
            timeout_timer.stop()
            timeout_timer.deleteLater()
        return captured_state

    def _poll_current_view_state(self) -> None:
        current_index = self.view_tabs.currentIndex()
        if current_index >= 0:
            self._capture_tab_state(current_index)

    @staticmethod
    def _should_persist_document_view_session(session: dict | None) -> bool:
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
        self._save_directory_view_states(directory)
        self._rebuild_tree_marker_cache()

    def _reset_document_views(
        self,
        path: Path | None,
        *,
        initial_state: dict | None = None,
    ) -> None:
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
        data = self._tab_data(index)
        if data is None:
            return
        path_text = str(data.get("path") or "").strip()
        if not path_text:
            return
        path = Path(path_text)
        self.current_file = path
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
        preview = self._preview_widget_for_path(path)
        self.preview_stack.setCurrentWidget(preview)
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
            restore_state = wanted_state or {"scale": "page-width"}
            self._run_viewer_js(
                "window.__pdfexploreBridge && window.__pdfexploreBridge.restoreViewState && "
                f"window.__pdfexploreBridge.restoreViewState({json.dumps(restore_state)});"
            )
            self._apply_persistent_text_highlights()
            self._apply_active_search_to_viewer()
        else:
            self._viewer_ready_timer.start()
        self._rebuild_tree_marker_cache()

    def _add_document_view(self) -> None:
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
        visible = False
        if self.view_tabs.count() > 1:
            visible = True
        elif self.view_tabs.count() == 1 and self._tab_custom_label(0) is not None:
            visible = True
        self.view_tabs.setVisible(visible)
        self._rebuild_tree_marker_cache()

    def _on_view_tab_changed(self, index: int) -> None:
        previous_index = self._active_view_tab_index
        if previous_index >= 0 and previous_index != index:
            self._capture_tab_state(previous_index, blocking=True)
        self._active_view_tab_index = index
        if index < 0:
            return
        self._load_tab_index(index)
        self._persist_document_view_session(capture_current=False)

    def _on_view_tab_close_requested(self, index: int) -> None:
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
        if self._tab_label_anchor(tab_index) is None:
            return
        if self.view_tabs.currentIndex() != tab_index:
            self.view_tabs.setCurrentIndex(tab_index)
        self._return_view_tab_to_beginning(tab_index)

    def _on_view_tab_beginning_reset_requested(self, tab_index: int) -> None:
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
        if not path_key:
            return
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
        if open_counts.get(path_key, 0) > 1:
            self._tree_multi_view_marker_paths.add(path_key)
        else:
            self._tree_multi_view_marker_paths.discard(path_key)
        self._sync_tree_markers_to_model()

    def _rebuild_tree_marker_cache(self) -> None:
        root = self.root
        highlighted_paths: set[str] = set()
        multi_view_paths: set[str] = set()
        open_counts: dict[str, int] = {}
        for index in range(self.view_tabs.count()):
            data = self._tab_data(index)
            if not data:
                continue
            path_key = str(data.get("path_key") or "").strip()
            if path_key:
                open_counts[path_key] = open_counts.get(path_key, 0) + 1
        multi_view_paths.update(
            path_key for path_key, count in open_counts.items() if count > 1
        )

        for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
            directory = Path(dirpath)
            if VIEWS_FILE_NAME in filenames:
                sessions = self._directory_view_states(directory)
                for file_name, session in sessions.items():
                    if self._should_persist_document_view_session(session):
                        multi_view_paths.add(self._path_key(directory / file_name))
            if HIGHLIGHTING_FILE_NAME not in filenames:
                continue
            highlights_by_file = self._directory_text_highlights(directory)
            for file_name, entries in highlights_by_file.items():
                if self._normalize_text_highlight_entries(entries):
                    highlighted_paths.add(self._path_key(directory / file_name))

        self._tree_multi_view_marker_paths = multi_view_paths
        self._tree_highlight_marker_paths = highlighted_paths
        self._sync_tree_markers_to_model()

    def _sync_tree_markers_to_model(self) -> None:
        self.model.set_multi_view_path_keys(self._tree_multi_view_marker_paths)
        self.model.set_persistent_highlight_path_keys(self._tree_highlight_marker_paths)
        self.tree.viewport().update()

    def _directory_view_states(self, directory: Path) -> dict[str, dict]:
        key = self._directory_key(directory)
        cached = self._persisted_view_sessions_by_dir.get(key)
        if cached is not None:
            return cached
        sessions: dict[str, dict] = {}
        file_path = directory / VIEWS_FILE_NAME
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
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

    def _save_directory_view_states(self, directory: Path) -> None:
        key = self._directory_key(directory)
        sessions = self._persisted_view_sessions_by_dir.get(key, {})
        file_path = directory / VIEWS_FILE_NAME
        try:
            serializable: dict[str, dict] = {}
            for file_name, session in sessions.items():
                if not isinstance(file_name, str) or not isinstance(session, dict):
                    continue
                if not self._should_persist_document_view_session(session):
                    continue
                serializable[file_name] = self._clone_json_compatible_dict(session)
            if serializable:
                file_path.write_text(
                    json.dumps({"files": serializable}, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            elif file_path.exists():
                file_path.unlink()
        except Exception:
            pass

    def _view_state_for_path_key(self, path_key: str | None) -> dict | None:
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
        return directory / HIGHLIGHTING_FILE_NAME

    def _directory_text_highlights(self, directory: Path) -> dict[str, list[dict]]:
        key = self._directory_key(directory)
        cached = self._persisted_text_highlights_by_dir.get(key)
        if cached is not None:
            return cached
        highlights_by_file: dict[str, list[dict]] = {}
        file_path = self._highlighting_file_path(directory)
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            files = payload.get("files", payload)
            if isinstance(files, dict):
                for file_name, raw_entries in files.items():
                    if isinstance(file_name, str):
                        highlights_by_file[file_name] = self._normalize_text_highlight_entries(raw_entries)
        self._persisted_text_highlights_by_dir[key] = highlights_by_file
        return highlights_by_file

    def _save_directory_text_highlights(self, directory: Path) -> None:
        key = self._directory_key(directory)
        highlights = self._persisted_text_highlights_by_dir.get(key, {})
        file_path = self._highlighting_file_path(directory)
        try:
            if highlights:
                file_path.write_text(
                    json.dumps({"files": highlights}, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            elif file_path.exists():
                file_path.unlink()
        except Exception:
            pass

    def _normalize_text_highlight_entries(self, raw_entries) -> list[dict[str, int | str]]:
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
        token = self._next_text_highlight_id
        self._next_text_highlight_id += 1
        return f"pdfhl-{token}"

    def _load_text_highlights_for_path_key(self, path_key: str | None) -> list[dict]:
        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return []
        directory, file_name = resolved
        return list(self._directory_text_highlights(directory).get(file_name, []))

    def _persist_text_highlights_for_path_key(self, path_key: str | None, entries: list[dict]) -> None:
        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return
        directory, file_name = resolved
        highlights = self._directory_text_highlights(directory)
        normalized = self._normalize_text_highlight_entries(entries)
        if normalized:
            highlights[file_name] = normalized
        else:
            highlights.pop(file_name, None)
        self._save_directory_text_highlights(directory)
        self._refresh_tree_marker_cache_for_path(path_key)

    def _on_preview_load_finished(self, path_key: str, ok: bool) -> None:
        if not ok:
            self.statusBar().showMessage("Failed to load PDF viewer", 4000)
            return
        self._viewer_bridge_ready_by_path[path_key] = False
        if self._current_preview_path_key() == path_key:
            self._viewer_ready_timer.start()

    def _ensure_viewer_bridge_ready(self) -> None:
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
            if result is not True:
                return
            self._viewer_ready_timer.stop()
            self._viewer_bridge_ready_by_path[path_key] = True
            restore_state = (
                self._viewer_pending_restore_state_by_path.get(path_key)
                or {"scale": "page-width"}
            )
            self._viewer_pending_restore_state_by_path[path_key] = None
            self._run_viewer_js(
                "window.__pdfexploreBridge && window.__pdfexploreBridge.restoreViewState && "
                f"window.__pdfexploreBridge.restoreViewState({json.dumps(restore_state)});"
            )
            self._apply_persistent_text_highlights()
            self._apply_active_search_to_viewer()

        self._run_viewer_js(js, _on_ready)

    def _apply_persistent_text_highlights(self) -> None:
        payload = self._normalize_text_highlight_entries(self._current_text_highlights)
        self._run_viewer_js(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.setPersistentHighlights && "
            f"window.__pdfexploreBridge.setPersistentHighlights({json.dumps(payload)});"
        )

    def _show_preview_context_menu(self, pos: QPoint) -> None:
        js = (
            "window.__pdfexploreBridge && window.__pdfexploreBridge.getSelectionInfo && "
            f"window.__pdfexploreBridge.getSelectionInfo({int(pos.x())}, {int(pos.y())});"
        )

        def _on_info(result) -> None:
            info = result if isinstance(result, dict) else {}
            self._show_preview_context_menu_with_info(pos, info)

        self._run_viewer_js(js, _on_info)

    def _show_preview_context_menu_with_info(self, pos: QPoint, info: dict) -> None:
        menu = QMenu(self)
        selected_text = str(info.get("selectedText", "") or "")
        page = info.get("page")
        start = info.get("start")
        end = info.get("end")
        clicked_highlight_id = str(info.get("clickedHighlightId", "") or "").strip()

        highlight_action = None
        highlight_important_action = None
        if selected_text.strip():
            highlight_action = menu.addAction("Highlight")
            highlight_important_action = menu.addAction("Highlight Important")

        remove_action = None
        if clicked_highlight_id or selected_text.strip():
            remove_action = menu.addAction("Remove Highlight")

        copy_action = None
        if selected_text.strip():
            menu.addSeparator()
            copy_action = menu.addAction("Copy Selected Text")

        preview = self._current_preview_widget()
        if preview is None:
            return
        chosen = menu.exec(preview.mapToGlobal(pos))
        if chosen is None:
            return
        if highlight_action is not None and chosen == highlight_action:
            self._add_persistent_preview_highlight(info, kind="normal")
            return
        if highlight_important_action is not None and chosen == highlight_important_action:
            self._add_persistent_preview_highlight(info, kind="important")
            return
        if remove_action is not None and chosen == remove_action:
            self._remove_persistent_preview_highlight(info)
            return
        if copy_action is not None and chosen == copy_action:
            QApplication.clipboard().setText(selected_text, QClipboard.Mode.Clipboard)
            self.statusBar().showMessage("Copied selected text", 2500)

    def _add_persistent_preview_highlight(self, info: dict, *, kind: str) -> None:
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
        entries = self._normalize_text_highlight_entries(self._current_text_highlights)
        entries.append(
            {
                "id": self._new_text_highlight_id(),
                "page": page,
                "start": start,
                "end": end,
                "kind": "important" if kind == "important" else "normal",
                "text": str(info.get("selectedText", "") or ""),
            }
        )
        path_key = self._path_key(self.current_file)
        self._current_text_highlights = entries
        self._persist_text_highlights_for_path_key(path_key, entries)
        self._apply_persistent_text_highlights()
        self.statusBar().showMessage("Important highlight added" if kind == "important" else "Highlight added", 2500)

    def _remove_persistent_preview_highlight(self, info: dict) -> None:
        if self.current_file is None:
            return
        entries = self._normalize_text_highlight_entries(self._current_text_highlights)
        if not entries:
            self.statusBar().showMessage("No persistent highlights to remove", 2500)
            return
        clicked_highlight_id = str(info.get("clickedHighlightId", "") or "").strip()
        remaining: list[dict] = []
        removed = False
        if clicked_highlight_id:
            for entry in entries:
                if str(entry.get("id")) == clicked_highlight_id:
                    removed = True
                    continue
                remaining.append(entry)
        else:
            try:
                page = int(info.get("page", 0))
                start = int(info.get("start", -1))
                end = int(info.get("end", -1))
            except Exception:
                page = 0
                start = -1
                end = -1
            for entry in entries:
                same_page = int(entry.get("page", 0)) == page
                overlaps = same_page and int(entry["start"]) < end and int(entry["end"]) > start
                if overlaps:
                    removed = True
                    continue
                remaining.append(entry)
        if not removed:
            self.statusBar().showMessage("No highlighted block selected", 2500)
            return
        path_key = self._path_key(self.current_file)
        self._current_text_highlights = remaining
        self._persist_text_highlights_for_path_key(path_key, remaining)
        self._apply_persistent_text_highlights()
        self.statusBar().showMessage("Highlight removed", 2500)

    def _copy_destination_is_directory(self) -> bool:
        return bool(self.copy_directory_radio.isChecked())

    def _default_copy_destination_directory(self) -> Path:
        if isinstance(self._copy_destination_directory, Path) and self._copy_destination_directory.is_dir():
            return self._copy_destination_directory
        return self._effective_root_for_persistence()

    def _prompt_copy_destination_directory(self) -> Path | None:
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

        self._save_directory_view_states(target_directory)
        self._save_directory_text_highlights(target_directory)
        return color_updates, view_updates, highlight_updates

    def _copy_files_to_directory_with_metadata(
        self, files: list[Path]
    ) -> tuple[int, int, int, int, int, Path] | None:
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

    def _copy_highlighted_files_to_clipboard(self, color_value: str, color_name: str) -> None:
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
        scope = self._highlight_scope_directory()
        self.setWindowTitle(f"pdfexplore - {scope}")

    def closeEvent(self, event) -> None:  # noqa: N802
        current_index = self.view_tabs.currentIndex()
        if current_index >= 0:
            self._capture_tab_state(current_index, blocking=True)
        self._persist_document_view_session(capture_current=False)
        self._persist_effective_root()
        super().closeEvent(event)


def _default_root_from_config() -> Path:
    fallback = Path.home()
    config_path = PdfExploreWindow._config_file_path()
    try:
        if not config_path.exists():
            return fallback
        raw = config_path.read_text(encoding="utf-8").strip()
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
    except Exception:
        pass
    return fallback


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse and highlight PDF files.")
    parser.add_argument("path", nargs="?", help="Root directory or PDF file to open")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
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
