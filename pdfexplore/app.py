#!/usr/bin/env python3
"""pdfexplore: read-only PDF browser with search and highlights."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pypdf import PdfReader
from PySide6.QtCore import QDir, QMimeData, QPoint, QSize, Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QAction, QClipboard, QIcon
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
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
        self._persisted_view_states_by_dir: dict[str, dict[str, dict]] = {}
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
        self._viewer_bridge_ready = False
        self._viewer_pending_restore_state: dict | None = None
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

        self.preview = QWebEngineView()
        preview_settings = self.preview.settings()
        preview_settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False
        )
        preview_settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True
        )
        self.preview.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.preview.customContextMenuRequested.connect(self._show_preview_context_menu)
        self.preview.loadFinished.connect(self._on_preview_load_finished)

        self.view_tabs = ViewTabBar()
        self.view_tabs.setDocumentMode(True)
        self.view_tabs.setMovable(False)
        self.view_tabs.setDrawBase(False)
        self.view_tabs.setExpanding(False)
        self.view_tabs.setUsesScrollButtons(True)
        self.view_tabs.setTabsClosable(True)
        self.view_tabs.setElideMode(Qt.TextElideMode.ElideNone)
        self.view_tabs.currentChanged.connect(self._on_view_tab_changed)
        self.view_tabs.tabCloseRequested.connect(self._on_view_tab_close_requested)
        self.view_tabs.beginningResetRequested.connect(
            lambda _index: self._run_viewer_js("window.__pdfexploreBridge && window.__pdfexploreBridge.goToTop && window.__pdfexploreBridge.goToTop();")
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
        preview_layout.addWidget(self.preview, 1)

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

        self._set_root_directory(self.root)
        self._update_window_title()

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

    def _go_up_directory(self) -> None:
        parent = self.root.parent
        if parent == self.root:
            return
        self._set_root_directory(parent)

    def _refresh_directory_view(self, _checked: bool = False) -> None:
        self.statusBar().showMessage("Refreshing directory view...")
        self._set_root_directory(self.root)
        self.statusBar().showMessage(f"Refreshed {self.root}", 2500)

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
        if self.current_file is None or not self._viewer_bridge_ready:
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
        if not self._viewer_bridge_ready:
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

    def _run_viewer_js(self, source: str, callback=None) -> None:
        if callback is None:
            self.preview.page().runJavaScript(source)
            return
        self.preview.page().runJavaScript(source, callback)

    def _ensure_current_tab(self) -> int:
        if self.view_tabs.count() > 0 and self.view_tabs.currentIndex() >= 0:
            return self.view_tabs.currentIndex()
        tab_index = self.view_tabs.addTab("PDF")
        self.view_tabs.setTabData(tab_index, self._new_tab_data())
        self.view_tabs.setCurrentIndex(tab_index)
        self._refresh_view_tabs_visibility()
        return tab_index

    def _new_tab_data(self, path: Path | None = None) -> dict:
        data = {
            "view_id": self._next_view_id,
            "sequence": self._next_view_sequence,
            "color_slot": self._next_tab_color_index,
            "progress": 0.0,
            "custom_label": None,
            "custom_label_anchor_scroll_y": 0.0,
            "custom_label_anchor_top_line": 1,
            "path": str(path) if isinstance(path, Path) else "",
            "path_key": self._path_key(path) if isinstance(path, Path) else "",
            "state": {},
        }
        self._next_view_id += 1
        self._next_view_sequence += 1
        self._next_tab_color_index = (self._next_tab_color_index + 1) % max(
            1, len(ViewTabBar.PASTEL_SEQUENCE)
        )
        return data

    def _tab_data(self, index: int) -> dict | None:
        if index < 0 or index >= self.view_tabs.count():
            return None
        data = self.view_tabs.tabData(index)
        return data if isinstance(data, dict) else None

    def _set_tab_data(self, index: int, data: dict) -> None:
        self.view_tabs.setTabData(index, data)

    def _capture_tab_state(self, index: int) -> None:
        data = self._tab_data(index)
        if data is None or index != self.view_tabs.currentIndex() or not self._viewer_bridge_ready:
            return

        def _on_state(result) -> None:
            if not isinstance(result, dict):
                return
            data["state"] = result
            pages = max(1, int(result.get("pagesCount", 0) or 1))
            page = max(1, int(result.get("page", 1) or 1))
            data["progress"] = max(0.0, min(1.0, page / pages))
            self._set_tab_data(index, data)
            path_key = str(data.get("path_key") or "").strip()
            if path_key:
                self._persist_view_state_for_path_key(path_key, result)

        self._run_viewer_js(
            "window.__pdfexploreBridge && window.__pdfexploreBridge.getViewState && window.__pdfexploreBridge.getViewState();",
            _on_state,
        )

    def _poll_current_view_state(self) -> None:
        current_index = self.view_tabs.currentIndex()
        if current_index >= 0:
            self._capture_tab_state(current_index)

    def _open_path_in_active_view(self, path: Path) -> None:
        if not path.is_file():
            return
        current_index = self._ensure_current_tab()
        if self._active_view_tab_index >= 0 and self._active_view_tab_index != current_index:
            self._capture_tab_state(self._active_view_tab_index)
        else:
            self._capture_tab_state(current_index)

        data = self._tab_data(current_index) or self._new_tab_data(path)
        data["path"] = str(path)
        data["path_key"] = self._path_key(path)
        data["state"] = self._view_state_for_path_key(data["path_key"]) or data.get("state") or {}
        self._set_tab_data(current_index, data)
        self.view_tabs.setTabText(current_index, path.name)
        self._active_view_tab_index = current_index
        self._rebuild_tree_marker_cache()
        self._load_tab_index(current_index)

    def _load_tab_index(self, index: int) -> None:
        data = self._tab_data(index)
        if data is None:
            return
        path_text = str(data.get("path") or "").strip()
        if not path_text:
            return
        path = Path(path_text)
        self.current_file = path
        self.path_label.setText(str(path))
        self.path_label.setToolTip(str(path))
        self._update_window_title()
        self._current_text_highlights = self._load_text_highlights_for_path_key(self._path_key(path))
        self._viewer_bridge_ready = False
        self._viewer_pending_restore_state = data.get("state") if isinstance(data.get("state"), dict) else None
        self.preview.setUrl(self._viewer_url_for_pdf(path))
        self._rebuild_tree_marker_cache()

    def _add_document_view(self) -> None:
        if self.current_file is None:
            return
        current_index = self._ensure_current_tab()
        self._capture_tab_state(current_index)
        current_data = self._tab_data(current_index) or {}
        duplicate_path = Path(str(current_data.get("path") or self.current_file))
        tab_index = self.view_tabs.addTab(duplicate_path.name)
        new_data = self._new_tab_data(duplicate_path)
        new_data["state"] = dict(current_data.get("state") or {})
        self._set_tab_data(tab_index, new_data)
        self.view_tabs.setCurrentIndex(tab_index)
        self._refresh_view_tabs_visibility()
        self._refresh_tree_marker_cache_for_path(self._path_key(duplicate_path))

    def _refresh_view_tabs_visibility(self) -> None:
        self.view_tabs.setVisible(self.view_tabs.count() > 0)
        self._rebuild_tree_marker_cache()

    def _on_view_tab_changed(self, index: int) -> None:
        previous_index = self._active_view_tab_index
        if previous_index >= 0 and previous_index != index:
            self._capture_tab_state(previous_index)
        self._active_view_tab_index = index
        if index < 0:
            return
        self._load_tab_index(index)

    def _on_view_tab_close_requested(self, index: int) -> None:
        data = self._tab_data(index)
        if data is not None:
            self._capture_tab_state(index)
            path_key = str(data.get("path_key") or "").strip()
            if path_key:
                self._persist_view_state_for_path_key(path_key, data.get("state") or {})
        self.view_tabs.removeTab(index)
        if self.view_tabs.count() == 0:
            self.current_file = None
            self.path_label.setText("")
            self.path_label.setToolTip("")
            self.preview.setUrl(QUrl("about:blank"))
            self._active_view_tab_index = -1
        else:
            self._active_view_tab_index = self.view_tabs.currentIndex()
        self._refresh_view_tabs_visibility()

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
        open_counts: dict[str, int] = {}
        for index in range(self.view_tabs.count()):
            data = self._tab_data(index)
            if not data:
                continue
            path_key = str(data.get("path_key") or "").strip()
            if path_key:
                open_counts[path_key] = open_counts.get(path_key, 0) + 1
        self._tree_multi_view_marker_paths = {
            path_key for path_key, count in open_counts.items() if count > 1
        }

        for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
            if HIGHLIGHTING_FILE_NAME not in filenames:
                continue
            directory = Path(dirpath)
            highlights_by_file = self._directory_text_highlights(directory)
            for file_name, entries in highlights_by_file.items():
                if self._normalize_text_highlight_entries(entries):
                    highlighted_paths.add(self._path_key(directory / file_name))

        self._tree_highlight_marker_paths = highlighted_paths
        self._sync_tree_markers_to_model()

    def _sync_tree_markers_to_model(self) -> None:
        self.model.set_multi_view_path_keys(self._tree_multi_view_marker_paths)
        self.model.set_persistent_highlight_path_keys(self._tree_highlight_marker_paths)
        self.tree.viewport().update()

    def _directory_view_states(self, directory: Path) -> dict[str, dict]:
        key = self._directory_key(directory)
        cached = self._persisted_view_states_by_dir.get(key)
        if cached is not None:
            return cached
        states: dict[str, dict] = {}
        file_path = directory / VIEWS_FILE_NAME
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            files = payload.get("files", payload)
            if isinstance(files, dict):
                for file_name, raw_state in files.items():
                    if isinstance(file_name, str) and isinstance(raw_state, dict):
                        states[file_name] = dict(raw_state)
        self._persisted_view_states_by_dir[key] = states
        return states

    def _save_directory_view_states(self, directory: Path) -> None:
        key = self._directory_key(directory)
        states = self._persisted_view_states_by_dir.get(key, {})
        file_path = directory / VIEWS_FILE_NAME
        try:
            if states:
                file_path.write_text(
                    json.dumps({"files": states}, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            elif file_path.exists():
                file_path.unlink()
        except Exception:
            pass

    def _view_state_for_path_key(self, path_key: str | None) -> dict | None:
        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return None
        directory, file_name = resolved
        state = self._directory_view_states(directory).get(file_name)
        return dict(state) if isinstance(state, dict) else None

    def _persist_view_state_for_path_key(self, path_key: str | None, state: dict) -> None:
        resolved = self._path_directory_and_name(path_key)
        if resolved is None:
            return
        directory, file_name = resolved
        states = self._directory_view_states(directory)
        if isinstance(state, dict) and state:
            states[file_name] = dict(state)
        else:
            states.pop(file_name, None)
        self._save_directory_view_states(directory)

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

    def _on_preview_load_finished(self, ok: bool) -> None:
        if not ok:
            self.statusBar().showMessage("Failed to load PDF viewer", 4000)
            return
        self._viewer_bridge_ready = False
        self._viewer_ready_timer.start()

    def _ensure_viewer_bridge_ready(self) -> None:
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
            self._viewer_bridge_ready = True
            restore_state = self._viewer_pending_restore_state or {"scale": "page-width"}
            self._viewer_pending_restore_state = None
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

        chosen = menu.exec(self.preview.mapToGlobal(pos))
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
            source_state = self._view_state_for_path_key(source_key)
            if isinstance(source_state, dict) and source_state:
                target_states[destination_path.name] = dict(source_state)
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
            self._capture_tab_state(current_index)
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
        root = _default_root_from_config()
        initial_file = None

    app = QApplication.instance() or QApplication(sys.argv[:1])
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
