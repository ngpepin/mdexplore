from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QPoint, QRect, Qt, QUrl
from PySide6.QtGui import QBrush, QColor, QIcon
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu, QSizePolicy

import mdexplore
from mdexplore_app import search as search_query
from mdexplore_app.file_coordination import load_files_payload, update_files_sidecar
from mdexplore_app.workers import SearchScanWorker


class WindowLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="mdexplore-layout-")
        root = Path(self._tempdir.name)
        self.window = mdexplore.MdExploreWindow(
            root=root,
            app_icon=QIcon(),
            config_path=root / ".mdexplore.cfg",
            gpu_context_available=False,
        )
        self.window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.window.show()
        QApplication.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        QApplication.processEvents()
        self._tempdir.cleanup()

    def test_long_path_label_does_not_raise_window_minimum_width(self) -> None:
        self.assertEqual(
            self.window.path_label.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Ignored,
        )
        baseline_width = self.window.centralWidget().minimumSizeHint().width()
        long_text = ("very/deep/document/path/" * 24) + "target-file.md"
        self.window.path_label.setText(long_text)
        self.window.path_label.setToolTip(long_text)
        QApplication.processEvents()
        expanded_width = self.window.centralWidget().minimumSizeHint().width()

        self.assertLessEqual(
            expanded_width,
            baseline_width + 80,
            "Long document paths should not make the main window effectively unshrinkable",
        )

    def test_preview_layout_shortcuts_paginate_pages_side_by_side_and_toggle(self) -> None:
        root = Path(self._tempdir.name)
        preview_path = root / "paged-preview.md"
        markdown_text = "\n\n".join(
            f"## Section {index}\n\n"
            f"Paragraph {index} contains enough rendered text to exercise page flow. "
            "The quick brown fox jumps over the lazy dog."
            for index in range(1, 141)
        )
        preview_path.write_text(markdown_text, encoding="utf-8")
        rendered = self.window.renderer.render_document(
            markdown_text,
            preview_path.name,
        )
        self.window.preview.setHtml(
            rendered,
            QUrl.fromLocalFile(f"{root}/"),
        )

        def run_js_json(expression: str) -> dict:
            results: list[str] = []
            self.window.preview.page().runJavaScript(
                f"JSON.stringify({expression})",
                lambda value: results.append(str(value)),
            )
            deadline = time.monotonic() + 3.0
            while not results and time.monotonic() < deadline:
                QApplication.processEvents()
                QTest.qWait(10)
            self.assertTrue(results)
            payload = json.loads(results[-1])
            self.assertIsInstance(payload, dict)
            return payload

        load_deadline = time.monotonic() + 5.0
        while time.monotonic() < load_deadline:
            ready = run_js_json(
                "({ ready: document.readyState === 'complete', "
                "hasContent: (document.querySelector('main')?.textContent || '')"
                ".includes('Section 140') })"
            )
            if ready.get("ready") and ready.get("hasContent"):
                break
            QTest.qWait(20)
        else:
            self.fail("Timed out loading paginated Markdown fixture")

        self.window.activateWindow()
        self.window.preview.setFocus()
        QApplication.processEvents()
        # Offscreen Qt occasionally exposes no focus widget after many prior
        # WebEngine windows; route through the active window in that case.
        key_target = QApplication.focusWidget() or self.window
        if key_target is not self.window:
            self.assertTrue(
                key_target is self.window.preview
                or self.window.preview.isAncestorOf(key_target)
            )
        layout_shortcuts = {
            shortcut.key().toString(): shortcut
            for shortcut in self.window._preview_layout_shortcuts
        }
        self.assertEqual(set(layout_shortcuts), {"Ctrl+7", "Ctrl+8", "Ctrl+9"})

        def press_layout_key(key, chord: str, expected_count: int | None) -> None:
            QTest.keyClick(
                key_target,
                key,
                Qt.KeyboardModifier.ControlModifier,
            )
            QApplication.processEvents()
            if self.window._preview_multi_up_count != expected_count:
                # Headless Qt can lack an active window after earlier
                # WebEngine teardowns, so exercise the verified shortcut
                # registration directly as a deterministic fallback.
                layout_shortcuts[chord].activated.emit()
                QApplication.processEvents()

        def wait_for_layout(expected_count: int) -> dict:
            payload: dict = {}
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                payload = run_js_json(
                    "(() => {"
                    " const main = document.querySelector('main');"
                    " const pages = Array.from(document.querySelectorAll("
                    "   'main > .mdexplore-layout-page'"
                    " ));"
                    " const firstTop = pages[0]?.getBoundingClientRect().top || 0;"
                    " const firstRowCount = pages.filter((page) =>"
                    "   Math.abs(page.getBoundingClientRect().top - firstTop) < 3"
                    " ).length;"
                    " const state = window.__mdexplorePageLayout?.getState?.() || {};"
                    " return {"
                    "   active: main?.classList.contains('mdexplore-page-layout') || false,"
                    "   count: Number(state.count || 0),"
                    "   pageCount: pages.length,"
                    "   firstRowCount,"
                    "   hasLastSection: (main?.textContent || '').includes('Section 140')"
                    " };"
                    "})()"
                )
                if expected_count:
                    if (
                        payload.get("active")
                        and int(payload.get("count", 0)) == expected_count
                        and int(payload.get("firstRowCount", 0)) == expected_count
                    ):
                        return payload
                elif not payload.get("active") and int(payload.get("pageCount", 0)) == 0:
                    return payload
                QTest.qWait(20)
            self.fail(
                f"Timed out waiting for {expected_count}-up Markdown layout: {payload}"
            )

        def centre_layout_page(page_number: int, marker: str) -> None:
            # Let the layout-change anchor complete before simulating a later
            # user scroll to a different row.
            QTest.qWait(220)
            positioned = run_js_json(
                "(() => {"
                " const page = Array.from(document.querySelectorAll("
                "   'main > .mdexplore-layout-page'"
                " )).find((candidate) =>"
                f"   Number(candidate.dataset.pageNumber || 0) === {page_number}"
                " );"
                " const content = page?.querySelector("
                "   '.mdexplore-layout-page-content'"
                " );"
                " const anchor = content?.firstElementChild;"
                " if (!page || !anchor) return { positioned: false };"
                f" anchor.dataset.exitAnchorTest = {json.dumps(marker)};"
                " page.scrollIntoView({ block: 'center', inline: 'center' });"
                " return { positioned: true };"
                "})()"
            )
            self.assertTrue(positioned.get("positioned"))
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                location = run_js_json(
                    "(() => {"
                    " const page = Array.from(document.querySelectorAll("
                    "   'main > .mdexplore-layout-page'"
                    " )).find((candidate) =>"
                    f"   Number(candidate.dataset.pageNumber || 0) === {page_number}"
                    " );"
                    " const rect = page?.getBoundingClientRect();"
                    " return {"
                    "   centred: Boolean(rect) && rect.top <= innerHeight / 2"
                    "     && rect.bottom >= innerHeight / 2"
                    " };"
                    "})()"
                )
                if location.get("centred"):
                    return
                QTest.qWait(20)
            self.fail(f"Page {page_number} did not reach the viewport centre")

        def assert_exit_page(page_number: int, marker: str) -> None:
            QTest.qWait(240)
            selector = json.dumps(f'[data-exit-anchor-test="{marker}"]')
            exit_state = run_js_json(
                "(() => {"
                " const state = window.__mdexplorePageLayout?.getState?.() || {};"
                f" const anchor = document.querySelector({selector});"
                " const rect = anchor?.getBoundingClientRect();"
                " return {"
                "   lastExitPageNumber: Number(state.lastExitPageNumber || 0),"
                "   anchorVisible: Boolean(rect) && rect.bottom >= 0"
                "     && rect.top < innerHeight,"
                "   anchorTop: rect?.top ?? null"
                " };"
                "})()"
            )
            self.assertEqual(exit_state.get("lastExitPageNumber"), page_number)
            self.assertTrue(exit_state.get("anchorVisible"))
            self.assertLess(abs(float(exit_state.get("anchorTop") or 0)), 160)

        def assert_marker_page_centred(page_number: int, marker: str) -> None:
            QTest.qWait(180)
            selector = json.dumps(f'[data-exit-anchor-test="{marker}"]')
            layout_state = run_js_json(
                "(() => {"
                " const state = window.__mdexplorePageLayout?.getState?.() || {};"
                f" const anchor = document.querySelector({selector});"
                " const page = anchor?.closest('.mdexplore-layout-page');"
                " const rect = page?.getBoundingClientRect();"
                " return {"
                "   pageNumber: Number(page?.dataset.pageNumber || 0),"
                "   lastLayoutAnchorPageNumber: Number("
                "     state.lastLayoutAnchorPageNumber || 0"
                "   ),"
                "   centred: Boolean(rect) && rect.top <= innerHeight / 2"
                "     && rect.bottom >= innerHeight / 2,"
                "   pageTop: rect?.top ?? null,"
                "   pageBottom: rect?.bottom ?? null,"
                "   viewportHeight: innerHeight,"
                "   scrollY: window.scrollY"
                " };"
                "})()"
            )
            self.assertEqual(layout_state.get("pageNumber"), page_number)
            self.assertEqual(
                layout_state.get("lastLayoutAnchorPageNumber"),
                page_number,
            )
            self.assertTrue(layout_state.get("centred"), layout_state)

        self.window._set_preview_zoom_factor(1.2)
        press_layout_key(Qt.Key.Key_7, "Ctrl+7", 6)
        self.assertEqual(self.window._preview_multi_up_count, 6)
        self.assertAlmostEqual(self.window.preview.zoomFactor(), 1.0)
        six_up = wait_for_layout(6)
        self.assertGreaterEqual(int(six_up.get("pageCount", 0)), 7)
        self.assertTrue(six_up.get("hasLastSection"))
        self.assertEqual(self.window._preview_zoom_overlay.text(), "6-Up")

        press_layout_key(Qt.Key.Key_8, "Ctrl+8", 3)
        self.assertEqual(self.window._preview_multi_up_count, 3)
        three_up = wait_for_layout(3)
        self.assertEqual(int(three_up.get("firstRowCount", 0)), 3)

        centre_layout_page(5, "three-up-centre")
        press_layout_key(Qt.Key.Key_8, "Ctrl+8", None)
        wait_for_layout(0)
        assert_exit_page(5, "three-up-centre")

        # Re-entering a grid from a scrolled continuous preview keeps the same
        # content page, as does changing the number of columns afterward.
        press_layout_key(Qt.Key.Key_8, "Ctrl+8", 3)
        wait_for_layout(3)
        assert_marker_page_centred(5, "three-up-centre")

        press_layout_key(Qt.Key.Key_9, "Ctrl+9", 2)
        self.assertEqual(self.window._preview_multi_up_count, 2)
        two_up = wait_for_layout(2)
        self.assertEqual(int(two_up.get("firstRowCount", 0)), 2)
        assert_marker_page_centred(5, "three-up-centre")

        centre_layout_page(3, "facing-left")

        press_layout_key(Qt.Key.Key_9, "Ctrl+9", None)
        continuous = wait_for_layout(0)
        assert_exit_page(3, "facing-left")
        self.assertIsNone(self.window._preview_multi_up_count)
        self.assertAlmostEqual(self.window.preview.zoomFactor(), 1.2)
        self.assertTrue(continuous.get("hasLastSection"))

        QTest.keyClick(
            key_target,
            Qt.Key.Key_7,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        QApplication.processEvents()
        self.assertIsNone(self.window._preview_multi_up_count)
        self.assertAlmostEqual(self.window.preview.zoomFactor(), 1.2)

        press_layout_key(Qt.Key.Key_7, "Ctrl+7", 6)
        self.assertEqual(self.window._preview_multi_up_count, 6)
        wait_for_layout(6)

        self.window._prepare_preview_zoom_for_pdf_export()
        QApplication.processEvents()
        self.assertAlmostEqual(self.window.preview.zoomFactor(), 1.0)
        export_layout = wait_for_layout(0)
        self.assertTrue(export_layout.get("hasLastSection"))

        self.window._restore_preview_zoom_after_pdf_export()
        QApplication.processEvents()
        self.assertEqual(self.window._preview_multi_up_count, 6)
        self.assertAlmostEqual(self.window.preview.zoomFactor(), 1.0)
        wait_for_layout(6)

        QTest.keyClick(
            key_target,
            Qt.Key.Key_0,
            Qt.KeyboardModifier.ControlModifier,
        )
        QApplication.processEvents()
        if self.window._preview_multi_up_count is not None:
            self.window._reset_preview_zoom()
            QApplication.processEvents()
        wait_for_layout(0)
        self.assertIsNone(self.window._preview_multi_up_count)
        self.assertAlmostEqual(self.window.preview.zoomFactor(), 1.0)

    def test_save_preview_base64_image_prompts_and_converts_to_png(self) -> None:
        gif_data_uri = (
            "data:image/gif;base64,"
            "R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
        )
        destination_without_suffix = Path(self._tempdir.name) / "saved-image"

        with patch.object(
            mdexplore.QFileDialog,
            "getSaveFileName",
            return_value=(str(destination_without_suffix), "PNG Image (*.png)"),
        ) as save_dialog:
            self.window._save_preview_image_as_png(
                gif_data_uri,
                click_x=12,
                click_y=24,
            )

        output_path = destination_without_suffix.with_suffix(".png")
        self.assertTrue(output_path.is_file())
        self.assertTrue(output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertFalse(mdexplore.QImage(str(output_path)).isNull())
        self.assertEqual(save_dialog.call_args.args[1], "Save Image")
        self.assertEqual(save_dialog.call_args.args[3], "PNG Image (*.png)")

    def test_preview_context_menu_replaces_native_save_image_action(self) -> None:
        media_url = "data:image/png;base64,AAAA"
        standard_menu = QMenu(self.window.preview)
        native_save_action = self.window.preview.pageAction(
            QWebEnginePage.WebAction.DownloadImageToDisk
        )
        native_save_action.setText("Save Image")
        standard_menu.addAction(native_save_action)

        menu = QMenu(self.window.preview)
        with patch.object(
            self.window, "_save_preview_image_as_png"
        ) as save_image:
            custom_save_action = self.window._append_preview_standard_menu_actions(
                menu,
                standard_menu,
                image_media_url=media_url,
                click_x=7,
                click_y=11,
            )
            candidates = [
                action
                for action in menu.actions()
                if action.text().replace("&", "").strip() == "Save Image"
            ]
            self.assertEqual(len(candidates), 1)
            self.assertIs(candidates[0], custom_save_action)
            self.assertIsNot(candidates[0], native_save_action)
            candidates[0].trigger()

        save_image.assert_called_once_with(media_url, click_x=7, click_y=11)

    def test_preview_image_png_filename_uses_source_stem(self) -> None:
        self.assertEqual(
            self.window._preview_image_png_filename(
                "https://example.test/images/figure.final.jpg?download=1"
            ),
            "figure.final.png",
        )
        self.assertEqual(
            self.window._preview_image_png_filename("data:image/png;base64,AAAA"),
            "image.png",
        )

    def test_single_view_zoom_and_position_persist_across_runs(self) -> None:
        root = Path(self._tempdir.name)
        markdown_path = root / "single-state.md"
        markdown_path.write_text("# Saved state\n\nContent\n", encoding="utf-8")
        path_key = self.window._path_key(markdown_path)
        self.window.current_file = markdown_path
        self.window._reset_document_views(initial_scroll=720.0, initial_line=38)
        self.window.preview.setZoomFactor(1.3)

        self.window._persist_document_view_session(path_key, capture_current=False)

        sidecar = root / ".mdexplore-views.json"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        persisted = payload["files"][markdown_path.name]
        self.assertEqual(persisted["preview"]["layout"], 0)
        self.assertAlmostEqual(persisted["preview"]["zoom_factor"], 1.3)

        reopened = mdexplore.MdExploreWindow(
            root=root,
            app_icon=QIcon(),
            config_path=root / ".mdexplore-reopened.cfg",
            gpu_context_available=False,
        )
        reopened.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        reopened.current_file = markdown_path
        reopened._load_persisted_document_view_session(path_key)
        self.assertTrue(reopened._restore_document_view_session(path_key))
        restored_state = reopened._current_view_state() or {}
        self.assertAlmostEqual(float(restored_state.get("scroll_y", 0.0)), 720.0)
        self.assertEqual(int(restored_state.get("top_line", 0)), 38)
        self.assertIsNone(reopened._preview_multi_up_count)
        self.assertAlmostEqual(reopened.preview.zoomFactor(), 1.3)
        reopened.current_file = None
        reopened.close()
        QApplication.processEvents()

    def test_page_layout_and_active_view_tab_persist_across_runs(self) -> None:
        root = Path(self._tempdir.name)
        markdown_path = root / "multi-state.md"
        markdown_path.write_text("# Saved tabs\n\nContent\n", encoding="utf-8")
        path_key = self.window._path_key(markdown_path)
        self.window.current_file = markdown_path
        self.window._reset_document_views(initial_scroll=110.0, initial_line=8)
        second_view_id = self.window._create_document_view(
            980.0,
            74,
            make_current=True,
        )
        self.window._preview_multi_up_count = 2
        self.window._preview_multi_up_baseline_zoom = 1.2
        self.window.preview.setZoomFactor(1.0)

        self.window._persist_document_view_session(path_key, capture_current=False)

        reopened = mdexplore.MdExploreWindow(
            root=root,
            app_icon=QIcon(),
            config_path=root / ".mdexplore-tabs-reopened.cfg",
            gpu_context_available=False,
        )
        reopened.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        reopened.current_file = markdown_path
        reopened._load_persisted_document_view_session(path_key)
        self.assertTrue(reopened._restore_document_view_session(path_key))
        self.assertEqual(reopened._active_view_id, second_view_id)
        self.assertEqual(reopened.view_tabs.count(), 2)
        self.assertEqual(reopened._preview_multi_up_count, 2)
        self.assertAlmostEqual(
            float(reopened._preview_multi_up_baseline_zoom or 0.0),
            1.2,
        )
        restored_state = reopened._current_view_state() or {}
        self.assertAlmostEqual(float(restored_state.get("scroll_y", 0.0)), 980.0)
        self.assertEqual(int(restored_state.get("top_line", 0)), 74)
        reopened.current_file = None
        reopened.close()
        QApplication.processEvents()

    def test_stale_view_cache_merge_preserves_another_instance_entry(self) -> None:
        root = Path(self._tempdir.name)
        local_sessions = self.window._directory_view_sessions(root)
        external_session = {
            "view_states": {"1": {"scroll_y": 440.0, "top_line": 22}},
            "tabs": [{"view_id": 1, "sequence": 1, "color_slot": 0}],
            "active_view_id": 1,
            "preview": {"layout": 0, "zoom_factor": 1.0},
        }
        update_files_sidecar(
            root / ".mdexplore-views.json",
            {"external.md": external_session},
        )
        local_session = {
            "view_states": {"1": {"scroll_y": 880.0, "top_line": 44}},
            "tabs": [{"view_id": 1, "sequence": 1, "color_slot": 0}],
            "active_view_id": 1,
            "preview": {"layout": 3, "zoom_factor": 1.25},
        }
        local_sessions["local.md"] = local_session

        self.window._save_directory_view_sessions(
            root,
            changed_file_names={"local.md"},
        )

        committed = load_files_payload(root / ".mdexplore-views.json")
        self.assertEqual(set(committed), {"external.md", "local.md"})
        self.assertEqual(committed["external.md"], external_session)
        self.assertEqual(committed["local.md"], local_session)

    def test_safe_path_helpers_swallow_permission_errors(self) -> None:
        class _DeniedPath:
            def is_dir(self):
                raise PermissionError("denied")

            def is_file(self):
                raise PermissionError("denied")

            def resolve(self):
                raise PermissionError("denied")

        denied = _DeniedPath()

        self.assertFalse(self.window._safe_is_dir(denied))
        self.assertFalse(self.window._safe_is_file(denied))
        self.assertIs(self.window._safe_resolve(denied), denied)

    def test_search_defaults_limit_gil_contention_and_use_small_batches(self) -> None:
        self.assertEqual(self.window.DEFAULT_SEARCH_SCAN_MAX_THREADS, 1)
        self.assertEqual(self.window.SEARCH_WORKER_CHUNK_SIZE, 8)

    def test_visible_search_candidate_scan_does_not_resolve_paths_on_ui_thread(
        self,
    ) -> None:
        root = Path(self._tempdir.name)
        target = root / "candidate.md"
        target.write_text("alpha", encoding="utf-8")
        self.window._refresh_directory_view()
        QApplication.processEvents()
        self._wait_for_tree_index(target)
        self.window._invalidate_visible_tree_markdown_cache()

        with patch.object(
            self.window,
            "_path_key",
            side_effect=AssertionError("UI candidate scan resolved a path"),
        ):
            candidates = self.window._list_visible_markdown_files_in_tree()

        self.assertIn(target, candidates)

    def _wait_for_tree_index(self, path: Path, timeout_seconds: float = 2.0):
        deadline = time.monotonic() + timeout_seconds
        index = self.window.model.index(str(path))
        while not index.isValid() and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.01)
            index = self.window.model.index(str(path))
        self.assertTrue(index.isValid(), f"Tree index did not load for {path}")
        return index

    def test_directory_context_menu_make_root_uses_selected_folder(self) -> None:
        root = Path(self._tempdir.name)
        target = root / "nested-root"
        target.mkdir()

        self.window._refresh_directory_view()
        QApplication.processEvents()
        index = self._wait_for_tree_index(target)
        self.window.tree.scrollTo(index)
        QApplication.processEvents()
        pos = self.window.tree.visualRect(index).center()

        class _FakeAction:
            def __init__(self, text: str) -> None:
                self.text = text

        class _FakeMenu:
            def __init__(self, *_args, **_kwargs) -> None:
                self.actions: list[_FakeAction] = []

            def addAction(self, text: str):
                action = _FakeAction(text)
                self.actions.append(action)
                return action

            def addSeparator(self) -> None:
                return None

            def exec(self, *_args, **_kwargs):
                return next(
                    action for action in self.actions if action.text == "Make Root"
                )

        with patch("mdexplore.QMenu", _FakeMenu), patch.object(
            self.window, "_set_root_directory"
        ) as set_root_mock:
            self.window._show_tree_context_menu(pos)

        set_root_mock.assert_called_once_with(target)

    def test_folder_search_hit_counts_are_independent_of_selected_scope(self) -> None:
        root = Path(self._tempdir.name)
        selected_folder = root / "selected"
        other_folder = root / "other" / "nested"
        selected_folder.mkdir()
        other_folder.mkdir(parents=True)
        selected_file = selected_folder / "selected.md"
        selected_nested_folder = selected_folder / "nested"
        selected_nested_folder.mkdir()
        selected_nested_file = selected_nested_folder / "nested.md"
        other_file = other_folder / "other.md"
        selected_file.write_text("selected", encoding="utf-8")
        selected_nested_file.write_text("nested", encoding="utf-8")
        other_file.write_text("other", encoding="utf-8")

        self.window.model.set_search_match_counts(
            {selected_file: 2, selected_nested_file: 8, other_file: 3}
        )
        self.window.model.set_effective_scope_directory(selected_folder)

        self.assertEqual(
            self.window.model.search_hit_count_for_directory(selected_folder), 2
        )
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(selected_nested_folder), 1
        )
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(other_folder), 1
        )
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(other_folder.parent), 1
        )

        selected_index = self._wait_for_tree_index(selected_folder)
        other_index = self._wait_for_tree_index(other_folder.parent)
        expected_color = QColor(self.window.model.DIRECTORY_SEARCH_MATCH_COLOR)
        pill_color = QColor(self.window.tree.itemDelegate()._DIR_SEARCH_PILL_BG)
        self.assertLess(expected_color.lightness(), pill_color.lightness())
        for index in (selected_index, other_index):
            foreground = self.window.model.data(
                index,
                Qt.ItemDataRole.ForegroundRole,
            )
            self.assertIsInstance(foreground, QBrush)
            self.assertEqual(foreground.color(), expected_color)

    def test_directory_pill_layout_does_not_elide_a_short_name(self) -> None:
        delegate = self.window.tree.itemDelegate()
        natural_width = self.window.tree.fontMetrics().horizontalAdvance("Books")
        text_rect = QRect(20, 4, 240, 24)

        label_rect, pill_rect = delegate._directory_search_layout_rects(
            text_rect,
            natural_width,
            30,
            7,
            18,
        )

        self.assertGreaterEqual(
            label_rect.width(),
            natural_width + delegate._DIR_SEARCH_LABEL_SLACK,
        )
        self.assertEqual(
            pill_rect.left() - label_rect.right() - 1,
            delegate._DIR_SEARCH_PILL_GAP,
        )

    def test_folder_search_counts_and_repaints_follow_visible_symlink_directory(
        self,
    ) -> None:
        root = Path(self._tempdir.name)
        publisher = root / "publisher"
        publisher.mkdir()
        target = publisher / "book.md"
        target.write_text("alpha", encoding="utf-8")
        topic = root / "topic"
        try:
            topic.symlink_to(publisher, target_is_directory=True)
        except OSError:
            self.skipTest("directory symlinks are not supported")

        self.window._refresh_directory_view()
        QApplication.processEvents()
        topic_index = self._wait_for_tree_index(topic)
        changed_paths: list[Path] = []

        def _record_changed_path(top_left, _bottom_right, _roles) -> None:
            if top_left.isValid():
                changed_paths.append(Path(self.window.model.filePath(top_left)))

        self.window.model.dataChanged.connect(_record_changed_path)
        self.window.model.set_search_match_counts(
            {target: 3},
            directory_match_paths=[topic / target.name],
        )

        self.assertEqual(
            self.window.model.search_hit_count_for_directory(topic),
            1,
        )
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(publisher),
            0,
        )
        self.assertIn(topic, changed_paths)
        self.assertTrue(topic_index.isValid())

    def test_search_publishes_symlink_directory_pills_before_all_workers_finish(
        self,
    ) -> None:
        root = Path(self._tempdir.name)
        publisher = root / "publisher"
        topic = root / "topic"
        publisher.mkdir()
        topic.mkdir()
        target = publisher / "book.md"
        target.write_text("alpha", encoding="utf-8")
        visible_link = topic / "linked-book.md"
        try:
            visible_link.symlink_to(target)
        except OSError:
            self.skipTest("file symlinks are not supported")

        canonical_key = self.window._path_key(target)
        self.window._search_scan_request_id = 7
        self.window._search_scan_total_candidates = 2
        self.window._search_scan_scope = root
        display_path = self.window._path_identity_without_io(visible_link)
        self.window._search_scan_candidate_order = {}
        self.window._search_scan_display_candidate_order = {
            display_path: 0,
        }
        self.window._search_scan_display_paths_by_key = {}
        self.window._search_scan_expected_workers = 2
        self.window._search_scan_completed_workers = 0
        self.window._search_scan_match_counts = {}
        self.window._search_scan_filename_match_paths = set()
        self.window._search_scan_published_match_count = 0

        predicate = search_query.compile_match_predicate("alpha")
        hit_counter = search_query.compile_match_hit_counter("alpha")
        worker = SearchScanWorker(7, [], predicate, hit_counter, [])
        worker.matched_display_paths_by_key = {
            canonical_key: {display_path},
        }
        self.window._active_search_scan_workers.add(worker)

        self.window._on_search_scan_finished(
            worker.worker_token,
            7,
            [canonical_key],
            {canonical_key: 2},
            [],
            "",
        )

        self.assertEqual(self.window._search_scan_completed_workers, 1)
        self.assertEqual(self.window._search_scan_published_match_count, 1)
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(topic),
            1,
        )
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(publisher),
            0,
        )
        self.assertFalse(self.window._search_progress_publish_timer.isActive())

    def test_search_coalesces_later_progressive_pill_updates(self) -> None:
        root = Path(self._tempdir.name)
        paths = [root / f"doc-{number}.md" for number in range(3)]
        for path in paths:
            path.write_text("alpha", encoding="utf-8")
        keys = [self.window._path_key(path) for path in paths]

        self.window._search_scan_request_id = 11
        self.window._search_scan_total_candidates = 3
        self.window._search_scan_scope = root
        self.window._search_scan_candidate_order = {
            path_key: index for index, path_key in enumerate(keys)
        }
        self.window._search_scan_display_paths_by_key = {
            path_key: {self.window._path_identity_without_io(path)}
            for path_key, path in zip(keys, paths)
        }
        self.window._search_scan_expected_workers = 3
        self.window._search_scan_completed_workers = 0
        self.window._search_scan_match_counts = {}
        self.window._search_scan_filename_match_paths = set()
        self.window._search_scan_published_match_count = 0

        self.window._on_search_scan_finished(
            object(), 11, [keys[0]], {keys[0]: 1}, [], ""
        )
        self.assertEqual(len(self.window.current_match_files), 1)

        self.window._on_search_scan_finished(
            object(), 11, [keys[1]], {keys[1]: 1}, [], ""
        )
        self.assertTrue(self.window._search_progress_publish_timer.isActive())
        self.assertEqual(len(self.window.current_match_files), 1)

        self.window._search_progress_publish_timer.stop()
        self.window._publish_search_scan_progress()
        self.assertEqual(len(self.window.current_match_files), 2)

        self.window._on_search_scan_finished(
            object(), 11, [keys[2]], {keys[2]: 1}, [], ""
        )
        self.assertEqual(len(self.window.current_match_files), 3)
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(root),
            3,
        )

    def test_search_worker_completion_removes_the_exact_worker(self) -> None:
        predicate = search_query.compile_match_predicate("alpha")
        hit_counter = search_query.compile_match_hit_counter("alpha")
        first = SearchScanWorker(7, [], predicate, hit_counter, [])
        second = SearchScanWorker(7, [], predicate, hit_counter, [])
        self.window._active_search_scan_workers = {first, second}
        self.window._search_scan_request_id = 7
        self.window._search_scan_expected_workers = 2
        self.window._search_scan_completed_workers = 0

        self.window._on_search_scan_finished(
            first.worker_token,
            7,
            [],
            {},
            [],
            "",
        )

        self.assertNotIn(first, self.window._active_search_scan_workers)
        self.assertIn(second, self.window._active_search_scan_workers)
        self.assertEqual(self.window._search_scan_completed_workers, 1)

    def test_expand_and_collapse_buttons_are_between_refresh_and_pdf(self) -> None:
        button_labels = [
            button.text()
            for button in self.window.centralWidget().findChildren(
                mdexplore.QPushButton
            )
        ]
        refresh_index = button_labels.index("Refresh")
        self.assertEqual(
            button_labels[refresh_index : refresh_index + 4],
            ["Refresh", "Expand", "Collapse", "PDF"],
        )

    def test_expand_opens_markdown_bearing_branches_and_ancestors(self) -> None:
        root = Path(self._tempdir.name)
        markdown_parent = root / "reports"
        markdown_directory = markdown_parent / "quarterly"
        markdown_directory.mkdir(parents=True)
        (markdown_directory / "results.md").write_text("# Results", encoding="utf-8")

        direct_markdown_directory = root / "notes"
        direct_markdown_directory.mkdir()
        (direct_markdown_directory / "todo.md").write_text("# Todo", encoding="utf-8")

        empty_directory = root / "empty" / "nested"
        empty_directory.mkdir(parents=True)

        self.window._refresh_directory_view()
        QApplication.processEvents()
        self.window.expand_btn.click()
        QApplication.processEvents()

        reports_index = self._wait_for_tree_index(markdown_parent)
        quarterly_index = self._wait_for_tree_index(markdown_directory)
        notes_index = self._wait_for_tree_index(direct_markdown_directory)
        empty_index = self._wait_for_tree_index(root / "empty")

        self.assertTrue(self.window.tree.isExpanded(reports_index))
        self.assertTrue(self.window.tree.isExpanded(quarterly_index))
        self.assertTrue(self.window.tree.isExpanded(notes_index))
        self.assertFalse(self.window.tree.isExpanded(empty_index))

    def test_collapse_closes_all_directory_branches(self) -> None:
        root = Path(self._tempdir.name)
        markdown_parent = root / "reports"
        markdown_directory = markdown_parent / "quarterly"
        markdown_directory.mkdir(parents=True)
        (markdown_directory / "results.md").write_text("# Results", encoding="utf-8")

        self.window._refresh_directory_view()
        QApplication.processEvents()
        self.window.expand_btn.click()
        QApplication.processEvents()

        reports_index = self._wait_for_tree_index(markdown_parent)
        quarterly_index = self._wait_for_tree_index(markdown_directory)
        self.assertTrue(self.window.tree.isExpanded(reports_index))
        self.assertTrue(self.window.tree.isExpanded(quarterly_index))

        self.window.collapse_btn.click()
        QApplication.processEvents()

        self.assertFalse(self.window.tree.isExpanded(reports_index))
        self.assertFalse(self.window.tree.isExpanded(quarterly_index))

    def test_collapsed_directory_is_removed_from_search_and_loses_its_pill(
        self,
    ) -> None:
        root = Path(self._tempdir.name)
        folder = root / "reports"
        folder.mkdir()
        hidden_match = folder / "result.md"
        hidden_match.write_text("needle\n", encoding="utf-8")

        self.window._refresh_directory_view()
        QApplication.processEvents()
        folder_index = self._wait_for_tree_index(folder)
        self.window.tree.expand(folder_index)
        QApplication.processEvents()
        self._wait_for_tree_index(hidden_match)

        self.window.match_input.setText("needle")
        self.window._run_match_search_now()
        deadline = time.monotonic() + 5.0
        while not self.window.current_match_files and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.005)

        self.assertIn(hidden_match.resolve(), self.window.current_match_files)
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(folder),
            1,
        )
        pre_collapse_request_id = self.window._search_scan_request_id

        self.window.tree.collapse(folder_index)

        self.assertFalse(self.window.tree.isExpanded(folder_index))
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(folder),
            0,
        )
        self.assertNotIn(hidden_match, self.window._list_visible_markdown_files_in_tree())
        self.assertNotIn(hidden_match.resolve(), self.window.current_match_files)

        # A worker result from the expanded-tree request must not restore the
        # collapsed directory's file match or pill.
        hidden_key = self.window._path_key(hidden_match)
        self.window._on_search_scan_finished(
            object(),
            pre_collapse_request_id,
            [hidden_key],
            {hidden_key: 1},
            [],
            "",
        )
        QApplication.processEvents()
        self.assertEqual(
            self.window.model.search_hit_count_for_directory(folder),
            0,
        )
        self.assertNotIn(hidden_match.resolve(), self.window.current_match_files)

    def test_status_bar_shows_mmdr_version_before_gpu(self) -> None:
        self.window.close()
        QApplication.processEvents()

        with patch.object(
            mdexplore.MarkdownRenderer,
            "mermaid_runtime_status_text",
            return_value="mmdr 0.2.2",
        ):
            self.window = mdexplore.MdExploreWindow(
                root=Path(self._tempdir.name),
                app_icon=QIcon(),
                config_path=Path(self._tempdir.name) / ".mdexplore.cfg",
                gpu_context_available=True,
            )

        self.window.show()
        QApplication.processEvents()

        status_widgets = [
            widget.text()
            for widget in self.window.statusBar().findChildren(mdexplore.QLabel)
            if widget.text() in {"mmdr 0.2.2", "GPU"}
        ]
        self.assertEqual(status_widgets, ["mmdr 0.2.2", "GPU"])
        self.assertEqual(
            self.window._mermaid_runtime_status_label.styleSheet(),
            self.window._gpu_status_label.styleSheet(),
        )


if __name__ == "__main__":
    unittest.main()
