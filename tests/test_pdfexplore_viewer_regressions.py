from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-gpu --disable-software-rasterizer",
)

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from pdfexplore.app import PdfExploreWindow


def _create_pdf_with_lines(path: Path, prefix: str, count: int) -> None:
    from reportlab.pdfgen import canvas

    writer = canvas.Canvas(str(path))
    y = 780
    for index in range(1, count + 1):
        writer.drawString(72, y, f"{prefix} line {index}")
        y -= 18
        if y < 72:
            writer.showPage()
            y = 780
    writer.save()


class PdfExploreViewerRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="pdfexplore-viewer-reg-")
        self.root = Path(self._tempdir.name)
        self.first_pdf = self.root / "first.pdf"
        self.second_pdf = self.root / "second.pdf"
        _create_pdf_with_lines(self.first_pdf, "Alpha", 180)
        _create_pdf_with_lines(self.second_pdf, "Beta", 40)
        self.window = PdfExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.root / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        self.window.show()
        self._app.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        self._app.processEvents()
        self._tempdir.cleanup()

    def wait_ms(self, milliseconds: int) -> None:
        QTest.qWait(int(milliseconds))

    def wait_until(self, predicate, *, timeout_ms: int, step_ms: int = 50) -> None:
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        while time.monotonic() < deadline:
            self._app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, step_ms)
            if predicate():
                return
            QTest.qWait(step_ms)
        self.fail(f"Timed out after {timeout_ms} ms waiting for condition")

    def run_current_viewer_js(self, script: str):
        preview = self.window._current_preview_widget()
        self.assertIsNotNone(preview)
        assert preview is not None
        loop = QEventLoop()
        holder: dict[str, object] = {}

        def _done(result) -> None:
            holder["result"] = result
            loop.quit()

        preview.page().runJavaScript(script, _done)
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        if "result" not in holder:
            self.fail("Timed out waiting for viewer JavaScript result")
        return holder["result"]

    def run_current_viewer_js_json(self, expression: str) -> dict:
        raw = self.run_current_viewer_js(f"(() => JSON.stringify({expression}))();")
        self.assertIsInstance(raw, str)
        payload = json.loads(raw)
        self.assertIsInstance(payload, dict)
        return payload

    def await_callback_result(self, invoker, *, timeout_ms: int = 10000):
        loop = QEventLoop()
        holder: dict[str, object] = {}

        def _done(result) -> None:
            holder["result"] = result
            loop.quit()

        invoker(_done)
        QTimer.singleShot(timeout_ms, loop.quit)
        loop.exec()
        if "result" not in holder:
            self.fail("Timed out waiting for callback result")
        return holder["result"]

    def _open_and_wait_for_viewer(self, path: Path) -> None:
        self.window._open_path_in_active_view(path)
        path_key = self.window._path_key(path)
        self.wait_until(
            lambda: self.window._viewer_bridge_ready_by_path.get(path_key, False),
            timeout_ms=20000,
        )
        self.wait_until(
            lambda: bool(
                self.run_current_viewer_js(
                    "(() => !!document.querySelector('.textLayer span'))();"
                )
            ),
            timeout_ms=20000,
        )

    def _seed_alpha_selection_info(self) -> dict:
        selected = self.run_current_viewer_js(
            """
(() => {
  const layer = document.querySelector('.textLayer');
  if (!layer) return false;
  const walker = document.createTreeWalker(layer, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const value = node.nodeValue || "";
    const index = value.indexOf("Alpha");
    if (index < 0) continue;
    const selection = window.getSelection();
    const range = document.createRange();
    range.setStart(node, index);
    range.setEnd(node, index + 5);
    selection.removeAllRanges();
    selection.addRange(range);
    return true;
  }
  return false;
})();
"""
        )
        self.assertEqual(selected, True)
        result = self.await_callback_result(
            lambda done: self.window._request_preview_context_menu_selection_info(
                0, 0, "Alpha", done
            )
        )
        self.assertIsInstance(result, dict)
        return dict(result)

    def test_pdfjs_viewer_container_remains_the_bounded_scroll_host(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)
        metrics = self.run_current_viewer_js_json(
            "(() => {"
            " const container = document.getElementById('viewerContainer');"
            " const style = container ? getComputedStyle(container) : null;"
            " return {"
            "   position: style?.position || '',"
            "   overflowY: style?.overflowY || '',"
            "   clientHeight: Number(container?.clientHeight || 0),"
            "   scrollHeight: Number(container?.scrollHeight || 0),"
            "   viewportHeight: Number(window.innerHeight || 0)"
            " };"
            "})()"
        )

        self.assertEqual(metrics.get("position"), "absolute")
        self.assertIn(metrics.get("overflowY"), {"auto", "scroll"})
        self.assertGreater(
            float(metrics.get("scrollHeight", 0)),
            float(metrics.get("clientHeight", 0)) + 100,
        )
        self.assertLessEqual(
            float(metrics.get("clientHeight", 0)),
            float(metrics.get("viewportHeight", 0)) + 10,
        )

    def test_scroll_state_restores_after_switching_documents_in_live_viewer(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)
        desired_state = {
            "page": 3,
            "pagesCount": 5,
            "scale": "page-width",
            "scrollTop": 1500,
            "scrollRatio": 0.5,
        }
        self.run_current_viewer_js(
            f"window.__pdfexploreBridge.restoreViewState({json.dumps(desired_state)});"
        )
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "window.__pdfexploreBridge.getViewState()"
            ).get("scrollTop", 0)
            >= 1400,
            timeout_ms=12000,
        )
        settled_before = self.run_current_viewer_js_json(
            "window.__pdfexploreBridge.getViewState()"
        )

        self._open_and_wait_for_viewer(self.second_pdf)
        self._open_and_wait_for_viewer(self.first_pdf)
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "window.__pdfexploreBridge.getViewState()"
            ).get("scrollTop", 0)
            >= 1400,
            timeout_ms=12000,
        )
        restored = self.run_current_viewer_js_json(
            "window.__pdfexploreBridge.getViewState()"
        )

        self.assertGreaterEqual(float(restored.get("scrollTop", 0)), 1400.0)
        self.assertLessEqual(
            abs(float(restored.get("scrollTop", 0)) - float(settled_before.get("scrollTop", 0))),
            220.0,
        )

    def test_dark_mode_applies_to_loaded_and_new_pdf_viewers(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)

        self.window.dark_mode_btn.click()
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({"
                " active: window.__pdfexploreBridge.isDarkModeActive(),"
                " classActive: document.documentElement.classList.contains('pdfexplore-dark-mode'),"
                " filter: getComputedStyle(document.querySelector('.canvasWrapper')).filter"
                "})"
            ).get("active")
            is True,
            timeout_ms=5000,
        )
        first_dark_state = self.run_current_viewer_js_json(
            "({"
            " active: window.__pdfexploreBridge.isDarkModeActive(),"
            " classActive: document.documentElement.classList.contains('pdfexplore-dark-mode'),"
            " filter: getComputedStyle(document.querySelector('.canvasWrapper')).filter"
            "})"
        )
        self.assertEqual(first_dark_state.get("classActive"), True)
        self.assertNotEqual(first_dark_state.get("filter"), "none")
        self.assertEqual(self.window.dark_mode_btn.text(), "Light")

        self._open_and_wait_for_viewer(self.second_pdf)
        second_dark_state = self.run_current_viewer_js_json(
            "({"
            " active: window.__pdfexploreBridge.isDarkModeActive(),"
            " classActive: document.documentElement.classList.contains('pdfexplore-dark-mode')"
            "})"
        )
        self.assertEqual(second_dark_state.get("active"), True)
        self.assertEqual(second_dark_state.get("classActive"), True)

        self.window.dark_mode_btn.click()
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({active: window.__pdfexploreBridge.isDarkModeActive()})"
            ).get("active")
            is False,
            timeout_ms=5000,
        )
        self.assertEqual(self.window.dark_mode_btn.text(), "Dark")

    def test_persistent_highlight_overlay_reappears_after_switch_and_reopen(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)
        selection_info = self._seed_alpha_selection_info()
        self.assertEqual(selection_info.get("selectedText"), "Alpha")
        self.window._add_persistent_preview_highlight(
            selection_info,
            kind="normal",
            selected_text_hint="Alpha",
        )
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({count: document.querySelectorAll('.pdfexplore-highlight-rect.normal').length})"
            ).get("count", 0)
            > 0,
            timeout_ms=12000,
        )
        initial_overlay = self.run_current_viewer_js_json(
            "({count: document.querySelectorAll('.pdfexplore-highlight-rect.normal').length, "
            "ids: Array.from(document.querySelectorAll('.pdfexplore-highlight-rect.normal')).map((n) => n.dataset.highlightId || '')})"
        )
        self.assertGreaterEqual(int(initial_overlay.get("count", 0)), 1)

        self._open_and_wait_for_viewer(self.second_pdf)
        self._open_and_wait_for_viewer(self.first_pdf)
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({count: document.querySelectorAll('.pdfexplore-highlight-rect.normal').length})"
            ).get("count", 0)
            > 0,
            timeout_ms=12000,
        )
        switched_overlay = self.run_current_viewer_js_json(
            "({count: document.querySelectorAll('.pdfexplore-highlight-rect.normal').length, "
            "ids: Array.from(document.querySelectorAll('.pdfexplore-highlight-rect.normal')).map((n) => n.dataset.highlightId || '')})"
        )
        self.assertEqual(
            list(switched_overlay.get("ids", [])),
            list(initial_overlay.get("ids", [])),
        )

        self.window.close()
        self._app.processEvents()
        reopened = PdfExploreWindow(
            root=self.root,
            app_icon=QIcon(),
            config_path=self.root / ".pdfexplore.cfg",
            gpu_context_available=False,
        )
        self.window = reopened
        self.window.show()
        self._app.processEvents()
        self._open_and_wait_for_viewer(self.first_pdf)
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({count: document.querySelectorAll('.pdfexplore-highlight-rect.normal').length})"
            ).get("count", 0)
            > 0,
            timeout_ms=12000,
        )
        reopened_overlay = self.run_current_viewer_js_json(
            "({count: document.querySelectorAll('.pdfexplore-highlight-rect.normal').length, "
            "ids: Array.from(document.querySelectorAll('.pdfexplore-highlight-rect.normal')).map((n) => n.dataset.highlightId || '')})"
        )
        self.assertEqual(
            list(reopened_overlay.get("ids", [])),
            list(initial_overlay.get("ids", [])),
        )

    def test_persistent_overlay_stays_stable_when_idle_scrolled_and_reapplied(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)
        payload = [
            {
                "id": "stable-overlay",
                "page": 1,
                "start": 0,
                "end": 5,
                "kind": "normal",
                "text": "Alpha",
            }
        ]
        self.window._current_text_highlights = payload
        self.window._apply_persistent_text_highlights()
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({count: document.querySelectorAll('.pdfexplore-highlight-rect.normal[data-highlight-id=\"stable-overlay\"]').length})"
            ).get("count", 0)
            > 0,
            timeout_ms=12000,
        )

        # Let the bridge's delayed rendering retries (and the window's restore
        # retries) settle before checking for an observer-driven repaint loop.
        self.wait_ms(1800)
        tagged = self.run_current_viewer_js_json(
            "(() => {"
            " window.__pdfexploreTestStableOverlay = document.querySelector("
            "   '.pdfexplore-highlight-rect.normal[data-highlight-id=\"stable-overlay\"]'"
            " );"
            " return {tagged: !!window.__pdfexploreTestStableOverlay};"
            "})()"
        )
        self.assertEqual(tagged.get("tagged"), True)

        self.wait_ms(300)
        idle_state = self.run_current_viewer_js_json(
            "(() => {"
            " const current = document.querySelector("
            "   '.pdfexplore-highlight-rect.normal[data-highlight-id=\"stable-overlay\"]'"
            " );"
            " return {"
            "   same: current === window.__pdfexploreTestStableOverlay,"
            "   connected: !!window.__pdfexploreTestStableOverlay?.isConnected"
            " };"
            "})()"
        )
        self.assertEqual(idle_state.get("same"), True)
        self.assertEqual(idle_state.get("connected"), True)

        self.run_current_viewer_js(
            "(() => {"
            f" window.__pdfexploreBridge.setPersistentHighlights({json.dumps(payload)});"
            " const container = document.getElementById('viewerContainer');"
            " if (container) container.dispatchEvent(new Event('scroll'));"
            " return true;"
            "})()"
        )
        self.wait_ms(350)
        reapplied_state = self.run_current_viewer_js_json(
            "(() => {"
            " const current = document.querySelector("
            "   '.pdfexplore-highlight-rect.normal[data-highlight-id=\"stable-overlay\"]'"
            " );"
            " return {"
            "   same: current === window.__pdfexploreTestStableOverlay,"
            "   connected: !!window.__pdfexploreTestStableOverlay?.isConnected"
            " };"
            "})()"
        )
        self.assertEqual(reapplied_state.get("same"), True)
        self.assertEqual(reapplied_state.get("connected"), True)

    def test_search_scrollbar_indicators_render_and_clear(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)
        self.run_current_viewer_js(
            "window.__pdfexploreBridge.setSearchTerms([{text: 'Alpha', caseSensitive: false}]);"
        )
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({count: document.querySelectorAll('.pdfexplore-search-indicator').length})"
            ).get("count", 0)
            > 0,
            timeout_ms=12000,
        )
        active_count = self.run_current_viewer_js_json(
            "({count: document.querySelectorAll('.pdfexplore-search-indicator').length})"
        ).get("count", 0)
        self.assertGreater(int(active_count), 0)

        self.run_current_viewer_js("window.__pdfexploreBridge.clearSearchTerms();")
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({count: document.querySelectorAll('.pdfexplore-search-indicator').length})"
            ).get("count", 1)
            == 0,
            timeout_ms=12000,
        )

    def test_search_scrollbar_indicators_cover_unrendered_pages(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)
        self.run_current_viewer_js(
            "window.__pdfexploreBridge.setSearchTerms([{text: 'line 170', caseSensitive: false}]);"
        )
        self.wait_until(
            lambda: any(
                int(page) > 1
                for page in self.run_current_viewer_js_json(
                    "({pages: Array.from(document.querySelectorAll('.pdfexplore-search-indicator')).map((n) => Number(n.dataset.pageNumber || 0))})"
                ).get("pages", [])
            ),
            timeout_ms=12000,
        )

    def test_identical_search_terms_and_scroll_preserve_marker_identity(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)
        self.run_current_viewer_js(
            "window.__pdfexploreBridge.setSearchTerms([{text: 'line 170', caseSensitive: false}]);"
        )
        self.wait_until(
            lambda: any(
                int(page) > 1
                for page in self.run_current_viewer_js_json(
                    "({pages: Array.from(document.querySelectorAll('.pdfexplore-search-indicator')).map((n) => Number(n.dataset.pageNumber || 0))})"
                ).get("pages", [])
            ),
            timeout_ms=12000,
        )
        self.wait_ms(500)
        tagged = self.run_current_viewer_js_json(
            "(() => {"
            " window.__pdfexploreTestSearchMarker = document.querySelector("
            "   '.pdfexplore-search-indicator[data-page-number]'"
            " );"
            " return {tagged: !!window.__pdfexploreTestSearchMarker};"
            "})()"
        )
        self.assertEqual(tagged.get("tagged"), True)

        self.run_current_viewer_js(
            "(() => {"
            " window.__pdfexploreBridge.setSearchTerms("
            "   [{text: 'line 170', caseSensitive: false}]"
            " );"
            " const container = document.getElementById('viewerContainer');"
            " if (container) container.dispatchEvent(new Event('scroll'));"
            " return true;"
            "})()"
        )
        self.wait_ms(400)
        marker_state = self.run_current_viewer_js_json(
            "(() => {"
            " const current = document.querySelector("
            "   '.pdfexplore-search-indicator[data-page-number]'"
            " );"
            " return {"
            "   same: current === window.__pdfexploreTestSearchMarker,"
            "   connected: !!window.__pdfexploreTestSearchMarker?.isConnected"
            " };"
            "})()"
        )
        self.assertEqual(marker_state.get("same"), True)
        self.assertEqual(marker_state.get("connected"), True)

    def test_search_scrollbar_indicator_click_jumps_to_target_page(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)
        self.run_current_viewer_js(
            "window.__pdfexploreBridge.setSearchTerms([{text: 'line 170', caseSensitive: false}]);"
        )
        self.wait_until(
            lambda: any(
                int(page) > 1
                for page in self.run_current_viewer_js_json(
                    "({pages: Array.from(document.querySelectorAll('.pdfexplore-search-indicator')).map((n) => Number(n.dataset.pageNumber || 0))})"
                ).get("pages", [])
            ),
            timeout_ms=12000,
        )

        clicked_payload = self.run_current_viewer_js_json(
            "(() => {"
            " const marker = Array.from(document.querySelectorAll('.pdfexplore-search-indicator'))"
            "   .find((n) => Number(n.dataset.pageNumber || 0) > 1);"
            " if (!marker) return {clicked: false, page: Number((window.PDFViewerApplication && window.PDFViewerApplication.page) || 0)};"
            " const rect = marker.getBoundingClientRect();"
            " marker.dispatchEvent(new MouseEvent('mousedown', {"
            "   bubbles: true,"
            "   cancelable: true,"
            "   button: 0,"
            "   clientX: rect.left + (rect.width / 2),"
            "   clientY: rect.top + (rect.height / 2),"
            " }));"
            " return {clicked: true, page: Number((window.PDFViewerApplication && window.PDFViewerApplication.page) || 0)};"
            "})()"
        )
        self.assertEqual(clicked_payload.get("clicked"), True)
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({page: Number((window.PDFViewerApplication && window.PDFViewerApplication.page) || 0)})"
            ).get("page", 0)
            > 1,
            timeout_ms=12000,
        )

    def test_highlight_and_search_markers_use_opposite_clickable_gutters(self) -> None:
        multi_page_pdf = self.root / "marker-gutters.pdf"
        _create_pdf_with_lines(multi_page_pdf, "Alpha", 210)
        self._open_and_wait_for_viewer(multi_page_pdf)

        self.window._current_text_highlights = [
            {
                "id": "normal-page-two",
                "page": 2,
                "start": 0,
                "end": 48,
                "kind": "normal",
                "text": "Alpha line 43 Alpha line 44",
            },
            {
                "id": "important-page-four",
                "page": 4,
                "start": 0,
                "end": 5,
                "kind": "important",
                "text": "Alpha",
            },
        ]
        self.window._apply_persistent_text_highlights()
        self.run_current_viewer_js(
            "window.__pdfexploreBridge.setSearchTerms([{text: 'line 170', caseSensitive: false}], []);"
        )

        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({"
                " highlights: document.querySelectorAll('.pdfexplore-highlight-indicator').length,"
                " searches: document.querySelectorAll('.pdfexplore-search-indicator').length"
                "})"
            ).get("highlights", 0)
            >= 2
            and self.run_current_viewer_js_json(
                "({count: document.querySelectorAll('.pdfexplore-search-indicator').length})"
            ).get("count", 0)
            > 0,
            timeout_ms=12000,
        )
        gutter_state = self.run_current_viewer_js_json(
            "(() => {"
            " const leftRail = document.querySelector('.pdfexplore-highlight-indicator-rail');"
            " const rightRail = document.querySelector('.pdfexplore-search-indicator-rail');"
            " const normal = document.querySelector('.pdfexplore-highlight-indicator.normal');"
            " const important = document.querySelector('.pdfexplore-highlight-indicator.important');"
            " return {"
            "   left: leftRail?.getBoundingClientRect().left ?? -1,"
            "   right: rightRail?.getBoundingClientRect().left ?? -1,"
            "   normalColor: normal ? getComputedStyle(normal).backgroundColor : '',"
            "   importantColor: important ? getComputedStyle(important).backgroundColor : '',"
            "   importantPage: Number(important?.dataset.pageNumber || 0),"
            "   importantId: String(important?.dataset.highlightId || '')"
            " };"
            "})()"
        )
        self.assertGreaterEqual(float(gutter_state.get("left", -1)), 0.0)
        self.assertGreater(float(gutter_state.get("right", -1)), float(gutter_state.get("left", -1)))
        self.assertNotEqual(gutter_state.get("normalColor"), gutter_state.get("importantColor"))
        self.assertEqual(gutter_state.get("importantPage"), 4)
        self.assertEqual(gutter_state.get("importantId"), "important-page-four")

        clicked = self.run_current_viewer_js_json(
            "(() => {"
            " const marker = document.querySelector('.pdfexplore-highlight-indicator.important');"
            " if (!marker) return {clicked: false};"
            " const rect = marker.getBoundingClientRect();"
            " marker.dispatchEvent(new MouseEvent('mousedown', {"
            "   bubbles: true, cancelable: true, button: 0,"
            "   clientX: rect.left + (rect.width / 2),"
            "   clientY: rect.top + (rect.height / 2)"
            " }));"
            " return {clicked: true};"
            "})()"
        )
        self.assertEqual(clicked.get("clicked"), True)
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({page: Number((window.PDFViewerApplication && window.PDFViewerApplication.page) || 0)})"
            ).get("page", 0)
            == 4,
            timeout_ms=12000,
        )
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({count: document.querySelectorAll('.pdfexplore-highlight-rect.important[data-highlight-id=\"important-page-four\"]').length})"
            ).get("count", 0)
            > 0,
            timeout_ms=12000,
        )

    def test_near_search_markers_only_cover_qualifying_pdf_pages(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)
        self.run_current_viewer_js(
            "window.__pdfexploreBridge.setSearchTerms("
            "[{text: 'Alpha', caseSensitive: false}, {text: 'line 170', caseSensitive: false}],"
            "[[{text: 'Alpha', caseSensitive: false}, {text: 'line 170', caseSensitive: false}]],"
            "true"
            ");"
        )
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({count: document.querySelectorAll('.pdfexplore-search-indicator').length})"
            ).get("count", 0)
            > 0,
            timeout_ms=12000,
        )
        marker_pages = self.run_current_viewer_js_json(
            "({pages: Array.from(new Set(Array.from(document.querySelectorAll('.pdfexplore-search-indicator')).map((node) => Number(node.dataset.pageNumber || 0))))})"
        ).get("pages", [])
        self.assertTrue(marker_pages)
        self.assertTrue(all(int(page) > 1 for page in marker_pages))
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({page: Number((window.PDFViewerApplication && window.PDFViewerApplication.page) || 0)})"
            ).get("page", 0)
            > 1,
            timeout_ms=12000,
        )

        self.run_current_viewer_js(
            "window.__pdfexploreBridge.setSearchTerms("
            "[{text: 'Alpha', caseSensitive: false}, {text: 'term-that-is-absent', caseSensitive: false}],"
            "[[{text: 'Alpha', caseSensitive: false}, {text: 'term-that-is-absent', caseSensitive: false}]]"
            ");"
        )
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "({count: document.querySelectorAll('.pdfexplore-search-indicator').length})"
            ).get("count", 1)
            == 0,
            timeout_ms=12000,
        )

    def test_highlight_overlay_tracks_selected_text_geometry(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)
        selection_rect = self.run_current_viewer_js_json(
            "(() => { "
            "const layer = document.querySelector('.page[data-page-number=\"1\"] .textLayer'); "
            "const walker = document.createTreeWalker(layer, NodeFilter.SHOW_TEXT); "
            "while (walker.nextNode()) { "
            "const node = walker.currentNode; "
            "const value = node.nodeValue || ''; "
            "const index = value.indexOf('Alpha'); "
            "if (index < 0) continue; "
            "const selection = window.getSelection(); "
            "const range = document.createRange(); "
            "range.setStart(node, index); "
            "range.setEnd(node, index + 5); "
            "selection.removeAllRanges(); "
            "selection.addRange(range); "
            "const rect = range.getBoundingClientRect(); "
            "return { top: rect.top, left: rect.left, width: rect.width, height: rect.height }; "
            "} "
            "return {}; "
            "})()"
        )
        selection_info = self.await_callback_result(
            lambda done: self.window._request_preview_context_menu_selection_info(
                0, 0, "Alpha", done
            )
        )
        self.window._add_persistent_preview_highlight(
            dict(selection_info),
            kind="normal",
            selected_text_hint="Alpha",
        )
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "(() => { "
                "const overlay = document.querySelector('.page[data-page-number=\"1\"] .pdfexplore-highlight-rect.normal'); "
                "if (!overlay) return {}; "
                "const rect = overlay.getBoundingClientRect(); "
                "return { top: rect.top, left: rect.left, width: rect.width, height: rect.height }; "
                "})()"
            ).get("width", 0)
            > 0,
            timeout_ms=12000,
        )
        overlay_rect = self.run_current_viewer_js_json(
            "(() => { "
            "const overlay = document.querySelector('.page[data-page-number=\"1\"] .pdfexplore-highlight-rect.normal'); "
            "const rect = overlay?.getBoundingClientRect(); "
            "return rect ? { top: rect.top, left: rect.left, width: rect.width, height: rect.height } : {}; "
            "})()"
        )
        # pdf.js may finish a page-fit scroll adjustment while the persistent
        # overlay is being created. Compare both rectangles in the same settled
        # viewport rather than treating the earlier screen coordinate as fixed.
        live_selection_rect = self.run_current_viewer_js_json(
            "(() => { "
            "const selection = window.getSelection(); "
            "if (!selection || selection.rangeCount < 1) return {}; "
            "const rect = selection.getRangeAt(0).getBoundingClientRect(); "
            "return { top: rect.top, left: rect.left, width: rect.width, height: rect.height }; "
            "})()"
        )
        self.assertGreater(float(selection_rect.get("width", 0)), 0)
        self.assertLessEqual(
            abs(
                float(overlay_rect.get("top", 0))
                - float(live_selection_rect.get("top", 0))
            ),
            12.0,
        )
        self.assertLessEqual(
            abs(
                float(overlay_rect.get("left", 0))
                - float(live_selection_rect.get("left", 0))
            ),
            12.0,
        )

    def test_persisted_later_page_highlight_becomes_visible_after_in_document_navigation(self) -> None:
        multi_page_pdf = self.root / "later-pages.pdf"
        _create_pdf_with_lines(multi_page_pdf, "Alpha", 210)
        self._open_and_wait_for_viewer(multi_page_pdf)
        self.wait_ms(1500)

        self.window._current_text_highlights = [
            {
                "id": "late-page-highlight",
                "page": 4,
                "start": 7,
                "end": 12,
                "kind": "normal",
                "text": "Alpha",
            }
        ]
        self.window._apply_persistent_text_highlights()

        self.run_current_viewer_js(
            "window.PDFViewerApplication.pdfViewer.scrollPageIntoView({ pageNumber: 4 });"
        )
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "(() => { "
                "const page = document.querySelector('.page[data-page-number=\"4\"]'); "
                "const layer = page?.querySelector('.textLayer'); "
                "const overlay = page?.querySelector('.pdfexplore-highlight-rect.normal'); "
                "const rect = overlay?.getBoundingClientRect(); "
                "return { "
                "hasLayer: !!layer, "
                "overlayCount: page ? page.querySelectorAll('.pdfexplore-highlight-rect.normal').length : 0, "
                "visibleCount: page ? Array.from(page.querySelectorAll('.pdfexplore-highlight-rect.normal')).filter((node) => { "
                "const r = node.getBoundingClientRect(); "
                "return r.bottom > 0 && r.top < window.innerHeight; "
                "}).length : 0, "
                "rect: rect ? { top: rect.top, bottom: rect.bottom } : null "
                "}; "
                "})()"
            ).get("visibleCount", 0)
            > 0,
            timeout_ms=12000,
        )

        payload = self.run_current_viewer_js_json(
            "(() => { "
            "const page = document.querySelector('.page[data-page-number=\"4\"]'); "
            "return { "
            "overlayCount: page ? page.querySelectorAll('.pdfexplore-highlight-rect.normal').length : 0, "
            "visibleCount: page ? Array.from(page.querySelectorAll('.pdfexplore-highlight-rect.normal')).filter((node) => { "
            "const r = node.getBoundingClientRect(); "
            "return r.bottom > 0 && r.top < window.innerHeight; "
            "}).length : 0 "
            "}; "
            "})()"
        )
        self.assertGreaterEqual(int(payload.get("overlayCount", 0)), 1)
        self.assertGreaterEqual(int(payload.get("visibleCount", 0)), 1)

    def test_preview_zoom_increases_and_decreases_pdfjs_scale(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)
        self.wait_ms(1000)
        baseline = self.run_current_viewer_js_json(
            "window.__pdfexploreBridge.getZoomState()"
        )
        self.window._zoom_preview_in()
        self.wait_until(
            lambda: float(
                self.run_current_viewer_js_json(
                    "window.__pdfexploreBridge.getZoomState()"
                ).get("currentScale", 0.0)
            )
            > float(baseline.get("currentScale", 0.0)),
            timeout_ms=6000,
        )
        zoomed_in = self.run_current_viewer_js_json(
            "window.__pdfexploreBridge.getZoomState()"
        )

        self.window._zoom_preview_out()
        self.wait_until(
            lambda: float(
                self.run_current_viewer_js_json(
                    "window.__pdfexploreBridge.getZoomState()"
                ).get("currentScale", 0.0)
            )
            < float(zoomed_in.get("currentScale", 0.0)),
            timeout_ms=6000,
        )
        zoomed_out = self.run_current_viewer_js_json(
            "window.__pdfexploreBridge.getZoomState()"
        )

        self.window._reset_preview_zoom()
        self.wait_until(
            lambda: str(
                self.run_current_viewer_js_json(
                    "window.__pdfexploreBridge.getZoomState()"
                ).get("currentScaleValue", "")
            )
            == "page-width",
            timeout_ms=6000,
        )
        reset_state = self.run_current_viewer_js_json(
            "window.__pdfexploreBridge.getZoomState()"
        )

        self.assertGreater(
            float(zoomed_in.get("currentScale", 0.0)),
            float(baseline.get("currentScale", 0.0)),
        )
        self.assertLess(
            float(zoomed_out.get("currentScale", 0.0)),
            float(zoomed_in.get("currentScale", 0.0)),
        )
        self.assertEqual(str(reset_state.get("currentScaleValue", "")), "page-width")

    def test_three_up_does_not_persist_scroll_and_restores_one_page_zoom(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)

        desired_state = {
            "page": 3,
            "pagesCount": 5,
            "scale": "1.3",
            "scrollTop": 1500,
            "scrollRatio": 0.5,
        }
        self.run_current_viewer_js(
            f"window.__pdfexploreBridge.restoreViewState({json.dumps(desired_state)});"
        )
        self.wait_until(
            lambda: self.run_current_viewer_js_json(
                "window.__pdfexploreBridge.getViewState()"
            ).get("scrollTop", 0)
            >= 1400,
            timeout_ms=12000,
        )
        settled_before = self.run_current_viewer_js_json(
            "window.__pdfexploreBridge.getViewState()"
        )

        toggled = self.run_current_viewer_js_json(
            "window.__pdfexploreBridge.toggleThreeUpMode()"
        )
        self.assertTrue(bool(toggled.get("threeUpActive")))

        self.run_current_viewer_js("window.__pdfexploreBridge.setZoomScale(1.6);")
        self.wait_until(
            lambda: float(
                self.run_current_viewer_js_json(
                    "window.__pdfexploreBridge.getZoomState()"
                ).get("currentScale", 0.0)
            )
            >= 1.58,
            timeout_ms=8000,
        )

        self.run_current_viewer_js(
            "(() => { const c = document.getElementById('viewerContainer'); if (c) { c.scrollLeft = 1400; c.scrollTop = 400; } return true; })();"
        )

        self._open_and_wait_for_viewer(self.second_pdf)
        self._open_and_wait_for_viewer(self.first_pdf)

        self.wait_until(
            lambda: self.run_current_viewer_js(
                "window.__pdfexploreBridge.isThreeUpActive();"
            )
            is False,
            timeout_ms=12000,
        )

        restored_zoom = self.run_current_viewer_js_json(
            "window.__pdfexploreBridge.getZoomState()"
        )
        restored_view = self.run_current_viewer_js_json(
            "window.__pdfexploreBridge.getViewState()"
        )

        self.assertFalse(bool(restored_zoom.get("threeUpActive")))
        self.assertLessEqual(
            abs(float(restored_zoom.get("currentScale", 0.0)) - 1.6),
            0.12,
        )
        self.assertLessEqual(
            abs(float(restored_view.get("scrollTop", 0)) - float(settled_before.get("scrollTop", 0))),
            260.0,
        )

    def test_three_up_toggle_responds_to_ctrl_backslash_keys_in_viewer(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)

        initial_state = self.run_current_viewer_js(
            "window.__pdfexploreBridge.isThreeUpActive();"
        )
        self.assertEqual(initial_state, False)

        after_ctrl_backslash = self.run_current_viewer_js_json(
            """
(() => {
    const event = new KeyboardEvent('keydown', {
        key: String.fromCharCode(92),
        code: 'Backslash',
        ctrlKey: true,
        bubbles: true,
        cancelable: true,
    });
    document.dispatchEvent(event);
    return {
        threeUpActive: window.__pdfexploreBridge.isThreeUpActive(),
        scrollMode: window.PDFViewerApplication.pdfViewer.scrollMode,
    };
})()
"""
        )
        self.assertEqual(bool(after_ctrl_backslash.get("threeUpActive")), True)
        self.assertEqual(int(after_ctrl_backslash.get("scrollMode", -1)), 2)

        after_ctrl_shift_bar = self.run_current_viewer_js(
            """
(() => {
  const event = new KeyboardEvent('keydown', {
    key: '|',
    code: 'Backslash',
    ctrlKey: true,
    shiftKey: true,
    bubbles: true,
    cancelable: true,
  });
  document.dispatchEvent(event);
  return window.__pdfexploreBridge.isThreeUpActive();
})();
"""
        )
        self.assertEqual(after_ctrl_shift_bar, False)

    def test_three_up_exit_page_selection_depends_on_exit_action(self) -> None:
        self._open_and_wait_for_viewer(self.first_pdf)

        entry_state = {
            "page": 1,
            "pagesCount": 6,
            "scale": "page-width",
            "scrollTop": 0,
            "scrollRatio": 0,
        }
        self.run_current_viewer_js(
            f"window.__pdfexploreBridge.restoreViewState({json.dumps(entry_state)});"
        )
        self.wait_until(
            lambda: int(
                self.run_current_viewer_js_json(
                    "window.__pdfexploreBridge.getViewState()"
                ).get("page", 0)
            )
            == 1,
            timeout_ms=12000,
        )

        self.run_current_viewer_js("window.__pdfexploreBridge.toggleThreeUpMode();")
        self.wait_until(
            lambda: bool(
                self.run_current_viewer_js("window.__pdfexploreBridge.isThreeUpActive();")
            ),
            timeout_ms=12000,
        )
        self.run_current_viewer_js(
            "(() => { const c = document.getElementById('viewerContainer'); if (c) { c.scrollTop = 2200; c.scrollLeft = 320; } return true; })();"
        )

        center_page = int(
            self.run_current_viewer_js_json(
                """
(() => {
  const c = document.getElementById('viewerContainer');
  const pages = Array.from(document.querySelectorAll('#viewer .page[data-page-number]'));
  if (!c || !pages.length) {
    return { centerPage: Number(window.PDFViewerApplication.page || 1) };
  }
  const cr = c.getBoundingClientRect();
  const cx = cr.left + (c.clientWidth / 2);
  const cy = cr.top + (c.clientHeight / 2);
  let bestPage = Number(window.PDFViewerApplication.page || 1);
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const page of pages) {
    const pageNumber = Number(page.dataset.pageNumber || 0);
    if (!pageNumber) {
      continue;
    }
    const rect = page.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) {
      continue;
    }
    const dx = (rect.left + (rect.width / 2)) - cx;
    const dy = (rect.top + (rect.height / 2)) - cy;
    const distance = (dx * dx) + (dy * dy);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestPage = pageNumber;
    }
  }
  return { centerPage: bestPage };
})()
"""
            ).get("centerPage", 1)
        )
        self.assertGreaterEqual(center_page, 1)

        self.run_current_viewer_js("window.__pdfexploreBridge.toggleThreeUpMode();")
        self.wait_until(
            lambda: self.run_current_viewer_js(
                "window.__pdfexploreBridge.isThreeUpActive();"
            )
            is False,
            timeout_ms=12000,
        )
        self.wait_until(
            lambda: int(
                self.run_current_viewer_js_json(
                    "window.__pdfexploreBridge.getViewState()"
                ).get("page", 0)
            )
            == center_page,
            timeout_ms=12000,
        )
        toggle_exit_page = int(
            self.run_current_viewer_js_json(
                "window.__pdfexploreBridge.getViewState()"
            ).get("page", 0)
        )
        self.assertEqual(toggle_exit_page, center_page)

        self.run_current_viewer_js(
            f"window.__pdfexploreBridge.restoreViewState({json.dumps(entry_state)});"
        )
        self.wait_until(
            lambda: int(
                self.run_current_viewer_js_json(
                    "window.__pdfexploreBridge.getViewState()"
                ).get("page", 0)
            )
            == 1,
            timeout_ms=12000,
        )

        self.run_current_viewer_js("window.__pdfexploreBridge.toggleThreeUpMode();")
        self.wait_until(
            lambda: bool(
                self.run_current_viewer_js("window.__pdfexploreBridge.isThreeUpActive();")
            ),
            timeout_ms=12000,
        )
        self.run_current_viewer_js(
            "(() => { const c = document.getElementById('viewerContainer'); if (c) { c.scrollTop = 2200; c.scrollLeft = 320; } return true; })();"
        )
        self.run_current_viewer_js("window.__pdfexploreBridge.setOnePageZoom100();")
        self.wait_until(
            lambda: self.run_current_viewer_js(
                "window.__pdfexploreBridge.isThreeUpActive();"
            )
            is False,
            timeout_ms=12000,
        )
        zoom_exit_page = int(
            self.run_current_viewer_js_json(
                "window.__pdfexploreBridge.getViewState()"
            ).get("page", 0)
        )
        self.assertEqual(zoom_exit_page, 1)


if __name__ == "__main__":
    unittest.main()
