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
        self.assertLessEqual(
            abs(float(overlay_rect.get("top", 0)) - float(selection_rect.get("top", 0))),
            12.0,
        )
        self.assertLessEqual(
            abs(float(overlay_rect.get("left", 0)) - float(selection_rect.get("left", 0))),
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


if __name__ == "__main__":
    unittest.main()
