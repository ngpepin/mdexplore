import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-gpu --disable-software-rasterizer",
)

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import mdexplore


FIXTURE_FILE_NAME = "testdoc.md"
FIXTURE_SIDECARS = [
    ".mdexplore-colors.json",
    ".mdexplore-highlighting.json",
    ".mdexplore-views.json",
]
SCROLL_TOLERANCE_PX = 180.0
LINE_TOLERANCE = 3
SETTLE_WAIT_MS = 2200
SEARCH_MARK_TEXTS_JS = """
(() => JSON.stringify(
  Array.from(document.querySelectorAll('[data-mdexplore-search-mark="1"]')).map(
    (node) => node.textContent || ""
  )
))();
"""
PERSISTENT_HIGHLIGHT_TEXTS_JS = """
(() => JSON.stringify(
  Array.from(
    document.querySelectorAll('span[data-mdexplore-persistent-highlight="1"]')
  ).map((node) => ({
    text: node.textContent || "",
    id: node.getAttribute("data-mdexplore-persistent-highlight-id") || "",
    kind: node.getAttribute("data-mdexplore-persistent-highlight-kind") || "",
  }))
))();
"""
SEARCH_VARIANTS_DOCUMENT_TEXT = """# Search Fixture

Alpha beta GAMMA.
Exact Case and other phrase appear here.
Program Director's RAG pipeline.
program director's rag pipeline.
The quick brown fox.
They said hello.
A later the appears here.
Joe met Anne Smith yesterday.
Another joe met anne smith today.
draft copy only.
"""
MULTIWORD_NEAR_DOCUMENT_TEXT = """# Multiword NEAR Fixture

Exact Case and other phrase appear here.

Program Director's RAG pipeline.

Joe met Anne Smith yesterday.
"""
REPEATED_NEAR_DOCUMENT_TEXT = """# Repeated NEAR Fixture

Nicolas Pepin

Other text between rows.

Nicolas Pepin
"""
SEARCH_MARKER_POSITION_TEXT = "\n\n".join(
    [f"Line {index}" for index in range(1, 40)]
    + ["UNIQUEHIT early marker"]
    + [f"Line {index}" for index in range(41, 220)]
    + ["UNIQUEHIT late marker"]
    + [f"Line {index}" for index in range(221, 320)]
) + "\n"

PROBE_JS = """
(() => {
  const visibleCount = (selector) =>
    Array.from(document.querySelectorAll(selector)).filter((node) => {
      const rect = node.getBoundingClientRect();
      return !!rect && rect.height > 0 && rect.bottom > 0 && rect.top < window.innerHeight;
    }).length;

  const topBandY = 12;
  const viewportHeight = Math.max(1, Number(window.innerHeight) || 0);
  const taggedNodes = Array.from(document.querySelectorAll("[data-md-line-start]"));
  let crossingLine = null;
  let crossingTop = -Infinity;
  let aboveLine = null;
  let aboveBottom = -Infinity;
  let belowLine = null;
  let belowTop = Infinity;
  for (const node of taggedNodes) {
    const rawValue = parseInt(node.getAttribute("data-md-line-start") || "", 10);
    if (Number.isNaN(rawValue)) continue;
    const lineValue = rawValue + 1;
    const rect = node.getBoundingClientRect();
    if (!rect) continue;
    if (!Number.isFinite(rect.top) || !Number.isFinite(rect.bottom)) continue;
    if (rect.height <= 0) continue;
    if (rect.bottom <= 0 || rect.top >= viewportHeight) continue;
    if (rect.top <= topBandY && rect.bottom > topBandY) {
      if (
        rect.top > crossingTop ||
        (rect.top === crossingTop &&
          (crossingLine === null || lineValue < crossingLine))
      ) {
        crossingTop = rect.top;
        crossingLine = lineValue;
      }
      continue;
    }
    if (rect.bottom <= topBandY) {
      if (
        rect.bottom > aboveBottom ||
        (rect.bottom === aboveBottom &&
          (aboveLine === null || lineValue > aboveLine))
      ) {
        aboveBottom = rect.bottom;
        aboveLine = lineValue;
      }
      continue;
    }
    if (
      rect.top < belowTop ||
      (rect.top === belowTop && (belowLine === null || lineValue < belowLine))
    ) {
      belowTop = rect.top;
      belowLine = lineValue;
    }
  }
  const topLine = crossingLine ?? aboveLine ?? belowLine ?? 1;
  return JSON.stringify({
    scrollY: window.scrollY,
    topLine,
    viewportHeight: window.innerHeight,
    viewMarkers: document.querySelectorAll(".mdexplore-scroll-view-marker").length,
    highlightMarkers: document.querySelectorAll(".mdexplore-scroll-highlight-marker").length,
    searchMarkers: document.querySelectorAll(".mdexplore-scroll-hit-marker").length,
    visibleHighlights: visibleCount('span[data-mdexplore-persistent-highlight="1"]'),
    visibleSearchMarks: visibleCount('[data-mdexplore-search-mark="1"]'),
  });
})();
"""


def marker_geometry_js(selector: str) -> str:
    selector_json = json.dumps(selector)
    return f"""
(() => {{
  const nodes = Array.from(document.querySelectorAll({selector_json}));
  return JSON.stringify(
    nodes.map((node, index) => {{
      const rect = node.getBoundingClientRect();
      return {{
        index,
        top: rect ? rect.top : 0,
        height: rect ? rect.height : 0,
      }};
    }})
  );
}})();
"""


def click_marker_js(selector: str, index: int) -> str:
    selector_json = json.dumps(selector)
    return f"""
(() => {{
  const nodes = Array.from(document.querySelectorAll({selector_json}));
  const node = nodes[{int(index)}];
  if (!node) {{
    return "missing";
  }}
  const rect = node.getBoundingClientRect();
  const eventOptions = {{
    bubbles: true,
    cancelable: true,
    button: 0,
    clientX: rect.left + Math.max(1, rect.width / 2),
    clientY: rect.top + Math.max(1, rect.height / 2),
  }};
  if (typeof PointerEvent === "function") {{
    node.dispatchEvent(new PointerEvent("pointerdown", eventOptions));
  }}
  node.dispatchEvent(new MouseEvent("mousedown", eventOptions));
  return "ok";
}})();
"""


class PreviewRegressionHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        source_root = repo_root / "test"
        self._tempdir = tempfile.TemporaryDirectory(
            prefix="mdexplore-preview-regression-"
        )
        self.root = Path(self._tempdir.name)
        shutil.copy2(source_root / FIXTURE_FILE_NAME, self.root / FIXTURE_FILE_NAME)
        for sidecar_name in FIXTURE_SIDECARS:
            sidecar = source_root / sidecar_name
            if sidecar.exists():
                shutil.copy2(sidecar, self.root / sidecar_name)
        self.fixture_path = self.root / FIXTURE_FILE_NAME
        self.window = mdexplore.MdExploreWindow(
            self.root,
            mdexplore._build_markdown_icon(),
            self.root / ".mdexplore-test.cfg",
            mermaid_backend=mdexplore.MERMAID_BACKEND_JS,
            gpu_context_available=False,
            debug_mode=False,
        )
        self.window.show()
        self.window._load_preview(self.fixture_path)
        self.wait_until(
            lambda: not self.window._preview_load_in_progress, timeout_ms=20000
        )
        self.wait_ms(SETTLE_WAIT_MS)

    def tearDown(self) -> None:
        self.window.close()
        self.wait_ms(120)
        self._tempdir.cleanup()

    def wait_ms(self, milliseconds: int) -> None:
        QTest.qWait(int(milliseconds))

    def wait_until(self, predicate, *, timeout_ms: int, step_ms: int = 50) -> None:
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        while time.monotonic() < deadline:
            self.app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, step_ms)
            if predicate():
                return
            QTest.qWait(step_ms)
        self.fail(f"Timed out after {timeout_ms} ms waiting for condition")

    def run_js(self, script: str):
        loop = QEventLoop()
        holder: dict[str, object] = {}

        def _done(result) -> None:
            holder["result"] = result
            loop.quit()

        self.window.preview.page().runJavaScript(script, _done)
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        if "result" not in holder:
            self.fail("Timed out waiting for preview JavaScript result")
        return holder["result"]

    def probe(self) -> dict:
        raw = self.run_js(PROBE_JS)
        if isinstance(raw, dict):
            return raw
        self.assertIsInstance(raw, str)
        payload = json.loads(raw)
        self.assertIsInstance(payload, dict)
        return payload

    def marker_geometry(self, selector: str) -> list[dict]:
        raw = self.run_js(marker_geometry_js(selector))
        self.assertIsInstance(raw, str)
        payload = json.loads(raw)
        self.assertIsInstance(payload, list)
        return payload

    def click_marker(self, selector: str, index: int) -> None:
        result = self.run_js(click_marker_js(selector, index))
        self.assertEqual(result, "ok")

    def load_markdown_text(self, file_name: str, text: str) -> Path:
        path = self.root / file_name
        path.write_text(text, encoding="utf-8")
        self.window._load_preview(path)
        self.wait_until(
            lambda: not self.window._preview_load_in_progress, timeout_ms=20000
        )
        self.wait_ms(SETTLE_WAIT_MS)
        return path

    def search_mark_texts(self) -> list[str]:
        raw = self.run_js(SEARCH_MARK_TEXTS_JS)
        self.assertIsInstance(raw, str)
        payload = json.loads(raw)
        self.assertIsInstance(payload, list)
        return [str(item) for item in payload]

    def persistent_highlight_spans(self) -> list[dict]:
        raw = self.run_js(PERSISTENT_HIGHLIGHT_TEXTS_JS)
        self.assertIsInstance(raw, str)
        payload = json.loads(raw)
        self.assertIsInstance(payload, list)
        return [dict(item) for item in payload]

    def persistent_highlight_text_by_id(self) -> dict[str, str]:
        grouped: dict[str, str] = {}
        for item in self.persistent_highlight_spans():
            entry_id = str(item.get("id", "")).strip()
            if not entry_id:
                continue
            grouped[entry_id] = grouped.get(entry_id, "") + str(item.get("text", ""))
        return grouped

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

    def apply_persistent_highlights(self, entries: list[dict]) -> dict:
        self.window._current_preview_text_highlights = list(entries)
        expected_key = self.window._current_preview_path_key()
        self.assertIsNotNone(expected_key)
        result = self.await_callback_result(
            lambda done: self.window._apply_persistent_preview_highlights(
                expected_key, completion=done
            )
        )
        self.assertIsInstance(result, dict)
        return dict(result)

    def request_live_highlight_target(self) -> dict:
        result = self.await_callback_result(
            lambda done: self.window._request_live_preview_highlight_target(done)
        )
        self.assertIsInstance(result, dict)
        return dict(result)

    def request_live_selection_offsets(self, hint: str = "") -> dict:
        result = self.await_callback_result(
            lambda done: self.window._request_live_preview_selection_offsets(hint, done)
        )
        self.assertIsInstance(result, dict)
        return dict(result)

    def request_preview_context_menu_selection_info(
        self, click_x: int, click_y: int, hint: str = ""
    ) -> dict:
        result = self.await_callback_result(
            lambda done: self.window._request_preview_context_menu_selection_info(
                int(click_x), int(click_y), hint, done
            )
        )
        self.assertIsInstance(result, dict)
        return dict(result)

    def run_search_and_expect_highlights(
        self, query: str, expected_texts: list[str]
    ) -> None:
        current_path = self.window.current_file
        self.assertIsNotNone(current_path)

        self.window.match_input.setText(query)
        self.window._run_match_search_now()

        def matches_expected() -> bool:
            return self.search_mark_texts() == expected_texts

        self.wait_until(matches_expected, timeout_ms=6000)
        self.assertEqual(self.search_mark_texts(), expected_texts)

        current_match_paths = {path.resolve() for path in self.window.current_match_files}
        if expected_texts:
            self.assertIn(current_path.resolve(), current_match_paths)
        else:
            self.assertNotIn(current_path.resolve(), current_match_paths)

    def current_file_hit_count(self) -> int:
        current_path = self.window.current_file
        self.assertIsNotNone(current_path)
        path_key = self.window.model._path_key(current_path)
        return int(self.window.model._search_match_counts.get(path_key, 0))

    def search_marker_projection(self) -> list[dict]:
        raw = self.run_js(
            """
(() => {
  const marks = Array.from(document.querySelectorAll('[data-mdexplore-search-mark="1"]'));
  const markers = Array.from(document.querySelectorAll('.mdexplore-scroll-hit-marker'));
  const scrollHeight = Math.max(
    1,
    document.documentElement ? document.documentElement.scrollHeight : 0,
    document.body ? document.body.scrollHeight : 0
  );
  const scrollableHeight = Math.max(1, scrollHeight - window.innerHeight);
  const trackHeight = window.innerHeight;
  return JSON.stringify({
    trackHeight,
    scrollHeight,
    scrollableHeight,
    markTops: marks.map((mark) => {
      const rect = mark.getBoundingClientRect();
      return window.scrollY + rect.top;
    }),
    markerTops: markers.map((marker) => {
      const rect = marker.getBoundingClientRect();
      return rect.top;
    }),
  });
})();
"""
        )
        self.assertIsInstance(raw, str)
        payload = json.loads(raw)
        self.assertIsInstance(payload, dict)
        mark_tops = sorted(float(value) for value in payload.get("markTops", []))
        marker_tops = sorted(float(value) for value in payload.get("markerTops", []))
        track_height = float(payload.get("trackHeight", 0))
        scrollable_height = float(payload.get("scrollableHeight", 1))
        expected = [
            max(0.0, min(track_height - 4.0, (top / scrollable_height) * track_height))
            for top in mark_tops
        ]
        return [
            {
                "expected": expected[index],
                "actual": marker_tops[index],
            }
            for index in range(min(len(expected), len(marker_tops)))
        ]

    def farthest_marker_index(self, selector: str) -> int:
        probe = self.probe()
        center_y = float(probe["viewportHeight"]) * 0.5
        geometry = self.marker_geometry(selector)
        self.assertTrue(geometry, f"No marker geometry found for {selector}")
        return max(
            range(len(geometry)),
            key=lambda idx: abs(
                float(geometry[idx].get("top", 0.0))
                + (float(geometry[idx].get("height", 0.0)) * 0.5)
                - center_y
            ),
        )

    def assert_view_matches_state(self, view_id: int, expected_state: dict) -> None:
        probe = self.probe()
        current_state = self.window._view_states.get(view_id)
        self.assertIsNotNone(current_state)
        self.assertEqual(self.window._active_view_id, view_id)
        self.assertLessEqual(
            abs(float(probe["scrollY"]) - float(expected_state["scroll_y"])),
            SCROLL_TOLERANCE_PX,
        )
        self.assertLessEqual(
            abs(int(probe["topLine"]) - int(expected_state["top_line"])),
            LINE_TOLERANCE,
        )
        self.assertLessEqual(
            abs(float(current_state["scroll_y"]) - float(expected_state["scroll_y"])),
            SCROLL_TOLERANCE_PX,
        )
        self.assertLessEqual(
            abs(int(current_state["top_line"]) - int(expected_state["top_line"])),
            LINE_TOLERANCE,
        )


class SearchHighlightRegressionTests(PreviewRegressionHarness):
    def test_search_highlighting_preserves_quoted_trailing_space(self) -> None:
        precise_path = self.root / "quoted-trailing-space.md"
        precise_path.write_text("The quick brown fox.\nThey said hello.\n", encoding="utf-8")
        self.window._load_preview(precise_path)
        self.wait_until(
            lambda: not self.window._preview_load_in_progress, timeout_ms=20000
        )
        self.wait_ms(SETTLE_WAIT_MS)

        self.window.match_input.setText("'The '")
        self.window._run_match_search_now()
        self.wait_until(
            lambda: int(self.probe()["visibleSearchMarks"]) > 0, timeout_ms=6000
        )

        raw = self.run_js(
            """
(() => JSON.stringify(
  Array.from(document.querySelectorAll('[data-mdexplore-search-mark="1"]')).map(
    (node) => node.textContent || ""
  )
))();
"""
        )
        self.assertIsInstance(raw, str)
        texts = json.loads(raw)
        self.assertEqual(texts, ["The "])

    def test_near_highlighting_requires_distinct_occurrences_for_overlapping_terms(self) -> None:
        precise_path = self.root / "close-overlap.md"
        precise_path.write_text(
            "The quick brown fox.\nThey said hello.\nA later the appears here.\n",
            encoding="utf-8",
        )
        self.window._load_preview(precise_path)
        self.wait_until(
            lambda: not self.window._preview_load_in_progress, timeout_ms=20000
        )
        self.wait_ms(SETTLE_WAIT_MS)

        self.window.match_input.setText("""NEAR('The ', the)""")
        self.window._run_match_search_now()
        self.wait_until(
            lambda: int(self.probe()["visibleSearchMarks"]) >= 2, timeout_ms=6000
        )

        raw = self.run_js(
            """
(() => JSON.stringify(
  Array.from(document.querySelectorAll('[data-mdexplore-search-mark="1"]')).map(
    (node) => node.textContent || ""
  )
))();
"""
        )
        self.assertIsInstance(raw, str)
        texts = json.loads(raw)
        self.assertEqual(texts, ["The ", "the"])
        self.assertEqual(self.current_file_hit_count(), 1)

    def test_near_highlighting_marks_all_multiword_terms_in_focus_window(self) -> None:
        self.load_markdown_text("near-multiword.md", MULTIWORD_NEAR_DOCUMENT_TEXT)

        cases = [
            ('NEAR("other phrase", "Exact Case")', ["Exact Case", "other phrase"]),
            (
                """NEAR("Program Director's RAG", pipeline)""",
                ["Program Director's RAG", "pipeline"],
            ),
            (
                """NEAR(pipeline, "Program Director's RAG")""",
                ["Program Director's RAG", "pipeline"],
            ),
            ('NEAR("Joe", "Anne Smith")', ["Joe", "Anne Smith"]),
        ]

        for query, expected_texts in cases:
            with self.subTest(query=query):
                self.run_search_and_expect_highlights(query, expected_texts)
                self.assertEqual(self.current_file_hit_count(), 1)

    def test_near_highlighting_marks_all_repeated_windows(self) -> None:
        self.load_markdown_text("near-repeated.md", REPEATED_NEAR_DOCUMENT_TEXT)
        self.run_search_and_expect_highlights(
            """NEAR('Nicolas', 'Pepin')""",
            ["Nicolas", "Pepin", "Nicolas", "Pepin"],
        )
        self.assertEqual(self.current_file_hit_count(), 2)


class SameDocumentSearchSyntaxRegressionTests(PreviewRegressionHarness):
    def test_same_document_search_variants_highlight_expected_fragments(self) -> None:
        self.load_markdown_text("search-variants.md", SEARCH_VARIANTS_DOCUMENT_TEXT)

        cases = [
            ("alpha", ["Alpha"]),
            ('"Program Director\'s RAG"', ["Program Director's RAG", "program director's rag"]),
            (r"""'Program Director\'s RAG'""", ["Program Director's RAG"]),
            ("the", ["the", "The", "The", "the", "the"]),
            ("'The '", ["The "]),
            ('"The "', ["The ", "the "]),
            ("alpha gamma", ["Alpha", "GAMMA"]),
            ("AND(alpha, gamma)", ["Alpha", "GAMMA"]),
            ("OR ('Exact Case' gamma)", ["GAMMA", "Exact Case"]),
            ('AND (Joe "Anne Smith")', ["Joe", "Anne Smith", "joe", "anne smith"]),
            ("Joe NOT zebra", ["Joe", "joe"]),
            ("Joe NOT draft", []),
            ("NEAR('The ', the)", ["The ", "the"]),
            ("NEAR ('The ', the)", ["The ", "the"]),
            ("OR(alpha, 'Exact Case' gamma)", ["Alpha", "GAMMA", "Exact Case"]),
        ]

        for query, expected_texts in cases:
            with self.subTest(query=query):
                self.run_search_and_expect_highlights(query, expected_texts)
                expected_hit_count = 1 if query.startswith("NEAR") else len(expected_texts)
                self.assertEqual(self.current_file_hit_count(), expected_hit_count)


class PreviewMarkerRegressionTests(PreviewRegressionHarness):
    def test_persistent_highlight_offsets_ignore_embedded_formatting_newlines(self) -> None:
        self.load_markdown_text(
            "persistent-highlight-hard-break.md",
            "# Hard Break Fixture\n\nalpha  \nbeta gamma\n",
        )

        for needle in ("beta", "gamma"):
            with self.subTest(needle=needle):
                self.apply_persistent_highlights([])
                needle_len = len(needle)
                selection_seed = self.run_js(
                    f"""
(() => {{
  const root = document.querySelector("main") || document.body;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {{
    const node = walker.currentNode;
    const value = node.nodeValue || "";
    const index = value.indexOf({json.dumps(needle)});
    if (index < 0) continue;
    const selection = window.getSelection();
    const range = document.createRange();
    range.setStart(node, index);
    range.setEnd(node, index + {needle_len});
    selection.removeAllRanges();
    selection.addRange(range);
    return {json.dumps(needle)};
  }}
  return "missing";
}})();
"""
                )
                self.assertEqual(selection_seed, needle)

                selected_offsets = self.request_live_selection_offsets(needle)
                self.assertTrue(bool(selected_offsets.get("hasSelection")))
                self.assertEqual(selected_offsets.get("selectedText"), needle)
                start = int(selected_offsets["selectionOffsetStart"])
                end = int(selected_offsets["selectionOffsetEnd"])
                self.assertEqual(
                    end - start,
                    len(needle),
                    "Embedded formatting newlines should not inflate logical highlight width",
                )

                apply_result = self.apply_persistent_highlights(
                    [{"id": f"case-{needle}", "start": start, "end": end, "kind": "normal"}]
                )
                self.assertGreaterEqual(int(apply_result.get("applied", 0)), 1)
                self.assertEqual(
                    self.persistent_highlight_spans(),
                    [{"text": needle, "id": f"case-{needle}", "kind": "normal"}],
                )

    def test_persistent_highlights_revisit_document_without_drift(self) -> None:
        target_path = self.load_markdown_text(
            "persistent-highlight-revisit-hard-break.md",
            "# Revisit Fixture\n\nalpha  \nbeta gamma\n",
        )
        other_path = self.load_markdown_text(
            "persistent-highlight-revisit-other.md",
            "# Other Fixture\n\nunrelated text\n",
        )
        self.window._load_preview(target_path)
        self.wait_until(
            lambda: not self.window._preview_load_in_progress, timeout_ms=20000
        )
        self.wait_ms(SETTLE_WAIT_MS)

        selection_seed = self.run_js(
            """
(() => {
  const root = document.querySelector("main") || document.body;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const value = node.nodeValue || "";
    const index = value.indexOf("gamma");
    if (index < 0) continue;
    const selection = window.getSelection();
    const range = document.createRange();
    range.setStart(node, index);
    range.setEnd(node, index + 5);
    selection.removeAllRanges();
    selection.addRange(range);
    return "gamma";
  }
  return "missing-gamma";
})();
"""
        )
        self.assertEqual(selection_seed, "gamma")
        selected_offsets = self.request_live_selection_offsets("gamma")
        start = int(selected_offsets["selectionOffsetStart"])
        end = int(selected_offsets["selectionOffsetEnd"])
        self.assertEqual(end - start, 5)

        apply_result = self.apply_persistent_highlights(
            [{"id": "revisit-case", "start": start, "end": end, "kind": "normal"}]
        )
        self.assertGreaterEqual(int(apply_result.get("applied", 0)), 1)
        path_key = self.window._current_preview_path_key()
        self.assertIsNotNone(path_key)
        self.window._persist_text_highlights_for_path_key(
            path_key,
            self.window._current_preview_text_highlights,
        )
        self.assertEqual(
            self.persistent_highlight_spans(),
            [{"text": "gamma", "id": "revisit-case", "kind": "normal"}],
        )

    def test_pipelines_highlight_important_survives_document_switch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        source_root = repo_root / "test"
        target_path = self.root / "Pipelines.md"
        other_path = self.root / "paper.md"
        target_path.write_text(
            (source_root / "Pipelines.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        other_path.write_text(
            (source_root / "paper.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        highlight_text = "10) A few packaging ideas for Upwork projects"
        drift_text = "ackage these as fixed-scope offers"

        self.window._persisted_text_highlights_by_dir.clear()
        self.window._load_preview(target_path)
        self.wait_until(
            lambda: not self.window._preview_load_in_progress, timeout_ms=20000
        )
        self.wait_ms(SETTLE_WAIT_MS)

        current_key = self.window._current_preview_path_key()
        self.assertIsNotNone(current_key)
        self.window._current_preview_text_highlights = []
        self.window._persist_text_highlights_for_path_key(current_key, [])
        self.apply_persistent_highlights([])

        selection_seed = self.run_js(
            f"""
(() => {{
  const root = document.querySelector("main") || document.body;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {{
    const node = walker.currentNode;
    const value = node.nodeValue || "";
    const index = value.indexOf({json.dumps(highlight_text)});
    if (index < 0) continue;
    const selection = window.getSelection();
    const range = document.createRange();
    range.setStart(node, index);
    range.setEnd(node, index + {len(highlight_text)});
    selection.removeAllRanges();
    selection.addRange(range);
    return {json.dumps(highlight_text)};
  }}
  return "missing-highlight-text";
}})();
"""
        )
        self.assertEqual(selection_seed, highlight_text)

        selection_info = self.request_live_selection_offsets(highlight_text)
        self.assertTrue(bool(selection_info.get("hasSelection")))
        self.window._add_persistent_preview_highlight(
            selection_info,
            highlight_text,
            kind=mdexplore.PREVIEW_HIGHLIGHT_KIND_IMPORTANT,
        )
        self.wait_until(
            lambda: any(
                item.get("kind") == "important"
                for item in self.persistent_highlight_spans()
            ),
            timeout_ms=6000,
        )
        self.assertTrue(
            any(
                str(entry.get("anchor_text", "")).strip() == highlight_text
                for entry in self.window._current_preview_text_highlights
                if isinstance(entry, dict)
            ),
            self.window._current_preview_text_highlights,
        )

        highlighted_before = self.persistent_highlight_text_by_id()
        self.assertTrue(highlighted_before)
        self.assertIn(highlight_text, list(highlighted_before.values()))
        self.assertNotIn(drift_text, " ".join(highlighted_before.values()))

        self.window._load_preview(other_path)
        self.wait_until(
            lambda: not self.window._preview_load_in_progress, timeout_ms=20000
        )
        self.wait_ms(SETTLE_WAIT_MS)

        self.window._load_preview(target_path)
        self.wait_until(
            lambda: not self.window._preview_load_in_progress, timeout_ms=20000
        )
        self.wait_ms(SETTLE_WAIT_MS)
        self.wait_until(
            lambda: bool(self.persistent_highlight_spans()),
            timeout_ms=6000,
        )
        self.assertTrue(
            any(
                str(entry.get("anchor_text", "")).strip() == highlight_text
                for entry in self.window._current_preview_text_highlights
                if isinstance(entry, dict)
            )
        )

        highlighted_after = self.persistent_highlight_text_by_id()
        self.assertTrue(highlighted_after)
        self.assertIn(highlight_text, list(highlighted_after.values()))
        self.assertNotIn(drift_text, " ".join(highlighted_after.values()))

    def test_legacy_source_offset_highlights_resolve_on_load_and_revisit(self) -> None:
        doc_text = """# Legacy Highlight Fixture

Intro paragraph.

> [!NOTE]
> Legacy source offsets were captured before preview-text normalization.

- **Built stakeholder-facing artifacts** (demo scripts, release notes, runbooks) to reduce onboarding and support overhead.  
**Skills:** Security automation • Risk-as-code
"""
        target_path = self.root / "legacy-source-highlight.md"
        target_path.write_text(doc_text, encoding="utf-8")
        other_path = self.root / "legacy-source-highlight-other.md"
        other_path.write_text("# Other\n\nNothing to see here.\n", encoding="utf-8")

        anchor_text = (
            "Built stakeholder-facing artifacts (demo scripts, release notes, runbooks)"
        )
        source_start = doc_text.index("Built stakeholder-facing artifacts")
        source_end = source_start + len(anchor_text)
        sidecar_path = self.root / ".mdexplore-highlighting.json"
        sidecar_payload = {"files": {target_path.name: [{
            "id": "legacy-source-case",
            "start": source_start,
            "end": source_end,
            "kind": "normal",
        }]}}
        sidecar_path.write_text(
            json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.window._persisted_text_highlights_by_dir.clear()

        self.window._load_preview(target_path)
        self.wait_until(
            lambda: not self.window._preview_load_in_progress, timeout_ms=20000
        )
        self.wait_ms(SETTLE_WAIT_MS)
        self.wait_until(
            lambda: "legacy-source-case" in self.persistent_highlight_text_by_id(),
            timeout_ms=4000,
        )
        self.assertEqual(
            self.persistent_highlight_text_by_id().get("legacy-source-case"),
            anchor_text,
        )

        self.window._load_preview(other_path)
        self.wait_until(
            lambda: not self.window._preview_load_in_progress, timeout_ms=20000
        )
        self.wait_ms(SETTLE_WAIT_MS)

        self.window._load_preview(target_path)
        self.wait_until(
            lambda: not self.window._preview_load_in_progress, timeout_ms=20000
        )
        self.wait_ms(SETTLE_WAIT_MS)
        self.wait_until(
            lambda: "legacy-source-case" in self.persistent_highlight_text_by_id(),
            timeout_ms=4000,
        )
        self.assertEqual(
            self.persistent_highlight_text_by_id().get("legacy-source-case"),
            anchor_text,
        )

    def test_persistent_highlight_apply_and_live_probes_match_dom_ranges(self) -> None:
        tall_text = "\n\n".join(
            [f"filler line {index}" for index in range(1, 80)]
            + ["alpha beta gamma"]
            + [f"tail line {index}" for index in range(80, 140)]
        ) + "\n"
        self.load_markdown_text("persistent-highlight-apply.md", tall_text)
        selection_seed = self.run_js(
            """
(() => {
  const root = document.querySelector("main") || document.body;
  if (!root) return "missing-root";
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const value = node.nodeValue || "";
    const index = value.indexOf("beta");
    if (index < 0) continue;
    const selection = window.getSelection();
    const range = document.createRange();
    range.setStart(node, index);
    range.setEnd(node, index + 4);
    selection.removeAllRanges();
    selection.addRange(range);
    return "beta";
  }
  return "missing-beta";
})();
"""
        )
        self.assertEqual(selection_seed, "beta")
        selected_offsets = self.request_live_selection_offsets()
        self.assertTrue(bool(selected_offsets.get("hasSelection")))
        self.assertEqual(selected_offsets.get("selectedText"), "beta")
        start = int(selected_offsets["selectionOffsetStart"])
        end = int(selected_offsets["selectionOffsetEnd"])
        self.assertGreaterEqual(start, 0)
        self.assertGreater(end, start)

        apply_result = self.apply_persistent_highlights(
            [{"id": "case1", "start": start, "end": end, "kind": "normal"}]
        )
        self.assertGreaterEqual(int(apply_result.get("applied", 0)), 1)
        self.wait_until(
            lambda: bool(self.persistent_highlight_spans())
            and int(self.probe()["highlightMarkers"]) >= 1,
            timeout_ms=4000,
        )

        spans = self.persistent_highlight_spans()
        self.assertEqual(
            spans,
            [{"text": "beta", "id": "case1", "kind": "normal"}],
        )
        self.assertGreaterEqual(int(self.probe()["highlightMarkers"]), 1)

        context_result = self.run_js(
            """
(() => {
  const node = document.querySelector('span[data-mdexplore-persistent-highlight="1"]');
  if (!node) return "missing";
  const rect = node.getBoundingClientRect();
  const options = {
    bubbles: true,
    cancelable: true,
    button: 2,
    clientX: rect.left + Math.max(1, rect.width / 2),
    clientY: rect.top + Math.max(1, rect.height / 2),
  };
  node.dispatchEvent(new MouseEvent("contextmenu", options));
  return "ok";
})();
"""
        )
        self.assertEqual(context_result, "ok")
        live_target = self.request_live_highlight_target()
        self.assertEqual(live_target.get("clickedHighlightId"), "case1")
        if live_target.get("clickedOffset") is not None:
            self.assertGreaterEqual(int(live_target["clickedOffset"]), start)
            self.assertLessEqual(int(live_target["clickedOffset"]), end)

        selection_result = self.run_js(
            """
(() => {
  const node = document.querySelector('span[data-mdexplore-persistent-highlight="1"]');
  if (!node) return "missing";
  const selection = window.getSelection();
  const range = document.createRange();
  range.selectNodeContents(node);
  selection.removeAllRanges();
  selection.addRange(range);
  return node.textContent || "";
})();
"""
        )
        self.assertEqual(selection_result, "beta")
        live_selection = self.request_live_selection_offsets()
        self.assertTrue(bool(live_selection.get("hasSelection")))
        self.assertEqual(live_selection.get("selectedText"), "beta")
        self.assertEqual(int(live_selection.get("selectionOffsetStart", -1)), start)
        self.assertEqual(int(live_selection.get("selectionOffsetEnd", -1)), end)

    def test_context_menu_probe_matches_selection_and_highlight_metadata(self) -> None:
        tall_text = "\n\n".join(
            [f"filler line {index}" for index in range(1, 60)]
            + ["alpha beta gamma"]
            + [f"tail line {index}" for index in range(60, 120)]
        ) + "\n"
        self.load_markdown_text("context-menu-probe.md", tall_text)

        seed_result = self.run_js(
            """
(() => {
  const root = document.querySelector("main") || document.body;
  if (!root) return "missing-root";
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const value = node.nodeValue || "";
    const index = value.indexOf("beta");
    if (index < 0) continue;
    const selection = window.getSelection();
    const range = document.createRange();
    range.setStart(node, index);
    range.setEnd(node, index + 4);
    selection.removeAllRanges();
    selection.addRange(range);
    return "beta";
  }
  return "missing-beta";
})();
"""
        )
        self.assertEqual(seed_result, "beta")
        selected_offsets = self.request_live_selection_offsets()
        start = int(selected_offsets["selectionOffsetStart"])
        end = int(selected_offsets["selectionOffsetEnd"])

        self.apply_persistent_highlights(
            [{"id": "case1", "start": start, "end": end, "kind": "normal"}]
        )
        self.wait_until(
            lambda: bool(self.persistent_highlight_spans()),
            timeout_ms=4000,
        )

        raw_coords = self.run_js(
            """
(() => {
  const node = document.querySelector('span[data-mdexplore-persistent-highlight="1"]');
  if (!node) return JSON.stringify({});
  node.scrollIntoView({ behavior: "auto", block: "center", inline: "nearest" });
  const selection = window.getSelection();
  const range = document.createRange();
  range.selectNodeContents(node);
  selection.removeAllRanges();
  selection.addRange(range);
  const rect = node.getBoundingClientRect();
  return JSON.stringify({
    x: Math.round(rect.left + Math.max(1, rect.width / 2)),
    y: Math.round(rect.top + Math.max(1, rect.height / 2)),
  });
})();
"""
        )
        self.assertIsInstance(raw_coords, str)
        coords = json.loads(raw_coords)
        self.wait_ms(150)

        probe = self.request_preview_context_menu_selection_info(
            int(coords["x"]), int(coords["y"]), "beta"
        )
        self.assertTrue(bool(probe.get("hasSelection")))
        self.assertEqual(probe.get("selectedText"), "beta")
        self.assertEqual(int(probe.get("selectionOffsetStart", -1)), start)
        self.assertEqual(int(probe.get("selectionOffsetEnd", -1)), end)
        self.assertEqual(probe.get("clickedHighlightId"), "case1")
        self.assertIn("case1", list(probe.get("selectedHighlightIds") or []))
        self.assertTrue(bool(probe.get("selectionHasHighlightedPart")))
        self.assertFalse(bool(probe.get("selectionHasUnhighlightedPart")))

    def test_search_marker_positions_track_scrollable_document_offsets(self) -> None:
        self.load_markdown_text("search-marker-positions.md", SEARCH_MARKER_POSITION_TEXT)
        self.window.match_input.setText("UNIQUEHIT")
        self.window._run_match_search_now()
        self.wait_until(
            lambda: int(self.probe()["searchMarkers"]) >= 2, timeout_ms=6000
        )

        projections = self.search_marker_projection()
        self.assertEqual(len(projections), 2)
        for entry in projections:
            self.assertLessEqual(abs(entry["actual"] - entry["expected"]), 3.0)

    def test_view_tab_round_trip_restores_saved_positions_without_drift(self) -> None:
        self.assertEqual(self.window.view_tabs.count(), 3)
        start_index = self.window.view_tabs.currentIndex()
        start_view_id = self.window._tab_view_id(start_index)
        other_index = 2 if start_index == 1 else 1
        other_view_id = self.window._tab_view_id(other_index)
        self.assertIsNotNone(start_view_id)
        self.assertIsNotNone(other_view_id)
        start_state = dict(self.window._view_states[int(start_view_id)])
        other_state = dict(self.window._view_states[int(other_view_id)])

        self.window.view_tabs.setCurrentIndex(other_index)
        self.wait_ms(1600)
        self.assert_view_matches_state(int(other_view_id), other_state)
        switch_probe = self.probe()
        self.wait_ms(900)
        settled_probe = self.probe()
        self.assertLessEqual(
            abs(float(switch_probe["scrollY"]) - float(settled_probe["scrollY"])), 80.0
        )
        self.assertLessEqual(
            abs(int(switch_probe["topLine"]) - int(settled_probe["topLine"])), 1
        )

        self.window.view_tabs.setCurrentIndex(start_index)
        self.wait_ms(1600)
        self.assert_view_matches_state(int(start_view_id), start_state)
        return_probe = self.probe()
        self.wait_ms(900)
        settled_return_probe = self.probe()
        self.assertLessEqual(
            abs(float(return_probe["scrollY"]) - float(settled_return_probe["scrollY"])),
            80.0,
        )
        self.assertLessEqual(
            abs(int(return_probe["topLine"]) - int(settled_return_probe["topLine"])), 1
        )

    def test_view_markers_restore_same_saved_positions_as_tabs(self) -> None:
        marker_entries = sorted(
            self.window._current_named_view_marker_entries(),
            key=lambda entry: (int(entry["top_line"]), int(entry["view_id"])),
        )
        self.assertEqual(self.probe()["viewMarkers"], len(marker_entries))
        self.assertGreaterEqual(len(marker_entries), 2)

        first_marker_view_id = int(marker_entries[0]["view_id"])
        second_marker_view_id = int(marker_entries[1]["view_id"])
        first_state = dict(self.window._view_states[first_marker_view_id])
        second_state = dict(self.window._view_states[second_marker_view_id])

        self.click_marker(".mdexplore-scroll-view-marker", 0)
        self.wait_until(
            lambda: self.window._active_view_id == first_marker_view_id,
            timeout_ms=2500,
        )
        self.wait_ms(1200)
        self.assert_view_matches_state(first_marker_view_id, first_state)

        self.click_marker(".mdexplore-scroll-view-marker", 1)
        self.wait_until(
            lambda: self.window._active_view_id == second_marker_view_id,
            timeout_ms=2500,
        )
        self.wait_ms(1200)
        self.assert_view_matches_state(second_marker_view_id, second_state)

    def test_switching_from_newly_labeled_view_restores_existing_view_position(self) -> None:
        self.assertGreaterEqual(self.window.view_tabs.count(), 2)
        target_index = 0 if self.window.view_tabs.currentIndex() != 0 else 1
        target_view_id = self.window._tab_view_id(target_index)
        self.assertIsNotNone(target_view_id)
        target_state = dict(self.window._view_states[int(target_view_id)])

        self.run_js("window.scrollTo(0, 2600);")
        self.wait_ms(500)
        self.window._capture_current_preview_scroll(force=True)
        self.wait_ms(250)

        before_count = self.window.view_tabs.count()
        self.window._add_document_view()
        self.wait_until(
            lambda: self.window.view_tabs.count() == before_count + 1,
            timeout_ms=2000,
        )
        self.wait_ms(800)

        self.run_js("window.scrollTo(0, 4300);")
        self.wait_ms(700)
        self.window._capture_current_preview_scroll(force=True)
        self.wait_ms(250)

        with patch("mdexplore.QInputDialog.getText", return_value=("Repro Label", True)):
            self.window._edit_view_tab_label(self.window.view_tabs.currentIndex())
        self.wait_ms(450)

        self.window.view_tabs.setCurrentIndex(target_index)
        self.wait_ms(1700)
        self.assert_view_matches_state(int(target_view_id), target_state)

    def test_tab_switch_uses_selection_time_restore_snapshot(self) -> None:
        self.assertGreaterEqual(self.window.view_tabs.count(), 2)
        target_index = 0 if self.window.view_tabs.currentIndex() != 0 else 1
        target_view_id = self.window._tab_view_id(target_index)
        self.assertIsNotNone(target_view_id)
        path_key = self.window._current_preview_path_key()
        self.assertIsNotNone(path_key)
        target_state = dict(self.window._view_states[int(target_view_id)])

        self.run_js("window.scrollTo(0, 4300);")
        self.wait_ms(500)
        self.window._capture_current_preview_scroll(force=True)
        self.wait_ms(250)

        scroll_key = f"{path_key}::view:{int(target_view_id)}"
        wrong_scroll = max(0.0, float(target_state["scroll_y"]) + 3200.0)
        wrong_top_line = max(1, int(target_state["top_line"]) + 80)

        self.window.view_tabs.setCurrentIndex(target_index)
        self.window._preview_scroll_positions[scroll_key] = wrong_scroll
        self.window._preview_scroll_positions[str(path_key)] = wrong_scroll
        self.window._view_states[int(target_view_id)]["scroll_y"] = wrong_scroll
        self.window._view_states[int(target_view_id)]["top_line"] = wrong_top_line

        self.wait_ms(1700)
        self.assert_view_matches_state(int(target_view_id), target_state)

    def test_persistent_highlight_markers_navigate_to_visible_highlights(self) -> None:
        before = self.probe()
        self.assertGreater(before["highlightMarkers"], 0)
        marker_index = self.farthest_marker_index(".mdexplore-scroll-highlight-marker")

        self.click_marker(".mdexplore-scroll-highlight-marker", marker_index)
        self.wait_ms(700)
        after = self.probe()

        self.assertGreater(after["visibleHighlights"], 0)
        self.assertTrue(
            abs(float(after["scrollY"]) - float(before["scrollY"])) > 120.0
            or abs(int(after["topLine"]) - int(before["topLine"])) > 6
        )

    def test_search_markers_navigate_to_visible_search_hits(self) -> None:
        self.window.match_input.setText("product")
        self.window._run_match_search_now()
        self.wait_until(
            lambda: int(self.probe()["searchMarkers"]) > 0, timeout_ms=6000
        )
        before = self.probe()
        marker_index = self.farthest_marker_index(".mdexplore-scroll-hit-marker")

        self.click_marker(".mdexplore-scroll-hit-marker", marker_index)
        self.wait_ms(700)
        after = self.probe()

        self.assertGreater(after["searchMarkers"], 0)
        self.assertGreater(after["visibleSearchMarks"], 0)
        self.assertTrue(
            abs(float(after["scrollY"]) - float(before["scrollY"])) > 120.0
            or abs(int(after["topLine"]) - int(before["topLine"])) > 6
        )


if __name__ == "__main__":
    unittest.main()
