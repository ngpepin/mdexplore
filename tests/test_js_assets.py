from __future__ import annotations

import json
import unittest

from mdexplore_app.constants import JS_ASSET_DIR, PDF_LANDSCAPE_PAGE_TOKEN
from mdexplore_app.js import get_js_asset, preload_js_assets, render_js_asset


class JsAssetTests(unittest.TestCase):
    def test_required_preview_scripts_are_preloaded(self) -> None:
        sources = preload_js_assets(force_reload=True)
        self.assertTrue(JS_ASSET_DIR.is_dir())
        self.assertIn("preview/apply_inline_image_sources.js", sources)
        self.assertIn("preview/apply_persistent_highlights.js", sources)
        self.assertIn("preview/clear_search_highlights.js", sources)
        self.assertIn("preview/collect_diagram_view_state.js", sources)
        self.assertIn("preview/context_menu_selection_probe.js", sources)
        self.assertIn("preview/harvest_mermaid_cache.js", sources)
        self.assertIn("preview/highlight_search_terms.js", sources)
        self.assertIn("preview/live_highlight_target.js", sources)
        self.assertIn("preview/live_selection_offsets.js", sources)
        self.assertIn("preview/probe_active_top_line.js", sources)
        self.assertIn("preview/probe_pdf_diagram_readiness.js", sources)
        self.assertIn("preview/probe_restore_overlay_readiness.js", sources)
        self.assertIn("preview/trigger_client_renderers.js", sources)
        self.assertIn("preview/update_named_view_markers.js", sources)
        self.assertIn("pdf/precheck_layout.js", sources)
        self.assertIn("pdf/preprint_normalize.js", sources)
        self.assertIn("pdf/restore_preview_palette.js", sources)
        self.assertTrue(sources["preview/apply_inline_image_sources.js"].strip())
        self.assertTrue(sources["preview/apply_persistent_highlights.js"].strip())
        self.assertTrue(sources["preview/clear_search_highlights.js"].strip())
        self.assertTrue(sources["preview/collect_diagram_view_state.js"].strip())
        self.assertTrue(sources["preview/context_menu_selection_probe.js"].strip())
        self.assertTrue(sources["preview/harvest_mermaid_cache.js"].strip())
        self.assertTrue(sources["preview/highlight_search_terms.js"].strip())
        self.assertTrue(sources["preview/live_highlight_target.js"].strip())
        self.assertTrue(sources["preview/live_selection_offsets.js"].strip())
        self.assertTrue(sources["preview/probe_active_top_line.js"].strip())
        self.assertTrue(sources["preview/probe_pdf_diagram_readiness.js"].strip())
        self.assertTrue(sources["preview/probe_restore_overlay_readiness.js"].strip())
        self.assertTrue(sources["preview/trigger_client_renderers.js"].strip())
        self.assertTrue(sources["preview/update_named_view_markers.js"].strip())
        self.assertTrue(sources["pdf/precheck_layout.js"].strip())
        self.assertTrue(sources["pdf/preprint_normalize.js"].strip())
        self.assertTrue(sources["pdf/restore_preview_palette.js"].strip())

    def test_rendered_inline_image_template_reports_applied_digests(self) -> None:
        rendered = render_js_asset(
            "preview/apply_inline_image_sources.js",
            {"__INLINE_IMAGE_URLS_JSON__": '{"abc":"file:///tmp/image.png"}'},
        )
        self.assertIn("appliedDigests", rendered)
        self.assertIn('"abc":"file:///tmp/image.png"', rendered)
        self.assertNotIn("__INLINE_IMAGE_URLS_JSON__", rendered)

    def test_rendered_preview_search_template_has_no_unresolved_placeholders(self) -> None:
        rendered = render_js_asset(
            "preview/highlight_search_terms.js",
            {
                "__TERMS_JSON__": '[{"text":"alpha","caseSensitive":false}]',
                "__SCROLL_BOOL__": "true",
                "__NEAR_WORD_GAP__": "50",
                "__NEAR_GROUPS_JSON__": "[]",
            },
        )
        self.assertIn('const shouldScroll = true;', rendered)
        self.assertIn('const nearWordGap = 50;', rendered)
        self.assertNotIn("__TERMS_JSON__", rendered)
        self.assertNotIn("__SCROLL_BOOL__", rendered)
        self.assertNotIn("__NEAR_WORD_GAP__", rendered)
        self.assertNotIn("__NEAR_GROUPS_JSON__", rendered)
        self.assertIn("['\\u2018\\u2019\\u02bc]", rendered)

    def test_rendered_live_selection_template_has_no_unresolved_placeholders(self) -> None:
        rendered = render_js_asset(
            "preview/live_selection_offsets.js",
            {"__SELECTED_HINT__": json.dumps("Selected text")},
        )
        self.assertIn('selectedText = "Selected text";', rendered)
        self.assertNotIn("__SELECTED_HINT__", rendered)

    def test_rendered_context_menu_probe_template_has_no_unresolved_placeholders(
        self,
    ) -> None:
        rendered = render_js_asset(
            "preview/context_menu_selection_probe.js",
            {
                "__CLICK_X__": "12",
                "__CLICK_Y__": "34",
                "__SELECTED_HINT__": json.dumps("Hint text"),
            },
        )
        self.assertIn('const hintedText = "Hint text";', rendered)
        self.assertIn("elementFromClick(12, 34)", rendered)
        self.assertNotIn("__CLICK_X__", rendered)
        self.assertNotIn("__CLICK_Y__", rendered)
        self.assertNotIn("__SELECTED_HINT__", rendered)

    def test_rendered_named_view_marker_update_template_has_no_unresolved_placeholders(
        self,
    ) -> None:
        rendered = render_js_asset(
            "preview/update_named_view_markers.js",
            {"__PAYLOAD_JSON__": '[{"view_id":1,"top_line":12,"scroll_y":100.0,"color":"#abc"}]'},
        )
        self.assertIn(
            'window.__mdexploreNamedViewMarkers = [{"view_id":1,"top_line":12,"scroll_y":100.0,"color":"#abc"}];',
            rendered,
        )
        self.assertNotIn("__PAYLOAD_JSON__", rendered)

    def test_rendered_apply_persistent_highlights_template_has_no_unresolved_placeholders(
        self,
    ) -> None:
        rendered = render_js_asset(
            "preview/apply_persistent_highlights.js",
            {
                "__PAYLOAD__": '[{"id":"case1","start":1,"end":5,"kind":"normal"}]',
                "__COLOR__": json.dumps("rgba(1,2,3,0.4)"),
                "__IMPORTANT_COLOR__": json.dumps("rgba(4,5,6,0.7)"),
                "__IMPORTANT_TEXT_COLOR__": json.dumps("#010203"),
                "__MARKER_COLOR__": json.dumps("rgba(7,8,9,0.8)"),
                "__IMPORTANT_MARKER_COLOR__": json.dumps("rgba(9,8,7,0.9)"),
                "__NORMAL_KIND__": "normal",
                "__IMPORTANT_KIND__": "important",
                "__OFFSET_SPACE_PREVIEW__": "preview_text_v2",
                "__OFFSET_SPACE_SOURCE__": "markdown_source_v1",
            },
        )
        self.assertIn('const incoming = [{"id":"case1","start":1,"end":5,"kind":"normal"}];', rendered)
        self.assertIn('const highlightColor = "rgba(1,2,3,0.4)";', rendered)
        self.assertIn('const importantHighlightTextColor = "#010203";', rendered)
        self.assertNotIn("__OFFSET_SPACE_PREVIEW__", rendered)
        self.assertNotIn("__OFFSET_SPACE_SOURCE__", rendered)
        self.assertNotIn("__PAYLOAD__", rendered)
        self.assertNotIn("__IMPORTANT_KIND__", rendered)

    def test_rendered_pdf_precheck_template_has_no_unresolved_placeholders(self) -> None:
        rendered = render_js_asset(
            "pdf/precheck_layout.js",
            {
                "__MDEXPLORE_FORCE_MERMAID__": "true",
                "__MDEXPLORE_RESET_MERMAID__": "false",
                "__MDEXPLORE_PRINT_LAYOUT_KNOBS__": "{}",
                "__MDEXPLORE_LANDSCAPE_PAGE_TOKEN_JSON__": json.dumps(
                    PDF_LANDSCAPE_PAGE_TOKEN
                ),
            },
        )
        self.assertIn("startPdfMermaidCleanRender(true);", rendered)
        self.assertIn(
            'const requestedBackend = String(window.__mdexplorePdfMermaidBackend || "js").toLowerCase();',
            rendered,
        )
        self.assertIn(
            'const backend = requestedBackend === "rust" ? "rust" : "js";',
            rendered,
        )
        self.assertNotIn(
            'const backend = String(window.__mdexploreMermaidBackend || "js").toLowerCase();',
            rendered,
        )
        self.assertIn('block.dataset.mdexploreMermaidRenderer = "js";', rendered)
        self.assertIn(
            'const previewBackend = String(window.__mdexploreMermaidBackend || "js").toLowerCase();',
            rendered,
        )
        self.assertIn("block.__mdexploreRustSvgMarkup = rustMarkup;", rendered)
        self.assertIn("autoCache[hashKey] = rustMarkup;", rendered)
        self.assertIn("const rustFallbackMarkupFor = (block) => {", rendered)
        self.assertIn("const restoreRustFallbackForPdf = (block) => {", rendered)
        self.assertIn("const isMermaidErrorSvgMarkup = (svgMarkup) => {", rendered)
        self.assertIn("const cleanupMermaidRenderArtifacts = (renderId = \"\") => {", rendered)
        self.assertIn("suppressErrorRendering: true", rendered)
        self.assertIn("cleanupMermaidRenderArtifacts(renderId);", rendered)
        self.assertIn("cleanupMermaidRenderArtifacts();", rendered)
        self.assertIn("for (const candidate of [pdfMarkup, storedMarkup, autoMarkup])", rendered)
        self.assertIn("if (restoreRustFallbackForPdf(block)) {", rendered)
        self.assertIn('block.dataset.mdexploreMermaidRenderer = "rust";', rendered)
        self.assertIn('svg.querySelector(".error-icon, .error-text")', rendered)
        self.assertIn('throw new Error("Mermaid returned its syntax-error SVG")', rendered)
        render_catch = rendered.index("          } catch (renderError) {")
        fallback_in_catch = rendered.index(
            "if (restoreRustFallbackForPdf(block)) {", render_catch
        )
        failure_in_catch = rendered.index("renderFailures += 1;", fallback_in_catch)
        self.assertLess(fallback_in_catch, failure_in_catch)
        self.assertLess(
            rendered.index("block.__mdexploreRustSvgMarkup = rustMarkup;"),
            rendered.index("block.dataset.mdexploreMermaidRenderer = \"js\";"),
        )
        self.assertIn("if (false) {", rendered)
        self.assertIn(
            f'const landscapeTokenText = "{PDF_LANDSCAPE_PAGE_TOKEN}";', rendered
        )

    def test_pdf_restore_reinstates_rust_mermaid_before_client_rerender(self) -> None:
        source = get_js_asset("pdf/restore_preview_palette.js")
        self.assertIn("const restoreRustMermaidBlocks = () => {", source)
        self.assertIn("window.__mdexploreRemoveMermaidRenderArtifacts();", source)
        self.assertIn('id.startsWith("dmdexplore_")', source)
        self.assertIn('id.startsWith("imdexplore_")', source)
        self.assertIn(
            'String(window.__mdexploreMermaidBackend || "js").toLowerCase() !== "rust"',
            source,
        )
        self.assertIn("block.innerHTML = rustMarkup;", source)
        self.assertIn('block.dataset.mdexploreMermaidRenderer = "rust";', source)
        self.assertIn('window.__mdexploreApplyMermaidPostStyles(block, "auto");', source)
        self.assertIn('block.querySelectorAll(".mdexplore-mermaid-renderer-toggle")', source)
        self.assertIn('rendererToggle.textContent = "R";', source)
        self.assertIn('rendererToggle.title = "Rust renderer active; switch to JavaScript";', source)
        self.assertIn("const restoredRustCount = restoreRustMermaidBlocks();", source)
        renderer_assignment = source.index('block.dataset.mdexploreMermaidRenderer = "rust";')
        toolbar_restore = source.index('window.__mdexploreApplyMermaidPostStyles(block, "auto");')
        toggle_sync = source.index('rendererToggle.textContent = "R";')
        self.assertLess(renderer_assignment, toolbar_restore)
        self.assertLess(toolbar_restore, toggle_sync)
        self.assertIn("const restoredAllRustBlocks =", source)
        self.assertIn('window.__mdexploreMermaidPaletteMode = "auto";', source)
        self.assertIn("forceMermaid: !restoredAllRustBlocks", source)
        self.assertLess(
            source.index("const restoredRustCount = restoreRustMermaidBlocks();"),
            source.index("window.__mdexploreRunClientRenderers"),
        )

    def test_pdf_capture_restores_preview_before_background_write(self) -> None:
        source = (JS_ASSET_DIR.parent.parent / "mdexplore.py").read_text(encoding="utf-8")
        render_ready = source.split("    def _on_pdf_render_ready(", 1)[1].split(
            "    def _on_pdf_export_finished(", 1
        )[0]
        export_finished = source.split("    def _on_pdf_export_finished(", 1)[1].split(
            "    def _edit_current_file(", 1
        )[0]
        self.assertIn("self._restore_preview_mermaid_palette(source_key)", render_ready)
        self.assertIn("self._restore_preview_zoom_after_pdf_export()", render_ready)
        self.assertLess(
            render_ready.index("self._restore_preview_mermaid_palette(source_key)"),
            render_ready.index("worker = PdfExportWorker"),
        )
        self.assertNotIn("self._restore_preview_mermaid_palette(source_key)", export_finished)

    def test_missing_placeholder_replacements_raise(self) -> None:
        with self.assertRaises(ValueError):
            render_js_asset("preview/highlight_search_terms.js")

    def test_get_js_asset_returns_exact_source(self) -> None:
        source = get_js_asset("preview/clear_search_highlights.js")
        self.assertIn("data-mdexplore-search-mark", source)


if __name__ == "__main__":
    unittest.main()
