from __future__ import annotations

import json
import unittest

from mdexplore_app.constants import JS_ASSET_DIR, PDF_LANDSCAPE_PAGE_TOKEN
from mdexplore_app.js import get_js_asset, preload_js_assets, render_js_asset


class JsAssetTests(unittest.TestCase):
    def test_required_preview_scripts_are_preloaded(self) -> None:
        sources = preload_js_assets(force_reload=True)
        self.assertTrue(JS_ASSET_DIR.is_dir())
        self.assertIn("preview/clear_search_highlights.js", sources)
        self.assertIn("preview/highlight_search_terms.js", sources)
        self.assertIn("pdf/precheck_layout.js", sources)
        self.assertIn("pdf/preprint_normalize.js", sources)
        self.assertIn("pdf/restore_preview_palette.js", sources)
        self.assertTrue(sources["preview/clear_search_highlights.js"].strip())
        self.assertTrue(sources["preview/highlight_search_terms.js"].strip())
        self.assertTrue(sources["pdf/precheck_layout.js"].strip())
        self.assertTrue(sources["pdf/preprint_normalize.js"].strip())
        self.assertTrue(sources["pdf/restore_preview_palette.js"].strip())

    def test_rendered_preview_search_template_has_no_unresolved_placeholders(self) -> None:
        rendered = render_js_asset(
            "preview/highlight_search_terms.js",
            {
                "__TERMS_JSON__": '[{"text":"alpha","caseSensitive":false}]',
                "__SCROLL_BOOL__": "true",
                "__CLOSE_WORD_GAP__": "50",
                "__CLOSE_GROUPS_JSON__": "[]",
            },
        )
        self.assertIn('const shouldScroll = true;', rendered)
        self.assertIn('const closeWordGap = 50;', rendered)
        self.assertNotIn("__TERMS_JSON__", rendered)
        self.assertNotIn("__SCROLL_BOOL__", rendered)
        self.assertNotIn("__CLOSE_WORD_GAP__", rendered)
        self.assertNotIn("__CLOSE_GROUPS_JSON__", rendered)

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
        self.assertIn("if (false) {", rendered)
        self.assertIn(
            f'const landscapeTokenText = "{PDF_LANDSCAPE_PAGE_TOKEN}";', rendered
        )

    def test_missing_placeholder_replacements_raise(self) -> None:
        with self.assertRaises(ValueError):
            render_js_asset("preview/highlight_search_terms.js")

    def test_get_js_asset_returns_exact_source(self) -> None:
        source = get_js_asset("preview/clear_search_highlights.js")
        self.assertIn("data-mdexplore-search-mark", source)


if __name__ == "__main__":
    unittest.main()
