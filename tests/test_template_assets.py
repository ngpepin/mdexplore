from __future__ import annotations

import unittest

import mdexplore
from mdexplore_app.constants import TEMPLATE_ASSET_DIR
from mdexplore_app.templates import (
    get_template_asset,
    preload_template_assets,
    render_template_asset,
)


class TemplateAssetTests(unittest.TestCase):
    def test_required_preview_template_is_preloaded(self) -> None:
        sources = preload_template_assets(force_reload=True)
        self.assertTrue(TEMPLATE_ASSET_DIR.is_dir())
        self.assertIn("preview/document.html", sources)
        self.assertTrue(sources["preview/document.html"].strip())

    def test_rendered_preview_document_template_has_no_unresolved_placeholders(
        self,
    ) -> None:
        rendered = render_template_asset(
            "preview/document.html",
            {
                "__ESCAPED_TITLE__": "Preview Title",
                "__MATHJAX_SOURCES_JSON__": "[]",
                "__MERMAID_SOURCES_JSON__": "[]",
                "__MERMAID_BACKEND_JSON__": '"js"',
                "__TOTAL_SOURCE_LINES_JSON__": "42",
                "__PERSISTENT_HIGHLIGHT_MARKER_COLOR_JSON__": '"rgba(1,2,3,0.4)"',
                "__PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR_JSON__": '"rgba(4,5,6,0.7)"',
                "__PREVIEW_HIGHLIGHT_KIND_NORMAL_JSON__": '"normal"',
                "__PREVIEW_HIGHLIGHT_KIND_IMPORTANT_JSON__": '"important"',
                "__MERMAID_CACHE_RESTORE_BATCH_SIZE__": "2",
                "__MERMAID_CACHE_TOKEN_JSON__": '"{}"',
                "__DIAGRAM_VIEW_STATE_TOKEN_JSON__": '"{}"',
                "__BODY_HTML__": "<p>Rendered body</p>",
            },
        )
        self.assertIn("<title>Preview Title</title>", rendered)
        self.assertIn("<main><p>Rendered body</p></main>", rendered)
        self.assertNotIn("__ESCAPED_TITLE__", rendered)
        self.assertNotIn("__BODY_HTML__", rendered)

    def test_missing_template_placeholder_replacements_raise(self) -> None:
        with self.assertRaises(ValueError):
            render_template_asset("preview/document.html")

    def test_get_template_asset_returns_exact_source(self) -> None:
        source = get_template_asset("preview/document.html")
        self.assertIn("window.__mdexploreLoadMathJaxScript", source)

    def test_markdown_renderer_uses_external_preview_template(self) -> None:
        renderer = mdexplore.MarkdownRenderer()
        rendered = renderer.render_document("# Heading", 'Doc "Title"')
        self.assertIn("<title>Doc &quot;Title&quot;</title>", rendered)
        self.assertIn("<main><h1", rendered)
        self.assertIn(">Heading</h1>", rendered)
        self.assertIn("window.__mdexploreMathJaxSources = [", rendered)
        self.assertNotIn("__ESCAPED_TITLE__", rendered)
        self.assertNotIn("__BODY_HTML__", rendered)


if __name__ == "__main__":
    unittest.main()
