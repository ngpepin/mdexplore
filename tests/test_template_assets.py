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
        self.assertIn("inlineMath: [['\\\\(', '\\\\)']]", source)
        self.assertIn("displayMath: [['\\\\[', '\\\\]']]", source)
        self.assertNotIn("inlineMath: [['$', '$']", source)

    def test_preview_template_routes_relative_markdown_links_through_custom_scheme(
        self,
    ) -> None:
        source = get_template_asset("preview/document.html")
        self.assertIn("isRelativeMarkdownHref", source)
        self.assertIn("mdexplore://open-relative/", source)

    def test_markdown_renderer_uses_external_preview_template(self) -> None:
        renderer = mdexplore.MarkdownRenderer()
        rendered = renderer.render_document("# Heading", 'Doc "Title"')
        self.assertIn("<title>Doc &quot;Title&quot;</title>", rendered)
        self.assertIn("<main><h1", rendered)
        self.assertIn(">Heading</h1>", rendered)
        self.assertIn("window.__mdexploreMathJaxSources = [", rendered)
        self.assertNotIn("__ESCAPED_TITLE__", rendered)
        self.assertNotIn("__BODY_HTML__", rendered)

    def test_markdown_renderer_neutralizes_raw_script_tags_in_prose(self) -> None:
        renderer = mdexplore.MarkdownRenderer()
        source = (
            "Our parser accepts strings such as "
            "<script>alert('xss')</script> as illustrative text."
        )

        rendered = renderer.render_document(source, "Script example")
        main_html = rendered.split("<main>", 1)[1].split("</main>", 1)[0]

        self.assertNotIn("<script", main_html.casefold())
        self.assertNotIn("</script", main_html.casefold())
        self.assertIn("&lt;script", main_html.casefold())
        self.assertIn("&lt;/script", main_html.casefold())
        self.assertIn("alert(", main_html.casefold())
        self.assertEqual(rendered.casefold().count("<script>"), 2)

    def test_markdown_renderer_does_not_parse_currency_as_math(self) -> None:
        renderer = mdexplore.MarkdownRenderer()
        tokens = renderer._md.parse(
            "## The $15,000 AI Bill. Your $20 Subscription Is a Delusion."
        )
        inline = next(token for token in tokens if token.type == "inline")
        self.assertNotIn("math_inline", [child.type for child in inline.children or []])
        self.assertIn("$15,000", inline.content)
        self.assertIn("$20", inline.content)

    def test_markdown_renderer_keeps_regular_inline_math(self) -> None:
        renderer = mdexplore.MarkdownRenderer()
        tokens = renderer._md.parse("Inline math $x^2 + y^2$ remains math.")
        inline = next(token for token in tokens if token.type == "inline")
        math_tokens = [child for child in inline.children or [] if child.type == "math_inline"]
        self.assertEqual([token.content for token in math_tokens], ["x^2 + y^2"])
        rendered = renderer._md.render("Inline math $x^2 + y^2$ remains math.")
        self.assertIn('class="mdexplore-math-inline"', rendered)
        self.assertIn("\\(x^2 + y^2\\)", rendered)

    def test_currency_heading_is_not_exposed_to_mathjax_dollar_scanning(self) -> None:
        renderer = mdexplore.MarkdownRenderer()
        source = "## 11. *The $15,000 AI Bill. Your $20 Subscription Is a Delusion.*"
        rendered = renderer.render_document(source, "Currency heading")
        self.assertIn("$15,000", rendered)
        self.assertIn("$20", rendered)
        self.assertNotIn("mdexplore-math-inline", rendered)
        self.assertFalse(
            mdexplore.MdExploreWindow._detect_special_features_from_html(rendered)[0]
        )

    def test_rendered_math_marker_drives_html_feature_detection(self) -> None:
        renderer = mdexplore.MarkdownRenderer()
        rendered = renderer.render_document("Inline $x + 2$ math.", "Math")
        self.assertIn("mdexplore-math-inline", rendered)
        self.assertTrue(
            mdexplore.MdExploreWindow._detect_special_features_from_html(rendered)[0]
        )

    def test_currency_does_not_trigger_math_feature_detection(self) -> None:
        source = "The service costs $15,000 while the plan costs $20 annually."
        renderer = mdexplore.MarkdownRenderer()
        self.assertFalse(renderer._markdown_contains_math(source))
        self.assertFalse(
            mdexplore.MdExploreWindow._detect_special_features_from_markdown(source)[0]
        )
        self.assertTrue(renderer._should_use_cmark_fast_path(source))

    def test_inline_math_triggers_math_feature_detection(self) -> None:
        source = "Inline math $2 + x$ remains supported."
        renderer = mdexplore.MarkdownRenderer()
        self.assertTrue(renderer._markdown_contains_math(source))
        self.assertTrue(
            mdexplore.MdExploreWindow._detect_special_features_from_markdown(source)[0]
        )
        self.assertFalse(renderer._should_use_cmark_fast_path(source))

    def test_markdown_renderer_normalizes_single_dollar_display_math(self) -> None:
        renderer = mdexplore.MarkdownRenderer()
        source = "$\n\\text{Expected value}\n= x + y\n$\n"
        prepared = renderer._prepare_markdown_for_render(source)
        self.assertEqual(prepared.count("\n"), source.count("\n"))
        self.assertTrue(prepared.startswith("$$\n"))
        self.assertTrue(prepared.endswith("$$\n"))
        tokens = renderer._md.parse(prepared)
        self.assertEqual([token.type for token in tokens], ["math_block"])

    def test_markdown_renderer_hides_chatgpt_writing_wrapper(self) -> None:
        renderer = mdexplore.MarkdownRenderer()
        source = (
            ':::writing{variant="document" id="48271"}\n\n'
            "# Heading\n\nBody\n\n:::\n"
        )
        prepared = renderer._prepare_markdown_for_render(source)
        self.assertEqual(prepared.count("\n"), source.count("\n"))
        self.assertNotIn(":::writing", prepared)
        self.assertNotIn("\n:::\n", prepared)
        self.assertIn("# Heading", prepared)
        self.assertIn("Body", prepared)

    def test_markdown_normalization_ignores_code_fences(self) -> None:
        renderer = mdexplore.MarkdownRenderer()
        source = (
            '```text\n:::writing{variant="document" id="48271"}\n$\n```\n'
        )
        self.assertEqual(renderer._prepare_markdown_for_render(source), source)


if __name__ == "__main__":
    unittest.main()
