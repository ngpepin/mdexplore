"""Regressions for Markdown exported with ChatGPT DIL layout markup."""

from __future__ import annotations

from pathlib import Path
import re
import unittest

from mdexplore_app.dil_markdown import normalize_dil_markdown
from mdexplore_app.dil_widgets import render_known_dil_widgets


REPORT = (
    Path(__file__).resolve().parents[1]
    / "issues/SVG-rendering/2026-11-02 Create AIgile Summary.md"
)


class DilMarkdownTests(unittest.TestCase):
    def test_reported_document_preserves_prose_and_links(self) -> None:
        original = REPORT.read_text(encoding="utf-8")
        normalized = normalize_dil_markdown(original)
        section = normalized.split("### 10.2 Comparing", 1)[1].split("### 10.3 Representative", 1)[0]
        self.assertEqual(len(original.splitlines()), len(normalized.splitlines()))
        self.assertEqual(section.count("Best suited to teams"), 3)
        self.assertIn("The adapter installs an extension", section)
        self.assertIn("The main advantage is process enforcement", section)
        self.assertIn("[AIgile with SpecKit](https://www.goaigile.net/docs/tutorials/speckit/)", section)
        self.assertNotIn("<Link", section)
        self.assertNotIn("```", section)
        self.assertNotIn("****", section)
        self.assertIsNone(re.search(r"(?m)^ {4,}\S", section))

    def test_reported_repository_tree_retains_nested_structure(self) -> None:
        original = REPORT.read_text(encoding="utf-8")
        normalized = normalize_dil_markdown(original)
        tree_section = normalized.split("### 10.1 Recommended", 1)[1].split("### 10.2 Comparing", 1)[0]
        self.assertIn('<pre><code class="language-text">', tree_section)
        self.assertIn(".aigile/\n├── constitution.md\n├── standards/\n│   ├── api.md", tree_section)
        self.assertIn("</code></pre>", tree_section)
        self.assertNotIn("<CodeBlock", tree_section)

    def test_actual_code_fence_inside_layout_is_preserved(self) -> None:
        source = "<box gap={2}>\n\n```python\nprint(42)\n```\n\n  Normal prose.\n</box>\n"
        normalized = normalize_dil_markdown(source)
        self.assertIn("```python\nprint(42)\n```", normalized)
        self.assertIn("\nNormal prose.\n", normalized)
        self.assertEqual(source.count("\n"), normalized.count("\n"))

    def test_aigile_widgets_render_without_evaluating_dil(self) -> None:
        original = REPORT.read_text(encoding="utf-8")
        widgets = render_known_dil_widgets(original)
        self.assertEqual(original.count("\n"), widgets.count("\n"))
        self.assertEqual(widgets.count('class="aigile-roadmap-day"'), 3)
        self.assertEqual(widgets.count('class="aigile-roadmap-phase"'), 3)
        self.assertEqual(widgets.count('class="aigile-readiness-area"'), 7)
        self.assertEqual(widgets.count('class="aigile-rating"'), 35)
        self.assertIn('data-aigile-readiness="1"', widgets)
        self.assertIn('data-aigile-prepare', widgets)
        self.assertIn('Agile and product delivery', widgets)
        self.assertIn('Phase 3 - Evaluate and institutionalize', widgets)
        self.assertNotIn('{@body const dimensions', widgets)
        self.assertNotIn('{#if answered===7}', widgets)
        normalized = normalize_dil_markdown(widgets)
        self.assertIn('<section class="aigile-readiness"', normalized)
        self.assertIn('<section class="aigile-roadmap"', normalized)
        self.assertEqual(original.count("\n"), normalized.count("\n"))

    def test_unsupported_widget_shape_remains_literal_and_inert(self) -> None:
        source = '{@body const dimensions = [{name:"<script>",detail:"evil"}]}\n'
        self.assertEqual(source, render_known_dil_widgets(source))
        self.assertEqual('Simple paragraph\n', render_known_dil_widgets('Simple paragraph\n'))

    def test_aigile_dimension_labels_are_html_escaped(self) -> None:
        source = REPORT.read_text(encoding="utf-8")
        altered = source.replace('Agile and product delivery', 'Agile & <test> delivery', 1)
        rendered = render_known_dil_widgets(altered)
        self.assertIn('Agile &amp; &lt;test&gt; delivery', rendered)
        self.assertNotIn('>Agile & <test> delivery<', rendered)

    def test_document_without_dil_is_unchanged(self) -> None:
        source = "# Plain Markdown\n\n```svg\n<svg></svg>\n```\n"
        self.assertEqual(source, normalize_dil_markdown(source))


if __name__ == "__main__":
    unittest.main()
