from __future__ import annotations

import unittest

from mdexplore_app.constants import PDF_LANDSCAPE_PAGE_TOKEN
from mdexplore_app.pdf import (
    _classify_pdf_page_flags,
    _page_looks_like_table_of_contents,
)


class PdfLayoutHintTests(unittest.TestCase):
    def test_table_of_contents_heading_is_detected(self) -> None:
        self.assertTrue(
            _page_looks_like_table_of_contents(
                "Table of Contents\nIntroduction ........ 1\nArchitecture ........ 4"
            )
        )
        self.assertTrue(
            _page_looks_like_table_of_contents(
                "Preface\n\nContents\nOverview 1\nAppendix 9"
            )
        )
        self.assertFalse(
            _page_looks_like_table_of_contents(
                "Architecture Overview\nThis page discusses contents and scope."
            )
        )

    def test_toc_pages_do_not_inherit_landscape_from_heading_matches(self) -> None:
        landscape_flags, diagram_flags, toc_flags = _classify_pdf_page_flags(
            [
                (
                    "Table of Contents\n"
                    "Overview ........ 1\n"
                    "Wide Sequence Diagram ........ 7\n"
                    f"{PDF_LANDSCAPE_PAGE_TOKEN}\n"
                ),
                "Wide Sequence Diagram\nRendered diagram body",
            ],
            {
                "landscapeHeadings": ["Wide Sequence Diagram"],
                "diagramHeadings": ["Wide Sequence Diagram"],
            },
        )
        self.assertEqual([True, False], toc_flags)
        self.assertEqual([False, True], landscape_flags)
        self.assertEqual([False, True], diagram_flags)

    def test_non_toc_heading_match_still_marks_landscape_page(self) -> None:
        landscape_flags, diagram_flags, toc_flags = _classify_pdf_page_flags(
            ["Wide Sequence Diagram\nRendered diagram body"],
            {
                "landscapeHeadings": ["Wide Sequence Diagram"],
                "diagramHeadings": ["Wide Sequence Diagram"],
            },
        )
        self.assertEqual([False], toc_flags)
        self.assertEqual([True], landscape_flags)
        self.assertEqual([True], diagram_flags)


if __name__ == "__main__":
    unittest.main()
