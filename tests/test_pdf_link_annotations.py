from __future__ import annotations

import unittest
from io import BytesIO

from mdexplore_app.pdf import stamp_pdf_page_numbers


class PdfLinkAnnotationTests(unittest.TestCase):
    @staticmethod
    def _source_pdf() -> bytes:
        from reportlab.pdfgen import canvas

        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=(612, 792))
        pdf.bookmarkPage("footnote-target")
        pdf.drawString(72, 700, "Footnote destination")
        pdf.showPage()
        pdf.drawString(72, 700, "External bibliography source")
        pdf.linkURL(
            "https://example.com/source",
            (72, 695, 180, 715),
            relative=0,
        )
        pdf.drawString(72, 650, "Internal footnote reference")
        pdf.linkRect(
            "",
            "footnote-target",
            (72, 645, 170, 665),
            relative=0,
        )
        pdf.save()
        return buffer.getvalue()

    @staticmethod
    def _link_annotations(page):
        return [
            ref.get_object()
            for ref in (page.get("/Annots") or [])
            if str(ref.get_object().get("/Subtype")) == "/Link"
        ]

    def test_page_layout_moves_link_hitboxes_with_content(self) -> None:
        from pypdf import PdfReader

        source = PdfReader(BytesIO(self._source_pdf()), strict=False)
        original_links = self._link_annotations(source.pages[1])
        self.assertEqual(2, len(original_links))
        original_rects = [tuple(float(value) for value in link["/Rect"]) for link in original_links]

        exported = PdfReader(
            BytesIO(stamp_pdf_page_numbers(self._source_pdf())), strict=False
        )
        exported_links = self._link_annotations(exported.pages[1])
        self.assertEqual(2, len(exported_links))
        exported_rects = [tuple(float(value) for value in link["/Rect"]) for link in exported_links]

        self.assertNotEqual(original_rects, exported_rects)
        source_vertical_gap = original_rects[0][1] - original_rects[1][1]
        exported_vertical_gap = exported_rects[0][1] - exported_rects[1][1]
        scale = exported_vertical_gap / source_vertical_gap
        self.assertGreater(scale, 0.5)
        self.assertLess(scale, 1.0)

        for original_rect, exported_rect in zip(original_rects, exported_rects):
            self.assertAlmostEqual(
                (exported_rect[2] - exported_rect[0])
                / (original_rect[2] - original_rect[0]),
                scale,
                places=5,
            )
            self.assertAlmostEqual(
                (exported_rect[3] - exported_rect[1])
                / (original_rect[3] - original_rect[1]),
                scale,
                places=5,
            )

    def test_external_and_internal_link_targets_survive_export(self) -> None:
        from pypdf import PdfReader

        exported = PdfReader(
            BytesIO(stamp_pdf_page_numbers(self._source_pdf())), strict=False
        )
        links = self._link_annotations(exported.pages[1])
        self.assertEqual(2, len(links))

        external = next(link for link in links if link.get("/A"))
        self.assertEqual("/URI", str(external["/A"]["/S"]))
        self.assertEqual("https://example.com/source", external["/A"]["/URI"])

        internal = next(link for link in links if link.get("/Dest"))
        destination = internal["/Dest"]
        self.assertGreaterEqual(len(destination), 2)
        self.assertEqual("/Fit", str(destination[1]))
        self.assertEqual(
            exported.pages[0].indirect_reference.idnum,
            destination[0].idnum,
        )


if __name__ == "__main__":
    unittest.main()
