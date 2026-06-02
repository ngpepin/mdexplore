from __future__ import annotations

from pathlib import Path
import unittest


class PdfExploreAssetTests(unittest.TestCase):
    def test_pdfjs_viewer_bundle_exists(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.assertTrue((root / "pdfexplore" / "vendor" / "pdfjs" / "web" / "viewer.html").is_file())
        self.assertTrue((root / "pdfexplore" / "vendor" / "pdfjs" / "web" / "viewer.mjs").is_file())
        self.assertTrue((root / "pdfexplore" / "vendor" / "pdfjs" / "build" / "pdf.mjs").is_file())
        self.assertTrue((root / "pdfexplore" / "vendor" / "pdfjs" / "build" / "pdf.worker.mjs").is_file())

    def test_viewer_bridge_asset_exists(self) -> None:
        root = Path(__file__).resolve().parent.parent
        bridge = root / "pdfexplore" / "assets" / "viewer_bridge.js"
        self.assertTrue(bridge.is_file())
        source = bridge.read_text(encoding="utf-8")
        self.assertIn("__pdfexploreBridge", source)
        self.assertIn("lastSelectionPayload", source)
        self.assertIn("captureViewState()", source)
        self.assertIn("getZoomState()", source)
        self.assertIn("setZoomScale(", source)
        self.assertIn("resetZoom()", source)
        self.assertIn("toggleThreeUpMode", source)
        self.assertIn("isThreeUpActive", source)
        self.assertIn("setOnePageZoom100", source)
        self.assertIn('currentApp.eventBus.on("updateviewarea"', source)
        self.assertIn("container.addEventListener(\"scroll\"", source)
        self.assertIn("window.setTimeout(() => {", source)
        self.assertIn('currentViewer.currentPageNumber = page', source)
        self.assertIn('currentApp.eventBus.on("pagerendered", reapply)', source)
        self.assertIn('currentApp.eventBus.on("textlayerrendered", refreshFromPdfJs)', source)
        self.assertIn("rgba(102, 86, 178, 0.36)", source)
        self.assertIn("rgba(225, 214, 255, 0.76)", source)


if __name__ == "__main__":
    unittest.main()
