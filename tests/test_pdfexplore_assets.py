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
        self.assertIn("setDarkMode", source)
        self.assertIn("isDarkModeActive", source)
        self.assertIn("html.pdfexplore-dark-mode .page", source)
        self.assertIn("filter: invert(90%) hue-rotate(180deg)", source)
        self.assertIn("toggleThreeUpMode", source)
        self.assertIn("isThreeUpActive", source)
        self.assertIn("setOnePageZoom100", source)
        self.assertIn('currentApp.eventBus.on("updateviewarea"', source)
        self.assertIn("target.addEventListener(\"scroll\"", source)
        self.assertIn("window.setTimeout(() => {", source)
        self.assertIn('currentViewer.currentPageNumber = page', source)
        self.assertIn('currentApp.eventBus.on("pagerendered", reapply)', source)
        self.assertIn('currentApp.eventBus.on("textlayerrendered", refreshFromPdfJs)', source)
        self.assertIn("rgba(102, 86, 178, 0.36)", source)
        self.assertIn("rgba(225, 214, 255, 0.76)", source)
        self.assertIn("pdfexplore-search-indicator", source)
        self.assertIn("pdfexplore-highlight-indicator", source)
        self.assertIn("refreshPersistentHighlightIndicators", source)
        self.assertIn("jumpToPersistentHighlightTarget", source)
        self.assertIn("collectNearFocusWindows", source)
        self.assertIn("normalizedNearTermGroups", source)
        self.assertIn("collectSearchIndicatorEntriesForTerms", source)
        self.assertIn("SEARCH_INDICATOR_PUBLISH_INTERVAL_MS", source)
        self.assertIn("scheduleSearchIndicatorPublish", source)
        self.assertIn("persistentEntriesSignature", source)
        self.assertIn("searchRequestSignature", source)
        self.assertIn("isBridgeOverlayNode", source)
        self.assertIn("mutationNeedsHighlightRefresh", source)
        self.assertIn("some(mutationNeedsHighlightRefresh)", source)
        self.assertIn("viewRestoreId", source)
        self.assertIn("restoreIsCurrent", source)
        self.assertIn("__pdfexplore_user_activity__", source)
        self.assertIn('document.addEventListener("pointerdown"', source)

    def test_cached_badge_asset_exists(self) -> None:
        root = Path(__file__).resolve().parent.parent
        cached_badge = root / "pdfexplore" / "assets" / "cached.png"
        self.assertTrue(cached_badge.is_file())


if __name__ == "__main__":
    unittest.main()
