from __future__ import annotations

import unittest

from pdfexplore.app import PdfExploreWindow
from pdfexplore.settings import APP_SETTINGS


class PdfExploreResponsivenessSettingsTests(unittest.TestCase):
    def test_responsiveness_defaults_are_conservative_and_bounded(self) -> None:
        self.assertEqual(APP_SETTINGS.get("search_thread_pool_max_threads"), 1)
        self.assertEqual(APP_SETTINGS.get("search_worker_chunk_size"), 8)
        self.assertEqual(APP_SETTINGS.get("search_progress_publish_interval_ms"), 100)
        self.assertEqual(APP_SETTINGS.get("preview_widget_cache_max_entries"), 2)
        self.assertIs(APP_SETTINGS.get("prefetch_enabled"), False)

        self.assertEqual(PdfExploreWindow.SEARCH_THREAD_POOL_MAX_THREADS, 1)
        self.assertEqual(PdfExploreWindow.SEARCH_WORKER_CHUNK_SIZE, 8)
        self.assertEqual(PdfExploreWindow.SEARCH_PROGRESS_PUBLISH_INTERVAL_MS, 100)
        self.assertEqual(PdfExploreWindow.PREVIEW_WIDGET_CACHE_MAX_ENTRIES, 2)
        self.assertIs(PdfExploreWindow.PREFETCH_ENABLED, False)


if __name__ == "__main__":
    unittest.main()
