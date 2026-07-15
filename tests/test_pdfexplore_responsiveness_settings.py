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
        self.assertIs(APP_SETTINGS.get("prefetch_enabled"), True)
        self.assertEqual(APP_SETTINGS.get("prefetch_idle_seconds"), 10.0)
        self.assertEqual(APP_SETTINGS.get("prefetch_batch_cooldown_seconds"), 1.0)
        self.assertEqual(
            APP_SETTINGS.get("pdf_text_disk_cache_touch_interval_seconds"),
            60.0,
        )
        self.assertEqual(
            APP_SETTINGS.get("multi_instance_activity_touch_interval_seconds"),
            0.25,
        )
        self.assertEqual(
            APP_SETTINGS.get("multi_instance_activity_probe_interval_seconds"),
            0.5,
        )

        self.assertEqual(PdfExploreWindow.SEARCH_THREAD_POOL_MAX_THREADS, 1)
        self.assertEqual(PdfExploreWindow.SEARCH_WORKER_CHUNK_SIZE, 8)
        self.assertEqual(PdfExploreWindow.SEARCH_PROGRESS_PUBLISH_INTERVAL_MS, 100)
        self.assertEqual(PdfExploreWindow.PREVIEW_WIDGET_CACHE_MAX_ENTRIES, 2)
        self.assertIs(PdfExploreWindow.PREFETCH_ENABLED, True)
        self.assertEqual(PdfExploreWindow.PREFETCH_IDLE_SECONDS, 10.0)
        self.assertEqual(PdfExploreWindow.PREFETCH_BATCH_COOLDOWN_SECONDS, 1.0)
        self.assertEqual(
            PdfExploreWindow.PDF_TEXT_DISK_CACHE_TOUCH_INTERVAL_SECONDS,
            60.0,
        )
        self.assertEqual(
            PdfExploreWindow.MULTI_INSTANCE_ACTIVITY_TOUCH_INTERVAL_SECONDS,
            0.25,
        )
        self.assertEqual(
            PdfExploreWindow.MULTI_INSTANCE_ACTIVITY_PROBE_INTERVAL_SECONDS,
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
