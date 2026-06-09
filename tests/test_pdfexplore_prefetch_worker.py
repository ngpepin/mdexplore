from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pdfexplore.workers import PdfTextPrefetchWorker


class PdfTextPrefetchWorkerTests(unittest.TestCase):
    def test_prefetch_worker_processes_all_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdfexplore-prefetch-all-") as tmpdir:
            root = Path(tmpdir)
            paths = [root / f"doc-{index}.pdf" for index in range(3)]
            for path in paths:
                path.write_text("dummy", encoding="utf-8")

            calls: list[Path] = []

            def loader(path: Path) -> str:
                calls.append(path)
                return "cached text"

            worker = PdfTextPrefetchWorker(
                request_id=11,
                paths=paths,
                content_loader=loader,
                should_abort=lambda: False,
            )

            emitted: dict[str, object] = {}

            def _on_finished(
                request_id: int,
                prefetched_count: int,
                skipped_count: int,
                error_text: str,
            ) -> None:
                emitted["request_id"] = request_id
                emitted["prefetched_count"] = prefetched_count
                emitted["skipped_count"] = skipped_count
                emitted["error_text"] = error_text

            worker.signals.finished.connect(_on_finished)
            worker.run()

            self.assertEqual(emitted.get("request_id"), 11)
            self.assertEqual(emitted.get("prefetched_count"), 3)
            self.assertEqual(emitted.get("skipped_count"), 0)
            self.assertEqual(emitted.get("error_text"), "")
            self.assertEqual(calls, paths)

    def test_prefetch_worker_stops_when_aborted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdfexplore-prefetch-abort-") as tmpdir:
            root = Path(tmpdir)
            first = root / "first.pdf"
            second = root / "second.pdf"
            first.write_text("dummy", encoding="utf-8")
            second.write_text("dummy", encoding="utf-8")

            calls: list[Path] = []
            abort_state = {"value": False}

            def loader(path: Path) -> str:
                calls.append(path)
                abort_state["value"] = True
                return "cached text"

            worker = PdfTextPrefetchWorker(
                request_id=12,
                paths=[first, second],
                content_loader=loader,
                should_abort=lambda: abort_state["value"],
            )

            emitted: dict[str, object] = {}

            def _on_finished(
                request_id: int,
                prefetched_count: int,
                skipped_count: int,
                error_text: str,
            ) -> None:
                emitted["request_id"] = request_id
                emitted["prefetched_count"] = prefetched_count
                emitted["skipped_count"] = skipped_count
                emitted["error_text"] = error_text

            worker.signals.finished.connect(_on_finished)
            worker.run()

            self.assertEqual(emitted.get("request_id"), 12)
            self.assertEqual(emitted.get("prefetched_count"), 1)
            self.assertEqual(emitted.get("skipped_count"), 0)
            self.assertEqual(emitted.get("error_text"), "")
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
