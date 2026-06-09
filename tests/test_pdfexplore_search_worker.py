from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pdfexplore.workers import PdfSearchWorker


class PdfSearchWorkerAbortTests(unittest.TestCase):
    def test_worker_stops_early_when_request_is_aborted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdfexplore-worker-abort-") as tmpdir:
            root = Path(tmpdir)
            first = root / "first.pdf"
            second = root / "second.pdf"
            first.write_text("dummy", encoding="utf-8")
            second.write_text("dummy", encoding="utf-8")

            calls: list[Path] = []
            abort_flag = {"value": False}

            def content_loader(path: Path) -> str:
                calls.append(path)
                # Flip abort flag after first file load to emulate a stale request.
                if len(calls) == 1:
                    abort_flag["value"] = True
                return "needle"

            worker = PdfSearchWorker(
                request_id=7,
                paths=[first, second],
                predicate=lambda _filename, content: "needle" in content,
                hit_counter=lambda _filename, _content: 1,
                filename_patterns=[],
                content_loader=content_loader,
                should_abort=lambda: abort_flag["value"],
            )

            emitted: dict[str, object] = {}

            def _on_finished(
                request_id: int,
                matched_paths,
                match_counts,
                filename_match_paths,
                error_text: str,
            ) -> None:
                emitted["request_id"] = request_id
                emitted["matched_paths"] = list(matched_paths)
                emitted["match_counts"] = dict(match_counts)
                emitted["filename_match_paths"] = list(filename_match_paths)
                emitted["error_text"] = error_text

            worker.signals.finished.connect(_on_finished)
            worker.run()

            self.assertEqual(emitted.get("request_id"), 7)
            self.assertEqual(emitted.get("error_text"), "")
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(emitted.get("matched_paths", [])), 1)


if __name__ == "__main__":
    unittest.main()
