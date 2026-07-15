from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from mdexplore_app.file_coordination import advisory_file_lock
from pdfexplore.workers import BACKGROUND_LEASE_BUSY, PdfTextPrefetchWorker


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

    def test_cross_process_lease_allows_only_one_background_worker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdfexplore-prefetch-lease-") as tmpdir:
            root = Path(tmpdir)
            first_path = root / "first.pdf"
            second_path = root / "second.pdf"
            first_path.write_bytes(b"pdf")
            second_path.write_bytes(b"pdf")
            lease_path = root / "background.lock"
            entered = threading.Event()
            release = threading.Event()
            first_calls: list[Path] = []
            second_calls: list[Path] = []

            def first_loader(path: Path) -> str:
                first_calls.append(path)
                entered.set()
                release.wait(5.0)
                return "cached"

            first = PdfTextPrefetchWorker(
                21,
                [first_path],
                first_loader,
                lease_lock_path=lease_path,
            )
            second = PdfTextPrefetchWorker(
                22,
                [second_path],
                lambda path: second_calls.append(path) or "cached",
                lease_lock_path=lease_path,
            )
            second_result: dict[str, object] = {}
            second.signals.finished.connect(
                lambda request_id, prefetched, skipped, error: second_result.update(
                    request_id=request_id,
                    prefetched=prefetched,
                    skipped=skipped,
                    error=error,
                )
            )
            thread = threading.Thread(target=first.run)
            thread.start()
            self.assertTrue(entered.wait(3.0))
            second.run()
            release.set()
            thread.join(5.0)

            self.assertEqual(first_calls, [first_path])
            self.assertEqual(second_calls, [])
            self.assertEqual(second_result.get("request_id"), 22)
            self.assertEqual(second_result.get("error"), BACKGROUND_LEASE_BUSY)

    def test_finished_signal_runs_after_successful_lease_release(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdfexplore-prefetch-release-") as tmpdir:
            root = Path(tmpdir)
            source = root / "document.pdf"
            source.write_bytes(b"pdf")
            lease_path = root / "background.lock"
            lease_available_in_callback: list[bool] = []

            worker = PdfTextPrefetchWorker(
                23,
                [source],
                lambda _path: "cached",
                lease_lock_path=lease_path,
            )

            def _on_finished(
                _request_id: int,
                _prefetched: int,
                _skipped: int,
                _error: str,
            ) -> None:
                with advisory_file_lock(
                    lease_path,
                    exclusive=True,
                    blocking=False,
                ) as acquired:
                    lease_available_in_callback.append(acquired)

            worker.signals.finished.connect(_on_finished)
            worker.run()

            self.assertEqual(lease_available_in_callback, [True])

    def test_finished_signal_is_emitted_when_lease_setup_raises(self) -> None:
        worker = PdfTextPrefetchWorker(
            24,
            [Path("document.pdf")],
            lambda _path: "cached",
            lease_lock_path=Path("background.lock"),
        )
        emitted: dict[str, object] = {}
        worker.signals.finished.connect(
            lambda request_id, prefetched, skipped, error: emitted.update(
                request_id=request_id,
                prefetched=prefetched,
                skipped=skipped,
                error=error,
            )
        )

        with patch(
            "pdfexplore.workers.advisory_file_lock",
            side_effect=OSError("lease setup failed"),
        ):
            worker.run()

        self.assertEqual(emitted.get("request_id"), 24)
        self.assertEqual(emitted.get("prefetched"), 0)
        self.assertEqual(emitted.get("skipped"), 0)
        self.assertIn("lease setup failed", str(emitted.get("error")))


if __name__ == "__main__":
    unittest.main()
