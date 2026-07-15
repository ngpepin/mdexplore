from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest

from mdexplore_app.file_coordination import (
    advisory_file_lock,
    load_files_payload,
    update_files_sidecar,
)


def _update_sidecar_process(
    path_text: str,
    name: str,
    value: str,
    start_event,
) -> None:
    start_event.wait(5.0)
    update_files_sidecar(Path(path_text), {name: value})


def _hold_advisory_lock_process(
    lock_path_text: str,
    ready_event,
    release_event,
) -> None:
    with advisory_file_lock(Path(lock_path_text), blocking=True) as acquired:
        if acquired:
            ready_event.set()
            release_event.wait(10.0)


class FileCoordinationTests(unittest.TestCase):
    def test_two_processes_merge_distinct_sidecar_entries(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sidecar-process-merge-") as tmpdir:
            path = Path(tmpdir) / ".pdfexplore-highlighting.json"
            context = multiprocessing.get_context("spawn")
            start = context.Event()
            processes = [
                context.Process(
                    target=_update_sidecar_process,
                    args=(str(path), "a.pdf", "one", start),
                ),
                context.Process(
                    target=_update_sidecar_process,
                    args=(str(path), "b.pdf", "two", start),
                ),
            ]
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(10.0)
                self.assertEqual(process.exitcode, 0)

            self.assertEqual(
                load_files_payload(path),
                {"a.pdf": "one", "b.pdf": "two"},
            )
            self.assertEqual(json.loads(path.read_text())["files"]["a.pdf"], "one")
            self.assertFalse(any(path.parent.glob(f".{path.name}.tmp.*")))

    def test_tombstone_removes_only_requested_entry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sidecar-tombstone-") as tmpdir:
            path = Path(tmpdir) / ".pdfexplore-views.json"
            update_files_sidecar(path, {"a.pdf": {"value": 1}, "b.pdf": {"value": 2}})
            committed = update_files_sidecar(path, {"a.pdf": None})
            self.assertEqual(committed, {"b.pdf": {"value": 2}})

    def test_nonblocking_lock_excludes_other_process_and_releases_on_exit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="advisory-process-lock-") as tmpdir:
            lock_path = Path(tmpdir) / "shared.lock"
            context = multiprocessing.get_context("spawn")
            ready = context.Event()
            release = context.Event()
            process = context.Process(
                target=_hold_advisory_lock_process,
                args=(str(lock_path), ready, release),
            )
            process.start()
            self.assertTrue(ready.wait(5.0))
            with advisory_file_lock(lock_path, blocking=False) as acquired:
                self.assertFalse(acquired)
            release.set()
            process.join(10.0)
            self.assertEqual(process.exitcode, 0)
            with advisory_file_lock(lock_path, blocking=False) as acquired:
                self.assertTrue(acquired)

    def test_cooperative_lock_wait_stops_when_aborted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="advisory-abort-") as tmpdir:
            lock_path = Path(tmpdir) / "shared.lock"
            context = multiprocessing.get_context("spawn")
            ready = context.Event()
            release = context.Event()
            process = context.Process(
                target=_hold_advisory_lock_process,
                args=(str(lock_path), ready, release),
            )
            process.start()
            self.assertTrue(ready.wait(5.0))
            started = time.monotonic()
            with advisory_file_lock(
                lock_path,
                should_abort=lambda: time.monotonic() - started > 0.05,
            ) as acquired:
                self.assertFalse(acquired)
            self.assertLess(time.monotonic() - started, 0.5)
            release.set()
            process.join(10.0)
            self.assertEqual(process.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
