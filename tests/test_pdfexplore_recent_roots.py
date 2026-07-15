from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from pdfexplore.app import PdfExploreWindow, _default_root_from_config


class PdfExploreRecentRootHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="pdfexplore-recent-root-")
        self.base = Path(self._tempdir.name)
        self.initial_root = self.base / "root-0"
        self.initial_root.mkdir(parents=True, exist_ok=True)
        self.config_path = self.base / ".pdfexplore.cfg"
        self.window = PdfExploreWindow(
            root=self.initial_root,
            app_icon=QIcon(),
            config_path=self.config_path,
            gpu_context_available=False,
        )
        self.window.show()
        QApplication.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        QApplication.processEvents()
        self._tempdir.cleanup()

    def test_recent_roots_are_most_recent_first_and_capped(self) -> None:
        navigated_roots: list[Path] = []
        for idx in range(1, 39):
            target = self.base / f"root-{idx}"
            target.mkdir(parents=True, exist_ok=True)
            navigated_roots.append(target.resolve())
            self.window._recent_root_entered_at -= (
                self.window.MIN_RECENT_ROOT_DWELL_SECONDS + 1.0
            )
            self.window._set_root_directory(target)

        recent = list(self.window._recent_root_directories)
        self.assertEqual(len(recent), self.window.MAX_RECENT_ROOT_DIRECTORIES)
        self.assertNotIn(navigated_roots[-1], recent)
        self.assertEqual(recent[0], navigated_roots[-2])
        self.assertNotIn(self.initial_root.resolve(), recent)

    def test_recent_root_requires_minimum_dwell_before_recording(self) -> None:
        quick = self.base / "quick-root"
        quick.mkdir(parents=True, exist_ok=True)
        self.window._set_root_directory(quick)

        next_root = self.base / "next-root"
        next_root.mkdir(parents=True, exist_ok=True)
        self.window._set_root_directory(next_root)

        recent_keys = {
            self.window._path_key(path) for path in self.window._recent_root_directories
        }
        self.assertNotIn(self.window._path_key(quick), recent_keys)

    def test_recent_menu_reload_reads_latest_config(self) -> None:
        external = self.base / "external-root"
        external.mkdir(parents=True, exist_ok=True)
        payload = {
            self.window.CONFIG_DEFAULT_ROOT_KEY: str(external.resolve()),
            self.window.CONFIG_RECENT_ROOTS_KEY: [str(external.resolve())],
        }
        self.config_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        self.window._reload_recent_root_directories_before_menu_open()
        recent_keys = {
            self.window._path_key(path) for path in self.window._recent_root_directories
        }
        self.assertIn(self.window._path_key(external), recent_keys)

    def test_persist_effective_root_writes_json_with_recent_roots(self) -> None:
        target = self.base / "persisted-root"
        target.mkdir(parents=True, exist_ok=True)
        self.window._recent_root_entered_at -= (
            self.window.MIN_RECENT_ROOT_DWELL_SECONDS + 1.0
        )
        self.window._set_root_directory(target)

        end = self.base / "final-root"
        end.mkdir(parents=True, exist_ok=True)
        self.window._recent_root_entered_at -= (
            self.window.MIN_RECENT_ROOT_DWELL_SECONDS + 1.0
        )
        self.window._set_root_directory(end)

        self.window._persist_effective_root()
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))

        self.assertEqual(payload.get(self.window.CONFIG_DEFAULT_ROOT_KEY), str(end))
        self.assertIsInstance(payload.get(self.window.CONFIG_RECENT_ROOTS_KEY), list)
        self.assertGreaterEqual(len(payload[self.window.CONFIG_RECENT_ROOTS_KEY]), 1)
        self.assertEqual(payload[self.window.CONFIG_RECENT_ROOTS_KEY][0], str(target))

    def test_active_aged_config_lock_is_not_unlinked(self) -> None:
        lock_path = self.window._config_lock_file_path()
        lock_path.touch()
        old_time = time.time() - 125.0
        os.utime(lock_path, (old_time, old_time))
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            inode_before = lock_path.stat().st_ino
            self.window._read_config_payload()
            self.assertTrue(lock_path.exists())
            self.assertEqual(lock_path.stat().st_ino, inode_before)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def test_unlocked_aged_config_lock_inode_remains_stable(self) -> None:
        lock_path = self.window._config_lock_file_path()
        lock_path.touch()
        old_time = time.time() - 125.0
        os.utime(lock_path, (old_time, old_time))
        inode_before = lock_path.stat().st_ino

        self.window._read_config_payload()

        self.assertTrue(lock_path.exists())
        self.assertEqual(lock_path.stat().st_ino, inode_before)

    def test_local_recent_event_merges_with_newer_external_config(self) -> None:
        local = self.base / "local-root"
        external = self.base / "external-root"
        local.mkdir()
        external.mkdir()
        self.window._record_recent_root_directory(local)
        self.config_path.write_text(
            json.dumps(
                {
                    self.window.CONFIG_DEFAULT_ROOT_KEY: str(external),
                    self.window.CONFIG_RECENT_ROOTS_KEY: [str(external)],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        self.window._persist_effective_root()

        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload[self.window.CONFIG_RECENT_ROOTS_KEY][:2],
            [str(local.resolve()), str(external.resolve())],
        )

    def test_contended_config_save_retains_event_for_retry(self) -> None:
        local = self.base / "retry-root"
        local.mkdir()
        self.window._record_recent_root_directory(local)
        lock_path = self.window._config_lock_file_path()
        lock_path.touch()

        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.window._persist_effective_root()
            self.assertEqual(self.window._recent_root_events, [local])
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        self.window._persist_effective_root()

        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload[self.window.CONFIG_RECENT_ROOTS_KEY][0],
            str(local.resolve()),
        )
        self.assertEqual(self.window._recent_root_events, [])


class PdfExploreDefaultRootConfigTests(unittest.TestCase):
    def test_load_default_root_from_json_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdfexplore-runtime-cfg-") as tmpdir:
            home = Path(tmpdir)
            root = home / "pdfs"
            root.mkdir(parents=True, exist_ok=True)
            cfg = home / ".pdfexplore.cfg"
            cfg.write_text(
                json.dumps(
                    {
                        "default_root": str(root.resolve()),
                        "recent_roots": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(PdfExploreWindow, "_config_file_path", return_value=cfg):
                loaded = _default_root_from_config()
            self.assertEqual(loaded, root.resolve())


if __name__ == "__main__":
    unittest.main()
