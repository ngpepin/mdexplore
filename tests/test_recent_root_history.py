from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

import mdexplore
from mdexplore_app import runtime as runtime_helpers


class RecentRootHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="mdexplore-recent-root-")
        self.base = Path(self._tempdir.name)
        self.initial_root = self.base / "root-0"
        self.initial_root.mkdir(parents=True, exist_ok=True)
        self.config_path = self.base / ".mdexplore.cfg"
        self.window = mdexplore.MdExploreWindow(
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
        for idx in range(1, 19):
            target = self.base / f"root-{idx}"
            target.mkdir(parents=True, exist_ok=True)
            navigated_roots.append(target.resolve())
            self.window._recent_root_entered_at -= (
                self.window.MIN_RECENT_ROOT_DWELL_SECONDS + 1.0
            )
            self.window._set_root_directory(target)

        recent = list(self.window._recent_root_directories)
        self.assertEqual(len(recent), self.window.MAX_RECENT_ROOT_DIRECTORIES)
        # The current root is not recorded until user leaves it.
        self.assertNotIn(navigated_roots[-1], recent)
        self.assertEqual(recent[0], navigated_roots[-2])
        self.assertNotIn(navigated_roots[0], recent)

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
        self.window._copy_base64_images_enabled = True

        self.window._persist_effective_root()
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))

        self.assertEqual(payload.get(self.window.CONFIG_DEFAULT_ROOT_KEY), str(end))
        self.assertIsInstance(payload.get(self.window.CONFIG_RECENT_ROOTS_KEY), list)
        self.assertGreaterEqual(len(payload[self.window.CONFIG_RECENT_ROOTS_KEY]), 1)
        self.assertEqual(payload[self.window.CONFIG_RECENT_ROOTS_KEY][0], str(target))
        self.assertEqual(
            payload.get(self.window.CONFIG_COPY_BASE64_IMAGES_ENABLED_KEY), True
        )

    def test_load_copy_base64_toggle_state_from_config(self) -> None:
        payload = {
            self.window.CONFIG_DEFAULT_ROOT_KEY: str(self.initial_root.resolve()),
            self.window.CONFIG_RECENT_ROOTS_KEY: [],
            self.window.CONFIG_COPY_BASE64_IMAGES_ENABLED_KEY: True,
        }
        self.config_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        self.assertTrue(self.window._load_copy_base64_toggle_from_config())


class RuntimeConfigPayloadTests(unittest.TestCase):
    def test_load_default_root_from_json_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mdexplore-runtime-cfg-") as tmpdir:
            home = Path(tmpdir)
            root = home / "notes"
            root.mkdir(parents=True, exist_ok=True)
            cfg = home / ".mdexplore.cfg"
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
            with patch("pathlib.Path.home", return_value=home):
                loaded = runtime_helpers.load_default_root_from_config()
            self.assertEqual(loaded, root.resolve())


if __name__ == "__main__":
    unittest.main()
