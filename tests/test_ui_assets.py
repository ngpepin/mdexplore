from __future__ import annotations

import unittest

from mdexplore_app.constants import PROJECT_ROOT, UI_ASSET_DIR
from mdexplore_app.icons import ui_asset_path


UI_ASSET_FILENAMES = (
    "LiberationSansNarrow-Regular.ttf",
    "home.svg",
    "home2.svg",
    "home3.png",
    "home3.svg",
    "markdown.svg",
    "marker.svg",
    "pin.png",
    "refresh.png",
    "refresh.svg",
    "search-hit.svg",
    "views.svg",
    "views2.svg",
)


class UiAssetLayoutTests(unittest.TestCase):
    def test_ui_assets_live_under_assets_ui(self) -> None:
        self.assertTrue(UI_ASSET_DIR.is_dir())
        for filename in UI_ASSET_FILENAMES:
            with self.subTest(filename=filename):
                self.assertTrue((UI_ASSET_DIR / filename).is_file())
                self.assertFalse((PROJECT_ROOT / filename).exists())
                self.assertEqual(ui_asset_path(filename), UI_ASSET_DIR / filename)

    def test_app_icon_remains_at_repo_root(self) -> None:
        self.assertTrue((PROJECT_ROOT / "mdexplor-icon.png").is_file())

