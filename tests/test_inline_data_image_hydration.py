from __future__ import annotations

import unittest

import mdexplore


class _FakePage:
    def __init__(self, results: list[object]) -> None:
        self.results = list(results)
        self.scripts: list[str] = []

    def runJavaScript(self, script: str, callback) -> None:
        self.scripts.append(script)
        result = self.results.pop(0) if self.results else None
        callback(result)


class _FakePreview:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    def page(self) -> _FakePage:
        return self._page


class _InlineImageHarness:
    _apply_inline_data_image_urls_to_preview = (
        mdexplore.MdExploreWindow._apply_inline_data_image_urls_to_preview
    )
    _flush_pending_inline_data_image_urls_for_current_preview = (
        mdexplore.MdExploreWindow._flush_pending_inline_data_image_urls_for_current_preview
    )

    def __init__(self, path_key: str, results: list[object]) -> None:
        self.path_key = path_key
        self.preview = _FakePreview(_FakePage(results))
        self._inline_data_image_pending_apply_urls_by_path_key: dict[
            str, dict[str, str]
        ] = {}

    def _current_preview_path_key(self) -> str:
        return self.path_key


class InlineDataImageHydrationTests(unittest.TestCase):
    def test_flush_retains_urls_when_current_dom_has_no_placeholders(self) -> None:
        harness = _InlineImageHarness(
            "/tmp/doc.md",
            [{"updated": 0, "appliedDigests": []}],
        )
        harness._inline_data_image_pending_apply_urls_by_path_key["/tmp/doc.md"] = {
            "digest-a": "file:///tmp/a.png"
        }

        harness._flush_pending_inline_data_image_urls_for_current_preview()

        self.assertEqual(
            harness._inline_data_image_pending_apply_urls_by_path_key,
            {"/tmp/doc.md": {"digest-a": "file:///tmp/a.png"}},
        )

    def test_flush_removes_only_digests_confirmed_present_in_dom(self) -> None:
        harness = _InlineImageHarness(
            "/tmp/doc.md",
            [{"updated": 1, "appliedDigests": ["digest-a"]}],
        )
        harness._inline_data_image_pending_apply_urls_by_path_key["/tmp/doc.md"] = {
            "digest-a": "file:///tmp/a.png",
            "digest-b": "file:///tmp/b.png",
        }

        harness._flush_pending_inline_data_image_urls_for_current_preview()

        self.assertEqual(
            harness._inline_data_image_pending_apply_urls_by_path_key,
            {"/tmp/doc.md": {"digest-b": "file:///tmp/b.png"}},
        )

    def test_later_load_can_apply_url_retained_from_early_flush(self) -> None:
        harness = _InlineImageHarness(
            "/tmp/doc.md",
            [
                {"updated": 0, "appliedDigests": []},
                {"updated": 1, "appliedDigests": ["digest-a"]},
            ],
        )
        harness._inline_data_image_pending_apply_urls_by_path_key["/tmp/doc.md"] = {
            "digest-a": "file:///tmp/a.png"
        }

        harness._flush_pending_inline_data_image_urls_for_current_preview()
        harness._flush_pending_inline_data_image_urls_for_current_preview()

        self.assertNotIn(
            "/tmp/doc.md",
            harness._inline_data_image_pending_apply_urls_by_path_key,
        )
        self.assertEqual(len(harness.preview.page().scripts), 2)


if __name__ == "__main__":
    unittest.main()
