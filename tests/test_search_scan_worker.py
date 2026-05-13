import tempfile
import unittest
from pathlib import Path

from mdexplore_app import search as search_query
from mdexplore_app.workers import SearchScanWorker


class SearchScanWorkerFilenameSurfaceTests(unittest.TestCase):
    def _run_worker_once(self, query: str, target: Path):
        predicate = search_query.compile_match_predicate(
            query,
            strip_inline_image_data=False,
        )
        hit_counter = search_query.compile_match_hit_counter(
            query,
            strip_inline_image_data=False,
        )
        worker = SearchScanWorker(
            request_id=1,
            paths=[target],
            predicate=predicate,
            hit_counter=hit_counter,
            filename_patterns=[],
        )

        emitted = {}

        def _on_finished(
            request_id: int,
            matched_paths,
            match_counts,
            filename_match_paths,
            error_text: str,
        ) -> None:
            emitted["request_id"] = request_id
            emitted["matched_paths"] = matched_paths
            emitted["match_counts"] = match_counts
            emitted["filename_match_paths"] = filename_match_paths
            emitted["error_text"] = error_text

        worker.signals.finished.connect(_on_finished)
        worker.run()
        return emitted

    def test_extension_text_is_excluded_from_filename_query_surface(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mdexplore-search-worker-stem-") as tmpdir:
            root = Path(tmpdir)
            target = root / "alpha.md"
            target.write_text("content without target token\n", encoding="utf-8")

            result = self._run_worker_once("md", target)

            self.assertEqual(result["error_text"], "")
            self.assertEqual(result["matched_paths"], [])
            self.assertEqual(result["match_counts"], {})

    def test_boolean_and_uses_same_stem_surface_as_near(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mdexplore-search-worker-bool-") as tmpdir:
            root = Path(tmpdir)
            target = root / "alpha.md"
            target.write_text("contains beta token\n", encoding="utf-8")

            result = self._run_worker_once("md AND beta", target)

            self.assertEqual(result["error_text"], "")
            self.assertEqual(result["matched_paths"], [])
            self.assertEqual(result["match_counts"], {})


if __name__ == "__main__":
    unittest.main()
