import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest

import hfind


class HfindCliTests(unittest.TestCase):
    def _run_main(self, args: list[str]) -> tuple[int, list[str]]:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = hfind.main(args)
        lines = [line.strip() for line in stream.getvalue().splitlines() if line.strip()]
        return code, lines

    def test_default_filename_only_search_ignores_content(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-filename-only-") as tmpdir:
            root = Path(tmpdir)
            (root / "paul_notes.txt").write_text("nothing special\n", encoding="utf-8")
            (root / "random.txt").write_text("contains fred in content\n", encoding="utf-8")

            previous = Path.cwd()
            os.chdir(root)
            try:
                code, lines = self._run_main([
                    "--query",
                    "OR(fred, paul)",
                    "*.txt",
                ])
            finally:
                os.chdir(previous)

            self.assertEqual(code, 0)
            self.assertEqual(lines, ["paul_notes.txt"])

    def test_recursive_content_search_with_stackable_short_flags(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-recursive-content-") as tmpdir:
            root = Path(tmpdir)
            (root / "top.txt").write_text("top level\n", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "inside.txt").write_text("this contains fred\n", encoding="utf-8")

            previous = Path.cwd()
            os.chdir(root)
            try:
                code, lines = self._run_main([
                    "-cr",
                    "OR(fred, paul)",
                    "*.txt",
                ])
            finally:
                os.chdir(previous)

            self.assertEqual(code, 0)
            self.assertEqual(lines, [str(Path("nested") / "inside.txt")])

    def test_implicit_query_when_q_missing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-implicit-query-") as tmpdir:
            root = Path(tmpdir)
            (root / "alpha.txt").write_text("paul in content\n", encoding="utf-8")

            previous = Path.cwd()
            os.chdir(root)
            try:
                code, lines = self._run_main([
                    "-rc",
                    "OR(fred, paul)",
                    "*.txt",
                ])
            finally:
                os.chdir(previous)

            self.assertEqual(code, 0)
            self.assertEqual(lines, ["alpha.txt"])

    def test_mixed_case_sensitivity_for_content(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-case-sensitive-") as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "docs"
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "one.md").write_text(
                "Fred is here and paul is here\n", encoding="utf-8"
            )
            (source_dir / "two.md").write_text(
                "fred is lowercase and paul is here\n", encoding="utf-8"
            )

            code, lines = self._run_main(
                [
                    "-q",
                    "AND('Fred',paul)",
                    "-c",
                    str(source_dir / "*.md"),
                ]
            )

            self.assertEqual(code, 0)
            self.assertEqual(lines, [str(source_dir / "one.md")])


if __name__ == "__main__":
    unittest.main()
