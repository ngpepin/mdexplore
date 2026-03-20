import contextlib
import io
import os
from pathlib import Path
import re
import tempfile
import unittest

import hfind


class HfindCliTests(unittest.TestCase):
    ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

    @classmethod
    def _strip_ansi(cls, text: str) -> str:
        return cls.ANSI_ESCAPE_RE.sub("", text)

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
            self.assertEqual([self._strip_ansi(line) for line in lines], ["paul_notes.txt"])

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
            self.assertEqual(
                [self._strip_ansi(line) for line in lines],
                [str(Path("nested") / "inside.txt")],
            )

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
            self.assertEqual([self._strip_ansi(line) for line in lines], ["alpha.txt"])

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
            self.assertEqual(
                [self._strip_ansi(line) for line in lines],
                [str(source_dir / "one.md")],
            )

    def test_help_short_flag_prints_usage_and_exits_zero(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as raised:
                hfind._parse_args(["-h"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("Usage:", out.getvalue())

    def test_invalid_args_include_usage_text(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            hfind._parse_args(["--query"])
        self.assertIn("error: --query requires a value", str(raised.exception))
        self.assertIn("Usage:", str(raised.exception))

    def test_verbose_lists_matching_lines_with_highlights(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-verbose-lines-") as tmpdir:
            root = Path(tmpdir)
            source = root / "notes.md"
            source.write_text(
                "alpha line\n"
                "Beta has pipelines\n"
                "closing line\n",
                encoding="utf-8",
            )

            code, lines = self._run_main([
                "-v",
                "pipelines",
                str(root / "*.md"),
            ])

            self.assertEqual(code, 0)
            self.assertEqual(self._strip_ansi(lines[0]), str(source))
            self.assertTrue(any("2:" in line for line in lines[1:]))
            self.assertTrue(any("\x1b[33m" in line for line in lines[1:]))

    def test_verbose_can_stack_with_content_and_recursive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-verbose-stack-") as tmpdir:
            root = Path(tmpdir)
            nested = root / "a" / "b"
            nested.mkdir(parents=True, exist_ok=True)
            source = nested / "doc.txt"
            source.write_text(
                "nothing on this line\n"
                "alpha appears here\n"
                "and beta appears there\n",
                encoding="utf-8",
            )

            previous = Path.cwd()
            os.chdir(root)
            try:
                code, lines = self._run_main([
                    "-crv",
                    "NEAR(alpha,beta)",
                    "*.txt",
                ])
            finally:
                os.chdir(previous)

            self.assertEqual(code, 0)
            self.assertEqual(self._strip_ansi(lines[0]), str(Path("a") / "b" / "doc.txt"))
            self.assertTrue(any("2:" in line for line in lines[1:]))
            self.assertFalse(any("3:" in line for line in lines[1:]))
            self.assertTrue(any("\x1b[33malpha\x1b[0m" in line for line in lines[1:]))
            self.assertTrue(any("\x1b[33mbeta\x1b[0m" in line for line in lines[1:]))

    def test_filepath_output_uses_bold_green_style(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-path-style-") as tmpdir:
            root = Path(tmpdir)
            source = root / "demo.txt"
            source.write_text("hello\n", encoding="utf-8")

            code, lines = self._run_main([
                "demo",
                str(root / "*.txt"),
            ])

            self.assertEqual(code, 0)
            self.assertEqual(self._strip_ansi(lines[0]), str(source))
            self.assertTrue(lines[0].startswith("\x1b[1;32m"))
            self.assertTrue(lines[0].endswith("\x1b[0m"))

    def test_verbose_near_is_strict_to_near_windows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-near-strict-") as tmpdir:
            root = Path(tmpdir)
            source = root / "strict.md"
            source.write_text(
                "nicolas appears near the top\n"
                + " ".join(["gap"] * 60)
                + "\n"
                "email npepin@umiquity.com is far away\n",
                encoding="utf-8",
            )

            code, lines = self._run_main([
                "-cv",
                "NEAR(nicolas,pepin)",
                str(root / "*.md"),
            ])

            self.assertEqual(code, 1)
            self.assertEqual(lines, [])

    def test_verbose_near_contiguous_lines_only_number_first_line(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-near-contiguous-") as tmpdir:
            root = Path(tmpdir)
            source = root / "window.md"
            source.write_text(
                "nicolas appears first\n"
                "bridge line with no match\n"
                "pepin appears next\n",
                encoding="utf-8",
            )

            code, lines = self._run_main([
                "-cv",
                "NEAR(nicolas,pepin)",
                str(root / "*.md"),
            ])

            self.assertEqual(code, 0)
            self.assertEqual(self._strip_ansi(lines[0]), str(source))
            self.assertTrue(self._strip_ansi(lines[1]).startswith("1: "))
            self.assertNotIn(":", self._strip_ansi(lines[2]).split(" ", 1)[0])
            self.assertNotIn(":", self._strip_ansi(lines[3]).split(" ", 1)[0])

    def test_verbose_near_self_contained_contiguous_lines_are_each_numbered(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-near-self-contained-") as tmpdir:
            root = Path(tmpdir)
            source = root / "table.md"
            source.write_text(
                "nicolas pepin row one\n"
                "nicolas pepin row two\n"
                "nicolas pepin row three\n",
                encoding="utf-8",
            )

            code, lines = self._run_main([
                "-cv",
                "NEAR(nicolas,pepin)",
                str(root / "*.md"),
            ])

            self.assertEqual(code, 0)
            self.assertEqual(self._strip_ansi(lines[0]), str(source))
            self.assertTrue(self._strip_ansi(lines[1]).startswith("1: "))
            self.assertTrue(self._strip_ansi(lines[2]).startswith("2: "))
            self.assertTrue(self._strip_ansi(lines[3]).startswith("3: "))


if __name__ == "__main__":
    unittest.main()
