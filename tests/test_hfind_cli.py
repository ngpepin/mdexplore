import contextlib
import io
import os
from pathlib import Path
import re
import tempfile
import unittest

import hfind


def _create_pdf_with_text(path: Path, text: str) -> None:
    from reportlab.pdfgen import canvas

    writer = canvas.Canvas(str(path))
    writer.drawString(72, 720, text)
    writer.save()


class HfindCliTests(unittest.TestCase):
    ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
    OSC8_ESCAPE_RE = re.compile(r"\x1b\]8;;[^\x1b\x07]*(?:\x1b\\|\x07)")

    @classmethod
    def _strip_ansi(cls, text: str) -> str:
        cleaned = cls.OSC8_ESCAPE_RE.sub("", text)
        return cls.ANSI_ESCAPE_RE.sub("", cleaned)

    def _run_main(self, args: list[str]) -> tuple[int, list[str]]:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = hfind.main(args)
        lines = [line.strip() for line in stream.getvalue().splitlines() if line.strip()]
        return code, lines

    def _run_main_with_stderr(
        self, args: list[str]
    ) -> tuple[int, list[str], list[str]]:
        stdout_stream = io.StringIO()
        stderr_stream = io.StringIO()
        with contextlib.redirect_stdout(stdout_stream), contextlib.redirect_stderr(
            stderr_stream
        ):
            code = hfind.main(args)
        stdout_lines = [
            line.strip() for line in stdout_stream.getvalue().splitlines() if line.strip()
        ]
        stderr_lines = [
            line.strip() for line in stderr_stream.getvalue().splitlines() if line.strip()
        ]
        return code, stdout_lines, stderr_lines

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

    def test_content_search_ignores_inline_image_base64_data(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-base64-ignore-") as tmpdir:
            root = Path(tmpdir)
            base64_only = root / "base64.md"
            base64_only.write_text(
                "![img](data:image/png;base64,AAAAAniCoBBBB)\n",
                encoding="utf-8",
            )
            visible = root / "visible.md"
            visible.write_text("Nico appears in visible text\n", encoding="utf-8")

            code, lines = self._run_main([
                "-c",
                "Nico",
                str(root / "*.md"),
            ])

            self.assertEqual(code, 0)
            self.assertEqual([self._strip_ansi(line) for line in lines], [str(visible)])

    def test_verbose_ignores_inline_image_base64_data_for_line_hits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-base64-verbose-") as tmpdir:
            root = Path(tmpdir)
            source = root / "nico.md"
            source.write_text(
                "![img](data:image/png;base64,AAAAAniCoBBBB)\n",
                encoding="utf-8",
            )

            code, lines = self._run_main([
                "-cv",
                "Nico",
                str(root / "*.md"),
            ])

            self.assertEqual(code, 0)
            self.assertEqual(self._strip_ansi(lines[0]), str(source))
            self.assertEqual(self._strip_ansi(lines[1]), "(filename match only)")
            self.assertEqual(len(lines), 2)

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

    def test_missing_pattern_defaults_to_current_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-default-pattern-") as tmpdir:
            root = Path(tmpdir)
            (root / "john.md").write_text("x\n", encoding="utf-8")
            (root / "sarah.md").write_text("x\n", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "john_nested.md").write_text("x\n", encoding="utf-8")

            previous = Path.cwd()
            os.chdir(root)
            try:
                code, lines = self._run_main([
                    "john",
                ])
            finally:
                os.chdir(previous)

            self.assertEqual(code, 0)
            stripped = [self._strip_ansi(line) for line in lines]
            self.assertEqual(stripped, ["john.md"])

    def test_missing_pattern_defaults_recursive_with_r(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-default-pattern-r-") as tmpdir:
            root = Path(tmpdir)
            (root / "john.md").write_text("x\n", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "john_nested.md").write_text("x\n", encoding="utf-8")

            previous = Path.cwd()
            os.chdir(root)
            try:
                code, lines = self._run_main([
                    "-r",
                    "john",
                ])
            finally:
                os.chdir(previous)

            self.assertEqual(code, 0)
            stripped = [self._strip_ansi(line) for line in lines]
            self.assertCountEqual(
                stripped,
                ["john.md", str(Path("nested") / "john_nested.md")],
            )

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
                "-cv",
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

    def test_filepath_output_uses_bold_purple_style(self) -> None:
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
            self.assertIn("\x1b[1;35m", lines[0])
            self.assertIn("\x1b[0m", lines[0])

    def test_filepath_hyperlink_uri_encodes_spaces(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-path-space-") as tmpdir:
            root = Path(tmpdir)
            source = root / "demo file.txt"
            source.write_text("hello\n", encoding="utf-8")

            code, lines = self._run_main([
                "demo",
                str(root / "*.txt"),
            ])

            self.assertEqual(code, 0)
            self.assertIn("%20", lines[0])
            self.assertEqual(self._strip_ansi(lines[0]), str(source))

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

    def test_binary_files_are_not_skipped_in_filename_only_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-binary-skip-") as tmpdir:
            root = Path(tmpdir)
            text_file = root / "pepin-notes.txt"
            text_file.write_text("plain text\n", encoding="utf-8")

            binary_file = root / "pepin-binary.dat"
            binary_file.write_bytes(
                b"\x00\x01\x02\x03" + bytes(range(255, 200, -1)) * 64
            )

            code, lines = self._run_main([
                "pepin",
                str(root / "*"),
            ])

            self.assertEqual(code, 0)
            self.assertCountEqual(
                [self._strip_ansi(line) for line in lines],
                [str(binary_file), str(text_file)],
            )

    def test_binary_files_are_skipped_when_content_mode_enabled(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-binary-content-") as tmpdir:
            root = Path(tmpdir)
            text_file = root / "notes.txt"
            text_file.write_text("contains pepin\n", encoding="utf-8")

            binary_file = root / "blob.bin"
            binary_file.write_bytes(b"\x00\x01\x02\x03" + bytes(range(255, 200, -1)) * 64)

            code, lines = self._run_main([
                "-c",
                "pepin",
                str(root / "*"),
            ])

            self.assertEqual(code, 0)
            self.assertEqual([self._strip_ansi(line) for line in lines], [str(text_file)])

    def test_content_mode_still_matches_binary_filename(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-binary-filename-content-") as tmpdir:
            root = Path(tmpdir)
            binary_file = root / "conflicted-copy.mdb"
            binary_file.write_bytes(
                b"\x00\x01\x02\x03" + bytes(range(255, 200, -1)) * 64
            )

            code, lines = self._run_main([
                "-c",
                "conflicted",
                str(root / "*"),
            ])

            self.assertEqual(code, 0)
            self.assertEqual([self._strip_ansi(line) for line in lines], [str(binary_file)])

    def test_pdf_files_are_ignored_without_pdf_flag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-pdf-no-flag-") as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "resume.pdf"
            _create_pdf_with_text(pdf_path, "Nicolas Pepin")

            code, lines = self._run_main([
                "pepin",
                str(root / "*.pdf"),
            ])

            self.assertEqual(code, 1)
            self.assertEqual(lines, [])

    def test_pdf_files_are_searchable_with_pdf_flag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-pdf-flag-") as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "resume.pdf"
            _create_pdf_with_text(pdf_path, "Nicolas Pepin")

            code, lines = self._run_main([
                "-cp",
                "pepin",
                str(root / "*.pdf"),
            ])

            self.assertEqual(code, 0)
            self.assertEqual([self._strip_ansi(line) for line in lines], [str(pdf_path)])

    def test_pdf_flag_matches_uppercase_pdf_extension_with_lowercase_pattern(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-pdf-case-") as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "resume.PDF"
            _create_pdf_with_text(pdf_path, "contains the keyword")

            code, lines = self._run_main([
                "-cp",
                "the",
                str(root / "*.pdf"),
            ])

            self.assertEqual(code, 0)
            self.assertEqual([self._strip_ansi(line) for line in lines], [str(pdf_path)])

    def test_pdf_extraction_failure_still_allows_filename_matching(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-pdf-fallback-") as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "the-reference.pdf"
            pdf_path.write_bytes(b"not a real pdf")

            original_reader = hfind._read_pdf_text_if_possible
            hfind._read_pdf_text_if_possible = lambda _path: ""
            try:
                code, lines = self._run_main([
                    "-cp",
                    "the",
                    str(root / "*.pdf"),
                ])
            finally:
                hfind._read_pdf_text_if_possible = original_reader

            self.assertEqual(code, 0)
            self.assertEqual([self._strip_ansi(line) for line in lines], [str(pdf_path)])

    def test_sort_flag_waits_then_outputs_sorted_matches(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-sort-") as tmpdir:
            root = Path(tmpdir)
            (root / "z_conflicted.txt").write_text("x\n", encoding="utf-8")
            (root / "a_conflicted.txt").write_text("x\n", encoding="utf-8")
            (root / "m_conflicted.txt").write_text("x\n", encoding="utf-8")

            code, stdout_lines, stderr_lines = self._run_main_with_stderr(
                [
                    "-s",
                    "conflicted",
                    str(root / "*.txt"),
                ]
            )

            self.assertEqual(code, 0)
            self.assertIn(
                "One moment please... finding all matches to sort them",
                stderr_lines,
            )
            self.assertEqual(
                [self._strip_ansi(line) for line in stdout_lines],
                [
                    str(root / "a_conflicted.txt"),
                    str(root / "m_conflicted.txt"),
                    str(root / "z_conflicted.txt"),
                ],
            )

    def test_sort_mode_case_sensitivity_switch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-sort-case-mode-") as tmpdir:
            root = Path(tmpdir)
            (root / "a_conflicted.txt").write_text("x\n", encoding="utf-8")
            (root / "B_conflicted.txt").write_text("x\n", encoding="utf-8")
            (root / "c_conflicted.txt").write_text("x\n", encoding="utf-8")

            code_insensitive, stdout_insensitive, _stderr_insensitive = (
                self._run_main_with_stderr(
                    [
                        "-s",
                        "conflicted",
                        str(root / "*.txt"),
                    ]
                )
            )
            code_sensitive, stdout_sensitive, _stderr_sensitive = (
                self._run_main_with_stderr(
                    [
                        "-S",
                        "conflicted",
                        str(root / "*.txt"),
                    ]
                )
            )

            self.assertEqual(code_insensitive, 0)
            self.assertEqual(code_sensitive, 0)
            self.assertEqual(
                [self._strip_ansi(line) for line in stdout_insensitive],
                [
                    str(root / "a_conflicted.txt"),
                    str(root / "B_conflicted.txt"),
                    str(root / "c_conflicted.txt"),
                ],
            )
            self.assertEqual(
                [self._strip_ansi(line) for line in stdout_sensitive],
                [
                    str(root / "B_conflicted.txt"),
                    str(root / "a_conflicted.txt"),
                    str(root / "c_conflicted.txt"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
