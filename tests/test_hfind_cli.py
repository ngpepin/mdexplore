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

    def test_default_path_search_ignores_content(self) -> None:
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

    def test_default_search_matches_directory_path_components(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-default-path-target-") as tmpdir:
            root = Path(tmpdir)
            nested = root / "project" / "logs"
            nested.mkdir(parents=True, exist_ok=True)
            source = nested / "inside.txt"
            source.write_text("no content term\n", encoding="utf-8")

            previous = Path.cwd()
            os.chdir(root)
            try:
                code, lines = self._run_main([
                    "-r",
                    "project",
                    "*.txt",
                ])
            finally:
                os.chdir(previous)

            self.assertEqual(code, 0)
            self.assertEqual(
                [self._strip_ansi(line) for line in lines],
                [str(Path("project") / "logs" / "inside.txt")],
            )

    def test_base_switch_limits_search_target_to_basename(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-base-switch-") as tmpdir:
            root = Path(tmpdir)
            nested = root / "project" / "logs"
            nested.mkdir(parents=True, exist_ok=True)
            source = nested / "inside.txt"
            source.write_text("no content term\n", encoding="utf-8")

            previous = Path.cwd()
            os.chdir(root)
            try:
                code_project, lines_project = self._run_main([
                    "-rb",
                    "project",
                    "*.txt",
                ])
                code_inside, lines_inside = self._run_main([
                    "-rb",
                    "inside",
                    "*.txt",
                ])
            finally:
                os.chdir(previous)

            self.assertEqual(code_project, 1)
            self.assertEqual(lines_project, [])
            self.assertEqual(code_inside, 0)
            self.assertEqual(
                [self._strip_ansi(line) for line in lines_inside],
                [str(Path("project") / "logs" / "inside.txt")],
            )

    def test_near_matches_path_in_default_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-near-path-default-") as tmpdir:
            root = Path(tmpdir)
            nested = root / "hazmat" / "primitives"
            nested.mkdir(parents=True, exist_ok=True)
            source = nested / "vector.txt"
            source.write_text("content unrelated\n", encoding="utf-8")

            previous = Path.cwd()
            os.chdir(root)
            try:
                code, lines = self._run_main([
                    "-r",
                    "NEAR(hazmat,primitives)",
                    "*.txt",
                ])
            finally:
                os.chdir(previous)

            self.assertEqual(code, 0)
            self.assertEqual(
                [self._strip_ansi(line) for line in lines],
                [str(Path("hazmat") / "primitives" / "vector.txt")],
            )

    def test_near_base_mode_uses_basename_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-near-base-mode-") as tmpdir:
            root = Path(tmpdir)
            nested = root / "hazmat" / "primitives"
            nested.mkdir(parents=True, exist_ok=True)
            path_match_only = nested / "vector.txt"
            path_match_only.write_text("content unrelated\n", encoding="utf-8")
            basename_match = root / "hazmat-primitives.txt"
            basename_match.write_text("content unrelated\n", encoding="utf-8")

            previous = Path.cwd()
            os.chdir(root)
            try:
                code, lines = self._run_main([
                    "-rb",
                    "NEAR(hazmat,primitives)",
                    "*.txt",
                ])
            finally:
                os.chdir(previous)

            self.assertEqual(code, 0)
            self.assertEqual(
                [self._strip_ansi(line) for line in lines],
                [str(Path("hazmat-primitives.txt"))],
            )

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
            hfind._read_pdf_text_if_possible = lambda _path, **_kwargs: ""
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

    def test_wip_lists_checked_full_paths_and_excludes_nonmatches_from_sort(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-wip-sort-") as tmpdir:
            root = Path(tmpdir)
            matched = root / "z_match.txt"
            missed = root / "a_miss.txt"
            matched.write_text("needle\n", encoding="utf-8")
            missed.write_text("nothing here\n", encoding="utf-8")

            code, stdout_lines, _stderr_lines = self._run_main_with_stderr([
                "-cws",
                "needle",
                str(root / "*.txt"),
            ])

            self.assertEqual(code, 0)
            plain = [self._strip_ansi(line) for line in stdout_lines]
            self.assertIn(f"{matched.resolve()} [content read]", plain)
            self.assertIn(f"{missed.resolve()} [content read]", plain)
            self.assertEqual(plain[-1], str(matched))
            self.assertEqual(plain.count(str(missed)), 0)
            self.assertTrue(any("\x1b[90m" in line for line in stdout_lines[:-1]))

    def test_wip_reports_ocr_when_ocr_is_performed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-wip-ocr-") as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "scan.pdf"
            pdf_path.write_bytes(b"not a real pdf")

            original_reader = hfind._read_pdf_text_if_possible

            def _fake_reader(_path: Path, *, ocr_pdf: bool = False, operations=None):
                if operations is not None:
                    operations.extend(["PDF text", "OCR"])
                return "needle from OCR"

            hfind._read_pdf_text_if_possible = _fake_reader
            try:
                code, lines = self._run_main([
                    "-cpw",
                    "needle",
                    str(pdf_path),
                ])
            finally:
                hfind._read_pdf_text_if_possible = original_reader

            self.assertEqual(code, 0)
            plain = [self._strip_ansi(line) for line in lines]
            self.assertIn(f"{pdf_path.resolve()} [PDF text, OCR]", plain)
            self.assertEqual(plain[-1], str(pdf_path))

    def test_ocr_pdf_is_opt_in_for_scan_like_pdf(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-ocr-opt-in-") as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "scanned-document.pdf"
            from reportlab.pdfgen import canvas

            writer = canvas.Canvas(str(pdf_path))
            writer.rect(72, 650, 300, 100)
            writer.save()

            calls: list[Path] = []
            original_ocr = hfind._ocr_pdf_text_if_possible
            hfind._ocr_pdf_text_if_possible = lambda path: (
                calls.append(path) or "Scanned Nicolas Pepin record"
            )
            try:
                code_without_ocr, lines_without_ocr = self._run_main([
                    "-cp",
                    "pepin",
                    str(root / "*.pdf"),
                ])
                self.assertEqual(code_without_ocr, 1)
                self.assertEqual(lines_without_ocr, [])
                self.assertEqual(calls, [])

                code_with_ocr, lines_with_ocr = self._run_main([
                    "-c",
                    "--ocr-pdf",
                    "pepin",
                    str(root / "*.pdf"),
                ])
            finally:
                hfind._ocr_pdf_text_if_possible = original_ocr

            self.assertEqual(code_with_ocr, 0)
            self.assertEqual(
                [self._strip_ansi(line) for line in lines_with_ocr],
                [str(pdf_path)],
            )
            self.assertEqual(calls, [pdf_path])

    def test_ocr_pdf_does_not_ocr_normal_searchable_pdf(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-ocr-text-pdf-") as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "searchable.pdf"
            _create_pdf_with_text(
                pdf_path,
                "Nicolas Pepin searchable ordinary text document with enough words to avoid OCR",
            )

            original_ocr = hfind._ocr_pdf_text_if_possible

            def _unexpected_ocr(_path: Path) -> str:
                raise AssertionError("ordinary searchable PDF should not invoke OCR")

            hfind._ocr_pdf_text_if_possible = _unexpected_ocr
            try:
                code, lines = self._run_main([
                    "-c",
                    "--ocr-pdf",
                    "pepin",
                    str(root / "*.pdf"),
                ])
            finally:
                hfind._ocr_pdf_text_if_possible = original_ocr

            self.assertEqual(code, 0)
            self.assertEqual([self._strip_ansi(line) for line in lines], [str(pdf_path)])

    def test_exclude_option_is_repeatable_and_consumes_path_atomically(self) -> None:
        parsed = hfind._parse_args([
            "-e",
            "~/my-dir",
            "-e=~/my-dir-2",
            "--exclude",
            "/tmp/my-dir-3",
            "--exclude=/tmp/my-dir-4",
            "needle",
            "*.txt",
        ])
        self.assertEqual(parsed[0], "needle")
        self.assertEqual(parsed[11], ["*.txt"])
        self.assertEqual(
            parsed[12],
            ["~/my-dir", "~/my-dir-2", "/tmp/my-dir-3", "/tmp/my-dir-4"],
        )

        for args in (["-e"], ["--exclude"], ["-e="], ["--exclude="]):
            with self.assertRaises(SystemExit) as raised:
                hfind._parse_args(list(args))
            self.assertIn("requires one PATH", str(raised.exception))

    def test_exclude_omits_path_and_all_children(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-exclude-") as tmpdir:
            root = Path(tmpdir)
            keep = root / "keep"
            excluded = root / "excluded"
            nested = excluded / "nested"
            keep.mkdir()
            nested.mkdir(parents=True)
            keep_file = keep / "needle-keep.txt"
            excluded_file = excluded / "needle-direct.txt"
            nested_file = nested / "needle-nested.txt"
            keep_file.write_text("x\n", encoding="utf-8")
            excluded_file.write_text("x\n", encoding="utf-8")
            nested_file.write_text("x\n", encoding="utf-8")

            code, lines = self._run_main([
                "-r",
                "-e",
                str(excluded),
                "needle",
                str(root / "*.txt"),
            ])

            self.assertEqual(code, 0)
            self.assertEqual(
                [self._strip_ansi(line) for line in lines],
                [str(keep_file)],
            )

    def test_exclude_expands_home_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-exclude-home-") as tmpdir:
            root = Path(tmpdir)
            fake_home = root / "home"
            excluded = fake_home / "skip"
            included = fake_home / "keep"
            excluded.mkdir(parents=True)
            included.mkdir(parents=True)
            (excluded / "needle.txt").write_text("x\n", encoding="utf-8")
            included_file = included / "needle.txt"
            included_file.write_text("x\n", encoding="utf-8")

            original_home = os.environ.get("HOME")
            os.environ["HOME"] = str(fake_home)
            try:
                code, lines = self._run_main([
                    "-r",
                    "-e=~/skip",
                    "needle",
                    str(fake_home / "*.txt"),
                ])
            finally:
                if original_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = original_home

            self.assertEqual(code, 0)
            self.assertEqual(
                [self._strip_ansi(line) for line in lines],
                [str(included_file)],
            )

    def test_links_flag_is_opt_in_and_stackable(self) -> None:
        default = hfind._parse_args(["needle", "*.txt"])
        long_form = hfind._parse_args(["--links", "needle", "*.txt"])
        short_form = hfind._parse_args(["-l", "needle", "*.txt"])
        stacked = hfind._parse_args(["-crl", "needle", "*.txt"])

        self.assertFalse(default[13])
        self.assertTrue(long_form[13])
        self.assertTrue(short_form[13])
        self.assertTrue(stacked[13])
        self.assertTrue(stacked[1])
        self.assertTrue(stacked[3])

    def test_symlink_file_is_ignored_by_default_and_allowed_with_links(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-links-file-") as tmpdir:
            root = Path(tmpdir)
            target = root / "target.txt"
            target.write_text("needle\n", encoding="utf-8")
            link = root / "needle-link.txt"
            try:
                link.symlink_to(target)
            except (NotImplementedError, OSError):
                self.skipTest("symlink creation is not supported in this environment")

            code_default, lines_default = self._run_main([
                "needle-link",
                str(link),
            ])
            code_links, lines_links = self._run_main([
                "-l",
                "needle-link",
                str(link),
            ])

            self.assertEqual(code_default, 1)
            self.assertEqual(lines_default, [])
            self.assertEqual(code_links, 0)
            self.assertEqual(
                [self._strip_ansi(line) for line in lines_links],
                [str(link)],
            )

    def test_recursive_symlink_directory_is_not_followed_without_links(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-links-dir-") as tmpdir:
            root = Path(tmpdir)
            outside = root / "outside"
            outside.mkdir()
            target = outside / "needle.txt"
            target.write_text("x\n", encoding="utf-8")
            search_root = root / "search"
            search_root.mkdir()
            link_dir = search_root / "linked"
            try:
                link_dir.symlink_to(outside, target_is_directory=True)
            except (NotImplementedError, OSError):
                self.skipTest("symlink creation is not supported in this environment")

            pattern = str(search_root / "*.txt")
            code_default, lines_default = self._run_main(["-r", "needle", pattern])
            code_links, lines_links = self._run_main(["-rl", "needle", pattern])

            self.assertEqual(code_default, 1)
            self.assertEqual(lines_default, [])
            self.assertEqual(code_links, 0)
            self.assertEqual(
                [self._strip_ansi(line) for line in lines_links],
                [str(link_dir / "needle.txt")],
            )

    def test_first_non_switch_positional_is_query_when_q_is_omitted(self) -> None:
        parsed = hfind._parse_args([
            "-crvo",
            "--cpu-limit",
            "70",
            "needle",
            "*.pdf",
        ])
        self.assertEqual(parsed[0], "needle")
        self.assertTrue(parsed[1])
        self.assertTrue(parsed[3])
        self.assertTrue(parsed[4])
        self.assertTrue(parsed[5])
        self.assertTrue(parsed[6])
        self.assertFalse(parsed[7])
        self.assertEqual(parsed[8], 70.0)
        self.assertEqual(parsed[11], ["*.pdf"])

    def test_ocr_pdf_flag_implies_pdf_and_cpu_limit_parses(self) -> None:
        parsed = hfind._parse_args([
            "--ocr-pdf",
            "--cpu-limit",
            "72.5",
            "-c",
            "pepin",
            "*.pdf",
        ])
        self.assertTrue(parsed[5])
        self.assertTrue(parsed[6])
        self.assertFalse(parsed[7])
        self.assertEqual(parsed[8], 72.5)

        short_only = hfind._parse_args(["-o", "-c", "pepin", "*.pdf"])
        self.assertTrue(short_only[5])
        self.assertTrue(short_only[6])
        self.assertTrue(short_only[1])

        stacked = hfind._parse_args(["-cro", "pepin", "*.pdf"])
        self.assertTrue(stacked[1])
        self.assertTrue(stacked[3])
        self.assertTrue(stacked[5])
        self.assertTrue(stacked[6])

        wip = hfind._parse_args(["-crw", "pepin", "*.pdf"])
        self.assertTrue(wip[7])
        long_wip = hfind._parse_args(["--wip", "pepin", "*.pdf"])
        self.assertTrue(long_wip[7])

        with self.assertRaises(SystemExit) as raised:
            hfind._parse_args(["--cpu-limit", "101", "pepin", "*.pdf"])
        self.assertIn("between 0 and 100", str(raised.exception))

    def test_hfind_settings_file_exposes_runtime_defaults(self) -> None:
        self.assertEqual(hfind._SETTINGS_PATH.name, "hfind.settings.json")
        self.assertEqual(hfind.HFIND_SETTINGS["cpu_limit_percent"], 90.0)
        self.assertEqual(hfind.HFIND_SETTINGS["search_worker_min"], 4)
        self.assertEqual(hfind.HFIND_SETTINGS["search_worker_max"], 24)
        self.assertEqual(hfind.HFIND_SETTINGS["binary_sample_bytes"], 8192)
        self.assertEqual(hfind.HFIND_SETTINGS["ocr_render_dpi"], 160)
        self.assertEqual(hfind.HFIND_SETTINGS["pdf_text_timeout_seconds"], 20)

    def test_default_cpu_limit_is_ninety_percent(self) -> None:
        original = os.environ.pop("HFIND_CPU_LIMIT", None)
        try:
            self.assertEqual(hfind._configured_cpu_limit(), 90.0)
        finally:
            if original is not None:
                os.environ["HFIND_CPU_LIMIT"] = original

    def test_cpu_worker_limit_backs_off_and_recovers(self) -> None:
        self.assertEqual(hfind._adjust_worker_limit(8, 95.0, 80.0, 16), 6)
        self.assertEqual(hfind._adjust_worker_limit(6, 82.0, 80.0, 16), 4)
        self.assertEqual(hfind._adjust_worker_limit(4, 75.0, 80.0, 16), 4)
        self.assertEqual(hfind._adjust_worker_limit(4, 60.0, 80.0, 16), 5)
        self.assertEqual(hfind._adjust_worker_limit(8, 99.0, 0.0, 16), 8)

    def test_interrupt_returns_130_with_single_message(self) -> None:
        original_iter = hfind._iter_candidate_paths

        def _interrupted_iter(
            _patterns,
            _recursive,
            _excluded_paths=(),
            _follow_links=False,
        ):
            raise KeyboardInterrupt
            yield  # pragma: no cover

        hfind._iter_candidate_paths = _interrupted_iter
        try:
            code, stdout_lines, stderr_lines = self._run_main_with_stderr([
                "needle",
                "*.txt",
            ])
        finally:
            hfind._iter_candidate_paths = original_iter

        self.assertEqual(code, 130)
        self.assertEqual(stdout_lines, [])
        self.assertEqual(stderr_lines, ["Search interrupted by user."])

    def test_interrupt_termination_bypasses_normal_python_shutdown(self) -> None:
        original_exit = os._exit
        calls: list[int] = []

        def _fake_exit(code: int) -> None:
            calls.append(code)
            raise RuntimeError("hard exit intercepted")

        os._exit = _fake_exit
        try:
            with self.assertRaisesRegex(RuntimeError, "hard exit intercepted"):
                hfind._terminate_after_interrupt(130)
        finally:
            os._exit = original_exit

        self.assertEqual(calls, [130])

    def test_non_interrupt_termination_uses_system_exit(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            hfind._terminate_after_interrupt(0)
        self.assertEqual(raised.exception.code, 0)

    def test_recursive_glob_skips_unreadable_symlink_targets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hfind-unreadable-symlink-") as tmpdir:
            root = Path(tmpdir)
            visible = root / "visible.txt"
            visible.write_text("plain text\n", encoding="utf-8")

            restricted = root / "restricted"
            restricted.mkdir(parents=True, exist_ok=True)
            secret = restricted / "secret.txt"
            secret.write_text("hidden\n", encoding="utf-8")

            blocked_link = root / "blocked-doc"
            try:
                blocked_link.symlink_to(secret)
            except (NotImplementedError, OSError):
                self.skipTest("symlink creation is not supported in this environment")

            os.chmod(restricted, 0)
            try:
                code, lines = self._run_main([
                    "-r",
                    "visible",
                    str(root / "*"),
                ])
            finally:
                os.chmod(restricted, 0o700)

            self.assertEqual(code, 0)
            self.assertEqual([self._strip_ansi(line) for line in lines], [str(visible)])


if __name__ == "__main__":
    unittest.main()
