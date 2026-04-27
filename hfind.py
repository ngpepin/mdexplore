#!/usr/bin/env python3
"""Search files using mdexplore query syntax.

Examples:
  hfind.py --query "OR(fred, paul)" --content --recursive *.txt
  hfind.py -q "OR(fred, paul)" -cr *.txt
  hfind.py -cr "OR(fred, paul)" *.txt
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import glob
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import quote

from mdexplore_app import search as search_query


USAGE = """Usage:
    hfind.py --query QUERY [--content] [--recursive] [--verbose] [--pdf] [--sort|--sort-case-sensitive] [PATTERN ...]
    hfind.py -q QUERY [-c] [-r] [-v] [-p] [-s|-S] [PATTERN ...]
    hfind.py -crvps QUERY [PATTERN ...]

Notes:
  - If -q/--query is omitted, the first positional argument is used as QUERY.
    - If no PATTERN is provided, current directory is assumed (`*`, or `**/*` with -r).
  - Default search checks filename (including extension, no path).
  - --content/-c includes file contents in matching.
  - --recursive/-r expands each pattern recursively under its base directory.
    - --verbose/-v lists matching lines under each matched file with yellow hits.
    - --pdf/-p includes searchable text extracted from PDF files (only when -c is set).
    - --sort/-s waits for full scan and emits case-insensitively sorted results.
    - --sort-case-sensitive/-S waits for full scan and emits case-sensitively sorted results.

Examples:
    # Filename-only search (default): stem contains fred OR paul
    hfind.py --query "OR(fred, paul)" *.txt

    # Recursive content search with stacked flags
    hfind.py -cr "AND(product, roadmap)" "docs/*.md"

    # NEAR is strict (terms must be within 50 words)
    hfind.py -cv "NEAR(nicolas,pepin)" "notes/*.md"

    # Exclude files mentioning john
    hfind.py -rc "NOT(john)" "archive/*.md"

    # Single-space boundary intent in quoted terms
    hfind.py -cv "'Nico '" "people/*.md"

    # Include PDF text extraction
    hfind.py -rcvp "the" "library/*.pdf"
"""


ANSI_YELLOW = "\033[33m"
ANSI_BOLD_PURPLE = "\033[1;35m"
ANSI_RESET = "\033[0m"
OSC8_OPEN = "\033]8;;"
OSC8_CLOSE = "\a"
_BINARY_SAMPLE_BYTES = 8192


def _configured_search_workers() -> int:
    raw = os.environ.get("HFIND_SEARCH_THREADS", "").strip()
    try:
        configured = int(raw)
    except Exception:
        configured = 0
    if configured > 0:
        return configured
    return max(4, min(24, (os.cpu_count() or 2) * 3))


_MAX_SEARCH_WORKERS = _configured_search_workers()


def _style_filepath(path: Path) -> str:
    label = str(path)
    try:
        resolved = path.resolve()
        uri = "file://" + quote(str(resolved), safe="/:-._~")
    except Exception:
        uri = ""
    if not uri:
        return f"{ANSI_BOLD_PURPLE}{label}{ANSI_RESET}"
    return (
        f"{OSC8_OPEN}{uri}{OSC8_CLOSE}"
        f"{ANSI_BOLD_PURPLE}{label}{ANSI_RESET}"
        f"{OSC8_OPEN}{OSC8_CLOSE}"
    )


def _parse_args(
    argv: list[str],
) -> tuple[str, bool, bool, bool, bool, bool, bool, list[str]]:
    def _usage_error(message: str) -> SystemExit:
        return SystemExit(f"{message}\n\n{USAGE}")

    query: str | None = None
    include_content = False
    recursive = False
    verbose = False
    include_pdf = False
    sort_results = False
    sort_case_sensitive = False
    positionals: list[str] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            positionals.extend(argv[i + 1 :])
            break
        if arg in {"-h", "--help"}:
            print(USAGE)
            raise SystemExit(0)
        if arg == "--query":
            if i + 1 >= len(argv):
                raise _usage_error("error: --query requires a value")
            query = argv[i + 1]
            i += 2
            continue
        if arg == "--content":
            include_content = True
            i += 1
            continue
        if arg == "--recursive":
            recursive = True
            i += 1
            continue
        if arg == "--verbose":
            verbose = True
            i += 1
            continue
        if arg == "--pdf":
            include_pdf = True
            i += 1
            continue
        if arg == "--sort":
            sort_results = True
            sort_case_sensitive = False
            i += 1
            continue
        if arg == "--sort-case-sensitive":
            sort_results = True
            sort_case_sensitive = True
            i += 1
            continue
        if arg.startswith("-") and arg != "-":
            # Allow stacked short flags, e.g. -cr
            consumed_query = False
            for flag in arg[1:]:
                if flag == "c":
                    include_content = True
                    continue
                if flag == "r":
                    recursive = True
                    continue
                if flag == "v":
                    verbose = True
                    continue
                if flag == "p":
                    include_pdf = True
                    continue
                if flag == "s":
                    sort_results = True
                    sort_case_sensitive = False
                    continue
                if flag == "S":
                    sort_results = True
                    sort_case_sensitive = True
                    continue
                if flag == "q":
                    if i + 1 >= len(argv):
                        raise _usage_error("error: -q requires a value")
                    query = argv[i + 1]
                    consumed_query = True
                    continue
                raise _usage_error(f"error: unknown option -{flag}")
            i += 2 if consumed_query else 1
            continue

        positionals.append(arg)
        i += 1

    if query is None:
        if not positionals:
            raise _usage_error("error: missing query")
        query = positionals.pop(0)

    if not positionals:
        positionals = ["**/*" if recursive else "*"]

    return (
        query,
        include_content,
        recursive,
        verbose,
        include_pdf,
        sort_results,
        sort_case_sensitive,
        positionals,
    )


def _recursive_pattern(pattern: str) -> str:
    if "**" in pattern:
        return pattern
    parent = os.path.dirname(pattern)
    leaf = os.path.basename(pattern)
    if not parent:
        return os.path.join("**", leaf)
    return os.path.join(parent, "**", leaf)


def _pdf_pattern_variants(pattern: str) -> list[str]:
    """Return glob variants for case-insensitive PDF extension matching."""
    if not pattern.lower().endswith(".pdf"):
        return [pattern]
    # Keep original first, then add extension-class variant for Linux globbing.
    base = pattern[:-4]
    return [pattern, f"{base}.[pP][dD][fF]"]


def _iter_candidate_paths(patterns: list[str], recursive: bool):
    """Yield candidate files progressively as globbing discovers them."""
    seen: set[str] = set()

    for raw in patterns:
        raw_patterns = _pdf_pattern_variants(raw)
        matched_any = False

        # In recursive mode, prefer a recursive variant first, then fall back
        # to the original raw pattern if the recursive variant yields nothing.
        if recursive:
            for raw_pattern in raw_patterns:
                recursive_pattern = _recursive_pattern(raw_pattern)
                for item in glob.iglob(recursive_pattern, recursive=True):
                    matched_any = True
                    path = Path(item)
                    if not path.is_file():
                        continue
                    key = str(path.resolve())
                    if key in seen:
                        continue
                    seen.add(key)
                    yield path

            if not matched_any:
                for raw_pattern in raw_patterns:
                    for item in glob.iglob(raw_pattern, recursive=True):
                        matched_any = True
                        path = Path(item)
                        if not path.is_file():
                            continue
                        key = str(path.resolve())
                        if key in seen:
                            continue
                        seen.add(key)
                        yield path
        else:
            for raw_pattern in raw_patterns:
                for item in glob.iglob(raw_pattern, recursive=False):
                    matched_any = True
                    path = Path(item)
                    if not path.is_file():
                        continue
                    key = str(path.resolve())
                    if key in seen:
                        continue
                    seen.add(key)
                    yield path

        # If shell already expanded the pattern into explicit file args,
        # preserve them as literals too.
        if not matched_any and os.path.exists(raw):
            path = Path(raw)
            if path.is_file():
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    yield path


def _read_text_if_possible(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _read_pdf_text_if_possible(path: Path) -> str | None:
    try:
        from pypdf import PdfReader
    except Exception:
        PdfReader = None  # type: ignore[assignment]

    if PdfReader is not None:
        try:
            reader = PdfReader(str(path), strict=False)
            pages: list[str] = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                pages.append(page_text)
            extracted = "\n".join(pages).strip()
            if extracted:
                return extracted
        except Exception:
            pass

    # Fallback: poppler's pdftotext can decode some PDFs pypdf cannot.
    pdftotext_cmd = shutil.which("pdftotext")
    if pdftotext_cmd:
        try:
            result = subprocess.run(
                [pdftotext_cmd, "-q", "-layout", str(path), "-"],
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except Exception:
            pass

    # Keep file eligible for filename matching even if content extraction fails.
    return ""


def _is_clearly_binary_file(path: Path) -> bool:
    """Return whether a file appears to be binary from an initial byte sample."""
    try:
        with path.open("rb") as stream:
            sample = stream.read(_BINARY_SAMPLE_BYTES)
    except Exception:
        return False

    if not sample:
        return False

    # NUL bytes are a strong binary signal.
    if b"\x00" in sample:
        return True

    text_like = set(b"\n\r\t\f\b") | set(range(32, 127))
    non_text_count = sum(1 for byte in sample if byte not in text_like)
    return (non_text_count / len(sample)) > 0.30


def _line_spans(content: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for raw_line in content.splitlines(keepends=True):
        start = cursor
        cursor += len(raw_line)
        spans.append((start, cursor, raw_line.rstrip("\r\n")))
    if not spans and content == "":
        return []
    if content and content[-1] not in {"\n", "\r"}:
        return spans
    return spans


def _iter_term_ranges(
    text: str,
    terms: list[search_query.SearchTerm],
    *,
    enforce_near_boundaries: bool = False,
) -> list[tuple[int, int]]:
    searchable_text = search_query.strip_inline_image_data_uris(text or "")
    ranges: list[tuple[int, int]] = []
    for term_text, is_case_sensitive in terms:
        if not term_text:
            continue
        pattern = search_query.compile_term_pattern(
            term_text,
            bool(is_case_sensitive),
            enforce_word_boundaries=bool(enforce_near_boundaries),
        )
        for match in pattern.finditer(searchable_text):
            ranges.append(match.span())
    ranges.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
            continue
        merged.append((start, end))
    return merged


def _highlight_line(
    line_text: str,
    terms: list[search_query.SearchTerm],
    *,
    enforce_near_boundaries: bool = False,
) -> str:
    ranges = _iter_term_ranges(
        line_text,
        terms,
        enforce_near_boundaries=enforce_near_boundaries,
    )
    if not ranges:
        return line_text
    pieces: list[str] = []
    cursor = 0
    for start, end in ranges:
        if cursor < start:
            pieces.append(line_text[cursor:start])
        pieces.append(f"{ANSI_YELLOW}{line_text[start:end]}{ANSI_RESET}")
        cursor = end
    if cursor < len(line_text):
        pieces.append(line_text[cursor:])
    return "".join(pieces)


def _collect_verbose_line_numbers(content: str, query: str) -> set[int]:
    line_numbers: set[int] = set()
    searchable_content = search_query.strip_inline_image_data_uris(content or "")
    spans = _line_spans(content)
    if not spans:
        return line_numbers

    terms = search_query.extract_search_terms(query)
    near_groups = search_query.extract_near_term_groups(query)
    near_windows = search_query.collect_near_focus_windows(
        searchable_content, near_groups
    )

    # NEAR() is strict: only include lines that overlap qualifying NEAR windows.
    if near_groups:
        for window in near_windows:
            focus_start = int(window.get("start_char", 0))
            focus_end = int(window.get("end_char", focus_start))
            for index, (line_start, line_end, _line_text) in enumerate(spans, start=1):
                if line_end <= focus_start:
                    continue
                if line_start >= focus_end:
                    continue
                line_numbers.add(index)
        return line_numbers

    for index, (_start, _end, line_text) in enumerate(spans, start=1):
        if _iter_term_ranges(line_text, terms):
            line_numbers.add(index)

    return line_numbers


def _verbose_terms_for_query(query: str) -> tuple[list[search_query.SearchTerm], bool]:
    near_groups = search_query.extract_near_term_groups(query)
    if not near_groups:
        return search_query.extract_search_terms(query), False

    seen: set[str] = set()
    terms: list[search_query.SearchTerm] = []
    for group in near_groups:
        for term_text, is_case_sensitive in group:
            if not term_text:
                continue
            key = (
                f"S:{term_text}"
                if is_case_sensitive
                else f"I:{term_text.casefold()}"
            )
            if key in seen:
                continue
            seen.add(key)
            terms.append((term_text, bool(is_case_sensitive)))

    terms.sort(key=lambda item: len(item[0]), reverse=True)
    return terms, True


def _line_has_self_contained_near_match(
    line_text: str,
    near_groups: list[list[search_query.SearchTerm]],
) -> bool:
    if not near_groups:
        return False
    if not line_text:
        return False
    searchable_line_text = search_query.strip_inline_image_data_uris(line_text)
    return bool(
        search_query.collect_near_focus_windows(searchable_line_text, near_groups)
    )


def _print_verbose_result(path: Path, content: str, query: str) -> None:
    print(_style_filepath(path), flush=True)
    spans = _line_spans(content)
    hit_lines = _collect_verbose_line_numbers(content, query)
    terms, enforce_near_boundaries = _verbose_terms_for_query(query)
    near_groups = search_query.extract_near_term_groups(query)
    has_near_groups = bool(near_groups)
    if not hit_lines:
        print("  (filename match only)", flush=True)
        return

    if has_near_groups:
        sorted_lines = sorted(hit_lines)
        previous_line_no: int | None = None
        previous_was_self_contained = False
        for line_no in sorted_lines:
            _start, _end, line_text = spans[line_no - 1]
            rendered = _highlight_line(
                line_text,
                terms,
                enforce_near_boundaries=enforce_near_boundaries,
            )
            is_self_contained = _line_has_self_contained_near_match(
                line_text,
                near_groups,
            )

            # Number every line that independently satisfies NEAR().
            # For contiguous lines that require cross-line context, only
            # the first line in that dependent block is numbered.
            if is_self_contained:
                print(f"  {line_no}: {rendered}", flush=True)
            elif (
                previous_line_no is None
                or line_no != previous_line_no + 1
                or previous_was_self_contained
            ):
                print(f"  {line_no}: {rendered}", flush=True)
            else:
                print(f"      {rendered}", flush=True)
            previous_line_no = line_no
            previous_was_self_contained = is_self_contained
        return

    for line_no, (_start, _end, line_text) in enumerate(spans, start=1):
        if line_no not in hit_lines:
            continue
        rendered = _highlight_line(
            line_text,
            terms,
            enforce_near_boundaries=enforce_near_boundaries,
        )
        print(f"  {line_no}: {rendered}", flush=True)


def _scan_candidate_for_query(
    path: Path,
    predicate,
    *,
    include_content: bool,
    include_pdf: bool,
) -> tuple[Path, str, bool]:
    """Read one candidate and evaluate query match state."""
    is_pdf = path.suffix.lower() == ".pdf"
    file_name = path.name
    content = ""

    if include_content:
        if is_pdf:
            if include_pdf:
                text = _read_pdf_text_if_possible(path)
                content = text or ""
        else:
            if _is_clearly_binary_file(path):
                # Preserve filename matching in content mode even when the
                # file body is binary and cannot be meaningfully searched.
                content = ""
            else:
                text = _read_text_if_possible(path)
                if text is None:
                    # Keep filename matching active even when content cannot
                    # be read (permissions/encoding/path race, etc.).
                    content = ""
                else:
                    content = text

    searchable_content = search_query.strip_inline_image_data_uris(
        content,
        preserve_line_structure=False,
    )
    try:
        matched = bool(predicate(file_name, searchable_content))
    except Exception:
        matched = False
    return path, content, matched


def main(argv: list[str]) -> int:
    try:
        (
            query,
            include_content,
            recursive,
            verbose,
            include_pdf,
            sort_results,
            sort_case_sensitive,
            patterns,
        ) = _parse_args(argv)
        predicate = search_query.compile_match_predicate(
            query, strip_inline_image_data=False
        )

        if include_pdf and not include_content:
            print("note: --pdf has no effect unless --content/-c is set", file=sys.stderr)
        if sort_results:
            print(
                "One moment please... finding all matches to sort them",
                file=sys.stderr,
                flush=True,
            )

        match_count = 0
        candidate_iter = _iter_candidate_paths(patterns, recursive)
        saw_candidate = False
        buffered_matches: list[tuple[Path, str]] = []

        def _emit_match(path: Path, content: str) -> None:
            nonlocal match_count
            match_count += 1
            if sort_results:
                buffered_matches.append((path, content))
                return
            if verbose:
                if include_content:
                    _print_verbose_result(path, content, query)
                else:
                    print(_style_filepath(path), flush=True)
            else:
                print(_style_filepath(path), flush=True)

        worker_count = max(1, _MAX_SEARCH_WORKERS)
        if worker_count <= 1:
            for candidate_path in candidate_iter:
                saw_candidate = True
                path, content, matched = _scan_candidate_for_query(
                    candidate_path,
                    predicate,
                    include_content=include_content,
                    include_pdf=include_pdf,
                )
                if not matched:
                    continue
                _emit_match(path, content)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                pending_futures = set()
                candidate_exhausted = False
                max_in_flight = max(4, worker_count * 4)

                while True:
                    while (not candidate_exhausted) and len(pending_futures) < max_in_flight:
                        try:
                            path = next(candidate_iter)
                        except StopIteration:
                            candidate_exhausted = True
                            break
                        saw_candidate = True
                        pending_futures.add(
                            executor.submit(
                                _scan_candidate_for_query,
                                path,
                                predicate,
                                include_content=include_content,
                                include_pdf=include_pdf,
                            )
                        )

                    if not pending_futures:
                        break

                    done, pending_futures = wait(
                        pending_futures,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        try:
                            path, content, matched = future.result()
                        except Exception:
                            continue
                        if not matched:
                            continue
                        _emit_match(path, content)

        if not saw_candidate:
            return 1

        if sort_results and buffered_matches:
            if sort_case_sensitive:
                buffered_matches.sort(key=lambda item: str(item[0]))
            else:
                buffered_matches.sort(
                    key=lambda item: (str(item[0]).casefold(), str(item[0]))
                )
            for path, content in buffered_matches:
                if verbose:
                    if include_content:
                        _print_verbose_result(path, content, query)
                    else:
                        print(_style_filepath(path), flush=True)
                else:
                    print(_style_filepath(path), flush=True)

        return 0 if match_count else 1
    except KeyboardInterrupt:
        print("Search interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
