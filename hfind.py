#!/usr/bin/env python3
"""Search files using mdexplore query syntax.

Examples:
  hfind.py --query "OR(fred, paul)" --content --recursive *.txt
  hfind.py -q "OR(fred, paul)" -cr *.txt
  hfind.py -cr "OR(fred, paul)" *.txt
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
import sys

from mdexplore_app import search as search_query


USAGE = """Usage:
  hfind.py --query QUERY [--content] [--recursive] PATTERN [PATTERN ...]
  hfind.py -q QUERY [-c] [-r] PATTERN [PATTERN ...]
  hfind.py -cr QUERY PATTERN [PATTERN ...]

Notes:
  - If -q/--query is omitted, the first positional argument is used as QUERY.
  - Default search checks filename stem only (no extension, no path).
  - --content/-c includes file contents in matching.
  - --recursive/-r expands each pattern recursively under its base directory.
"""


def _parse_args(argv: list[str]) -> tuple[str, bool, bool, list[str]]:
    query: str | None = None
    include_content = False
    recursive = False
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
                raise SystemExit("error: --query requires a value")
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
                if flag == "q":
                    if i + 1 >= len(argv):
                        raise SystemExit("error: -q requires a value")
                    query = argv[i + 1]
                    consumed_query = True
                    continue
                raise SystemExit(f"error: unknown option -{flag}")
            i += 2 if consumed_query else 1
            continue

        positionals.append(arg)
        i += 1

    if query is None:
        if not positionals:
            raise SystemExit("error: missing query\n\n" + USAGE)
        query = positionals.pop(0)

    if not positionals:
        raise SystemExit("error: missing file pattern(s)\n\n" + USAGE)

    return query, include_content, recursive, positionals


def _recursive_pattern(pattern: str) -> str:
    if "**" in pattern:
        return pattern
    parent = os.path.dirname(pattern)
    leaf = os.path.basename(pattern)
    if not parent:
        return os.path.join("**", leaf)
    return os.path.join(parent, "**", leaf)


def _expand_patterns(patterns: list[str], recursive: bool) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []

    for raw in patterns:
        if recursive:
            expanded = glob.glob(_recursive_pattern(raw), recursive=True)
            if not expanded:
                expanded = glob.glob(raw, recursive=True)
        else:
            expanded = glob.glob(raw, recursive=False)

        # If shell already expanded the pattern into explicit file args,
        # preserve them as literals too.
        if not expanded and os.path.exists(raw):
            expanded = [raw]

        for item in expanded:
            path = Path(item)
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(path)

    return sorted(out, key=lambda p: str(p))


def _read_text_if_possible(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def main(argv: list[str]) -> int:
    query, include_content, recursive, patterns = _parse_args(argv)
    predicate = search_query.compile_match_predicate(query)

    candidates = _expand_patterns(patterns, recursive)
    matches: list[Path] = []

    for path in candidates:
        stem = path.stem
        content = ""
        if include_content:
            text = _read_text_if_possible(path)
            if text is None:
                continue
            content = text
        try:
            if predicate(stem, content):
                matches.append(path)
        except Exception:
            # Keep search resilient across odd parser/runtime edge cases.
            continue

    for path in matches:
        print(str(path))

    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
