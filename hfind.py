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
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote

from mdexplore_app import search as search_query


USAGE = """Usage:
    hfind.py --query QUERY [--base] [--content] [--recursive] [--verbose] [--pdf] [--ocr-pdf|-o] [--wip|-w] [--exclude PATH|-e PATH] [--links|-l] [--cpu-limit PERCENT] [--sort|--sort-case-sensitive] [PATTERN ...]
    hfind.py -q QUERY [-b] [-c] [-r] [-v] [-p] [-o] [-w] [-e PATH] [-l] [-s|-S] [--cpu-limit PERCENT] [PATTERN ...]
    hfind.py -bcrvlposw QUERY [PATTERN ...]

Notes:
  If -q/--query is omitted, the first non-switch positional argument is used as QUERY.
  If no PATTERN is provided, current directory is assumed (`*`, or `**/*` with -r).
  Default search checks the full discovered path (directories + filename).
  --base/-b switches matching target to basename only (filename + extension).
  --content/-c includes file contents in matching.
  --recursive/-r expands each pattern recursively under its base directory.
  --verbose/-v lists matching lines under each matched file with yellow hits.
  --pdf/-p includes searchable text extracted from PDF files (only when -c is set).
  --ocr-pdf/-o enables OCR fallback for PDFs that contain little/no extractable text;
    it implies --pdf and still requires -c. The -o form may be used alone or stacked
    with other short flags (for example, -cro or -crvo).
  --wip/-w prints each checked file as one gray full-path line, with performed
    operations such as content reading, PDF extraction, or OCR appended.
  --exclude/-e PATH excludes PATH and all of its descendants; repeat for multiple paths.
    Both '-e PATH' and '-e=PATH' forms are accepted, as are the long forms.
  Symlinks are ignored by default, including files reached through symlinked directories.
  --links/-l allows symlink files and traversal through symlinked directories.
  --cpu-limit PERCENT dynamically throttles concurrent work to keep observed system CPU
    near/below the target (default: 90).
  --sort/-s waits for full scan and emits case-insensitively sorted results.
  --sort-case-sensitive/-S waits for full scan and emits case-sensitively sorted results.

Examples:
    # Path search (default): path contains fred OR paul
    hfind.py --query "OR(fred, paul)" *.txt

    # Basename-only search (legacy behavior)
    hfind.py -b --query "OR(fred, paul)" *.txt

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

    NEAR behavior:
        - NEAR(...) is evaluated against the active search stream.
        - Default stream includes discovered path text.
        - With -b/--base, stream uses basename text only.
        - With -c/--content, file content is appended to the same stream.
"""


ANSI_YELLOW = "\033[33m"
ANSI_BOLD_PURPLE = "\033[1;35m"
ANSI_GRAY = "\033[90m"
ANSI_RESET = "\033[0m"
OSC8_OPEN = "\033]8;;"
OSC8_CLOSE = "\a"

_SETTINGS_PATH = Path(__file__).resolve().with_name("hfind.settings.json")


def _load_settings() -> dict[str, object]:
    try:
        parsed = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    section = parsed.get("hfind")
    return dict(section) if isinstance(section, dict) else {}


HFIND_SETTINGS = _load_settings()


def _setting_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(HFIND_SETTINGS.get(name, default))
    except Exception:
        value = default
    return max(minimum, value)


def _setting_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(HFIND_SETTINGS.get(name, default))
    except Exception:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


_SEARCH_WORKER_MIN = _setting_int("search_worker_min", 4, minimum=1)
_SEARCH_WORKER_MAX = _setting_int("search_worker_max", 24, minimum=_SEARCH_WORKER_MIN)
_SEARCH_WORKER_CPU_MULTIPLIER = _setting_int("search_worker_cpu_multiplier", 3, minimum=1)
_CPU_SAMPLE_INTERVAL_SECONDS = _setting_float("cpu_sample_interval_seconds", 0.20, minimum=0.01)
_BINARY_SAMPLE_BYTES = _setting_int("binary_sample_bytes", 8192, minimum=1)
_BINARY_NON_TEXT_RATIO_THRESHOLD = _setting_float(
    "binary_non_text_ratio_threshold", 0.30, minimum=0.0, maximum=1.0
)
_OCR_SPARSE_TEXT_ALNUM_THRESHOLD = _setting_int(
    "ocr_sparse_text_alnum_threshold", 32, minimum=0
)
_OCR_SPARSE_TEXT_ALNUM_PER_PAGE = _setting_int(
    "ocr_sparse_text_alnum_per_page", 24, minimum=0
)
_OCR_SCAN_PAGE_ALNUM_THRESHOLD = _setting_int(
    "ocr_scan_page_alnum_threshold", 8, minimum=0
)
_OCR_RENDER_DPI = _setting_int("ocr_render_dpi", 160, minimum=1)
_OCR_RENDER_TIMEOUT_SECONDS = _setting_float(
    "ocr_render_timeout_seconds", 120.0, minimum=0.1
)
_OCR_TESSERACT_PSM = _setting_int(
    "ocr_tesseract_page_segmentation_mode", 3, minimum=0
)
_OCR_TESSERACT_TIMEOUT_SECONDS = _setting_float(
    "ocr_tesseract_timeout_seconds", 90.0, minimum=0.1
)
_PDF_TEXT_TIMEOUT_SECONDS = _setting_float(
    "pdf_text_timeout_seconds", 20.0, minimum=0.1
)
_DEFAULT_CPU_LIMIT_PERCENT = _setting_float(
    "cpu_limit_percent", 90.0, minimum=0.0, maximum=100.0
)


def _configured_search_workers() -> int:
    raw = os.environ.get("HFIND_SEARCH_THREADS", "").strip()
    try:
        configured = int(raw)
    except Exception:
        configured = 0
    if configured > 0:
        return configured
    return max(
        _SEARCH_WORKER_MIN,
        min(_SEARCH_WORKER_MAX, (os.cpu_count() or 2) * _SEARCH_WORKER_CPU_MULTIPLIER),
    )


_MAX_SEARCH_WORKERS = _configured_search_workers()


def _configured_cpu_limit() -> float:
    raw = os.environ.get("HFIND_CPU_LIMIT", str(_DEFAULT_CPU_LIMIT_PERCENT)).strip()
    try:
        value = float(raw)
    except Exception:
        value = _DEFAULT_CPU_LIMIT_PERCENT
    if value <= 0:
        return 0.0
    return min(100.0, value)


def _read_cpu_times() -> tuple[int, int] | None:
    """Return cumulative (total, idle) CPU jiffies on Linux when available."""
    try:
        first_line = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0]
        fields = first_line.split()
        if not fields or fields[0] != "cpu":
            return None
        values = [int(value) for value in fields[1:]]
        if len(values) < 4:
            return None
        total = sum(values)
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return total, idle
    except Exception:
        return None


class _CpuUsageSampler:
    """Sample whole-system CPU usage without adding a psutil dependency."""

    def __init__(self) -> None:
        self._previous = _read_cpu_times()
        self._previous_time = time.monotonic()

    def sample(self, *, force: bool = False) -> float | None:
        now = time.monotonic()
        if not force and (now - self._previous_time) < _CPU_SAMPLE_INTERVAL_SECONDS:
            return None
        current = _read_cpu_times()
        previous = self._previous
        self._previous = current
        self._previous_time = now
        if current is None or previous is None:
            return None
        total_delta = current[0] - previous[0]
        idle_delta = current[1] - previous[1]
        if total_delta <= 0:
            return None
        busy_delta = max(0, total_delta - idle_delta)
        return max(0.0, min(100.0, (busy_delta / total_delta) * 100.0))


def _adjust_worker_limit(
    current_limit: int,
    cpu_percent: float | None,
    cpu_limit: float,
    maximum: int,
) -> int:
    """Adapt concurrency conservatively around the configured CPU ceiling."""
    current_limit = max(1, min(maximum, current_limit))
    if cpu_percent is None or cpu_limit <= 0 or maximum <= 1:
        return current_limit
    if cpu_percent > cpu_limit:
        # Back off quickly under pressure so expensive OCR/PDF jobs stop piling up.
        return max(1, min(current_limit - 1, int(current_limit * 0.75)))
    if cpu_percent < max(0.0, cpu_limit - 10.0) and current_limit < maximum:
        # Ramp back up slowly to avoid oscillating around the limit.
        return current_limit + 1
    return current_limit


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


def _style_wip_filepath(path: Path, operations: list[str]) -> str:
    try:
        resolved = path.resolve()
    except Exception:
        try:
            resolved = path.absolute()
        except Exception:
            resolved = path
    label = str(resolved)
    if operations:
        label += " [" + ", ".join(operations) + "]"
    try:
        uri = "file://" + quote(str(resolved), safe="/:-._~")
    except Exception:
        uri = ""
    if not uri:
        return f"{ANSI_GRAY}{label}{ANSI_RESET}"
    return (
        f"{OSC8_OPEN}{uri}{OSC8_CLOSE}"
        f"{ANSI_GRAY}{label}{ANSI_RESET}"
        f"{OSC8_OPEN}{OSC8_CLOSE}"
    )


def _parse_args(
    argv: list[str],
) -> tuple[
    str,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    float,
    bool,
    bool,
    list[str],
    list[str],
    bool,
]:
    def _usage_error(message: str) -> SystemExit:
        return SystemExit(f"{message}\n\n{USAGE}")

    query: str | None = None
    include_content = False
    search_base_only = False
    recursive = False
    verbose = False
    include_pdf = False
    ocr_pdf = False
    wip = False
    cpu_limit = _configured_cpu_limit()
    sort_results = False
    sort_case_sensitive = False
    positionals: list[str] = []
    excluded_paths: list[str] = []
    follow_links = False

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
        if arg == "--base":
            search_base_only = True
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
        if arg == "--ocr-pdf":
            ocr_pdf = True
            include_pdf = True
            i += 1
            continue
        if arg == "--wip":
            wip = True
            i += 1
            continue
        if arg == "--links":
            follow_links = True
            i += 1
            continue
        if arg == "--exclude" or arg == "-e":
            if i + 1 >= len(argv):
                raise _usage_error(f"error: {arg} requires one PATH")
            excluded_paths.append(argv[i + 1])
            i += 2
            continue
        if arg.startswith("--exclude=") or arg.startswith("-e="):
            value = arg.split("=", 1)[1]
            if not value:
                option = "--exclude" if arg.startswith("--exclude=") else "-e"
                raise _usage_error(f"error: {option} requires one PATH")
            excluded_paths.append(value)
            i += 1
            continue
        if arg == "--cpu-limit":
            if i + 1 >= len(argv):
                raise _usage_error("error: --cpu-limit requires a value")
            try:
                cpu_limit = float(argv[i + 1])
            except ValueError:
                raise _usage_error("error: --cpu-limit must be a number") from None
            if cpu_limit < 0 or cpu_limit > 100:
                raise _usage_error("error: --cpu-limit must be between 0 and 100")
            i += 2
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
                if flag == "b":
                    search_base_only = True
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
                if flag == "o":
                    ocr_pdf = True
                    include_pdf = True
                    continue
                if flag == "w":
                    wip = True
                    continue
                if flag == "l":
                    follow_links = True
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
        search_base_only,
        recursive,
        verbose,
        include_pdf,
        ocr_pdf,
        wip,
        cpu_limit,
        sort_results,
        sort_case_sensitive,
        positionals,
        excluded_paths,
        follow_links,
    )


def _normalized_exclude_path(raw_path: str) -> Path:
    expanded = Path(os.path.expanduser(raw_path))
    try:
        return expanded.resolve(strict=False)
    except Exception:
        try:
            return expanded.absolute()
        except Exception:
            return expanded


def _path_is_excluded(path: Path, excluded_paths: tuple[Path, ...]) -> bool:
    if not excluded_paths:
        return False
    try:
        candidate = path.resolve(strict=False)
    except Exception:
        try:
            candidate = path.absolute()
        except Exception:
            candidate = path
    return any(
        candidate == excluded or excluded in candidate.parents
        for excluded in excluded_paths
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


def _safe_is_file(path: Path) -> bool:
    """Return whether path is a regular file, skipping unreadable entries."""
    try:
        return path.is_file()
    except OSError:
        return False


def _pattern_static_root(pattern: str) -> Path:
    """Return the non-glob prefix that forms this pattern's search root."""
    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        current = Path(pattern_path.anchor)
        parts = pattern_path.parts[1:]
    else:
        current = Path.cwd()
        parts = pattern_path.parts

    saw_magic = False
    for part in parts:
        if glob.has_magic(part):
            saw_magic = True
            break
        current = current / part

    if not saw_magic:
        current = current.parent
    return current


def _path_uses_symlink(path: Path, boundary: Path | None = None) -> bool:
    """Return whether path uses a symlink at or below the pattern search root."""
    try:
        candidate = path.absolute()
    except Exception:
        candidate = path
    if boundary is None:
        boundary = candidate.parent
    else:
        try:
            boundary = boundary.absolute()
        except Exception:
            pass

    for part in (candidate, *candidate.parents):
        try:
            if part.is_symlink():
                return True
        except OSError:
            pass
        if part == boundary:
            break
    return False


def _iter_glob_matches(pattern: str, *, recursive: bool, follow_links: bool):
    """Yield glob matches without traversing symlink directories unless requested."""
    if follow_links:
        yield from glob.iglob(pattern, recursive=recursive)
        return

    boundary = _pattern_static_root(pattern)
    if _path_uses_symlink(boundary, boundary):
        return

    if not recursive:
        iterator = glob.iglob(pattern, recursive=False)
    else:
        pattern_path = Path(pattern)
        if pattern_path.is_absolute():
            anchor = Path(pattern_path.anchor)
            relative_pattern = pattern[len(pattern_path.anchor) :].lstrip(os.sep)
        else:
            anchor = Path(".")
            relative_pattern = pattern
        try:
            iterator = (str(item) for item in anchor.glob(relative_pattern))
        except (OSError, ValueError):
            return

    try:
        for item in iterator:
            path = Path(item)
            if _path_uses_symlink(path, boundary):
                continue
            yield str(path)
    except (OSError, ValueError):
        return


def _safe_resolved_key(path: Path) -> str:
    """Build a stable dedupe key without failing on unreadable symlink targets."""
    try:
        return str(path.resolve())
    except OSError:
        try:
            return str(path.absolute())
        except Exception:
            return str(path)


def _iter_candidate_paths(
    patterns: list[str],
    recursive: bool,
    excluded_paths: tuple[Path, ...] = (),
    follow_links: bool = False,
):
    """Yield candidates progressively, skipping exclusions and symlinks by default."""
    seen: set[str] = set()

    for raw in patterns:
        raw_patterns = _pdf_pattern_variants(raw)
        matched_any = False

        # In recursive mode, prefer a recursive variant first, then fall back
        # to the original raw pattern if the recursive variant yields nothing.
        if recursive:
            for raw_pattern in raw_patterns:
                recursive_pattern = _recursive_pattern(raw_pattern)
                for item in _iter_glob_matches(
                    recursive_pattern,
                    recursive=True,
                    follow_links=follow_links,
                ):
                    matched_any = True
                    path = Path(item)
                    if not _safe_is_file(path):
                        continue
                    if _path_is_excluded(path, excluded_paths):
                        continue
                    key = _safe_resolved_key(path)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield path

            if not matched_any:
                for raw_pattern in raw_patterns:
                    for item in _iter_glob_matches(
                        raw_pattern,
                        recursive=True,
                        follow_links=follow_links,
                    ):
                        matched_any = True
                        path = Path(item)
                        if not _safe_is_file(path):
                            continue
                        if _path_is_excluded(path, excluded_paths):
                            continue
                        key = _safe_resolved_key(path)
                        if key in seen:
                            continue
                        seen.add(key)
                        yield path
        else:
            for raw_pattern in raw_patterns:
                for item in _iter_glob_matches(
                    raw_pattern,
                    recursive=False,
                    follow_links=follow_links,
                ):
                    matched_any = True
                    path = Path(item)
                    if not _safe_is_file(path):
                        continue
                    if _path_is_excluded(path, excluded_paths):
                        continue
                    key = _safe_resolved_key(path)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield path

        # If shell already expanded the pattern into explicit file args,
        # preserve them as literals too.
        if not matched_any and os.path.exists(raw):
            path = Path(raw)
            if (
                _safe_is_file(path)
                and (follow_links or not _path_uses_symlink(path, path.parent))
                and not _path_is_excluded(path, excluded_paths)
            ):
                key = _safe_resolved_key(path)
                if key not in seen:
                    seen.add(key)
                    yield path


def _read_text_if_possible(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _pdf_text_is_sparse(text: str, page_count: int = 1) -> bool:
    """Return whether extracted PDF text is sparse enough to indicate a scan."""
    alnum_count = sum(1 for char in (text or "") if char.isalnum())
    threshold = max(
        _OCR_SPARSE_TEXT_ALNUM_THRESHOLD,
        max(1, page_count) * _OCR_SPARSE_TEXT_ALNUM_PER_PAGE,
    )
    return alnum_count < threshold


def _pdf_page_has_image(page) -> bool:
    """Best-effort detection of raster images on a pypdf page."""
    try:
        resources = page.get("/Resources")
        if resources is None:
            return False
        resources = resources.get_object()
        xobjects = resources.get("/XObject")
        if xobjects is None:
            return False
        xobjects = xobjects.get_object()
        for value in xobjects.values():
            try:
                obj = value.get_object()
                if obj.get("/Subtype") == "/Image":
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def _ocr_pdf_text_if_possible(path: Path) -> str:
    """OCR a likely scanned PDF using Poppler plus Tesseract when available."""
    pdftoppm_cmd = shutil.which("pdftoppm")
    tesseract_cmd = shutil.which("tesseract")
    if not pdftoppm_cmd or not tesseract_cmd:
        return ""

    try:
        with tempfile.TemporaryDirectory(prefix="hfind-ocr-") as tmpdir:
            prefix = Path(tmpdir) / "page"
            render = subprocess.run(
                [
                    pdftoppm_cmd,
                    "-jpeg",
                    "-r",
                    str(_OCR_RENDER_DPI),
                    str(path),
                    str(prefix),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=_OCR_RENDER_TIMEOUT_SECONDS,
            )
            if render.returncode != 0:
                return ""

            page_images = sorted(Path(tmpdir).glob("page-*.jpg"))
            if not page_images:
                return ""

            env = os.environ.copy()
            # Tesseract/OpenMP can otherwise multiply CPU use inside every
            # hfind worker. One OCR thread per subprocess lets hfind's worker
            # controller remain the single concurrency authority.
            env.setdefault("OMP_THREAD_LIMIT", "1")
            pages: list[str] = []
            for image_path in page_images:
                result = subprocess.run(
                    [
                        tesseract_cmd,
                        str(image_path),
                        "stdout",
                        "--psm",
                        str(_OCR_TESSERACT_PSM),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=_OCR_TESSERACT_TIMEOUT_SECONDS,
                    env=env,
                )
                if result.returncode == 0 and result.stdout.strip():
                    pages.append(result.stdout.strip())
            return "\n\n".join(pages).strip()
    except Exception:
        return ""


def _read_pdf_text_if_possible(
    path: Path,
    *,
    ocr_pdf: bool = False,
    operations: list[str] | None = None,
) -> str | None:
    best_text = ""
    page_count = 1
    if operations is not None:
        operations.append("PDF text")
    scan_like_page_count = 0
    try:
        from pypdf import PdfReader
    except Exception:
        PdfReader = None  # type: ignore[assignment]

    if PdfReader is not None:
        try:
            reader = PdfReader(str(path), strict=False)
            page_count = max(1, len(reader.pages))
            pages: list[str] = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                pages.append(page_text)
                page_alnum = sum(1 for char in page_text if char.isalnum())
                if page_alnum < _OCR_SCAN_PAGE_ALNUM_THRESHOLD and _pdf_page_has_image(page):
                    scan_like_page_count += 1
            extracted = "\n".join(pages).strip()
            if extracted:
                best_text = extracted
                if not _pdf_text_is_sparse(extracted, page_count):
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
                timeout=_PDF_TEXT_TIMEOUT_SECONDS,
            )
            if result.returncode == 0 and result.stdout.strip():
                poppler_text = result.stdout.strip()
                if len(poppler_text) > len(best_text):
                    best_text = poppler_text
                if not _pdf_text_is_sparse(poppler_text, page_count):
                    return poppler_text
        except Exception:
            pass

    # OCR is intentionally opt-in. Zero-text PDFs are eligible because image-only
    # scans commonly extract nothing. Sparse-but-nonempty PDFs are OCRed only when
    # pypdf also found raster-image evidence on at least half of their pages; this
    # avoids OCRing ordinary short text PDFs merely because they contain few words.
    scan_page_threshold = max(1, (page_count + 1) // 2)
    clearly_scan_like = (not best_text.strip()) or (
        _pdf_text_is_sparse(best_text, page_count)
        and scan_like_page_count >= scan_page_threshold
    )
    if ocr_pdf and clearly_scan_like:
        if operations is not None:
            operations.append("OCR")
        ocr_text = _ocr_pdf_text_if_possible(path)
        if ocr_text:
            return ocr_text

    # Keep file eligible for filename matching even if content extraction fails.
    return best_text


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
    return (non_text_count / len(sample)) > _BINARY_NON_TEXT_RATIO_THRESHOLD


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
    search_base_only: bool,
    include_pdf: bool,
    ocr_pdf: bool = False,
) -> tuple[Path, str, bool, list[str]]:
    """Read one candidate and evaluate query match state."""
    is_pdf = path.suffix.lower() == ".pdf"
    search_target = path.name if search_base_only else str(path)
    content = ""
    operations: list[str] = []

    if include_content:
        if is_pdf:
            if include_pdf:
                text = _read_pdf_text_if_possible(
                    path,
                    ocr_pdf=ocr_pdf,
                    operations=operations,
                )
                content = text or ""
        else:
            if _is_clearly_binary_file(path):
                # Preserve filename matching in content mode even when the
                # file body is binary and cannot be meaningfully searched.
                operations.append("binary skipped")
                content = ""
            else:
                operations.append("content read")
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
        matched = bool(predicate(search_target, searchable_content))
    except Exception:
        matched = False
    return path, content, matched, operations


def main(argv: list[str]) -> int:
    try:
        (
            query,
            include_content,
            search_base_only,
            recursive,
            verbose,
            include_pdf,
            ocr_pdf,
            wip,
            cpu_limit,
            sort_results,
            sort_case_sensitive,
            patterns,
            excluded_path_args,
            follow_links,
        ) = _parse_args(argv)
        predicate = search_query.compile_match_predicate(
            query, strip_inline_image_data=False
        )

        if include_pdf and not include_content:
            print("note: --pdf/--ocr-pdf has no effect unless --content/-c is set", file=sys.stderr)
        if sort_results:
            print(
                "One moment please... finding all matches to sort them",
                file=sys.stderr,
                flush=True,
            )

        match_count = 0
        excluded_paths = tuple(
            _normalized_exclude_path(path) for path in excluded_path_args
        )
        candidate_iter = _iter_candidate_paths(
            patterns,
            recursive,
            excluded_paths,
            follow_links,
        )
        saw_candidate = False
        buffered_matches: list[tuple[Path, str]] = []

        def _emit_wip(path: Path, operations: list[str]) -> None:
            if wip:
                print(_style_wip_filepath(path, operations), flush=True)

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
                path, content, matched, operations = _scan_candidate_for_query(
                    candidate_path,
                    predicate,
                    include_content=include_content,
                    search_base_only=search_base_only,
                    include_pdf=include_pdf,
                    ocr_pdf=ocr_pdf,
                )
                _emit_wip(path, operations)
                if not matched:
                    continue
                _emit_match(path, content)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                pending_futures = set()
                candidate_exhausted = False
                cpu_sampler = _CpuUsageSampler()
                if cpu_limit > 0:
                    active_worker_limit = min(
                        worker_count,
                        max(1, os.cpu_count() or 1),
                    )
                else:
                    active_worker_limit = worker_count

                while True:
                    cpu_percent = cpu_sampler.sample()
                    active_worker_limit = _adjust_worker_limit(
                        active_worker_limit,
                        cpu_percent,
                        cpu_limit,
                        worker_count,
                    )

                    while (
                        not candidate_exhausted
                        and len(pending_futures) < active_worker_limit
                    ):
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
                                search_base_only=search_base_only,
                                include_pdf=include_pdf,
                                ocr_pdf=ocr_pdf,
                            )
                        )

                    if not pending_futures:
                        break

                    done, pending_futures = wait(
                        pending_futures,
                        timeout=_CPU_SAMPLE_INTERVAL_SECONDS,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        # A timeout gives the controller a chance to observe
                        # sustained CPU pressure even while long OCR jobs run.
                        continue
                    for future in done:
                        try:
                            path, content, matched, operations = future.result()
                        except Exception:
                            continue
                        _emit_wip(path, operations)
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
        print("Search interrupted by user.", file=sys.stderr, flush=True)
        return 130


def _terminate_after_interrupt(exit_code: int) -> None:
    """Exit immediately after an already-handled Ctrl-C.

    ThreadPoolExecutor registers interpreter-shutdown callbacks that wait for
    worker threads. A second Ctrl-C during that shutdown can otherwise expose
    noisy threading/weakref/tempfile tracebacks even though hfind already
    handled the interruption. os._exit() intentionally bypasses those atexit
    callbacks only for the Ctrl-C exit path.
    """
    if exit_code != 130:
        raise SystemExit(exit_code)
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(130)


if __name__ == "__main__":
    _terminate_after_interrupt(main(sys.argv[1:]))
