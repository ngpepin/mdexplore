"""Shared BASE64 helpers with optional SIMD/native acceleration."""

from __future__ import annotations

import atexit
import base64 as _stdlib_base64
import ctypes
import json
from pathlib import Path
import platform
import subprocess
from threading import Lock
import threading
import time

try:
    import pybase64 as _simd_base64
except Exception:
    _simd_base64 = None

# Code-level safety switch for the vendor AVX2 backend.
# Set to False to disable vendor/fastbase64 and use pybase64/stdlib only.
USE_VENDOR_FASTBASE64 = True

# Debug switch: run vendor and fallback decode paths for every call, time both,
# and log results. Output bytes still come from vendor path when available.
DEBUG_BENCHMARK_VENDOR_AND_FALLBACK = False

# Adaptive routing: prefer pybase64 overall and route only sweet-spot payloads
# to vendor/fastbase64. Sweet-spot thresholds are continuously tuned from
# sampled vendor-vs-pybase64 benchmark data and persisted on disk.
ADAPTIVE_VENDOR_ROUTING_ENABLED = True
ADAPTIVE_VENDOR_DEFAULT_MIN_CHARS = 50_000
ADAPTIVE_VENDOR_DEFAULT_MAX_CHARS = 1_000_000
ADAPTIVE_VENDOR_MIN_BUCKET_SAMPLES = 12
ADAPTIVE_VENDOR_BOOTSTRAP_SAMPLES = 4
ADAPTIVE_VENDOR_SAMPLE_EVERY_N_CALLS = 48
ADAPTIVE_VENDOR_REQUIRED_ADVANTAGE = 0.97
ADAPTIVE_VENDOR_PERSIST_EVERY_SAMPLES = 20

_VENDOR_FASTBASE64_DIR = Path(__file__).resolve().parents[1] / "vendor" / "fastbase64"
_VENDOR_FASTBASE64_LIB_PATH = _VENDOR_FASTBASE64_DIR / "libfastbase64.so"
_VENDOR_FASTBASE64_BENCHMARK_LOG_PATH = (
    Path(__file__).resolve().parents[1] / "fastbase64-benchmark.log"
)
_VENDOR_FASTBASE64_ADAPTIVE_STATE_PATH = (
    Path(__file__).resolve().parents[1] / "fastbase64-adaptive.json"
)
_VENDOR_FASTBASE64_INIT_LOCK = Lock()
_VENDOR_FASTBASE64_BENCHMARK_LOG_LOCK = Lock()
_VENDOR_FASTBASE64_ADAPTIVE_STATE_LOCK = Lock()
_VENDOR_FASTBASE64_INIT_DONE = False
_VENDOR_FASTBASE64_BACKEND = None
_VENDOR_FASTBASE64_ADAPTIVE_STATE_LOADED = False
_VENDOR_FASTBASE64_ADAPTIVE_STATE: dict[str, object] = {}
_VENDOR_FASTBASE64_ADAPTIVE_UNSAVED_SAMPLES = 0
_VENDOR_FASTBASE64_ROUTING_CALL_COUNT = 0

_ADAPTIVE_BUCKET_LIMITS: tuple[int, ...] = (
    0,
    50_000,
    200_000,
    500_000,
    1_000_000,
    2_000_000,
    3_000_000,
    5_000_000,
)


def _new_adaptive_buckets() -> list[dict[str, int | None]]:
    """Return zeroed benchmark buckets used for adaptive routing decisions."""
    buckets: list[dict[str, int | None]] = []
    for idx, lower in enumerate(_ADAPTIVE_BUCKET_LIMITS):
        upper: int | None = None
        if idx + 1 < len(_ADAPTIVE_BUCKET_LIMITS):
            upper = _ADAPTIVE_BUCKET_LIMITS[idx + 1]
        buckets.append(
            {
                "min_chars": int(lower),
                "max_chars": upper,
                "samples": 0,
                "vendor_ns_total": 0,
                "fallback_ns_total": 0,
            }
        )
    return buckets


def _new_adaptive_state() -> dict[str, object]:
    """Return default adaptive state."""
    return {
        "version": 1,
        "benchmark_samples": 0,
        "updated_ns": time.time_ns(),
        "tuned_vendor_min_chars": ADAPTIVE_VENDOR_DEFAULT_MIN_CHARS,
        "tuned_vendor_max_chars": ADAPTIVE_VENDOR_DEFAULT_MAX_CHARS,
        "buckets": _new_adaptive_buckets(),
    }


def _int_from_json(value: object, default: int) -> int:
    """Best-effort integer coercion for persisted state values."""
    try:
        return int(value)
    except Exception:
        return int(default)


def _optional_int_from_json(value: object) -> int | None:
    """Best-effort optional integer coercion for persisted state values."""
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _adaptive_bucket_index(payload_chars: int) -> int:
    """Return bucket index for payload size in chars."""
    normalized = max(0, int(payload_chars))
    for idx, lower in enumerate(_ADAPTIVE_BUCKET_LIMITS):
        upper = None
        if idx + 1 < len(_ADAPTIVE_BUCKET_LIMITS):
            upper = _ADAPTIVE_BUCKET_LIMITS[idx + 1]
        if upper is None or (normalized >= lower and normalized < upper):
            return idx
    return len(_ADAPTIVE_BUCKET_LIMITS) - 1


def _load_adaptive_state() -> None:
    """Load adaptive benchmark state from disk once per process."""
    global _VENDOR_FASTBASE64_ADAPTIVE_STATE_LOADED
    global _VENDOR_FASTBASE64_ADAPTIVE_STATE
    if _VENDOR_FASTBASE64_ADAPTIVE_STATE_LOADED:
        return
    with _VENDOR_FASTBASE64_ADAPTIVE_STATE_LOCK:
        if _VENDOR_FASTBASE64_ADAPTIVE_STATE_LOADED:
            return
        state = _new_adaptive_state()
        try:
            if _VENDOR_FASTBASE64_ADAPTIVE_STATE_PATH.is_file():
                raw = json.loads(
                    _VENDOR_FASTBASE64_ADAPTIVE_STATE_PATH.read_text(
                        encoding="utf-8", errors="replace"
                    )
                )
                if isinstance(raw, dict):
                    state["benchmark_samples"] = max(
                        0,
                        _int_from_json(
                            raw.get(
                                "benchmark_samples", state.get("benchmark_samples", 0)
                            ),
                            0,
                        ),
                    )
                    tuned_min = _optional_int_from_json(
                        raw.get(
                            "tuned_vendor_min_chars",
                            state.get("tuned_vendor_min_chars"),
                        )
                    )
                    tuned_max = _optional_int_from_json(
                        raw.get(
                            "tuned_vendor_max_chars",
                            state.get("tuned_vendor_max_chars"),
                        )
                    )
                    if tuned_min is not None and tuned_min < 0:
                        tuned_min = 0
                    if tuned_max is not None and tuned_max <= 0:
                        tuned_max = None
                    state["tuned_vendor_min_chars"] = tuned_min
                    state["tuned_vendor_max_chars"] = tuned_max
                    raw_buckets = raw.get("buckets")
                    if isinstance(raw_buckets, list):
                        expected = _new_adaptive_buckets()
                        if len(raw_buckets) == len(expected):
                            for idx, expected_bucket in enumerate(expected):
                                current = raw_buckets[idx]
                                if not isinstance(current, dict):
                                    continue
                                expected_bucket["samples"] = max(
                                    0,
                                    _int_from_json(
                                        current.get("samples", expected_bucket["samples"]),
                                        0,
                                    ),
                                )
                                expected_bucket["vendor_ns_total"] = max(
                                    0,
                                    _int_from_json(
                                        current.get(
                                            "vendor_ns_total",
                                            expected_bucket["vendor_ns_total"],
                                        ),
                                        0,
                                    ),
                                )
                                expected_bucket["fallback_ns_total"] = max(
                                    0,
                                    _int_from_json(
                                        current.get(
                                            "fallback_ns_total",
                                            expected_bucket["fallback_ns_total"],
                                        ),
                                        0,
                                    ),
                                )
                            state["buckets"] = expected
        except Exception:
            pass
        _VENDOR_FASTBASE64_ADAPTIVE_STATE = state
        _VENDOR_FASTBASE64_ADAPTIVE_STATE_LOADED = True


def _persist_adaptive_state_locked(force: bool = False) -> None:
    """Persist adaptive state to disk atomically."""
    global _VENDOR_FASTBASE64_ADAPTIVE_UNSAVED_SAMPLES
    if not force and (
        _VENDOR_FASTBASE64_ADAPTIVE_UNSAVED_SAMPLES
        < ADAPTIVE_VENDOR_PERSIST_EVERY_SAMPLES
    ):
        return
    try:
        _VENDOR_FASTBASE64_ADAPTIVE_STATE["updated_ns"] = time.time_ns()
        _VENDOR_FASTBASE64_ADAPTIVE_STATE_PATH.parent.mkdir(
            parents=True, exist_ok=True
        )
        payload = json.dumps(
            _VENDOR_FASTBASE64_ADAPTIVE_STATE,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        temp_path = _VENDOR_FASTBASE64_ADAPTIVE_STATE_PATH.with_suffix(
            _VENDOR_FASTBASE64_ADAPTIVE_STATE_PATH.suffix + ".tmp"
        )
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(_VENDOR_FASTBASE64_ADAPTIVE_STATE_PATH)
        _VENDOR_FASTBASE64_ADAPTIVE_UNSAVED_SAMPLES = 0
    except Exception:
        # Adaptive persistence should never affect runtime decode behavior.
        return


def _flush_adaptive_state_at_exit() -> None:
    """Flush adaptive state at interpreter shutdown (best effort)."""
    if not _VENDOR_FASTBASE64_ADAPTIVE_STATE_LOADED:
        return
    with _VENDOR_FASTBASE64_ADAPTIVE_STATE_LOCK:
        _persist_adaptive_state_locked(force=True)


atexit.register(_flush_adaptive_state_at_exit)


def _recompute_tuned_vendor_range_locked() -> None:
    """Recompute vendor sweet-spot thresholds from persisted bucket timings."""
    raw_buckets = _VENDOR_FASTBASE64_ADAPTIVE_STATE.get("buckets")
    if not isinstance(raw_buckets, list):
        return

    better_ranges: list[tuple[int, int, int, int]] = []
    current_start: int | None = None
    current_gain = 0
    current_samples = 0

    for idx, bucket in enumerate(raw_buckets):
        if not isinstance(bucket, dict):
            is_vendor_better = False
            samples = 0
            gain = 0
        else:
            samples = max(0, _int_from_json(bucket.get("samples"), 0))
            vendor_total = max(0, _int_from_json(bucket.get("vendor_ns_total"), 0))
            fallback_total = max(0, _int_from_json(bucket.get("fallback_ns_total"), 0))
            is_vendor_better = (
                samples >= ADAPTIVE_VENDOR_MIN_BUCKET_SAMPLES
                and vendor_total > 0
                and fallback_total > 0
                and vendor_total <= int(fallback_total * ADAPTIVE_VENDOR_REQUIRED_ADVANTAGE)
            )
            gain = max(0, fallback_total - vendor_total)

        if is_vendor_better:
            if current_start is None:
                current_start = idx
                current_gain = gain
                current_samples = samples
            else:
                current_gain += gain
                current_samples += samples
            continue

        if current_start is not None:
            better_ranges.append((current_start, idx - 1, current_gain, current_samples))
            current_start = None
            current_gain = 0
            current_samples = 0

    if current_start is not None:
        better_ranges.append(
            (current_start, len(raw_buckets) - 1, current_gain, current_samples)
        )

    if better_ranges:
        best_start, best_end, _best_gain, _best_samples = max(
            better_ranges,
            key=lambda item: (
                item[2],  # total timing gain
                item[3],  # sample count
                (item[1] - item[0]),  # span
            ),
        )
        first_bucket = raw_buckets[best_start]
        last_bucket = raw_buckets[best_end]
        if isinstance(first_bucket, dict) and isinstance(last_bucket, dict):
            tuned_min = _optional_int_from_json(first_bucket.get("min_chars"))
            tuned_max = _optional_int_from_json(last_bucket.get("max_chars"))
            if tuned_min is not None and tuned_min < 0:
                tuned_min = 0
            if tuned_max is not None and tuned_min is not None and tuned_max <= tuned_min:
                tuned_max = None
            _VENDOR_FASTBASE64_ADAPTIVE_STATE["tuned_vendor_min_chars"] = tuned_min
            _VENDOR_FASTBASE64_ADAPTIVE_STATE["tuned_vendor_max_chars"] = tuned_max
            return

    benchmark_samples = max(
        0,
        _int_from_json(
            _VENDOR_FASTBASE64_ADAPTIVE_STATE.get("benchmark_samples"),
            0,
        ),
    )
    if benchmark_samples < ADAPTIVE_VENDOR_MIN_BUCKET_SAMPLES:
        _VENDOR_FASTBASE64_ADAPTIVE_STATE["tuned_vendor_min_chars"] = (
            ADAPTIVE_VENDOR_DEFAULT_MIN_CHARS
        )
        _VENDOR_FASTBASE64_ADAPTIVE_STATE["tuned_vendor_max_chars"] = (
            ADAPTIVE_VENDOR_DEFAULT_MAX_CHARS
        )
        return
    _VENDOR_FASTBASE64_ADAPTIVE_STATE["tuned_vendor_min_chars"] = None
    _VENDOR_FASTBASE64_ADAPTIVE_STATE["tuned_vendor_max_chars"] = None


def _record_adaptive_benchmark_sample(
    *,
    payload_chars: int,
    vendor_elapsed_ns: int,
    fallback_elapsed_ns: int,
    fallback_backend_name: str,
) -> None:
    """Update adaptive bucket timings from a sampled vendor-vs-fallback run."""
    global _VENDOR_FASTBASE64_ADAPTIVE_UNSAVED_SAMPLES
    if fallback_backend_name != "pybase64":
        return
    _load_adaptive_state()
    with _VENDOR_FASTBASE64_ADAPTIVE_STATE_LOCK:
        raw_buckets = _VENDOR_FASTBASE64_ADAPTIVE_STATE.get("buckets")
        if not isinstance(raw_buckets, list) or not raw_buckets:
            return
        idx = _adaptive_bucket_index(payload_chars)
        if idx < 0 or idx >= len(raw_buckets):
            return
        bucket = raw_buckets[idx]
        if not isinstance(bucket, dict):
            return
        bucket["samples"] = max(0, _int_from_json(bucket.get("samples"), 0)) + 1
        bucket["vendor_ns_total"] = max(
            0, _int_from_json(bucket.get("vendor_ns_total"), 0)
        ) + max(0, int(vendor_elapsed_ns))
        bucket["fallback_ns_total"] = max(
            0, _int_from_json(bucket.get("fallback_ns_total"), 0)
        ) + max(0, int(fallback_elapsed_ns))
        _VENDOR_FASTBASE64_ADAPTIVE_STATE["benchmark_samples"] = max(
            0,
            _int_from_json(
                _VENDOR_FASTBASE64_ADAPTIVE_STATE.get("benchmark_samples"),
                0,
            ),
        ) + 1
        _VENDOR_FASTBASE64_ADAPTIVE_UNSAVED_SAMPLES += 1
        _recompute_tuned_vendor_range_locked()
        _persist_adaptive_state_locked(force=False)


def _should_prefer_vendor_backend(
    payload_chars: int,
    *,
    fallback_backend_hint: str,
) -> bool:
    """Return whether vendor backend should be preferred for this payload."""
    if fallback_backend_hint != "pybase64":
        return True
    if not ADAPTIVE_VENDOR_ROUTING_ENABLED:
        return True
    _load_adaptive_state()
    with _VENDOR_FASTBASE64_ADAPTIVE_STATE_LOCK:
        tuned_min = _optional_int_from_json(
            _VENDOR_FASTBASE64_ADAPTIVE_STATE.get("tuned_vendor_min_chars")
        )
        tuned_max = _optional_int_from_json(
            _VENDOR_FASTBASE64_ADAPTIVE_STATE.get("tuned_vendor_max_chars")
        )
    normalized = max(0, int(payload_chars))
    if tuned_min is not None and normalized < tuned_min:
        return False
    if tuned_max is not None and normalized >= tuned_max:
        return False
    if tuned_min is None and tuned_max is None:
        return False
    return True


def _benchmark_decision(
    payload_chars: int,
    *,
    fallback_backend_hint: str,
) -> tuple[bool, bool]:
    """
    Decide whether to dual-run benchmark this decode.

    Returns `(should_benchmark, run_vendor_first_for_timing)`.
    """
    global _VENDOR_FASTBASE64_ROUTING_CALL_COUNT
    if fallback_backend_hint != "pybase64":
        return False, True
    _load_adaptive_state()
    with _VENDOR_FASTBASE64_ADAPTIVE_STATE_LOCK:
        _VENDOR_FASTBASE64_ROUTING_CALL_COUNT += 1
        call_count = _VENDOR_FASTBASE64_ROUTING_CALL_COUNT
        if DEBUG_BENCHMARK_VENDOR_AND_FALLBACK:
            return True, bool(call_count % 2 == 0)
        raw_buckets = _VENDOR_FASTBASE64_ADAPTIVE_STATE.get("buckets")
        if not isinstance(raw_buckets, list) or not raw_buckets:
            return False, bool(call_count % 2 == 0)
        bucket_idx = _adaptive_bucket_index(payload_chars)
        if bucket_idx < 0 or bucket_idx >= len(raw_buckets):
            return False, bool(call_count % 2 == 0)
        bucket = raw_buckets[bucket_idx]
        if not isinstance(bucket, dict):
            return False, bool(call_count % 2 == 0)
        bucket_samples = max(0, _int_from_json(bucket.get("samples"), 0))
        if bucket_samples < ADAPTIVE_VENDOR_BOOTSTRAP_SAMPLES:
            return True, bool(call_count % 2 == 0)
        return (
            bool(call_count % ADAPTIVE_VENDOR_SAMPLE_EVERY_N_CALLS == 0),
            bool(call_count % 2 == 0),
        )


def _cpu_supports_avx2() -> bool:
    """Best-effort AVX2 capability probe on Linux."""
    if platform.system() != "Linux":
        return False
    machine = platform.machine().strip().lower()
    if machine not in {"x86_64", "amd64"}:
        return False
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return "avx2" in cpuinfo.casefold()


def _ensure_vendor_fastbase64_shared_lib() -> bool:
    """Ensure vendor fastbase64 shared library exists."""
    if _VENDOR_FASTBASE64_LIB_PATH.is_file():
        return True
    if not _VENDOR_FASTBASE64_DIR.is_dir():
        return False
    makefile_path = _VENDOR_FASTBASE64_DIR / "Makefile"
    if not makefile_path.is_file():
        return False
    try:
        result = subprocess.run(
            ["make", "libfastbase64.so"],
            cwd=str(_VENDOR_FASTBASE64_DIR),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    return _VENDOR_FASTBASE64_LIB_PATH.is_file()


class _VendorFastBase64Backend:
    """ctypes bridge for vendor/fastbase64 shared library."""

    def __init__(self, lib_path: Path):
        if not lib_path.is_file():
            raise FileNotFoundError(lib_path)
        self._lib = ctypes.CDLL(str(lib_path))
        self._modp_error = ctypes.c_size_t(-1).value

        char_ptr = ctypes.POINTER(ctypes.c_char)
        self._encode = self._lib.fast_avx2_base64_encode
        self._encode.argtypes = [char_ptr, char_ptr, ctypes.c_size_t]
        self._encode.restype = ctypes.c_size_t

        self._decode = self._lib.fast_avx2_base64_decode
        self._decode.argtypes = [char_ptr, char_ptr, ctypes.c_size_t]
        self._decode.restype = ctypes.c_size_t

    def encode_ascii(self, raw_bytes: bytes) -> str | None:
        source = bytes(raw_bytes or b"")
        if not source:
            return ""
        source_len = len(source)
        destination_len = ((source_len + 2) // 3) * 4 + 1
        destination_buffer = ctypes.create_string_buffer(destination_len)
        source_buffer = ctypes.create_string_buffer(source, source_len)
        output_len = int(
            self._encode(destination_buffer, source_buffer, ctypes.c_size_t(source_len))
        )
        if output_len == self._modp_error:
            return None
        if output_len <= 0 or output_len > destination_len:
            return None
        encoded = destination_buffer.raw[: output_len - 1]
        try:
            return encoded.decode("ascii")
        except Exception:
            return None

    def decode_loose(self, payload: str) -> bytes | None:
        compact_payload = "".join(str(payload or "").split())
        if not compact_payload:
            return b""
        remainder = len(compact_payload) % 4
        if remainder:
            compact_payload += "=" * (4 - remainder)
        try:
            source = compact_payload.encode("ascii")
        except Exception:
            return None
        source_len = len(source)
        destination_len = (source_len // 4) * 3 + 2
        if destination_len <= 0:
            return b""
        destination_buffer = ctypes.create_string_buffer(destination_len)
        source_buffer = ctypes.create_string_buffer(source, source_len)
        output_len = int(
            self._decode(destination_buffer, source_buffer, ctypes.c_size_t(source_len))
        )
        if output_len == self._modp_error:
            return None
        if output_len < 0 or output_len > destination_len:
            return None
        return destination_buffer.raw[:output_len]


def _get_vendor_fastbase64_backend() -> _VendorFastBase64Backend | None:
    """Return initialized vendor fastbase64 backend when enabled/available."""
    global _VENDOR_FASTBASE64_INIT_DONE, _VENDOR_FASTBASE64_BACKEND
    if not USE_VENDOR_FASTBASE64:
        return None
    if _VENDOR_FASTBASE64_INIT_DONE:
        return _VENDOR_FASTBASE64_BACKEND
    with _VENDOR_FASTBASE64_INIT_LOCK:
        if _VENDOR_FASTBASE64_INIT_DONE:
            return _VENDOR_FASTBASE64_BACKEND
        _VENDOR_FASTBASE64_INIT_DONE = True
        if not _cpu_supports_avx2():
            _VENDOR_FASTBASE64_BACKEND = None
            return None
        if not _ensure_vendor_fastbase64_shared_lib():
            _VENDOR_FASTBASE64_BACKEND = None
            return None
        try:
            _VENDOR_FASTBASE64_BACKEND = _VendorFastBase64Backend(
                _VENDOR_FASTBASE64_LIB_PATH
            )
        except Exception:
            _VENDOR_FASTBASE64_BACKEND = None
        return _VENDOR_FASTBASE64_BACKEND


def using_simd_base64() -> bool:
    """Return whether SIMD/native BASE64 acceleration is available."""
    return _simd_base64 is not None


def _decode_with_fallback_backend(payload: str) -> tuple[bytes | None, str]:
    """Decode with non-vendor backend (`pybase64` preferred, stdlib fallback)."""
    if _simd_base64 is not None:
        try:
            return _simd_base64.b64decode(payload, validate=False), "pybase64"
        except Exception:
            pass
    try:
        return _stdlib_base64.b64decode(payload, validate=False), "stdlib"
    except Exception:
        return None, "stdlib"


def _decode_with_vendor_backend(
    vendor_backend: _VendorFastBase64Backend,
    payload: str,
) -> bytes | None:
    """Decode with vendor backend (best effort, no exceptions)."""
    try:
        return vendor_backend.decode_loose(payload)
    except Exception:
        return None


def _log_vendor_benchmark_result(
    *,
    payload_chars: int,
    vendor_elapsed_ns: int,
    compare_elapsed_ns: int,
    compare_backend_name: str,
    vendor_output: bytes | None,
    compare_output: bytes | None,
) -> None:
    """Append one vendor-vs-fallback decode benchmark record."""
    try:
        vendor_ok = vendor_output is not None
        compare_ok = compare_output is not None
        outputs_equal = (
            vendor_ok and compare_ok and bytes(vendor_output) == bytes(compare_output)
        )
        vendor_bytes = len(vendor_output or b"")
        compare_bytes = len(compare_output or b"")
        winner = "vendor" if vendor_elapsed_ns <= compare_elapsed_ns else "fallback"
        line = (
            f"ts_ns={time.time_ns()} "
            f"thread={threading.get_ident()} "
            f"payload_chars={int(payload_chars)} "
            f"vendor_ns={int(vendor_elapsed_ns)} "
            f"fallback_ns={int(compare_elapsed_ns)} "
            f"fallback_backend={compare_backend_name} "
            f"vendor_ok={1 if vendor_ok else 0} "
            f"fallback_ok={1 if compare_ok else 0} "
            f"equal={1 if outputs_equal else 0} "
            f"vendor_bytes={vendor_bytes} "
            f"fallback_bytes={compare_bytes} "
            f"winner={winner}\n"
        )
        with _VENDOR_FASTBASE64_BENCHMARK_LOG_LOCK:
            _VENDOR_FASTBASE64_BENCHMARK_LOG_PATH.parent.mkdir(
                parents=True, exist_ok=True
            )
            with _VENDOR_FASTBASE64_BENCHMARK_LOG_PATH.open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(line)
    except Exception:
        # Benchmark logging must never interfere with decode behavior.
        return


def b64encode_ascii(raw_bytes: bytes) -> str:
    """Encode bytes to ASCII BASE64 text with SIMD fast path when available."""
    vendor_backend = _get_vendor_fastbase64_backend()
    if vendor_backend is not None:
        try:
            encoded = vendor_backend.encode_ascii(raw_bytes)
            if encoded is not None:
                return encoded
        except Exception:
            pass
    if _simd_base64 is not None:
        try:
            return _simd_base64.b64encode(raw_bytes).decode("ascii")
        except Exception:
            pass
    return _stdlib_base64.b64encode(raw_bytes).decode("ascii")


def b64decode_loose(payload: str) -> bytes:
    """Decode BASE64 text allowing non-strict input (whitespace/padding quirks)."""
    payload_text = str(payload or "")
    payload_chars = len(payload_text)
    vendor_backend = _get_vendor_fastbase64_backend()
    if vendor_backend is None:
        decoded, _backend_name = _decode_with_fallback_backend(payload_text)
        if decoded is not None:
            return decoded
        return _stdlib_base64.b64decode(payload_text, validate=False)

    fallback_backend_hint = "pybase64" if _simd_base64 is not None else "stdlib"
    prefer_vendor = _should_prefer_vendor_backend(
        payload_chars,
        fallback_backend_hint=fallback_backend_hint,
    )
    should_benchmark, benchmark_vendor_first = _benchmark_decision(
        payload_chars,
        fallback_backend_hint=fallback_backend_hint,
    )

    if should_benchmark:
        vendor_decoded: bytes | None = None
        compare_decoded: bytes | None = None
        compare_backend_name = fallback_backend_hint
        vendor_elapsed_ns = 0
        compare_elapsed_ns = 0

        if benchmark_vendor_first:
            start_vendor_ns = time.perf_counter_ns()
            vendor_decoded = _decode_with_vendor_backend(vendor_backend, payload_text)
            vendor_elapsed_ns = time.perf_counter_ns() - start_vendor_ns

            start_compare_ns = time.perf_counter_ns()
            compare_decoded, compare_backend_name = _decode_with_fallback_backend(
                payload_text
            )
            compare_elapsed_ns = time.perf_counter_ns() - start_compare_ns
        else:
            start_compare_ns = time.perf_counter_ns()
            compare_decoded, compare_backend_name = _decode_with_fallback_backend(
                payload_text
            )
            compare_elapsed_ns = time.perf_counter_ns() - start_compare_ns

            start_vendor_ns = time.perf_counter_ns()
            vendor_decoded = _decode_with_vendor_backend(vendor_backend, payload_text)
            vendor_elapsed_ns = time.perf_counter_ns() - start_vendor_ns

        _log_vendor_benchmark_result(
            payload_chars=payload_chars,
            vendor_elapsed_ns=vendor_elapsed_ns,
            compare_elapsed_ns=compare_elapsed_ns,
            compare_backend_name=compare_backend_name,
            vendor_output=vendor_decoded,
            compare_output=compare_decoded,
        )
        _record_adaptive_benchmark_sample(
            payload_chars=payload_chars,
            vendor_elapsed_ns=vendor_elapsed_ns,
            fallback_elapsed_ns=compare_elapsed_ns,
            fallback_backend_name=compare_backend_name,
        )
        if prefer_vendor and vendor_decoded is not None:
            return vendor_decoded
        if compare_decoded is not None:
            return compare_decoded
        if vendor_decoded is not None:
            return vendor_decoded
    elif prefer_vendor:
        vendor_decoded = _decode_with_vendor_backend(vendor_backend, payload_text)
        if vendor_decoded is not None:
            return vendor_decoded
        compare_decoded, _backend_name = _decode_with_fallback_backend(payload_text)
        if compare_decoded is not None:
            return compare_decoded
    else:
        compare_decoded, _backend_name = _decode_with_fallback_backend(payload_text)
        if compare_decoded is not None:
            return compare_decoded
        vendor_decoded = _decode_with_vendor_backend(vendor_backend, payload_text)
        if vendor_decoded is not None:
            return vendor_decoded

    return _stdlib_base64.b64decode(payload_text, validate=False)
