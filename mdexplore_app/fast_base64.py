"""Shared BASE64 helpers with optional SIMD/native acceleration."""

from __future__ import annotations

import base64 as _stdlib_base64

try:
    import pybase64 as _simd_base64
except Exception:
    _simd_base64 = None


def using_simd_base64() -> bool:
    """Return whether SIMD/native BASE64 acceleration is available."""
    return _simd_base64 is not None


def b64encode_ascii(raw_bytes: bytes) -> str:
    """Encode bytes to ASCII BASE64 text with SIMD fast path when available."""
    if _simd_base64 is not None:
        try:
            return _simd_base64.b64encode(raw_bytes).decode("ascii")
        except Exception:
            pass
    return _stdlib_base64.b64encode(raw_bytes).decode("ascii")


def b64decode_loose(payload: str) -> bytes:
    """Decode BASE64 text allowing non-strict input (whitespace/padding quirks)."""
    if _simd_base64 is not None:
        try:
            return _simd_base64.b64decode(payload, validate=False)
        except Exception:
            pass
    return _stdlib_base64.b64decode(payload, validate=False)
