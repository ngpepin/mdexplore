"""JavaScript asset loading helpers for preview/page scripts."""

from __future__ import annotations

import re
from pathlib import Path

from .constants import JS_ASSET_DIR

_PLACEHOLDER_PATTERN = re.compile(r"__[A-Z0-9_]+__")
_JS_SOURCES: dict[str, str] | None = None


def _load_js_sources_from_disk() -> dict[str, str]:
    if not JS_ASSET_DIR.is_dir():
        raise FileNotFoundError(f"JavaScript asset directory not found: {JS_ASSET_DIR}")

    sources: dict[str, str] = {}
    for path in sorted(JS_ASSET_DIR.rglob("*.js")):
        key = path.relative_to(JS_ASSET_DIR).as_posix()
        sources[key] = path.read_text(encoding="utf-8")
    return sources


def preload_js_assets(*, force_reload: bool = False) -> dict[str, str]:
    """Load tracked JS assets into a shared in-memory registry."""
    global _JS_SOURCES
    if force_reload or _JS_SOURCES is None:
        _JS_SOURCES = _load_js_sources_from_disk()
    return dict(_JS_SOURCES)


def get_js_asset(script_name: str) -> str:
    """Return one preloaded JS asset by registry key."""
    sources = preload_js_assets()
    try:
        return sources[script_name]
    except KeyError as exc:
        raise FileNotFoundError(f"JavaScript asset not found: {script_name}") from exc


def render_js_asset(
    script_name: str, replacements: dict[str, str] | None = None
) -> str:
    """Render a JS asset template with explicit placeholder substitution."""
    script = get_js_asset(script_name)
    for placeholder, value in (replacements or {}).items():
        script = script.replace(placeholder, value)

    remaining = sorted(set(_PLACEHOLDER_PATTERN.findall(script)))
    if remaining:
        raise ValueError(
            f"Unresolved placeholders in JavaScript asset {script_name}: {remaining}"
        )
    return script

