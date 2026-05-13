"""HTML/template asset loading helpers for preview/page shells."""

from __future__ import annotations

import re

from .constants import TEMPLATE_ASSET_DIR

_PLACEHOLDER_PATTERN = re.compile(r"__[A-Z0-9_]+__")
_TEMPLATE_SOURCES: dict[str, str] | None = None


def _load_template_sources_from_disk() -> dict[str, str]:
    if not TEMPLATE_ASSET_DIR.is_dir():
        raise FileNotFoundError(
            f"Template asset directory not found: {TEMPLATE_ASSET_DIR}"
        )

    sources: dict[str, str] = {}
    for path in sorted(TEMPLATE_ASSET_DIR.rglob("*.html")):
        key = path.relative_to(TEMPLATE_ASSET_DIR).as_posix()
        sources[key] = path.read_text(encoding="utf-8")
    return sources


def preload_template_assets(*, force_reload: bool = False) -> dict[str, str]:
    """Load tracked HTML template assets into a shared in-memory registry."""
    global _TEMPLATE_SOURCES
    if force_reload or _TEMPLATE_SOURCES is None:
        _TEMPLATE_SOURCES = _load_template_sources_from_disk()
    return dict(_TEMPLATE_SOURCES)


def get_template_asset(template_name: str) -> str:
    """Return one preloaded HTML template asset by registry key."""
    sources = preload_template_assets()
    try:
        return sources[template_name]
    except KeyError as exc:
        raise FileNotFoundError(
            f"HTML template asset not found: {template_name}"
        ) from exc


def render_template_asset(
    template_name: str, replacements: dict[str, str] | None = None
) -> str:
    """Render an HTML template asset with explicit placeholder substitution."""
    replacements = replacements or {}
    template = get_template_asset(template_name)
    required_placeholders = sorted(set(_PLACEHOLDER_PATTERN.findall(template)))
    missing = [name for name in required_placeholders if name not in replacements]
    if missing:
        raise ValueError(
            f"Unresolved placeholders in HTML template asset {template_name}: {missing}"
        )

    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)

    remaining = [name for name in required_placeholders if name in template]
    if remaining:
        raise ValueError(
            f"Unresolved placeholders in HTML template asset {template_name}: {remaining}"
        )
    return template
