"""JSON-backed runtime settings for pdfexplore."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SETTINGS_PATH = _PROJECT_ROOT / "pdfexplore.settings.json"


def _load_settings() -> dict[str, Any]:
    try:
        raw = _SETTINGS_PATH.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except Exception:
        return {"app": {}, "viewer_bridge": {}, "tree": {}}
    if not isinstance(parsed, dict):
        return {"app": {}, "viewer_bridge": {}, "tree": {}}
    return parsed


SETTINGS = _load_settings()
APP_SETTINGS: dict[str, Any] = dict(SETTINGS.get("app") or {})
VIEWER_BRIDGE_SETTINGS: dict[str, Any] = dict(SETTINGS.get("viewer_bridge") or {})
TREE_SETTINGS: dict[str, Any] = dict(SETTINGS.get("tree") or {})
