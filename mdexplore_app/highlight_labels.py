from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def _normalized_color(color: str) -> str:
    return str(color or "").strip().lower()


def _load_labels(directory: Path, filename: str) -> dict[str, str]:
    path = Path(directory) / filename
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("labels"), dict):
        payload = payload["labels"]
    if not isinstance(payload, dict):
        return {}
    labels: dict[str, str] = {}
    for raw_color, raw_label in payload.items():
        if not isinstance(raw_color, str) or not isinstance(raw_label, str):
            continue
        color = _normalized_color(raw_color)
        label = raw_label.strip()
        if color and label:
            labels[color] = label
    return labels


def local_highlight_label(directory: Path, filename: str, color: str) -> str | None:
    return _load_labels(Path(directory), filename).get(_normalized_color(color))


def effective_highlight_label(directory: Path, filename: str, color: str) -> str | None:
    current = Path(directory).resolve()
    color_key = _normalized_color(color)
    while True:
        label = _load_labels(current, filename).get(color_key)
        if label:
            return label
        parent = current.parent
        if parent == current:
            return None
        current = parent


def label_assignment_directory(directory: Path, use_parent: bool) -> Path:
    current = Path(directory).resolve()
    if use_parent and current.parent != current:
        return current.parent
    return current


def highlight_label_menu_text(
    directory: Path,
    filename: str,
    color: str,
    color_name: str,
    max_chars: int = 25,
) -> str:
    text = effective_highlight_label(directory, filename, color) or str(color_name)
    limit = max(1, int(max_chars))
    if len(text) <= limit:
        return text
    if limit == 1:
        return text[:1]
    return text[: limit - 1] + "…"


def set_highlight_label(directory: Path, filename: str, color: str, label: str) -> None:
    directory = Path(directory).resolve()
    path = directory / filename
    labels = _load_labels(directory, filename)
    color_key = _normalized_color(color)
    cleaned = str(label or "").strip()
    if cleaned:
        labels[color_key] = cleaned
    else:
        labels.pop(color_key, None)

    if not labels:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return

    payload = {"version": 1, "labels": dict(sorted(labels.items()))}
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
