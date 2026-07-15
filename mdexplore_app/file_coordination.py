"""Small cross-process coordination helpers for shared sidecar/cache files."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Callable, Iterator


@contextmanager
def advisory_file_lock(
    lock_path: Path,
    *,
    exclusive: bool = True,
    blocking: bool = True,
    timeout_seconds: float | None = None,
    should_abort: Callable[[], bool] | None = None,
    poll_interval_seconds: float = 0.01,
) -> Iterator[bool]:
    """Hold one stable advisory lock inode for the duration of the context.

    Supplying ``should_abort`` or ``timeout_seconds`` turns a blocking request
    into cooperative non-blocking polling, allowing idle workers and shutdown
    paths to stop without becoming stuck behind another process.
    """
    handle = None
    acquired = False
    try:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("a+b")
        except OSError:
            # Shared caches are best-effort. Callers that require durability
            # can turn this false result into an explicit save error, while
            # search/prefetch workers can continue without disk coordination.
            yield False
            return
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        cooperative = bool(
            blocking and (callable(should_abort) or timeout_seconds is not None)
        )
        if not blocking or cooperative:
            operation |= fcntl.LOCK_NB
        if cooperative:
            deadline = (
                time.monotonic() + max(0.0, float(timeout_seconds))
                if timeout_seconds is not None
                else None
            )
            while True:
                if callable(should_abort) and should_abort():
                    break
                try:
                    fcntl.flock(handle.fileno(), operation)
                    acquired = True
                    break
                except BlockingIOError:
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    time.sleep(max(0.001, float(poll_interval_seconds)))
                except OSError:
                    break
        else:
            try:
                fcntl.flock(handle.fileno(), operation)
                acquired = True
            except (BlockingIOError, OSError):
                acquired = False
        yield acquired
    finally:
        if handle is not None:
            if acquired:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            try:
                handle.close()
            except OSError:
                pass


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace a text file atomically with a process-unique temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        temporary.write_text(text, encoding=encoding)
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def load_files_payload(path: Path) -> dict[str, object]:
    """Read either a wrapped or legacy flat JSON files mapping."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    files = payload.get("files", payload)
    if not isinstance(files, dict):
        return {}
    return {
        str(name): value
        for name, value in files.items()
        if isinstance(name, str)
    }


def update_files_sidecar(
    path: Path,
    updates: dict[str, object | None],
    *,
    replace_all: bool = False,
) -> dict[str, object]:
    """Merge named entries into a JSON sidecar under a cross-process lock.

    A value of ``None`` is a tombstone. The returned mapping is the exact state
    committed to disk, allowing callers to refresh stale per-instance caches.
    """
    lock_path = path.with_name(f"{path.name}.lock")
    with advisory_file_lock(lock_path, exclusive=True, blocking=True) as acquired:
        if not acquired:
            raise OSError(f"Unable to acquire sidecar lock: {lock_path}")
        merged = {} if replace_all else load_files_payload(path)
        for raw_name, value in updates.items():
            name = str(raw_name)
            if value is None:
                merged.pop(name, None)
            else:
                merged[name] = value
        if merged:
            atomic_write_text(
                path,
                json.dumps(
                    {"files": dict(sorted(merged.items()))},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
        else:
            path.unlink(missing_ok=True)
        return merged


def transform_files_sidecar_entry(
    path: Path,
    name: str,
    transform: Callable[[object | None], object | None],
) -> tuple[object | None, dict[str, object]]:
    """Atomically transform one named sidecar entry from its latest value.

    The callback runs while the sidecar's process lock is held. Returning
    ``None`` deletes the entry. The returned tuple contains the committed entry
    value and complete committed mapping, which lets callers replace stale
    in-process snapshots without performing an unlocked follow-up read.
    """
    lock_path = path.with_name(f"{path.name}.lock")
    with advisory_file_lock(lock_path, exclusive=True, blocking=True) as acquired:
        if not acquired:
            raise OSError(f"Unable to acquire sidecar lock: {lock_path}")
        merged = load_files_payload(path)
        entry_name = str(name)
        updated = transform(merged.get(entry_name))
        if updated is None:
            merged.pop(entry_name, None)
        else:
            merged[entry_name] = updated
        if merged:
            atomic_write_text(
                path,
                json.dumps(
                    {"files": dict(sorted(merged.items()))},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
        else:
            # Deleting the final entry is the transaction's durable commit.
            # Propagate failure so callers do not report a successful removal
            # while the sidecar still contains the supposedly deleted value.
            path.unlink(missing_ok=True)
        return merged.get(entry_name), merged
