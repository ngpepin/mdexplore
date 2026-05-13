"""Background workers used by pdfexplore."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal


class PdfSearchWorkerSignals(QObject):
    """Signals emitted by background PDF search workers."""

    finished = Signal(int, object, object, str)


class PdfSearchWorker(QRunnable):
    """Search direct child PDFs in one directory without blocking the UI thread."""

    def __init__(
        self,
        scope: Path,
        request_id: int,
        query: str,
        search_callback,
    ) -> None:
        super().__init__()
        self.scope = scope
        self.request_id = request_id
        self.query = query
        self.search_callback = search_callback
        self.signals = PdfSearchWorkerSignals()

    def run(self) -> None:
        try:
            matches, match_counts = self.search_callback(self.scope, self.query)
            self.signals.finished.emit(
                self.request_id,
                matches,
                match_counts,
                "",
            )
        except Exception as exc:
            self.signals.finished.emit(self.request_id, [], {}, str(exc))
