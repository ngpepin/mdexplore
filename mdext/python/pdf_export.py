#!/usr/bin/env python3
"""Render a self-contained mdExt HTML snapshot with Chromium's native PDF engine."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from PySide6.QtCore import QMarginsF, QTimer, QUrl
from PySide6.QtGui import QGuiApplication, QPageLayout, QPageSize
from PySide6.QtWebEngineCore import QWebEnginePage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print an mdExt HTML snapshot to a vector PDF")
    parser.add_argument("--html", required=True, help="Path to the prepared HTML snapshot")
    parser.add_argument("--output", required=True, help="Destination PDF path")
    parser.add_argument("--timeout-ms", type=int, default=60000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    html_path = Path(args.html).resolve()
    output_path = Path(args.output).resolve()
    if not html_path.is_file():
        print(f"HTML snapshot does not exist: {html_path}", file=sys.stderr)
        return 2

    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("QT_QUICK_BACKEND", "software")
        os.environ.setdefault("QT_OPENGL", "software")
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

    app = QGuiApplication(sys.argv[:1])
    page = QWebEnginePage()
    page_layout = QPageLayout(
        QPageSize(QPageSize.PageSizeId.Letter),
        QPageLayout.Orientation.Portrait,
        QMarginsF(0.6, 0.55, 0.6, 0.65),
        QPageLayout.Unit.Inch,
    )
    exit_code = 3

    def finish(code: int, message: str = "") -> None:
        nonlocal exit_code
        exit_code = code
        if message:
            print(message, file=sys.stderr)
        app.quit()

    def pdf_ready(data) -> None:
        payload = bytes(data)
        if not payload:
            finish(5, "Qt WebEngine returned an empty PDF payload")
            return
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(payload)
        except Exception as exc:
            finish(6, f"Could not write PDF: {exc}")
            return
        finish(0)

    def print_page() -> None:
        page.printToPdf(pdf_ready, page_layout)

    def page_ready(ok: bool) -> None:
        if not ok:
            finish(4, "Qt WebEngine could not load the prepared HTML snapshot")
            return
        # The webview already resolves MathJax/Mermaid before producing the
        # snapshot. One event-loop turn lets Chromium finish final print layout.
        QTimer.singleShot(0, print_page)

    page.loadFinished.connect(page_ready)
    page.load(QUrl.fromLocalFile(str(html_path)))
    QTimer.singleShot(max(1000, int(args.timeout_ms)), lambda: finish(7, "PDF export timed out"))
    app.exec()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
