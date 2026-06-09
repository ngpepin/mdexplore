"""Public package surface for pdfexplore.

This package exports:
- `PdfExploreWindow`: main Qt window class.
- `main`: CLI/launcher entrypoint used by script wrappers.
"""

from .app import PdfExploreWindow, main

__all__ = ["PdfExploreWindow", "main"]
