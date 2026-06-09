"""`python -m pdfexplore` launcher shim."""

from .app import main


if __name__ == "__main__":
    # Delegate all argument parsing and runtime setup to app.main().
    raise SystemExit(main())
