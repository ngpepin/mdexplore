# pdfexplore Notes

- Keep `pdfexplore` read-only with respect to PDF contents. Sidecars are allowed.
- Prefer sharing generic code with `mdexplore_app` when the behavior is not markdown-specific.
- Keep bundled `pdf.js` vendor files isolated under `pdfexplore/vendor/pdfjs/`.
- Persistent highlight behavior should remain resilient: malformed sidecars must not block browsing.
