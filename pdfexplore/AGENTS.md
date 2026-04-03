# pdfexplore Notes

- Keep `pdfexplore` read-only with respect to PDF contents. Sidecars are allowed.
- Prefer sharing generic code with `mdexplore_app` when the behavior is not markdown-specific.
- Keep bundled `pdf.js` vendor files isolated under `pdfexplore/vendor/pdfjs/`.
- Persistent highlight behavior should remain resilient: malformed sidecars must not block browsing.
- Preserve mdexplore-style UX contracts where they apply to PDFs:
  - same tree scope rules for search/copy/highlight operations,
  - same copy metadata merge semantics for directory-copy mode,
  - same view-tab model (multi-view per document, hidden single default tab, max-views cap, custom labels, return-to-beginning actions),
  - same sidecar persistence philosophy (persist meaningful view sessions, avoid noisy default-state writes).
- Maintain compatibility with existing `.pdfexplore-views.json` sidecars:
  - accept both legacy single-view shape and current session shape,
  - never hard-fail on malformed or partial entries.
- Keep PDF preview caching in-memory per document path so switching between already-opened PDFs does not require full widget recreation.
