# AGENTS.md

Quick maintenance notes for agent-driven edits to `mdexplore`.
For full architecture and behavior details, see `DEVELOPERS-AGENTS.md`.

## UI Controls That Must Be Preserved

- Top-left navigation actions include:
  - `Recent` (dropdown menu)
  - `^`
  - `Refresh`
  - `PDF`
  - `Add View`
  - `Edit`
- The `Recent` menu sits to the left of `^`.
- The `Recent` dropdown must show up to 35 retained root directories using this presentation:
  - first 10 shown most-recent-first,
  - then a separator,
  - then up to 25 remaining entries sorted alphabetically.
- A root should be recorded only after it has been active for at least 30 seconds and the user navigates to another root.

## Search + Scope Styling Rules

- Active search should run across markdown files currently visible in the tree
  (root + expanded branches), not just a single effective-scope directory.
- If search is active and tree visibility changes (expand/collapse/root/scope
  navigation), search should rerun automatically.
- Matching file rows should stay bold+italic with left-gutter hit-count pills.
- Filename-term matches should keep yellow filename text.
- Effective-root directory row should stay bold and:
  - aqua-blue (`#7fdfe8`) when no active search hits are under it,
  - yellow with an appended hit-count pill when active search has hits under it.
- Effective-root search-hit pill should mirror file-pill formatting (`1..99`,
  then `++`).
- `NEAR(...)` requires at least two terms. Six-term `NEAR(...)` queries are
  covered by regression testing, and the parser/matcher are intentionally
  variadic with no explicit upper bound beyond ordinary performance limits.

## Recent Directory Persistence Rules

- Config file: `~/.mdexplore.cfg`
- Lock file: `~/.mdexplore.cfg.lock`
- Config payload (JSON):
  - `default_root`: string path
  - `recent_roots`: array of string paths (max 35, rolling most-recent-first storage)
  - `copy_base64_images_enabled`: boolean toggle state for copy-time BASE64 embedding
- Writes occur on root navigation and again on shutdown.
- Backward compatibility: legacy plain-text config (single path line) must continue to load.
- Multi-instance behavior:
  - The `Recent` menu must re-read disk config each time the menu opens.
  - Config updates should use short non-blocking lock attempts; if lock acquisition fails, skip save silently.
  - Lock files older than 2 minutes should be deleted automatically and silently.

## Copy Toolbar Rules

- The copy area includes:
  - destination radios (`Clipboard`, `Directory`)
  - pin copy button
  - highlight-color copy buttons
  - BASE64 image toggle button (image icons on/off)
- PDF export should inline retrievable `<img>` sources to BASE64 data URIs before print; unresolved/broken links should remain unchanged.
- BASE64 toggle behavior:
  - Default is disabled on startup.
  - State should load from/persist to `~/.mdexplore.cfg`.
  - Tooltip when disabled: `Turn BASE64 image encoding on`.
  - Tooltip when enabled: `Turn BASE64 image encoding off`.
  - Applies only to copied outputs (clipboard staging and directory copy), never mutates source markdown files.
  - Retrievable image links (`file:`/relative paths/HTTP(S)) are embedded as BASE64 data URIs; unresolved links remain unchanged.

## Rendering and Performance Rules

- Markdown rendering should default to `cmarkgfm` fast path with automatic fallback to `markdown-it-py` for compatibility cases.
- `MDEXPLORE_MARKDOWN_ENGINE` should continue to support `cmark` (default), `markdown-it`, and `auto`.
- Startup-generated icon assets (app icon normalization and two-tone icon recolors)
  should persist under `~/.cache/mdexplore/icon-cache` and be reused on later
  launches unless source asset identity/render parameters change.
- Launcher runtime import verification should use and refresh
  `.venv/.runtime-import-check.sha256` (requirements-hash stamp) so the full
  import probe is not rerun on every launch.
- Shared BASE64 encode/decode helpers should remain in `mdexplore_app/fast_base64.py`, using adaptive routing between vendored `fastbase64` and `pybase64`, with stdlib fallback.
- Vendor `fastbase64` encode output must be validated before use; on malformed
  output, helpers should silently fall back to `pybase64`/stdlib so PlantUML
  and other `data:` URI consumers do not break.
- `MDEXPLORE_BASE64_IMAGE_THREADS` controls worker-pool size used for both preview inline data-image materialization and copy-time image-link prefetch.

## Per-Document View Persistence

- The existing per-directory sidecar is `.mdexplore-views.json` (plural
  `views`); do not introduce a competing `.mdexplore-view.json` file.
- Returning to a Markdown document across navigation or application restarts
  must restore its active view tab, tab positions, ordinary preview zoom, and
  active continuous/2-up/3-up/6-up layout.
- A meaningful position, non-default zoom, or page layout makes even an
  unlabeled single-view session eligible for persistence. An untouched
  single-view document at the beginning may still be omitted to avoid sidecar
  churn.
- A document with no saved view state opens in continuous page-width view at
  `100%` zoom and at the beginning.
- Tree view badges continue to mean that a document has multiple views; a
  single saved position or zoom must not create a multi-view badge.
- Preserve the existing merge-safe, atomic per-file sidecar update path so two
  running instances do not overwrite unrelated document sessions.

## Preview Image Context Menu

- Keep Qt WebEngine's native `Copy Image` action and behavior intact.
- Replace the native `Save Image` action with mdexplore's PNG-only save flow;
  do not rely on WebEngine's profile download handler for this action.
- `Save Image` must open a modal destination/filename chooser, suggest a
  `.png` filename, append `.png` when omitted, and rasterize the selected image
  as PNG. This includes images whose source is a BASE64 `data:` URI.
- Cancelling the chooser must not write a file. Conversion or write failures
  must produce a user-visible warning.

## Preview Page-Layout Hotkeys

- Markdown preview page layouts are `Ctrl+7` for 6-up, `Ctrl+8` for 3-up, and
  `Ctrl+9` for 2-up. These must paginate rendered Markdown into distinct page
  cards and arrange that many cards side-by-side per row; shrinking one
  continuous page is not an acceptable implementation.
- Each layout toggles back to continuous preview and the preview zoom that was
  active before entering the first page layout; switching layouts retains that
  baseline.
- When leaving a scrolled multipage layout, retain the page at the viewport's
  vertical centre. For odd-column layouts select the horizontally centred page;
  for 2-up facing pages select the left page. Anchor that page after unwrapping
  the grid instead of returning to the beginning or retaining an unrelated raw
  scroll offset.
- Entering a page layout from a scrolled continuous preview, or changing the
  number of pages displayed horizontally, must keep the same current content
  page centred after repagination. Never reset to the document beginning merely
  because the grid column count changed.
- Pagination must move the existing rendered DOM so links, selections,
  highlights, Mermaid/PlantUML interaction state, and source-line metadata are
  preserved. Do not clone or rerender the Markdown merely to change layout.
- PDF export must temporarily unwrap page cards into continuous source DOM and
  restore the active page layout afterward.
- These are literal unshifted number chords. Never implement them as `Ctrl+&`,
  `Ctrl+*`, `Ctrl+(`, or any `Ctrl+Shift+number` translation inferred from a
  particular keyboard layout.
- Keep each chord registered exactly once and regression-test delivery while
  the `QWebEngineView` preview has focus. `Ctrl+0` must continue to reset the
  preview to `100%` and leave any page layout.
