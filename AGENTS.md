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
- Shared BASE64 encode/decode helpers should remain in `mdexplore_app/fast_base64.py`, using adaptive routing between vendored `fastbase64` and `pybase64`, with stdlib fallback.
- Vendor `fastbase64` encode output must be validated before use; on malformed
  output, helpers should silently fall back to `pybase64`/stdlib so PlantUML
  and other `data:` URI consumers do not break.
- `MDEXPLORE_BASE64_IMAGE_THREADS` controls worker-pool size used for both preview inline data-image materialization and copy-time image-link prefetch.
