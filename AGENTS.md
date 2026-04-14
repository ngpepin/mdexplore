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
- The `Recent` dropdown must show up to 20 most recently navigated root directories, newest first.
- A root should be recorded only after it has been active for at least 30 seconds and the user navigates to another root.

## Recent Directory Persistence Rules

- Config file: `~/.mdexplore.cfg`
- Lock file: `~/.mdexplore.cfg.lock`
- Config payload (JSON):
  - `default_root`: string path
  - `recent_roots`: array of string paths (max 20, newest first)
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
