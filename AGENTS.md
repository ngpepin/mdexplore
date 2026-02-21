# AGENTS.md

Guidance for automated coding agents that add features or maintain `mdexplore`.

## Mission

Maintain a fast, reliable Markdown explorer for Ubuntu/Linux desktop with:

- Left-pane directory tree rooted at a target folder.
- Markdown-only file listing (`*.md`).
- Right-pane rendered preview with math and diagram support.
- `^`, `Quit`, and `Edit` actions.
- Top-right copy-by-color controls for clipboard file operations.

## Repository Map

- `mdexplore.py`: main application (Qt UI, renderer, file loading, cache, CLI path arg).
- `mdexplore.sh`: launcher (venv lifecycle + dependency install + app run).
- `requirements.txt`: Python runtime dependencies.
- `README.md`: user-facing setup and usage documentation.
- `mdexplore.desktop.sample`: user-adaptable `.desktop` launcher template.
- `mdexplor-icon.png`: primary app icon asset (preferred).
- `DESCRIPTION.md`: short repository summary and suggested topic tags.
- `LICENSE`: MIT license text.
- Runtime config file: `~/.mdexplore.cfg` (persisted last effective root).

## Runtime Assumptions

- Python `3.10+`.
- Linux desktop with GUI support.
- `PySide6` and `QtWebEngine` available via pip dependencies.
- Network access is expected for MathJax/Mermaid CDNs and PlantUML server rendering.
- VS Code `code` command may or may not be installed; app should fail gracefully when missing.

## Core Behavior You Must Preserve

- Without a path arg, default root is loaded from `~/.mdexplore.cfg` if valid;
  otherwise home directory is used.
- Both entrypoints support optional root path argument:
  - `mdexplore.sh [PATH]`
  - `mdexplore.py [PATH]`
- Tree shows directories and `.md` files only.
- Selecting a Markdown file updates preview quickly.
- `^` navigates one directory level up and re-roots the tree.
- Window title reflects effective root scope (selected directory if selected, otherwise current root).
- Effective root is persisted on close to `~/.mdexplore.cfg`.
- Linux desktop identity should remain `mdexplore` so launcher icon matching
  works (`QApplication.setDesktopFileName("mdexplore")` + desktop
  `StartupWMClass=mdexplore`).
- `Edit` opens currently selected file with `code`.
- `F5` refreshes preview for the selected file.
- File highlight colors are assigned from tree context menu and persisted per directory.
- Highlight state persists in `.mdexplore-colors.json` files where writable.
- `Clear All` in the context menu recursively removes highlight metadata after confirmation.
- Copy/Clear scope resolves in this order: selected directory, then most
  recently selected/expanded directory, then current root.
- Clipboard copy must preserve Nemo/Nautilus compatibility (`text/uri-list` plus `x-special/gnome-copied-files`).
- Preview context menu should keep standard actions and add
  `Copy Rendered Text` and `Copy Source Markdown` when there is a text
  selection.
- `Copy Rendered Text` should copy the selected preview text as plain text.
- `Copy Source Markdown` should map preview selection to source markdown line
  ranges and copy source text (not rendered plain text).
  If direct mapping fails, it should use selected-text matching and first/last
  line fuzzy matching against source markdown lines, then fall back to copying
  the full source file.

## Editing Rules

- Keep code ASCII unless file already requires Unicode.
- Prefer type hints and straightforward, explicit control flow.
- Avoid large framework changes unless requested.
- Do not add heavy dependencies without clear user value.
- Keep startup and preview interactions responsive.
- Preserve cache semantics unless changing performance behavior intentionally.

## Rendering Rules

- Markdown parser is `markdown-it-py`.
- Mermaid and MathJax are rendered client-side in web view.
- PlantUML fences are encoded and rendered through a PlantUML server URL.
- Maintain base URL behavior (`setHtml(..., base_url)`) so relative links/images resolve.
- If adding new embedded syntaxes, implement via fenced code handling and document it.

## Launcher Rules

- `mdexplore.sh` must be runnable from any working directory.
- It must keep venv handling deterministic:
  - create if missing
  - install dependencies
  - run app
  - invoke `.venv/bin/python` directly (no shell activation required)
  - do not use `exec -a` with Python (can break venv resolution)
- Launcher arg handling must tolerate `.desktop` `%u` behavior:
  - empty argument should behave like "no path"
  - `file://` URIs should be decoded
  - file arguments should resolve to parent directory
- Non-interactive launcher runs should log to
  `~/.cache/mdexplore/launcher.log` for desktop troubleshooting.
  Log retention is capped to the most recent 1000 lines.
- Launcher should verify key runtime imports (`markdown_it`, `linkify_it`,
  `PySide6.QtWebEngineWidgets`) and self-heal by reinstalling requirements if
  the environment is incomplete.
- `--help` should stay lightweight and not require venv activation.

## Quality Gates Before Finishing

Run at least:

```bash
bash -n mdexplore.sh
python3 -m py_compile mdexplore.py
```

If behavior changes, also run manual smoke tests:

1. Launch with no path.
2. Launch with explicit path.
3. Open `.md` file and verify render.
4. Verify `Edit` behavior with and without `code` in `PATH`.

## Documentation Requirements

When changing behavior:

- Update `README.md` usage and examples.
- Update this file if agent-facing workflow or guarantees changed.

## Common Feature Patterns

- New keyboard action: add `QAction` in `_add_shortcuts`.
- New toolbar action: add button in top bar setup and handler method.
- New markdown extension: extend custom fence handling in `MarkdownRenderer`.
- New CLI option: update argparse setup in `main`, then mirror wrapper behavior in `mdexplore.sh`.

## Out of Scope Unless Requested

- Packaging into Snap/Deb/AppImage.
- Multi-tab editing or embedded text editor.
- Background file watching/auto-reload loops.
- Cross-platform polish outside Linux desktop behavior.
