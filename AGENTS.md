# AGENTS.md

Guidance for automated coding agents that add features or maintain `mdexplore`.

## Mission

Maintain a fast, reliable Markdown explorer for Ubuntu/Linux desktop with:

- Left-pane directory tree rooted at a target folder.
- Markdown-only file listing (`*.md`).
- Right-pane rendered preview with math and diagram support.
- `^`, `Quit`, and `Edit` actions.

## Repository Map

- `mdexplore.py`: main application (Qt UI, renderer, file loading, cache, CLI path arg).
- `mdexplore.sh`: launcher (venv lifecycle + dependency install + app run).
- `requirements.txt`: Python runtime dependencies.
- `README.md`: user-facing setup and usage documentation.
- `DESCRIPTION.md`: short repository summary and suggested topic tags.
- `LICENSE`: MIT license text.

## Runtime Assumptions

- Python `3.10+`.
- Linux desktop with GUI support.
- `PySide6` and `QtWebEngine` available via pip dependencies.
- Network access is expected for MathJax/Mermaid CDNs and PlantUML server rendering.
- VS Code `code` command may or may not be installed; app should fail gracefully when missing.

## Core Behavior You Must Preserve

- The default root is current working directory when no path arg is provided.
- Both entrypoints support optional root path argument:
  - `mdexplore.sh [PATH]`
  - `mdexplore.py [PATH]`
- Tree shows directories and `.md` files only.
- Selecting a Markdown file updates preview quickly.
- `^` navigates one directory level up and re-roots the tree.
- `Edit` opens currently selected file with `code`.
- `F5` refreshes preview for the selected file.
- File highlight colors are assigned from tree context menu and persisted per directory.
- Highlight state persists in `.mdexplore-colors.json` files where writable.
- `Clear All` in the context menu recursively removes highlight metadata after confirmation.
- Copy-by-color buttons recurse from selected directory when a directory is selected, otherwise from current root.

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
