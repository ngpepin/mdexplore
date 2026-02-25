# AGENTS.md

Guidance for automated coding agents that add features or maintain `mdexplore`.

## Mission

Maintain a fast, reliable Markdown explorer for Ubuntu/Linux desktop with:

- Left-pane directory tree rooted at a target folder.
- Markdown-only file listing (`*.md`).
- Right-pane rendered preview with math and diagram support.
- `^`, `Refresh`, `PDF`, `Add View`, and `Edit` actions.
- Top-right copy-by-color controls for clipboard file operations.
- A pin button before copy-by-color controls to copy the currently previewed file.

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
- `pypdf` and `reportlab` available for PDF page-number stamping.
- Mermaid should prefer a local bundle when available and only use CDN fallback.
- MathJax should prefer a local bundle when available and only use CDN fallback.
- Java runtime is expected for local PlantUML rendering.
- VS Code `code` command may or may not be installed; app should fail gracefully when missing.

## Core Behavior You Must Preserve

- Without a path arg, default root is loaded from `~/.mdexplore.cfg` if valid;
  otherwise home directory is used.
- Both entrypoints support optional root path argument:
  - `mdexplore.sh [PATH]`
  - `mdexplore.py [PATH]`
- Tree shows directories and `.md` files only.
- Selecting a Markdown file updates preview quickly.
- GitHub/Obsidian-style markdown callouts (`> [!TYPE]`) should render as styled
  callout boxes in preview and PDF output.
- Navigating to a previously cached file must still stat-check mtime/size;
  if changed, it must re-render instead of showing stale cached HTML.
- PlantUML rendering jobs must run off the UI thread to keep the window responsive.
- PlantUML blocks should not block markdown preview; show placeholders first,
  then progressively replace each diagram as local render jobs complete.
- PlantUML progressive updates should be applied in-place so the preview scroll
  location is preserved.
- PlantUML progress should persist across file navigation so returning to a
  file shows completed diagrams without restarting work.
- Preview scroll position should be remembered per markdown file for the
  current application run.
- `^` navigates one directory level up and re-roots the tree.
- User-adjusted splitter width between tree and preview should persist across
  root changes for the current application run.
- Window title reflects effective root scope (selected directory if selected, otherwise current root).
- Effective root is persisted on close to `~/.mdexplore.cfg`.
- Linux desktop identity should remain `mdexplore` so launcher icon matching
  works (`QApplication.setDesktopFileName("mdexplore")` + desktop
  `StartupWMClass=mdexplore`).
- `Edit` opens currently selected file with `code`.
- `Add View` creates another tabbed view of the same current document, starting
  from the current top-visible line/scroll position.
- View tabs show top-visible source line numbers, are closeable, and capped at
  8 tabs per document.
- View tabs include a left-side mini bargraph indicating each view's position in
  the current document.
- View tabs should keep a stable soft-pastel color sequence based on open order.
- View tabs are draggable/reorderable; moving tabs must not change their assigned
  color.
- When assigning a new tab color and wrapping the palette, skip any color already
  used by open tabs.
- Per-document tabs should persist for the app run: switching to another markdown
  file and back must restore prior tabs, their order, and the previously selected
  tab.
- The tab strip should remain hidden when there is only one view.
- `Refresh` button and `F5` both refresh the current directory tree view
  (new files appear, deleted files disappear) without requiring app restart.
- `PDF` exports the currently previewed markdown rendering to
  `<source-basename>.pdf` beside the source file, with centered `N of M`
  numbering at the bottom of each page.
- PDF export should preflight preview readiness (MathJax/Mermaid/fonts) before
  snapshotting, and apply print-focused math styling to avoid cramped glyphs.
- If the currently previewed markdown file changes on disk, preview should
  auto-refresh and report that via status bar message.
- Status bar should show progress during long-running operations (preview
  loading/rendering, PlantUML completion, PDF export) and should not remain
  blank during idle gaps.
- Search input uses label `Search and highlight:` and includes an explicit in-field `X`
  clear action.
- Pressing `Enter` in search should bypass debounce and run search immediately.
- Non-quoted search terms should be case-insensitive; quoted terms should be
  case-sensitive.
- `CLOSE(...)` should be supported in search queries and require all listed
  terms to occur within `SEARCH_CLOSE_WORD_GAP` content words.
- Function-style operators should accept both no-space and spaced forms before
  `(` (for example `OR(...)` and `OR (...)`).
- `AND(...)`, `OR(...)`, and `CLOSE(...)` should accept comma-delimited,
  space-delimited, or mixed argument lists.
- `AND(...)` and `OR(...)` should be variadic (2+ arguments).
- If search is active and directory scope changes, search should rerun against
  the new directory scope.
- File highlight colors are assigned from tree context menu and persisted per directory.
- Highlight state persists in `.mdexplore-colors.json` files where writable.
- `Clear All` in the context menu recursively removes highlight metadata after confirmation.
- Copy/Clear scope resolves in this order: selected directory, then most
  recently selected/expanded directory, then current root.
- Clipboard copy must preserve Nemo/Nautilus compatibility (`text/uri-list` plus `x-special/gnome-copied-files`).
- The pin copy action should copy exactly the currently previewed markdown file
  using the same clipboard MIME compatibility guarantees.
- Preview context menu should keep standard actions and add
  `Copy Rendered Text` and `Copy Source Markdown` when there is a text
  selection.
- `Copy Rendered Text` should copy the selected preview text as plain text.
- `Copy Source Markdown` should map preview selection to source markdown line
  ranges and copy source text (not rendered plain text).
  If direct mapping fails, it should use selected-text matching and first/last
  line fuzzy matching against source markdown lines, then fall back to copying
  the full source file.
- While search is active, opening a matched markdown file should highlight
  matched terms in yellow in preview and scroll to the first match.

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
- Mermaid loading order is local-first, then CDN fallback.
- Mermaid rendering should use a dark-theme-friendly high-contrast palette when
  preview background is dark.
- Mermaid SVG render results should be cached in-memory (by diagram hash and
  render mode) for the current app run so revisiting documents avoids rerender.
- PDF export should temporarily switch Mermaid to a light-background-friendly
  palette, then restore preview palette after export.
- `MDEXPLORE_MERMAID_JS` can be used to force a specific local Mermaid script path.
- MathJax loading order is local-first, then CDN fallback.
- `MDEXPLORE_MATHJAX_JS` can be used to force a specific local MathJax script path.
- Callout syntax is parsed from blockquotes using `[!TYPE]` markers and rendered
  as non-collapsible callout containers.
- PlantUML fences are rendered locally through `plantuml.jar` (`java -jar ...`),
  defaulting to `vendor/plantuml/plantuml.jar` unless `PLANTUML_JAR` is set.
- PlantUML failures should render inline as:
  `PlantUML render failed with error ...`, including detailed stderr context
  (line numbers when available).
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
  `PySide6.QtWebEngineWidgets`, `pypdf`, `reportlab.pdfgen.canvas`) and
  self-heal by reinstalling requirements if the environment is incomplete.
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
