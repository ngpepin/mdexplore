# mdexplore

Fast Markdown explorer for Ubuntu/Linux desktop: browse `.md` files in a directory tree and preview fully rendered output instantly.

## Features

- Expandable left-pane directory tree rooted at a chosen folder.
- Shows Markdown files only (`*.md`), while still allowing folder navigation.
- Large right-pane rendered preview with scroll support.
- PlantUML diagram renders run in background workers so markdown preview stays responsive.
- Supports:
  - CommonMark + tables + strikethrough.
  - TeX/LaTeX math via MathJax.
  - Mermaid diagrams.
  - PlantUML diagrams (asynchronous local render with placeholders).
- Top actions:
  - `^` moves root up one directory level.
  - `Refresh` rescans the current directory view to pick up new/deleted files.
  - `Quit` closes the app.
  - `Edit` opens the selected file in VS Code (`code` CLI).
- Window title shows the current effective root path.
- Preview cache keyed by file timestamp and size for fast re-open.
- Navigating back to a cached file still performs a fresh stat check; changed files re-render automatically.
- `F5` refresh shortcut for directory view rescan (same behavior as `Refresh` button).
- If the currently previewed markdown file changes on disk, preview auto-refreshes and shows a status bar message.
- Manual tree/preview pane resizing is preserved across `^` root navigation for the current app run.
- Right-click a Markdown file to assign a highlight color in the tree.
- Highlight colors persist per directory in `.mdexplore-colors.json` files.
- Right-click menu includes `Clear All` to recursively remove all highlights from scope.
- Top-right color buttons copy matching highlighted files to clipboard.
- Search box includes an explicit `X` clear control that clears the query and removes bolded match markers.
- When search is active and a matched file is opened, preview matches are highlighted in yellow and the view scrolls to the first match.
- Preview scroll position is remembered per markdown file for the current app session.
- Right-click selected text in the preview pane to use:
  - `Copy Rendered Text` for plain rendered selection text.
  - `Copy Source Markdown` for markdown source content.
    Copies matching source markdown using direct range mapping first, then
    selected-text/fuzzy line matching as fallback, and finally the full source
    file if no match is possible.
- Clipboard copy uses file URI MIME formats compatible with Nemo/Nautilus paste.
- Last effective root is persisted to `~/.mdexplore.cfg` on exit.
  - If no directory is selected at quit time, the most recently selected/expanded
    directory is used.

## Requirements

- Ubuntu/Linux desktop with GUI.
- Python `3.10+`.
- `python3-venv` package available.
- Internet access for Mermaid only when no local Mermaid bundle is available.
- Internet access for MathJax only when no local MathJax bundle is available.
- Java runtime (`java` in `PATH`) for local PlantUML rendering.
- `plantuml.jar` available (project root by default, or set `PLANTUML_JAR`).
- Optional: VS Code `code` command in `PATH` for `Edit`.

## Quick Start

From any directory:

```bash
/path/to/mdexplore/mdexplore.sh
```

When no `PATH` is supplied, the app opens:

1. the path stored in `~/.mdexplore.cfg` (if valid), otherwise
2. your home directory.

To open a specific root directory:

```bash
/path/to/mdexplore/mdexplore.sh /path/to/notes
```

What the launcher does:

- Creates `.venv` inside the project if missing.
- Uses `.venv/bin/python` directly (does not alter your current shell session).
- Installs dependencies when `requirements.txt` changes.
- Runs the app.

## Usage

### Wrapper script

```bash
mdexplore.sh [PATH]
```

- `PATH` is optional.
- Supports plain paths and `file://` URIs (for `.desktop` `%u` launches).
- If a file path is passed, mdexplore opens its parent directory.
- If omitted, `~/.mdexplore.cfg` is used (falling back to home directory).
- `--help` prints usage.

### Direct Python run

```bash
python3 -m pip install -r /path/to/mdexplore/requirements.txt
python3 /path/to/mdexplore/mdexplore.py [PATH]
```

If `PATH` is omitted for direct run, the same config/home default rule applies.

### File Highlights

- In the file tree, right-click any `.md` file.
- Choose a highlight color (`Highlight Yellow`, then `... <Color>`), or clear it.
- Colors are persisted in `.mdexplore-colors.json` in each affected directory.
- If a directory is not writable, color persistence fails quietly by design.
- Use the top-right "Copy to clipboard:" color buttons to copy files
  with a given highlight color.
- Right-click a directory to access recursive `Clear All` for that subtree.
- Copy/Clear operations are recursive and use scope in this order:
  selected directory, else most recently selected/expanded directory, else root.

### Search and Match Highlighting

- Use `Search and highlight:` for non-recursive matching in the current effective scope.
- Matching files are shown in bold in the tree.
- Press `Enter` in the search field to run search immediately (skip debounce).
- Clicking the `X` in the search field clears search text and removes bolding.
- If search is active and you navigate to a different directory scope, search
  reruns automatically for that directory.
- Opening a matched file while search is active highlights matching text in yellow
  in the preview and scrolls to the first highlighted match.
- Non-quoted terms are case-insensitive.
- Quoted terms are case-sensitive.
- Function-style operators accept both no-space and spaced forms before `(`:
  `CLOSE(...)`/`CLOSE (...)`, `OR(...)`/`OR (...)`, `AND(...)`/`AND (...)`,
  and `NOT(...)`/`NOT (...)`.
- `CLOSE(...)` requires 2+ terms and matches only when all terms appear within
  50 words of each other in file content.

Search examples:

```text
joe
```

Matches `joe`, `JOE`, `JoE`, etc. (filename or content).

```text
"Anne Smith"
```

Case-sensitive exact phrase match.

```text
OR (Joe, "Fred")
```

Matches files containing either `Joe` (any case) or exact-case `"Fred"`.

```text
CLOSE(Fred "Anne Smith" Joe)
```

Matches only if all listed terms occur within 50 words of each other.

```text
Joe "Anne Smith" NOT draft
```

Implicit `AND`: equivalent to `Joe AND "Anne Smith" AND NOT draft`.

## PlantUML Local Configuration

The app renders PlantUML blocks locally with `plantuml.jar`.

- Default jar path: `/path/to/mdexplore/plantuml.jar`
- Override jar path with `PLANTUML_JAR`:

```bash
PLANTUML_JAR=/path/to/plantuml.jar /path/to/mdexplore/mdexplore.sh
```

Behavior details:

- Markdown preview loads immediately with `PlantUML rendering...` placeholders.
- Each diagram is replaced automatically as soon as it completes.
- Diagram replacements are applied in-place so scroll position is preserved.
- Failed diagrams show `PlantUML render failed with error ...` plus detailed
  stderr context (including line number when PlantUML provides one).
- Diagram progress continues while you browse other files; returning shows completed progress.

## MathJax Local-First Configuration

Math rendering is local-first:

- mdexplore first tries a local `tex-mml-chtml.js` bundle.
- If no local bundle is found, it falls back to CDN.

Local lookup order:

1. `MDEXPLORE_MATHJAX_JS` (explicit file path)
2. `mathjax/es5/tex-mml-chtml.js` under the repo
3. `mathjax/tex-mml-chtml.js` under the repo
4. `assets/mathjax/es5/tex-mml-chtml.js` under the repo
5. `vendor/mathjax/es5/tex-mml-chtml.js` under the repo
6. System paths such as `/usr/share/javascript/mathjax/es5/tex-mml-chtml.js`

Example override:

```bash
MDEXPLORE_MATHJAX_JS=/absolute/path/to/tex-mml-chtml.js /path/to/mdexplore/mdexplore.sh
```

## Mermaid Local-First Configuration

Mermaid rendering is local-first:

- mdexplore first tries a local `mermaid.min.js` bundle.
- If no local bundle is found, it falls back to CDN.

Local lookup order:

1. `MDEXPLORE_MERMAID_JS` (explicit file path)
2. `mermaid/mermaid.min.js` under the repo
3. `mermaid/dist/mermaid.min.js` under the repo
4. `assets/mermaid/mermaid.min.js` under the repo
5. `assets/mermaid/dist/mermaid.min.js` under the repo
6. `vendor/mermaid/mermaid.min.js` under the repo
7. `vendor/mermaid/dist/mermaid.min.js` under the repo
8. System paths such as `/usr/share/javascript/mermaid/mermaid.min.js`

Example override:

```bash
MDEXPLORE_MERMAID_JS=/absolute/path/to/mermaid.min.js /path/to/mdexplore/mdexplore.sh
```

## Project Structure

```text
mdexplore.py       # Qt application, file tree, renderer integration
mdexplore.sh       # launcher (venv create/install/run)
mdexplore.desktop.sample # sample desktop launcher entry for user customization
mdexplor-icon.png  # primary app icon asset (preferred)
requirements.txt   # Python runtime dependencies
README.md          # project docs
AGENTS.md          # coding-agent maintenance guide
DESCRIPTION.md     # repository description + suggested topic tags
LICENSE            # MIT license
```

## Development

Basic local checks:

```bash
bash -n mdexplore.sh
python3 -m py_compile mdexplore.py
```

## Troubleshooting

If you see `ModuleNotFoundError: No module named 'PySide6.QtWebEngineWidgets'`:

- Re-run `mdexplore.sh`; it now performs a runtime import check and auto-reinstalls
  dependencies when the venv is incomplete.
- If it still fails, run:
  - `rm -rf /path/to/mdexplore/.venv`
  - `/path/to/mdexplore/mdexplore.sh`

If `Edit` does nothing:

- Ensure VS Code is installed.
- Run `code --version` and confirm it is available in your `PATH`.

If the dock/menu still shows an old app icon:

- Ensure your launcher contains:
  - `Icon=/absolute/path/to/mdexplor-icon.png`
  - `StartupWMClass=mdexplore`
- Refresh desktop entries:
  - `update-desktop-database ~/.local/share/applications`
- Remove old pinned `mdexplore` favorites from the dock and pin it again.
- Log out/in if the shell still shows the previous cached icon.

If running the launcher appears to do nothing:

- Watch terminal output; the launcher now prints setup/launch status.
- First dependency install can take time because Qt packages are large.
- Ensure you are in a GUI session (`DISPLAY` or `WAYLAND_DISPLAY` must be set).
- For desktop/dock launches (`Terminal=false`), check launcher log:
  - `~/.cache/mdexplore/launcher.log`
  - log is auto-trimmed to the most recent 1000 lines

## Security Notes

- Markdown HTML is enabled (`html=True`), so preview untrusted Markdown with care.
- Mermaid uses local-first loading with CDN fallback.
- MathJax uses local-first loading with CDN fallback.

## Contributing

1. Keep changes focused and small.
2. Run syntax checks before submitting.
3. Update docs (`README.md`, `AGENTS.md`) when behavior changes.

## License

This project is licensed under the MIT License. See `LICENSE`.
