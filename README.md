# mdexplore

Fast Markdown explorer for Ubuntu/Linux desktop: browse `.md` files in a directory tree and preview fully rendered output instantly.

## Features

- Expandable left-pane directory tree rooted at a chosen folder.
- Shows Markdown files only (`*.md`), while still allowing folder navigation.
- Large right-pane rendered preview with scroll support.
- Supports:
  - CommonMark + tables + strikethrough.
  - TeX/LaTeX math via MathJax.
  - Mermaid diagrams.
  - PlantUML diagrams.
- Top actions:
  - `^` moves root up one directory level.
  - `Quit` closes the app.
  - `Edit` opens the selected file in VS Code (`code` CLI).
- Window title shows the current effective root path.
- Preview cache keyed by file timestamp and size for fast re-open.
- `F5` refresh shortcut for the currently selected file.
- Right-click a Markdown file to assign a highlight color in the tree.
- Highlight colors persist per directory in `.mdexplore-colors.json` files.
- Right-click menu includes `Clear All` to recursively remove all highlights from scope.
- Top-right color buttons copy matching highlighted files to clipboard.
- Clipboard copy uses file URI MIME formats compatible with Nemo/Nautilus paste.

## Requirements

- Ubuntu/Linux desktop with GUI.
- Python `3.10+`.
- `python3-venv` package available.
- Internet access for MathJax and Mermaid CDN scripts.
- Internet access to a PlantUML server (or your own internal server).
- Optional: VS Code `code` command in `PATH` for `Edit`.

## Quick Start

From any directory:

```bash
/path/to/mdexplore/mdexplore.sh
```

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
- If omitted, the current working directory is used.
- `--help` prints usage.

### Direct Python run

```bash
python3 -m pip install -r /path/to/mdexplore/requirements.txt
python3 /path/to/mdexplore/mdexplore.py [PATH]
```

### File Highlights

- In the file tree, right-click any `.md` file.
- Choose a highlight color (`Highlight Yellow`, then `... <Color>`), or clear it.
- Colors are persisted in `.mdexplore-colors.json` in each affected directory.
- If a directory is not writable, color persistence fails quietly by design.
- Use the top-right "Copy to clipboard files matching:" color buttons to copy files
  with a given highlight color.
- Right-click a directory to access recursive `Clear All` for that subtree.
- Copy/Clear operations are recursive and use the selected directory as scope when
  a directory is selected; otherwise they use the current root directory.

## PlantUML Server Configuration

The app renders PlantUML blocks through a server endpoint.

- Default: `https://www.plantuml.com/plantuml`
- Override:

```bash
PLANTUML_SERVER=https://your-server/plantuml /path/to/mdexplore/mdexplore.sh
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

- Ensure dependencies were installed from `requirements.txt` inside the launcher-managed venv.
- Re-run the launcher once to force environment setup.

If `Edit` does nothing:

- Ensure VS Code is installed.
- Run `code --version` and confirm it is available in your `PATH`.

If running the launcher appears to do nothing:

- Watch terminal output; the launcher now prints setup/launch status.
- First dependency install can take time because Qt packages are large.
- Ensure you are in a GUI session (`DISPLAY` or `WAYLAND_DISPLAY` must be set).

## Security Notes

- Markdown HTML is enabled (`html=True`), so preview untrusted Markdown with care.
- Mermaid and MathJax load from external CDNs at runtime.

## Contributing

1. Keep changes focused and small.
2. Run syntax checks before submitting.
3. Update docs (`README.md`, `AGENTS.md`) when behavior changes.

## License

This project is licensed under the MIT License. See `LICENSE`.
