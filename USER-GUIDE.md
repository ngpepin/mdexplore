# mdexplore User Guide

This guide explains `mdexplore` through practical workflows instead of just listing features.

If you want installation or developer-level internals, use `README.md` and `DEVELOPERS-AGENTS.md`.

## Who This Is For

Use this guide if you want to:

- browse large Markdown folders quickly,
- search notes with more than simple keyword matching,
- keep multiple reading positions in one long file,
- highlight and copy selected files/text for sharing,
- export reliable PDFs from rendered markdown.

## Quick Start (2 Minutes)

From the repo root:

```bash
./mdexplore.sh
```

Or open a specific folder:

```bash
./mdexplore.sh /path/to/notes
```

What to expect:

- Left pane: directory tree (`.md` files + folders).
- Right pane: rendered preview.
- Top controls: `Recent`, `^`, `Refresh`, `PDF`, `Add View`, `Edit`, copy controls, search box.

## Core Concept: Effective Root

Many actions depend on the **effective root** (the active scope).

- Selecting a directory makes that directory the effective root.
- Selecting a file makes that file’s parent directory the effective root.
- The window title shows the current effective root.
- Search uses visible files in the current tree state and updates as scope changes.

## Tutorial 1: Browse and Open Files Fast

Use case: you are exploring an unfamiliar notes tree.

1. Launch `mdexplore` on the target folder.
2. Expand folders in the left tree.
3. Click a markdown file to load preview.
4. Use `Refresh` if files were added/removed externally.
5. Use `^` to move up one directory level.

Tips:

- `F5` is the same as `Refresh`.
- `Edit` opens the selected file in VS Code (`code` must be in `PATH`).

## Tutorial 2: Find the Right File with Search

Use case: “find where I discussed a specific topic or phrase”.

1. Type into `Search and highlight:`.
2. Wait briefly for debounce, or press `Enter` to run immediately.
3. Matching files become bold+italic in the tree.
4. Open a matched file to see in-preview highlighted hits.
5. Click the field’s `X` to clear search and remove match formatting.

Query examples:

```text
backup strategy
"NVMe enclosure"
'Exact Case'
alpha AND beta
OR(mermaid, plantuml)
NEAR("error code", timeout)
```

Notes:

- Unquoted and double-quoted terms are case-insensitive.
- Single-quoted terms are case-sensitive.
- `NEAR(...)` requires all listed terms to occur within the configured window.

## Tutorial 3: Keep Multiple Reading Positions (Views)

Use case: one long file, multiple sections you jump between.

1. Open a long file.
2. Scroll to section A, click `Add View`.
3. Scroll to section B, click `Add View` again.
4. Switch tabs to jump between saved positions.
5. Right-click a tab to assign a custom label.
6. Use `Return to beginning` on labeled tabs as needed.

What persists:

- Multi-view state is saved in `.mdexplore-views.json`.
- Reopening the same file restores its saved views.

## Tutorial 4: Highlight What Matters

### A) File-level highlights in tree

Use case: mark files by workflow stage (todo/review/final).

1. Right-click a `.md` file in the tree.
2. Choose a highlight color.
3. Repeat for other files.

Useful actions:

- `Clear Highlight` for one file.
- `Clear in Directory` for non-recursive clear.
- `Clear All` for recursive clear.

Persistence:

- Stored per directory in `.mdexplore-colors.json`.

### B) Text highlights in preview

Use case: mark important passages inside documents.

1. Select text in preview.
2. Right-click and choose `Highlight` or `Highlight Important`.
3. Use `Remove Highlight` to delete.

Persistence and markers:

- Stored in `.mdexplore-highlighting.json`.
- Files with persisted text highlights get a marker badge in the tree.

## Tutorial 5: Copy Work Products (Clipboard or Directory)

Use case: collect selected files for sharing or downstream processing.

Top-right controls start with `Copy to:`.

### A) Clipboard mode

1. Leave `Clipboard` selected.
2. Click:
   - pin button: copy current file,
   - color buttons: copy files with that file-highlight color.

### B) Directory mode

1. Select `Directory`.
2. Click pin or a color button.
3. Choose destination folder.

Directory copy behavior:

- Markdown files are copied.
- Metadata sidecars (`colors/views/highlights`) are merged to destination.

### BASE64 image toggle (copy-time)

Use case: copied markdown should be self-contained.

1. Click the BASE64 icon button near copy controls.
2. When enabled, copied markdown rewrites retrievable image links into `data:` URIs.
3. Broken/unretrievable links are left unchanged.

Important:

- This affects copied output only.
- Source markdown files are never modified by this toggle.

## Tutorial 6: Export Clean PDFs

Use case: deliver a polished PDF from markdown notes/diagrams.

1. Open the markdown file you want.
2. Click `PDF`.
3. Wait for status-bar completion.
4. Output is written beside source file as `<name>.pdf`.

What mdexplore handles:

- math/diagram readiness before print,
- footer page numbering (`N of M`),
- diagram-aware print layout logic,
- image inlining for PDF export when sources are retrievable.

## Tutorial 7: Use Recent Locations Like a Workspace Queue

Use case: switch between several active project roots quickly.

1. Navigate to project roots during normal work.
2. Open `Recent` to see your latest directories (newest first).
3. Pick any entry to re-root instantly.

Behavior:

- Up to 20 recent roots are kept.
- Entries are persisted in `~/.mdexplore.cfg`.
- List refreshes when menu opens (multi-instance friendly).

## Tutorial 8: Use `hfind` for Terminal-Only Batch Search

Use case: search many files from shell scripts or remote sessions.

```bash
./hfind.sh -crv "NEAR(alpha,beta)" "./**/*.md"
```

Common patterns:

```bash
./hfind.sh --query "OR(fred, paul)" --content --recursive *.txt
./hfind.sh -q "AND('Fred',paul)" /path/to/dir/*.md
./hfind.sh -cv "'Nico '" "./**/*.md"
```

Use `hfind` when you do not need GUI preview/navigation.

## Keyboard Shortcuts

- `F5`: refresh tree.
- `Ctrl++`: preview zoom in.
- `Ctrl+-`: preview zoom out.
- `Ctrl+0`: preview zoom reset.
- `Enter` in search field: run search immediately.

## Troubleshooting (User-Focused)

If app does not appear:

- Run from terminal first: `./mdexplore.sh`.
- Wait on first run; dependency install can take time.
- Confirm GUI session (`DISPLAY` or `WAYLAND_DISPLAY` set).

If `Edit` does nothing:

- Ensure `code --version` works in your shell.

If preview feels stale:

- Press `Refresh` or `F5`.
- Reopen file from tree.

## Daily Workflow Recipe (Suggested)

1. Launch on your active notes root.
2. Use search to shortlist files.
3. Mark files with tree highlight colors.
4. Add view tabs in long docs for key sections.
5. Add preview text highlights for critical passages.
6. Copy selected files (with BASE64 toggle as needed).
7. Export final PDF when ready.

---

If you want deeper technical controls (render backends, performance knobs, local MathJax/Mermaid paths), see `README.md`.
