# pdfexplore

`pdfexplore` is the read-only PDF companion to `mdexplore`, with the same UI shell and operational model where it is applicable to PDFs.

## Features

- Left-pane tree rooted at the chosen directory, showing folders plus `*.pdf` files.
- Right-pane embedded PDF preview via bundled local `pdf.js`.
- Top actions:
  - `^` to move root up one directory.
  - `Refresh` to rescan the current tree root.
  - `Add View` to create another tabbed view of the same PDF at the current page/scroll location.
- `F5` refresh shortcut (same behavior as `Refresh`).
- Search box and color-apply controls matching `mdexplore` search syntax:
  - supports tokenized/quoted boolean queries via shared parser helpers,
  - searches non-recursively within direct-child PDFs of the selected directory,
  - highlights in-tree match counts and in-view search hits for matched files.
- Tree highlight colors and clear actions:
  - per-file color assignment via tree context menu,
  - `Clear in Directory` and `Clear All` with confirmation prompts,
  - persisted per-directory in sidecars.
- Copy controls aligned with `mdexplore`:
  - destination mode `Clipboard` or `Directory`,
  - pin action copies current PDF,
  - color actions copy all files with a selected highlight color in active scope,
  - directory copy merges color/view/highlight metadata into destination sidecars.
- Preview context menu:
  - `Copy Selected Text`,
  - `Highlight` / `Highlight Important`,
  - `Remove Highlight`.
- Persistent in-view text highlights with per-file sidecar storage.
- View tabs aligned with `mdexplore` behavior:
  - per-document multi-view tabs (max 8),
  - tab position progress bar driven by page progress,
  - custom tab labels via tab context menu,
  - labeled-tab `Return to beginning` support,
  - home/reset icon actions from shared `ViewTabBar`,
  - hide tab strip when only one unlabeled default view remains,
  - per-document tab sessions persisted/restored from `.pdfexplore-views.json`.
- PDF preview widget caching:
  - once a PDF preview widget is created, it remains cached in memory for the app run,
  - switching away and back avoids full viewer/widget re-instantiation.
- If the currently previewed PDF changes on disk, preview auto-refreshes and reports that in the status bar.
- `Refresh` preserves expanded folders/selection when possible and clears the preview if the active PDF was deleted.
- Effective root persisted to `~/.pdfexplore.cfg` on close (interactive terminal launch without path defaults to current working directory).

## Run

From repo root:

```bash
./pdfexplore.sh [PATH]
```

- `PATH` may be a directory or a single PDF file.
- If omitted:
  - interactive terminal launch uses current working directory,
  - otherwise falls back to `~/.pdfexplore.cfg`, then `$HOME`.

## Sidecars

`pdfexplore` writes sidecars beside PDFs/directories:

- `.pdfexplore-colors.json`
- `.pdfexplore-views.json`
- `.pdfexplore-highlighting.json`

`views` sidecars store multi-view sessions (tabs, active tab, per-tab state, custom labels when present). As with `mdexplore`, default untouched single-view sessions are not kept as persistent entries.

## Notes

- PDF content remains read-only; sidecars are the only persisted metadata outputs.
- The embedded viewer bundle is local under `pdfexplore/vendor/pdfjs/` (no remote dependency for core PDF viewing).
