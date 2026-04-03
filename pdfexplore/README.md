# pdfexplore

`pdfexplore` is a read-only PDF companion to `mdexplore`.

It keeps the same general shell:

- left-side filesystem tree scoped to `*.pdf`
- top-bar search and file-highlight controls
- tabbed preview area on the right
- copy-to-clipboard or copy-to-directory actions

It adds PDF-specific behavior through a bundled local `pdf.js` viewer:

- render and scroll through PDFs in-app
- text selection and copying
- content search across PDFs in the selected directory
- in-view search highlighting
- persistent text highlights stored beside the files

## Run

From the repo root:

```bash
./pdfexplore.sh [PATH]
```

`PATH` may be either a directory or a single PDF file. If omitted, `pdfexplore` uses `~/.pdfexplore.cfg` and falls back to `$HOME`.

## Sidecars

`pdfexplore` writes lightweight sidecars beside PDFs/directories:

- `.pdfexplore-colors.json`
- `.pdfexplore-views.json`
- `.pdfexplore-highlighting.json`

These store file color tags, last-view state, and persistent text highlights.

## Notes

- Rendering is read-only. There is no PDF editing or annotation write-back into the PDF itself.
- Search scope matches `mdexplore`: direct child files in the currently selected directory, not a recursive content search.
- The embedded viewer is bundled under `pdfexplore/vendor/pdfjs/`.
