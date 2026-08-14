# mdExt

`mdExt` is a Visual Studio Code Markdown preview extension derived from the rendering ideas in **mdexplore**. It provides a real read-only custom editor for `.md` and `.markdown` files.

## Features

- **Editor toolbar preview:** while viewing a Markdown document, click the mdExt preview button in the editor toolbar to keep the current Markdown editing surface open and open the mdExt preview beside it.
- **Open With support:** right-click a Markdown tab or file, choose **Reopen Editor With…**, then choose **mdExt Markdown (Read-only)**.
- **Edit from preview:** the `Edit` button immediately to the right of `Search` reopens the previewed document with the configured default Markdown editor. If mdExt itself is configured as the default for that Markdown file, `Edit` opens VS Code's Markdown Editor (not Markdown Preview); only when the Markdown Editor is unavailable does it fall back to the plain Text Editor.
- **Comfortable default zoom:** preview text starts two font-size steps larger than VS Code's editor font. `Alt++` and `Alt+-` still adjust it one pixel at a time.
- **Per-document zoom memory:** font-size adjustments are stored in VS Code workspace storage for each Markdown document and restored when it is previewed again. Stale zoom records are periodically garbage-collected, with a bounded retained history.
- **PDF export:** the `PDF` button creates a letter-sized **native Chromium PDF** through PySide6 Qt WebEngine, matching mdexplore's rendering architecture. Text stays selectable/searchable, SVG/MathJax/Mermaid content remains vector where Chromium supports it, and links are preserved instead of flattening each page to a screenshot.
- **Live updates:** previews refresh as the source document changes, with a configurable debounce delay.
- **Scroll sync:** when a source editor and mdExt preview are both visible, scrolling either one keeps the other aligned to the same approximate source line.
- **In-preview search:** the header magnifier opens a search line that uses mdexplore-compatible query parsing and match highlighting, including `NEAR(...)` support.
- **Preview selection copy:** select rendered preview text and right-click to choose **Copy Rendered Text** or **Copy Source Markdown**, matching mdexplore's preview copy choices. Rendered copy preserves the visible text; source copy uses the preview-to-source line mapping (with a source-text fallback) to copy the corresponding Markdown markup through VS Code's clipboard API.
- **Persistent preview highlights:** the preview can add `Highlight` and `Highlight Important` ranges, stores them in `.mdexplore-highlighting.json`, and reapplies them when the document is previewed again.
- **GitHub-flavoured Markdown:** tables, strikethrough, autolinks, task-list inputs, code fences, and syntax highlighting.
- **MathJax:** inline and display TeX rendering using the bundled local MathJax runtime.
- **Mermaid:** Rust-first rendering through the bundled Linux x64 `mmdr` binary, with the bundled Mermaid JavaScript runtime as fallback.
- **R/J renderer toggle:** every Mermaid diagram can switch between cached Rust output (`R`) and JavaScript output (`J`) when Rust output is available.
- **Diagram controls:** Fit, zoom, pan buttons, Ctrl/Cmd+wheel zoom, and pointer-drag panning.
- **PlantUML:** local SVG rendering through Java and the bundled PlantUML jar.
- **Embedded SVG:** fenced `svg` blocks and standalone raw `<svg>...</svg>` blocks render as cached SVG images instead of source markup; unchanged SVG source reuses the in-process cache while edits generate a new cache entry.
- **mdexplore-style callouts:** NOTE, TIP, IMPORTANT, WARNING, and CAUTION blocks.
- **Images and links:** relative images are resolved through VS Code webview resource URIs; relative Markdown links can reopen in mdExt.
- **PDF creation:** the preview header includes a `PDF` button that waits for fonts and images, switches the rendered preview to PDF-safe layout, snapshots the already-rendered DOM, then hands that HTML to PySide6 Qt WebEngine's native `printToPdf` path. The resulting `<filename>.pdf` is written beside the Markdown source without VS Code's print dialog and without html2canvas/jsPDF rasterization.
- **Source navigation:** double-click rendered content with source-line metadata to reveal the corresponding source line.

## Development

Open the `mdext` directory in Visual Studio Code and press `F5` to launch an Extension Development Host.

```bash
npm install
npm test
npm run check
npm run package
```

The package command creates a `.vsix` file in this directory. Install it through **Extensions: Install from VSIX…**.

For PDF export, mdExt needs a Python runtime containing PySide6 Qt WebEngine. From the mdexplore repository root, run `./install-update.sh` once (or `./install-update.sh --check` to verify without changing anything). mdExt searches the project `.venv` automatically; `mdExt.pdfPythonPath` can point to another compatible Python executable when needed.

## Commands

- `mdExt: Preview Current Markdown`
- `mdExt: Reopen Current Markdown With mdExt`
- `mdExt: Refresh Preview`
- `mdExt: Open Markdown Source`

The default preview shortcut is `Ctrl+Shift+V` (`Cmd+Shift+V` on macOS) while a Markdown resource is active. VS Code's built-in Text Editor, Markdown Preview, and newer Markdown Editor/Hybrid Markdown Editor are preserved in place while mdExt opens beside them. Other third-party/custom Markdown editors are first reopened with the built-in Text Editor, then mdExt opens to the right.

## Settings

- `mdExt.liveUpdateDelay`: live-preview debounce in milliseconds.
- `mdExt.mermaidBackend`: `rust` or `javascript`.
- `mdExt.plantUml.enabled`: enables local PlantUML rendering.
- `mdExt.openRelativeMarkdownLinksIn`: opens relative Markdown links as source documents or mdExt custom editors.

## Platform notes

The bundled `mmdr` executable targets **Linux x64**, matching mdexplore's current desktop platform. On other platforms, mdExt automatically uses Mermaid JavaScript. PlantUML requires a Java runtime in `PATH`.

## Security model

Preview HTML is sanitized before it enters the webview. The webview uses a restrictive Content Security Policy and loads Mermaid, MathJax, styles, and images only from allowed local resources, HTTPS, or embedded data images.

## License

MIT. Bundled third-party components retain their own upstream licenses.
