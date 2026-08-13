# mdExt

`mdExt` is a Visual Studio Code Markdown preview extension derived from the rendering ideas in **mdexplore**. It provides a real custom editor for `.md` and `.markdown` files.

## Features

- **Editor toolbar preview:** while viewing a Markdown document, click the mdExt preview button in the editor toolbar to keep the current Markdown editing surface open and open the mdExt preview beside it.
- **Open With support:** right-click a Markdown tab or file, choose **Reopen Editor With…**, then choose **mdExt Markdown Preview**.
- **Live updates:** previews refresh as the source document changes, with a configurable debounce delay.
- **Scroll sync:** when a source editor and mdExt preview are both visible, scrolling either one keeps the other aligned to the same approximate source line.
- **In-preview search:** the header magnifier opens a search line that uses mdexplore-compatible query parsing and match highlighting, including `NEAR(...)` support.
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
- **PDF creation:** the preview header includes a `PDF` button that waits for fonts and images, switches the rendered preview to PDF-safe layout, renders the preview in the webview, and writes `<filename>.pdf` beside the Markdown source without relying on VS Code's print dialog.
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
