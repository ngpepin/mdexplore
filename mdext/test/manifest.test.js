'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const root = path.resolve(__dirname, '..');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
const webviewSource = fs.readFileSync(path.join(root, 'src', 'webview.js'), 'utf8');
const previewSource = fs.readFileSync(path.join(root, 'media', 'preview.js'), 'utf8');
const previewCss = fs.readFileSync(path.join(root, 'media', 'preview.css'), 'utf8');

test('contributes an editor-toolbar preview action and Markdown custom editor', () => {
  assert.equal(manifest.contributes.viewsContainers, undefined);
  assert.equal(manifest.contributes.views, undefined);
  assert.equal(manifest.contributes.menus['view/title'], undefined);
  const toolbarAction = manifest.contributes.menus['editor/title'].find(
    (entry) => entry.command === 'mdExt.previewCurrent',
  );
  assert.ok(toolbarAction);
  assert.match(toolbarAction.when, /resourceExtname == \.md/);
  assert.deepEqual(manifest.contributes.commands.find(
    (entry) => entry.command === 'mdExt.previewCurrent',
  ).icon, {
    light: 'media/mdexplor-icon.png',
    dark: 'media/mdexplor-icon.png',
  });
  const editors = manifest.contributes.customEditors;
  const editor = editors.find((entry) => entry.viewType === 'mdExt.markdownEditor');
  assert.ok(editor);
  assert.equal(editor.displayName, 'mdExt Markdown Preview');
  assert.ok(editor.selector.some((entry) => entry.filenamePattern === '*.md'));
  assert.equal(editor.priority, 'option');
  const previewKeybinding = manifest.contributes.keybindings.find(
    (entry) => entry.command === 'mdExt.previewCurrent',
  );
  assert.ok(previewKeybinding);
  assert.match(previewKeybinding.when, /editorLangId == markdown/);
  assert.match(previewKeybinding.when, /resourceExtname == \.md/);
  assert.match(previewKeybinding.when, /resourceExtname == \.markdown/);
});

test('uses the bundled extension entry point', () => {
  assert.equal(manifest.main, './dist/extension.js');
});

test('bundled rendering assets exist', () => {
  const assets = [
    'media/mdexplor-icon.png',
    'python/pdf_export.py',
    'media/vendor/mermaid.min.js',
    'media/vendor/tex-svg.js',
    'vendor/plantuml/plantuml.jar',
    'bin/linux-x64/mmdr',
  ];
  for (const asset of assets) {
    const absolute = path.join(root, asset);
    assert.equal(fs.existsSync(absolute), true, `${asset} should exist`);
    assert.ok(fs.statSync(absolute).size > 1024, `${asset} should be non-empty`);
  }
});

test('MathJax only scans explicit inline delimiters so currency is not treated as TeX', () => {
  assert.ok(webviewSource.includes("inlineMath: [['\\\\\\\\(', '\\\\\\\\)']]"));
  assert.equal(webviewSource.includes("['$', '$']"), false);
});

test('preview header replaces Source with a direct PDF creation control', () => {
  assert.match(webviewSource, /id="pdf-button"[^>]*>PDF<\/button>/);
  assert.doesNotMatch(webviewSource, /id="open-source-button"/);
  assert.doesNotMatch(webviewSource, /html2pdf/);
  assert.match(webviewSource, /connect-src \$\{webview\.cspSource\} https: data: blob:/);
  assert.match(webviewSource, /img-src \$\{webview\.cspSource\} https: data: blob:/);
  assert.match(previewSource, /createPdfFromPreview/);
  assert.doesNotMatch(previewSource, /window\.print\(\)/);
  assert.doesNotMatch(previewSource, /resolvePdfFactory/);
  assert.doesNotMatch(previewSource, /renderPdfArrayBuffer/);
  assert.match(previewSource, /preparePdfHtmlSnapshot/);
  assert.match(previewSource, /type: 'createPdf'/);
  assert.match(previewSource, /prepareDiagramLayoutForPdf\(/);
  assert.match(previewSource, /restoreDiagramLayoutAfterPdfExport\(/);
  assert.match(previewSource, /inlinePdfExportImages\(/);
  assert.doesNotMatch(previewSource, /PDF_CHUNK_TARGET_CSS_HEIGHT_PX/);
  assert.doesNotMatch(previewSource, /html2canvas/);
  assert.doesNotMatch(previewSource, /jsPDF/);
  assert.match(previewSource, /collectPdfStyleText\(\)/);
  assert.match(previewSource, /await vscode\.postMessage/);
  assert.match(previewSource, /restorePdfExportImages\(/);
  assert.match(previewSource, /document\.fonts\?\.ready/);
  assert.match(previewCss, /mdext-pdf-export-mode/);
  assert.match(previewCss, /--vscode-editor-foreground: #111827/);
  assert.match(previewCss, /--mdext-muted: #4b5563/);
  assert.match(previewCss, /--mdext-border: #9ca3af/);
  assert.match(previewCss, /\.mdext-pdf-export-mode \.markdown-body a/);
  assert.match(previewCss, /\.mdext-pdf-export-mode \.markdown-body blockquote/);
  assert.match(previewCss, /--mdext-print-diagram-width/);
  assert.match(previewCss, /mdext-pdf-break-before/);
});

test('search is discoverable and Ctrl/Cmd+F opens mdexplore-style search', () => {
  assert.match(webviewSource, /<span>Search<\/span>/);
  assert.match(webviewSource, /Search this preview with mdexplore syntax/);
  assert.match(previewSource, /event\.ctrlKey \|\| event\.metaKey/);
  assert.match(previewSource, /setSearchVisibility\(true, \{ focus: true \}\)/);
  assert.match(previewSource, /type: 'setSearchQuery'/);
});

test('preview header places Edit immediately after Search and wires the edit request', () => {
  const searchIndex = webviewSource.indexOf('id="search-toggle-button"');
  const editIndex = webviewSource.indexOf('id="edit-button"');
  const pdfIndex = webviewSource.indexOf('id="pdf-button"');
  assert.ok(searchIndex >= 0);
  assert.ok(editIndex > searchIndex);
  assert.ok(pdfIndex > editIndex);
  assert.match(webviewSource, /id="edit-button"[^>]*>Edit<\/button>/);
  assert.match(previewSource, /editButton\?\.addEventListener\('click'/);
  assert.match(previewSource, /type: 'edit'/);
});

test('selected preview text exposes mdexplore-style copy and persistent highlight actions', () => {
  assert.match(webviewSource, /id="preview-context-menu"/);
  assert.match(webviewSource, /id="context-copy-rendered-button"[^>]*>Copy Rendered Text<\/button>/);
  assert.match(webviewSource, /id="context-copy-source-button"[^>]*>Copy Source Markdown<\/button>/);
  assert.match(webviewSource, /id="context-highlight-button"/);
  assert.match(webviewSource, /id="context-highlight-important-button"/);
  assert.match(previewSource, /content\.addEventListener\('contextmenu'/);
  assert.match(previewSource, /handleCopyRequest\('rendered'\)/);
  assert.match(previewSource, /handleCopyRequest\('source'\)/);
  assert.match(previewSource, /type: 'copyRenderedText'/);
  assert.match(previewSource, /type: 'copySourceMarkdown'/);
  assert.match(previewSource, /selectedSourceLineRange\(\)/);
  assert.match(previewSource, /handleHighlightRequest\(PREVIEW_HIGHLIGHT_KIND_NORMAL\)/);
  assert.match(previewSource, /handleHighlightRequest\(PREVIEW_HIGHLIGHT_KIND_IMPORTANT\)/);
});

test('preview render does not force a scroll report that can restart sync feedback', () => {
  assert.doesNotMatch(previewSource, /scheduleVisibleLineReport\(true\)/);
});

test('Alt plus and minus adjust preview font size without taking VS Code Ctrl zoom shortcuts', () => {
  assert.match(previewCss, /--mdext-preview-font-size: calc\(var\(--vscode-font-size\) \+ 2px\)/);
  assert.match(previewCss, /font-size: var\(--mdext-preview-font-size\)/);
  assert.match(previewSource, /event\.altKey && !event\.ctrlKey && !event\.metaKey/);
  assert.match(previewSource, /key === '\+' \|\| key === '='/);
  assert.match(previewSource, /adjustPreviewFontSize\(1\)/);
  assert.match(previewSource, /key === '-' \|\| key === '_'/);
  assert.match(previewSource, /adjustPreviewFontSize\(-1\)/);
  assert.match(previewSource, /Math\.max\(8, Math\.min\(40,/);
  assert.match(previewSource, /type: 'setPreviewFontSize'/);
  assert.match(previewSource, /message\.type === 'previewFontSize'/);
});

test('diagram cards use direct SVG dimensions instead of transform-only scaling', () => {
  assert.match(previewCss, /\.diagram-viewport \{ max-height:/);
  assert.doesNotMatch(previewCss, /\.diagram-viewport \{[^}]*min-height:/);
  assert.match(previewSource, /svg\.style\.width = `\$\{Math\.max\(1, size\.width \* state\.zoom\)\}px`/);
  assert.match(previewSource, /svg\.style\.height = `\$\{Math\.max\(1, size\.height \* state\.zoom\)\}px`/);
  assert.doesNotMatch(previewSource, /canvas\.style\.transform = `scale/);
});
