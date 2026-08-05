'use strict';

const path = require('node:path');
const { nonce } = require('./utils');

function resourceUri(webview, context, ...parts) {
  return webview.asWebviewUri(context.extensionUri.with({
    path: path.posix.join(context.extensionUri.path, ...parts),
  }));
}

function getWebviewHtml(webview, context) {
  const scriptNonce = nonce();
  const previewScript = resourceUri(webview, context, 'media', 'preview.js');
  const searchCompatScript = resourceUri(webview, context, 'media', 'searchCompat.js');
  const highlightCompatScript = resourceUri(webview, context, 'media', 'highlightCompat.js');
  const previewStyle = resourceUri(webview, context, 'media', 'preview.css');
  const highlightStyle = resourceUri(webview, context, 'media', 'vendor', 'highlight.css');
  const mermaidScript = resourceUri(webview, context, 'media', 'vendor', 'mermaid.min.js');
  const mathJaxScript = resourceUri(webview, context, 'media', 'vendor', 'tex-svg.js');
  const csp = [
    "default-src 'none'",
    `img-src ${webview.cspSource} https: data:`,
    `font-src ${webview.cspSource} data:`,
    `style-src ${webview.cspSource} 'unsafe-inline'`,
    `script-src 'nonce-${scriptNonce}' ${webview.cspSource}`,
  ].join('; ');

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <link rel="stylesheet" href="${previewStyle}">
  <link rel="stylesheet" href="${highlightStyle}">
  <script nonce="${scriptNonce}">
    window.MathJax = {
      startup: { typeset: false },
      tex: {
        inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
        displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']]
      },
      svg: { fontCache: 'global', scale: 1.05 },
      options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
    };
  </script>
  <script nonce="${scriptNonce}" src="${mermaidScript}"></script>
  <script nonce="${scriptNonce}" src="${mathJaxScript}"></script>
  <title>mdExt</title>
</head>
<body>
  <header class="preview-header">
    <div class="preview-header-row">
      <div class="document-identity">
        <strong id="document-title">mdExt</strong>
        <span id="document-path">Open a Markdown document to preview it.</span>
      </div>
      <div class="header-actions">
        <button id="search-toggle-button" class="icon-button" title="Search this preview" aria-label="Search this preview" aria-expanded="false" aria-controls="preview-searchbar">
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M10.5 3a7.5 7.5 0 1 1 0 15 7.5 7.5 0 0 1 0-15Zm0 2a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11Zm6.56 10.64 3.15 3.15-1.42 1.42-3.15-3.15 1.42-1.42Z" />
          </svg>
        </button>
        <button id="open-source-button" title="Open Markdown source">Source</button>
        <button id="refresh-button" title="Refresh preview">Refresh</button>
      </div>
    </div>
    <div id="preview-searchbar" class="preview-searchbar" hidden>
      <input id="preview-search-input" class="preview-search-input" type="search" spellcheck="false" placeholder="Search this preview with mdexplore syntax">
      <span id="preview-search-result" class="preview-search-result"></span>
      <button id="highlight-button" title="Highlight selected text" disabled>Highlight</button>
      <button id="highlight-important-button" title="Highlight selected text as important" disabled>Important</button>
      <button id="preview-search-close-button" class="icon-button" title="Close search" aria-label="Close search">
        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <path d="M6.4 5 12 10.6 17.6 5 19 6.4 13.4 12 19 17.6 17.6 19 12 13.4 6.4 19 5 17.6 10.6 12 5 6.4 6.4 5Z" />
        </svg>
      </button>
    </div>
  </header>
  <main id="preview-content" class="markdown-body">
    <section class="empty-state">
      <h2>mdExt Markdown Preview</h2>
      <p>Open a Markdown file, then select the mdExt Activity Bar icon or reopen the file with mdExt.</p>
    </section>
  </main>
  <div id="render-status" role="status" aria-live="polite"></div>
  <script nonce="${scriptNonce}" src="${searchCompatScript}"></script>
  <script nonce="${scriptNonce}" src="${highlightCompatScript}"></script>
  <script nonce="${scriptNonce}" src="${previewScript}"></script>
</body>
</html>`;
}

module.exports = { getWebviewHtml };
