'use strict';

const vscode = acquireVsCodeApi();
const content = document.getElementById('preview-content');
const titleNode = document.getElementById('document-title');
const pathNode = document.getElementById('document-path');
const statusNode = document.getElementById('render-status');
const refreshButton = document.getElementById('refresh-button');
const pdfButton = document.getElementById('pdf-button');
const searchToggleButton = document.getElementById('search-toggle-button');
const editButton = document.getElementById('edit-button');
const searchBar = document.getElementById('preview-searchbar');
const searchInput = document.getElementById('preview-search-input');
const searchResultNode = document.getElementById('preview-search-result');
const searchCloseButton = document.getElementById('preview-search-close-button');
const highlightButton = document.getElementById('highlight-button');
const highlightImportantButton = document.getElementById('highlight-important-button');
const contextMenu = document.getElementById('preview-context-menu');
const contextCopyRenderedButton = document.getElementById('context-copy-rendered-button');
const contextCopySourceButton = document.getElementById('context-copy-source-button');
const contextHighlightButton = document.getElementById('context-highlight-button');
const contextHighlightImportantButton = document.getElementById('context-highlight-important-button');

const PREVIEW_HIGHLIGHT_KIND_NORMAL = 'normal';
const PREVIEW_HIGHLIGHT_KIND_IMPORTANT = 'important';
const PREVIEW_PERSISTENT_HIGHLIGHT_COLOR = 'rgba(102, 86, 178, 0.36)';
const PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_COLOR = 'rgba(225, 214, 255, 0.76)';
const PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_TEXT_COLOR = '#170534';
const PREVIEW_PERSISTENT_HIGHLIGHT_MARKER_COLOR = 'rgba(112, 90, 188, 0.92)';
const PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR = 'rgba(154, 132, 220, 0.96)';
const PDF_PAGE_WIDTH_IN = 8.5;
const PDF_PAGE_HEIGHT_IN = 11;
const PDF_MARGIN_TOP_IN = 0.55;
const PDF_MARGIN_RIGHT_IN = 0.6;
const PDF_MARGIN_BOTTOM_IN = 0.65;
const PDF_MARGIN_LEFT_IN = 0.6;
const CSS_PX_PER_IN = 96;
const PDF_MIN_DIAGRAM_FONT_PT = 4;

let renderRevision = 0;
let mermaidConfigured = false;
let statusTimer = null;
let scrollReportTimer = null;
let suppressScrollEventsUntil = 0;
let lastReportedLine = -1;
let searchInputTimer = null;
let selectionRefreshTimer = null;
let searchBarVisible = false;
let previewFontSize = null;
let currentPersistentHighlights = [];
let currentSearchState = {
  query: '',
  terms: [],
  nearTermGroups: [],
  nearWordGap: 50,
  scrollToFirst: false,
};
let latestSelectionInfo = {
  hasSelection: false,
  selectedText: '',
  selectionOffsetStart: null,
  selectionOffsetEnd: null,
};

function persistentEntriesSignature(entries) {
  if (!Array.isArray(entries)) {
    return '[]';
  }
  return JSON.stringify(entries.map((entry) => ({
    id: String(entry?.id || ''),
    start: Number(entry?.start) || 0,
    end: Number(entry?.end) || 0,
    kind: String(entry?.kind || PREVIEW_HIGHLIGHT_KIND_NORMAL),
    anchor_text: String(entry?.anchor_text || ''),
    offset_space: String(entry?.offset_space || ''),
  })));
}

function normalizeSearchState(raw) {
  const query = String(raw?.query || '');
  const normalizeTerm = (item) => {
    if (!item || typeof item.text !== 'string' || !item.text.trim()) {
      return null;
    }
    return {
      text: item.text,
      caseSensitive: !!item.caseSensitive,
    };
  };
  const terms = Array.isArray(raw?.terms)
    ? raw.terms.map(normalizeTerm).filter(Boolean)
    : [];
  const nearTermGroups = Array.isArray(raw?.nearTermGroups)
    ? raw.nearTermGroups
      .map((group) => Array.isArray(group) ? group.map(normalizeTerm).filter(Boolean) : [])
      .filter((group) => group.length >= 2)
    : [];
  return {
    query,
    terms,
    nearTermGroups,
    nearWordGap: Math.max(1, Number(raw?.nearWordGap) || 50),
    scrollToFirst: !!raw?.scrollToFirst,
  };
}

function currentPreviewFontSize() {
  if (Number.isFinite(previewFontSize)) {
    return previewFontSize;
  }
  const computed = Number.parseFloat(getComputedStyle(document.body).fontSize);
  previewFontSize = Number.isFinite(computed) ? computed : 13;
  return previewFontSize;
}

function setPreviewFontSize(nextSize, announce = true, persist = true) {
  const size = Math.max(8, Math.min(40, Number(nextSize) || currentPreviewFontSize()));
  previewFontSize = Math.round(size * 10) / 10;
  document.documentElement.style.setProperty('--mdext-preview-font-size', `${previewFontSize}px`);
  if (persist) {
    vscode.postMessage({ type: 'setPreviewFontSize', size: previewFontSize });
  }
  if (announce) {
    setStatus(`Preview font: ${previewFontSize}px`);
  }
}

function adjustPreviewFontSize(delta) {
  setPreviewFontSize(currentPreviewFontSize() + delta);
}

function setStatus(message, persistent = false) {
  statusNode.textContent = String(message || '');
  statusNode.classList.toggle('visible', Boolean(message));
  if (statusTimer) {
    clearTimeout(statusTimer);
    statusTimer = null;
  }
  if (message && !persistent) {
    statusTimer = setTimeout(() => statusNode.classList.remove('visible'), 1600);
  }
}

async function waitForPdfAssets() {
  try {
    if (document.fonts?.ready) {
      await document.fonts.ready;
    }
  } catch {
    // Continue with the available fonts rather than blocking PDF creation.
  }

  const pendingImages = Array.from(content.querySelectorAll('img'))
    .filter((image) => !image.complete)
    .map((image) => new Promise((resolve) => {
      const done = () => resolve();
      image.addEventListener('load', done, { once: true });
      image.addEventListener('error', done, { once: true });
      setTimeout(done, 1500);
    }));
  if (pendingImages.length) {
    await Promise.all(pendingImages);
  }
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener('load', () => resolve(String(reader.result || '')));
    reader.addEventListener('error', () => reject(reader.error || new Error('Failed to read blob')));
    reader.readAsDataURL(blob);
  });
}

async function imageSourceToDataUrl(source) {
  const rawSource = String(source || '').trim();
  if (!rawSource || rawSource.toLowerCase().startsWith('data:')) {
    return rawSource;
  }

  try {
    const response = await fetch(rawSource);
    if (response.ok) {
      return await blobToDataUrl(await response.blob());
    }
  } catch {
    // Leave unresolved/broken image sources unchanged rather than rasterizing
    // the already-rendered element through a canvas fallback.
  }
  return '';
}

async function inlinePdfExportImages() {
  const images = Array.from(content.querySelectorAll('img'));
  const replacements = [];
  for (const image of images) {
    const attributeSource = String(image.getAttribute('src') || '').trim();
    const absoluteSource = String(image.currentSrc || image.src || '').trim();
    const source = attributeSource || absoluteSource;
    if (!source || source.toLowerCase().startsWith('data:')) {
      continue;
    }
    const dataUrl = await imageSourceToDataUrl(source);
    if (!dataUrl) {
      continue;
    }
    replacements.push({ image, source });
    image.setAttribute('src', dataUrl);
    image.src = dataUrl;
    try {
      await image.decode?.();
    } catch {
      // Data-URL decode failures should not abort PDF export after src replacement.
    }
  }
  return replacements;
}

function restorePdfExportImages(replacements) {
  if (!Array.isArray(replacements)) {
    return;
  }
  for (const item of replacements) {
    const image = item?.image;
    const source = String(item?.source || '');
    if (!(image instanceof HTMLImageElement) || !source) {
      continue;
    }
    image.setAttribute('src', source);
    image.src = source;
  }
}

function parseCssPixels(value) {
  const parsed = Number.parseFloat(String(value || '').replace(/px$/i, '').trim());
  return Number.isFinite(parsed) ? parsed : 0;
}

function intrinsicSvgSizeForPdf(svg) {
  if (!(svg instanceof SVGElement)) {
    return { width: 0, height: 0 };
  }
  const viewBox = String(svg.getAttribute('viewBox') || '').trim();
  if (viewBox) {
    const parts = viewBox.split(/[\s,]+/).map((part) => Number.parseFloat(part));
    if (parts.length === 4 && parts.every((value) => Number.isFinite(value))) {
      const width = Math.abs(parts[2]);
      const height = Math.abs(parts[3]);
      if (width > 0 && height > 0) {
        return { width, height };
      }
    }
  }
  const width = parseCssPixels(svg.getAttribute('width'));
  const height = parseCssPixels(svg.getAttribute('height'));
  if (width > 0 && height > 0) {
    return { width, height };
  }
  try {
    const box = svg.getBBox();
    if (box && box.width > 0 && box.height > 0) {
      return { width: box.width, height: box.height };
    }
  } catch {
    // Ignore SVG bbox failures and fall back to the live box below.
  }
  const rect = svg.getBoundingClientRect();
  return {
    width: Math.max(0, rect.width),
    height: Math.max(0, rect.height),
  };
}

function maxSvgFontPxForPdf(shell) {
  if (!(shell instanceof HTMLElement)) {
    return 12;
  }
  let maxFontPx = 0;
  for (const node of Array.from(shell.querySelectorAll('svg text, svg tspan, svg foreignObject, svg foreignObject *'))) {
    if (!(node instanceof Element)) {
      continue;
    }
    const rawFont = node.getAttribute('font-size')
      || ((node instanceof HTMLElement || node instanceof SVGElement) ? getComputedStyle(node).fontSize : '');
    const fontPx = parseCssPixels(rawFont);
    if (fontPx > maxFontPx) {
      maxFontPx = fontPx;
    }
  }
  return Math.max(12, maxFontPx);
}

function printablePdfSizePx() {
  return {
    width: Math.max(1, Math.round((PDF_PAGE_WIDTH_IN - PDF_MARGIN_LEFT_IN - PDF_MARGIN_RIGHT_IN) * CSS_PX_PER_IN)),
    height: Math.max(1, Math.round((PDF_PAGE_HEIGHT_IN - PDF_MARGIN_TOP_IN - PDF_MARGIN_BOTTOM_IN) * CSS_PX_PER_IN)),
  };
}

function restoreStyleAttribute(element, styleText) {
  if (!(element instanceof Element)) {
    return;
  }
  if (styleText === null) {
    element.removeAttribute('style');
    return;
  }
  element.setAttribute('style', styleText);
}

function prepareDiagramLayoutForPdf() {
  const shells = Array.from(content.querySelectorAll('.diagram-shell'));
  const printable = printablePdfSizePx();
  const minReadableFontPx = PDF_MIN_DIAGRAM_FONT_PT * (4 / 3);
  const restoreEntries = [];
  const contentTop = (content.getBoundingClientRect().top + window.scrollY) || 0;

  for (const shell of shells) {
    if (!(shell instanceof HTMLElement)) {
      continue;
    }
    const viewport = shell.querySelector('.diagram-viewport');
    const canvas = shell.querySelector('.diagram-canvas');
    const svg = shell.querySelector('.diagram-canvas svg');
    if (!(viewport instanceof HTMLElement) || !(canvas instanceof HTMLElement) || !(svg instanceof SVGElement)) {
      continue;
    }

    restoreEntries.push({
      shell,
      shellClassName: shell.className,
      shellStyle: shell.getAttribute('style'),
      viewport,
      viewportStyle: viewport.getAttribute('style'),
      canvas,
      canvasStyle: canvas.getAttribute('style'),
      svg,
      svgStyle: svg.getAttribute('style'),
    });

    const size = intrinsicSvgSizeForPdf(svg);
    if (!(size.width > 0 && size.height > 0)) {
      continue;
    }

    const widthScale = Math.min(1, printable.width / size.width);
    const pageFitScale = Math.min(widthScale, printable.height / size.height);
    const fontPx = maxSvgFontPxForPdf(shell);
    const fitFontPx = pageFitScale * fontPx;
    const keepOnOnePage = (size.height * widthScale) <= printable.height || fitFontPx >= minReadableFontPx || pageFitScale >= 0.42;
    const chosenScale = keepOnOnePage ? pageFitScale : widthScale;
    const diagramWidth = Math.max(1, Math.round(size.width * chosenScale));
    const diagramHeight = Math.max(1, Math.round(size.height * chosenScale));
    const relativeTop = Math.max(0, (shell.getBoundingClientRect().top + window.scrollY) - contentTop);
    const pageOffset = relativeTop % printable.height;
    const remaining = Math.max(0, printable.height - pageOffset);
    const shouldBreakBefore = keepOnOnePage && pageOffset > 1 && diagramHeight > (remaining + 1);

    shell.classList.toggle('mdext-pdf-allow-break', !keepOnOnePage);
    shell.classList.toggle('mdext-pdf-break-before', shouldBreakBefore);
    shell.style.setProperty('--mdext-print-section-width', `${printable.width}px`);
    shell.style.setProperty('--mdext-print-diagram-width', `${diagramWidth}px`);
    shell.style.setProperty('--mdext-print-diagram-max-width', `${diagramWidth}px`);
    if (keepOnOnePage) {
      shell.style.setProperty('--mdext-print-diagram-height', `${diagramHeight}px`);
      shell.style.setProperty('--mdext-print-diagram-max-height', `${diagramHeight}px`);
    } else {
      shell.style.removeProperty('--mdext-print-diagram-height');
      shell.style.removeProperty('--mdext-print-diagram-max-height');
    }

    viewport.scrollLeft = 0;
    viewport.scrollTop = 0;
    viewport.style.setProperty('overflow', keepOnOnePage ? 'hidden' : 'visible', 'important');
    viewport.style.setProperty('max-height', 'none', 'important');
    viewport.style.setProperty('scrollbar-width', 'none', 'important');
    viewport.style.setProperty('-ms-overflow-style', 'none', 'important');

    canvas.style.setProperty('width', 'auto', 'important');
    canvas.style.setProperty('min-width', '0', 'important');
    svg.style.removeProperty('transform');
    svg.style.setProperty('display', 'block', 'important');
    svg.style.setProperty('margin', '0 auto', 'important');
    svg.style.setProperty('width', 'var(--mdext-print-diagram-width, auto)', 'important');
    svg.style.setProperty('max-width', 'var(--mdext-print-diagram-max-width, 100%)', 'important');
    if (keepOnOnePage) {
      svg.style.setProperty('height', 'var(--mdext-print-diagram-height, auto)', 'important');
      svg.style.setProperty('max-height', 'var(--mdext-print-diagram-max-height, none)', 'important');
    } else {
      svg.style.removeProperty('height');
      svg.style.removeProperty('max-height');
    }
  }

  return restoreEntries;
}

function restoreDiagramLayoutAfterPdfExport(restoreEntries) {
  if (!Array.isArray(restoreEntries)) {
    return;
  }
  for (const entry of restoreEntries) {
    if (!(entry?.shell instanceof HTMLElement)) {
      continue;
    }
    entry.shell.className = entry.shellClassName;
    restoreStyleAttribute(entry.shell, entry.shellStyle);
    restoreStyleAttribute(entry.viewport, entry.viewportStyle);
    restoreStyleAttribute(entry.canvas, entry.canvasStyle);
    restoreStyleAttribute(entry.svg, entry.svgStyle);
  }
}

function escapePdfHtmlAttribute(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

async function collectPdfStyleText() {
  const blocks = [];
  for (const sheet of Array.from(document.styleSheets || [])) {
    try {
      const rules = Array.from(sheet.cssRules || []);
      if (rules.length) {
        blocks.push(rules.map((rule) => rule.cssText).join('\n'));
        continue;
      }
    } catch {
      // Linked webview styles can reject cssRules access. Fetch them below.
    }

    const href = String(sheet.href || '').trim();
    if (!href) {
      continue;
    }
    try {
      const response = await fetch(href);
      if (response.ok) {
        blocks.push(await response.text());
      }
    } catch {
      // A missing optional stylesheet should not prevent native PDF printing.
    }
  }
  return blocks.join('\n\n');
}

function preparePdfHtmlSnapshot(styleText) {
  const bodyClone = document.body.cloneNode(true);
  bodyClone.querySelectorAll('script').forEach((node) => node.remove());
  bodyClone.querySelectorAll('a[data-mdext-href]').forEach((anchor) => {
    const href = String(anchor.getAttribute('data-mdext-href') || '').trim();
    if (href) {
      anchor.setAttribute('href', href);
    }
  });
  bodyClone.querySelectorAll('img[loading]').forEach((image) => image.removeAttribute('loading'));
  bodyClone.classList.add('mdext-pdf-export-mode');

  const htmlStyle = document.documentElement.getAttribute('style');
  const bodyStyle = document.body.getAttribute('style');
  const htmlStyleAttribute = htmlStyle ? ` style="${escapePdfHtmlAttribute(htmlStyle)}"` : '';
  const bodyStyleAttribute = bodyStyle ? ` style="${escapePdfHtmlAttribute(bodyStyle)}"` : '';
  const title = escapePdfHtmlAttribute(document.title || 'mdExt PDF');

  return `<!DOCTYPE html>
<html lang="en" class="mdext-pdf-export-mode"${htmlStyleAttribute}>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title}</title>
  <style>${String(styleText || '').replace(/<\/style/gi, '<\\/style')}</style>
</head>
<body class="${escapePdfHtmlAttribute(bodyClone.className)}"${bodyStyleAttribute}>
${bodyClone.innerHTML}
</body>
</html>`;
}

async function createPdfFromPreview() {
  if (!pdfButton || pdfButton.disabled) {
    return;
  }

  pdfButton.disabled = true;
  setStatus('Preparing vector PDF…', true);
  const previousScrollY = window.scrollY;
  let imageRestoreState = [];
  let diagramLayoutRestoreState = [];
  document.documentElement.classList.add('mdext-pdf-export-mode');
  document.body.classList.add('mdext-pdf-export-mode');

  try {
    await waitForPdfAssets();
    diagramLayoutRestoreState = prepareDiagramLayoutForPdf();
    imageRestoreState = await inlinePdfExportImages();
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const styleText = await collectPdfStyleText();
    const html = preparePdfHtmlSnapshot(styleText);
    setStatus('Printing PDF…', true);
    const delivered = await vscode.postMessage({ type: 'createPdf', html });
    if (delivered === false) {
      throw new Error('PDF export message delivery failed');
    }
  } catch (error) {
    setStatus(`PDF export failed: ${error?.message || error}`, true);
  } finally {
    restorePdfExportImages(imageRestoreState);
    restoreDiagramLayoutAfterPdfExport(diagramLayoutRestoreState);
    document.documentElement.classList.remove('mdext-pdf-export-mode');
    document.body.classList.remove('mdext-pdf-export-mode');
    window.scrollTo({ top: previousScrollY, behavior: 'auto' });
    pdfButton.disabled = false;
  }
}

function decodeBase64Utf8(value) {
  if (!value) {
    return '';
  }
  try {
    const binary = atob(value);
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    return new TextDecoder('utf-8').decode(bytes);
  } catch {
    return '';
  }
}

function compatSearch() {
  return window.__mdextSearchCompat || null;
}

function compatHighlights() {
  return window.__mdextHighlightCompat || null;
}

function updateHighlightButtons() {
  const enabled = !!latestSelectionInfo.hasSelection;
  if (highlightButton) {
    highlightButton.disabled = !enabled;
  }
  if (highlightImportantButton) {
    highlightImportantButton.disabled = !enabled;
  }
}

function refreshSelectionState() {
  const compat = compatHighlights();
  if (!compat || typeof compat.getLiveSelectionOffsets !== 'function') {
    latestSelectionInfo = {
      hasSelection: false,
      selectedText: '',
      selectionOffsetStart: null,
      selectionOffsetEnd: null,
    };
    updateHighlightButtons();
    return;
  }
  const hintedSelection = typeof compat.currentSelectionText === 'function'
    ? compat.currentSelectionText()
    : '';
  latestSelectionInfo = compat.getLiveSelectionOffsets(hintedSelection);
  updateHighlightButtons();
}

function scheduleSelectionRefresh() {
  if (selectionRefreshTimer) {
    clearTimeout(selectionRefreshTimer);
  }
  selectionRefreshTimer = setTimeout(() => {
    selectionRefreshTimer = null;
    refreshSelectionState();
  }, 45);
}

function setSearchVisibility(visible, options = {}) {
  searchBarVisible = !!visible;
  if (searchBar) {
    searchBar.hidden = !searchBarVisible;
  }
  if (searchToggleButton) {
    searchToggleButton.setAttribute('aria-expanded', searchBarVisible ? 'true' : 'false');
    searchToggleButton.classList.toggle('active', searchBarVisible || !!currentSearchState.query.trim());
  }
  if (searchBarVisible && options.focus && searchInput) {
    requestAnimationFrame(() => {
      searchInput.focus();
      searchInput.select();
    });
  }
}

function updateSearchResult(matches, query) {
  if (!searchResultNode) {
    return;
  }
  if (!String(query || '').trim()) {
    searchResultNode.textContent = '';
    return;
  }
  if (matches <= 0) {
    searchResultNode.textContent = 'No matches';
    return;
  }
  searchResultNode.textContent = matches === 1 ? '1 match' : `${matches} matches`;
}

function applySearchState(state = currentSearchState) {
  currentSearchState = normalizeSearchState(state);
  if (searchInput && searchInput.value !== currentSearchState.query) {
    searchInput.value = currentSearchState.query;
  }
  if (searchToggleButton) {
    searchToggleButton.classList.toggle('active', searchBarVisible || !!currentSearchState.query.trim());
  }
  const compat = compatSearch();
  if (!compat) {
    updateSearchResult(0, currentSearchState.query);
    return { matches: 0 };
  }
  if (!currentSearchState.query.trim()) {
    compat.clearSearchHighlights?.();
    updateSearchResult(0, '');
    return { matches: 0 };
  }
  const result = compat.highlightSearchTerms(currentSearchState);
  updateSearchResult(Number(result?.matches) || 0, currentSearchState.query);
  return result || { matches: 0 };
}

function requestSearchUpdate(scrollToFirst = false) {
  vscode.postMessage({
    type: 'setSearchQuery',
    query: searchInput ? searchInput.value : '',
    scrollToFirst,
  });
}

function scheduleSearchUpdate(scrollToFirst = false) {
  if (searchInputTimer) {
    clearTimeout(searchInputTimer);
  }
  searchInputTimer = setTimeout(() => {
    searchInputTimer = null;
    requestSearchUpdate(scrollToFirst);
  }, 70);
}

function applyPersistentHighlights(entries, allowCoordinatorSync = true) {
  currentPersistentHighlights = Array.isArray(entries) ? entries : [];
  const compat = compatHighlights();
  if (!compat || typeof compat.applyPersistentHighlights !== 'function') {
    return { resolvedEntries: currentPersistentHighlights, applied: 0 };
  }
  const incomingSignature = persistentEntriesSignature(currentPersistentHighlights);
  const result = compat.applyPersistentHighlights({
    entries: currentPersistentHighlights,
    color: PREVIEW_PERSISTENT_HIGHLIGHT_COLOR,
    importantColor: PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_COLOR,
    importantTextColor: PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_TEXT_COLOR,
    markerColor: PREVIEW_PERSISTENT_HIGHLIGHT_MARKER_COLOR,
    importantMarkerColor: PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR,
  }) || { resolvedEntries: currentPersistentHighlights, applied: 0 };
  const resolvedEntries = Array.isArray(result.resolvedEntries)
    ? result.resolvedEntries
    : currentPersistentHighlights;
  currentPersistentHighlights = resolvedEntries;
  const resolvedSignature = persistentEntriesSignature(resolvedEntries);
  if (allowCoordinatorSync && resolvedSignature !== incomingSignature) {
    vscode.postMessage({ type: 'persistentHighlightsResolved', entries: resolvedEntries });
  }
  return result;
}

function hideContextMenu() {
  if (contextMenu) {
    contextMenu.hidden = true;
  }
}

function showContextMenu(x, y) {
  if (!contextMenu) {
    return;
  }
  contextMenu.hidden = false;
  const rect = contextMenu.getBoundingClientRect();
  const left = Math.max(6, Math.min(Number(x) || 0, window.innerWidth - rect.width - 6));
  const top = Math.max(6, Math.min(Number(y) || 0, window.innerHeight - rect.height - 6));
  contextMenu.style.left = `${left}px`;
  contextMenu.style.top = `${top}px`;
}

function selectedSourceLineRange() {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount < 1 || selection.isCollapsed) {
    return null;
  }
  const range = selection.getRangeAt(0);
  const taggedNodes = Array.from(content.querySelectorAll('[data-md-line-start]')).filter((node) => {
    try {
      return range.intersectsNode(node);
    } catch {
      return false;
    }
  });
  if (!taggedNodes.length) {
    return null;
  }
  const starts = taggedNodes
    .map((node) => parseLineAttribute(node, 'data-md-line-start'))
    .filter((value) => Number.isFinite(value));
  const ends = taggedNodes
    .map((node) => parseLineAttribute(node, 'data-md-line-end'))
    .filter((value) => Number.isFinite(value));
  if (!starts.length) {
    return null;
  }
  const startLine = Math.min(...starts);
  const endLine = ends.length ? Math.max(...ends) : startLine + 1;
  return { startLine, endLine: Math.max(startLine + 1, endLine) };
}

function handleCopyRequest(kind) {
  refreshSelectionState();
  if (!latestSelectionInfo.hasSelection) {
    setStatus('Select preview text to copy', true);
    return;
  }
  const selectedText = String(latestSelectionInfo.selectedText || '');
  if (kind === 'rendered') {
    vscode.postMessage({ type: 'copyRenderedText', text: selectedText });
    return;
  }
  const sourceRange = selectedSourceLineRange();
  vscode.postMessage({
    type: 'copySourceMarkdown',
    selectedText,
    startLine: sourceRange?.startLine,
    endLine: sourceRange?.endLine,
  });
}

function handleHighlightRequest(kind) {
  refreshSelectionState();
  if (!latestSelectionInfo.hasSelection) {
    setStatus('Select text to highlight', true);
    return;
  }
  vscode.postMessage({
    type: 'addPersistentHighlight',
    kind,
    start: Number(latestSelectionInfo.selectionOffsetStart),
    end: Number(latestSelectionInfo.selectionOffsetEnd),
    selectedText: String(latestSelectionInfo.selectedText || ''),
  });
}

function headerOffsetTop() {
  const header = document.querySelector('.preview-header');
  if (!header) {
    return 0;
  }
  return Math.max(0, Math.floor(header.getBoundingClientRect().bottom));
}

function parseLineAttribute(node, name) {
  const value = Number.parseInt(node?.getAttribute(name) || '', 10);
  return Number.isFinite(value) ? value : null;
}

function approximateTopVisibleLine() {
  const xPositions = [0.36, 0.52, 0.68]
    .map((ratio) => Math.max(18, Math.min(window.innerWidth - 18, Math.floor(window.innerWidth * ratio))));
  const top = headerOffsetTop();
  const yPositions = [10, 22, 38, 56, 78]
    .map((offset) => Math.min(window.innerHeight - 8, top + offset));

  for (const y of yPositions) {
    for (const x of xPositions) {
      const element = document.elementFromPoint(x, y);
      const tagged = element?.closest?.('[data-md-line-start]');
      const line = parseLineAttribute(tagged, 'data-md-line-start');
      if (Number.isFinite(line)) {
        return line;
      }
    }
  }

  let bestLine = 0;
  const limit = top + 8;
  for (const tagged of content.querySelectorAll('[data-md-line-start]')) {
    const rect = tagged.getBoundingClientRect();
    const line = parseLineAttribute(tagged, 'data-md-line-start');
    if (!Number.isFinite(line)) {
      continue;
    }
    if (rect.top <= limit) {
      bestLine = line;
      continue;
    }
    break;
  }
  return bestLine;
}

function findNodeForSourceLine(line) {
  const requestedLine = Math.max(0, Math.floor(Number(line) || 0));
  const taggedNodes = content.querySelectorAll('[data-md-line-start]');
  let before = null;
  let after = null;

  for (const tagged of taggedNodes) {
    const startLine = parseLineAttribute(tagged, 'data-md-line-start');
    const endLine = parseLineAttribute(tagged, 'data-md-line-end');
    if (!Number.isFinite(startLine)) {
      continue;
    }
    const exclusiveEnd = Number.isFinite(endLine) && endLine > startLine ? endLine : startLine + 1;
    if (requestedLine >= startLine && requestedLine < exclusiveEnd) {
      return tagged;
    }
    if (startLine <= requestedLine) {
      before = tagged;
      continue;
    }
    after = tagged;
    break;
  }

  return after || before || null;
}

function scrollToSourceLine(line) {
  const numericLine = Math.max(0, Math.floor(Number(line) || 0));
  if (numericLine === 0) {
    window.scrollTo({ top: 0, behavior: 'auto' });
    return true;
  }
  const target = findNodeForSourceLine(numericLine);
  if (!target) {
    return false;
  }
  const top = Math.max(0, Math.round(window.scrollY + target.getBoundingClientRect().top - headerOffsetTop() - 8));
  window.scrollTo({ top, behavior: 'auto' });
  return true;
}

function suppressScrollEvents(durationMs = 220) {
  suppressScrollEventsUntil = Math.max(suppressScrollEventsUntil, Date.now() + durationMs);
}

function reportVisibleLine(force = false) {
  if (!force && Date.now() < suppressScrollEventsUntil) {
    return;
  }
  const line = approximateTopVisibleLine();
  if (!force && line === lastReportedLine) {
    return;
  }
  lastReportedLine = line;
  vscode.postMessage({ type: 'previewScroll', line });
}

function scheduleVisibleLineReport(force = false) {
  if (scrollReportTimer) {
    clearTimeout(scrollReportTimer);
  }
  scrollReportTimer = setTimeout(() => {
    scrollReportTimer = null;
    reportVisibleLine(force);
  }, force ? 0 : 70);
}

function applySyncedScroll(line) {
  if (!Number.isFinite(line)) {
    return;
  }
  suppressScrollEvents();
  if (scrollToSourceLine(line)) {
    lastReportedLine = Math.max(0, Math.floor(line));
  }
}

function configureMermaid() {
  if (mermaidConfigured || !window.mermaid) {
    return;
  }
  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    suppressErrorRendering: true,
    theme: 'base',
    themeVariables: {
      background: 'transparent',
      primaryColor: '#1e293b',
      primaryBorderColor: '#93c5fd',
      primaryTextColor: '#e5e7eb',
      secondaryColor: '#172554',
      tertiaryColor: '#111827',
      lineColor: '#d1d5db',
      textColor: '#e5e7eb',
      edgeLabelBackground: '#0f172a',
      clusterBkg: '#1f2937',
      clusterBorder: '#94a3b8',
      actorBkg: '#1e293b',
      actorBorder: '#93c5fd',
      noteBkg: '#1f2937',
      noteBorderColor: '#93c5fd',
      fontFamily: 'var(--vscode-font-family)',
    },
  });
  mermaidConfigured = true;
}

function diagramNaturalSize(shell) {
  const svg = shell.querySelector('.diagram-canvas svg');
  if (!svg) {
    return { width: 800, height: 500 };
  }
  const viewBox = svg.viewBox && svg.viewBox.baseVal;
  const width = Number(viewBox?.width) || Number.parseFloat(svg.getAttribute('width')) || svg.getBoundingClientRect().width || 800;
  const height = Number(viewBox?.height) || Number.parseFloat(svg.getAttribute('height')) || svg.getBoundingClientRect().height || 500;
  return { width, height };
}

function installDiagramControls(shell) {
  const viewport = shell.querySelector('.diagram-viewport');
  const canvas = shell.querySelector('.diagram-canvas');
  const zoomLabel = shell.querySelector('.diagram-zoom-label');
  if (!viewport || !canvas || !zoomLabel) {
    return;
  }

  const state = { zoom: 1, dragging: false, x: 0, y: 0, left: 0, top: 0 };
  shell.__mdExtDiagramState = state;
  const applyZoom = (nextZoom) => {
    state.zoom = Math.max(0.1, Math.min(8, Number(nextZoom) || 1));
    const svg = shell.querySelector('.diagram-canvas svg');
    if (svg) {
      const size = diagramNaturalSize(shell);
      svg.style.width = `${Math.max(1, size.width * state.zoom)}px`;
      svg.style.height = `${Math.max(1, size.height * state.zoom)}px`;
    }
    zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
  };
  const fit = () => {
    const size = diagramNaturalSize(shell);
    const availableWidth = Math.max(160, viewport.clientWidth - 28);
    const availableHeight = Math.max(140, Math.min(680, window.innerHeight * 0.72));
    applyZoom(Math.min(1, availableWidth / size.width, availableHeight / size.height));
    viewport.scrollLeft = 0;
    viewport.scrollTop = 0;
  };
  shell.__mdExtFit = fit;

  shell.querySelector('.fit-button')?.addEventListener('click', fit);
  shell.querySelector('.zoom-out-button')?.addEventListener('click', () => applyZoom(state.zoom / 1.2));
  shell.querySelector('.zoom-in-button')?.addEventListener('click', () => applyZoom(state.zoom * 1.2));
  shell.querySelector('.pan-left-button')?.addEventListener('click', () => viewport.scrollBy({ left: -120, behavior: 'smooth' }));
  shell.querySelector('.pan-right-button')?.addEventListener('click', () => viewport.scrollBy({ left: 120, behavior: 'smooth' }));
  shell.querySelector('.pan-up-button')?.addEventListener('click', () => viewport.scrollBy({ top: -120, behavior: 'smooth' }));
  shell.querySelector('.pan-down-button')?.addEventListener('click', () => viewport.scrollBy({ top: 120, behavior: 'smooth' }));

  viewport.addEventListener('wheel', (event) => {
    if (!(event.ctrlKey || event.metaKey)) {
      return;
    }
    event.preventDefault();
    applyZoom(event.deltaY > 0 ? state.zoom / 1.12 : state.zoom * 1.12);
  }, { passive: false });
  viewport.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || event.target.closest('.diagram-toolbar')) {
      return;
    }
    state.dragging = true;
    state.x = event.clientX;
    state.y = event.clientY;
    state.left = viewport.scrollLeft;
    state.top = viewport.scrollTop;
    viewport.classList.add('dragging');
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener('pointermove', (event) => {
    if (!state.dragging) {
      return;
    }
    viewport.scrollLeft = state.left - (event.clientX - state.x);
    viewport.scrollTop = state.top - (event.clientY - state.y);
  });
  const finishDrag = () => {
    state.dragging = false;
    viewport.classList.remove('dragging');
  };
  viewport.addEventListener('pointerup', finishDrag);
  viewport.addEventListener('pointercancel', finishDrag);
  requestAnimationFrame(fit);
}

async function renderMermaidWithJavaScript(shell) {
  configureMermaid();
  const canvas = shell.querySelector('.diagram-canvas');
  const toggle = shell.querySelector('.renderer-toggle');
  const source = decodeBase64Utf8(shell.dataset.sourceB64 || '');
  if (!window.mermaid || !canvas || !source) {
    return false;
  }
  const previous = canvas.innerHTML;
  try {
    const renderId = `mdext-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const result = await window.mermaid.render(renderId, source);
    canvas.innerHTML = result.svg;
    if (typeof result.bindFunctions === 'function') {
      result.bindFunctions(canvas);
    }
    shell.dataset.renderer = 'javascript';
    if (toggle) {
      toggle.textContent = 'J';
      toggle.title = shell.dataset.rustSvgB64 ? 'JavaScript renderer active; switch to Rust' : 'JavaScript renderer active; Rust output unavailable';
    }
    requestAnimationFrame(() => shell.__mdExtFit?.());
    return true;
  } catch (error) {
    canvas.innerHTML = previous;
    setStatus(`Mermaid JavaScript render failed: ${error?.message || error}`, true);
    return false;
  }
}

function restoreRustMermaid(shell) {
  const canvas = shell.querySelector('.diagram-canvas');
  const toggle = shell.querySelector('.renderer-toggle');
  const svg = decodeBase64Utf8(shell.dataset.rustSvgB64 || '');
  if (!canvas || !svg.includes('<svg')) {
    return false;
  }
  canvas.innerHTML = svg;
  shell.dataset.renderer = 'rust';
  if (toggle) {
    toggle.textContent = 'R';
    toggle.title = 'Rust renderer active; switch to JavaScript';
  }
  requestAnimationFrame(() => shell.__mdExtFit?.());
  return true;
}

async function initializeDiagrams(revision) {
  const shells = Array.from(content.querySelectorAll('.diagram-shell'));
  for (const shell of shells) {
    if (revision !== renderRevision) {
      return;
    }
    installDiagramControls(shell);
    const toggle = shell.querySelector('.renderer-toggle');
    toggle?.addEventListener('click', async () => {
      toggle.disabled = true;
      try {
        if (shell.dataset.renderer === 'javascript') {
          restoreRustMermaid(shell);
        } else {
          await renderMermaidWithJavaScript(shell);
        }
      } finally {
        toggle.disabled = false;
      }
    });
    if (shell.dataset.kind === 'mermaid' && shell.dataset.renderer === 'javascript') {
      await renderMermaidWithJavaScript(shell);
    }
  }
}

async function typesetMath(revision) {
  if (!window.MathJax?.typesetPromise || revision !== renderRevision) {
    return;
  }
  try {
    window.MathJax.typesetClear?.([content]);
    await window.MathJax.typesetPromise([content]);
  } catch (error) {
    setStatus(`MathJax render warning: ${error?.message || error}`);
  }
}

async function applyRender(message) {
  const revision = ++renderRevision;
  const previousTop = window.scrollY;
  const previousLine = approximateTopVisibleLine();
  const messageLine = Number(message.scrollLine);
  currentSearchState = normalizeSearchState(message.searchState);
  if (message.previewFontSize !== null
      && message.previewFontSize !== undefined
      && Number.isFinite(Number(message.previewFontSize))) {
    setPreviewFontSize(Number(message.previewFontSize), false, false);
  }
  if (currentSearchState.query.trim()) {
    setSearchVisibility(true);
  } else if (searchToggleButton) {
    searchToggleButton.classList.toggle('active', searchBarVisible);
  }
  window.MathJax?.typesetClear?.([content]);
  content.innerHTML = String(message.body || '');
  titleNode.textContent = String(message.title || 'Markdown');
  pathNode.textContent = String(message.path || '');
  document.title = `${message.title || 'Markdown'} — mdExt`;
  await Promise.all([initializeDiagrams(revision), typesetMath(revision)]);
  if (revision === renderRevision) {
    applyPersistentHighlights(message.persistentHighlights, true);
    applySearchState(currentSearchState);
    latestSelectionInfo = {
      hasSelection: false,
      selectedText: '',
      selectionOffsetStart: null,
      selectionOffsetEnd: null,
    };
    updateHighlightButtons();
    suppressScrollEvents();
    const targetLine = Number.isFinite(messageLine) ? messageLine : previousLine;
    if (Number.isFinite(targetLine) && scrollToSourceLine(targetLine)) {
      lastReportedLine = Math.max(0, Math.floor(targetLine));
    } else {
      window.scrollTo({ top: previousTop, behavior: 'auto' });
    }
    setStatus('Preview updated');
  }
}

window.addEventListener('message', (event) => {
  const message = event.data || {};
  if (message.type === 'renderStarted') {
    setStatus('Rendering…', true);
  } else if (message.type === 'render') {
    applyRender(message);
  } else if (message.type === 'status') {
    setStatus(String(message.message || ''), Boolean(message.persistent));
  } else if (message.type === 'searchState') {
    const state = normalizeSearchState(message);
    currentSearchState = state;
    if (state.query.trim()) {
      setSearchVisibility(true);
    }
    applySearchState(state);
  } else if (message.type === 'persistentHighlights') {
    applyPersistentHighlights(message.entries, false);
    applySearchState(currentSearchState);
  } else if (message.type === 'previewFontSize') {
    setPreviewFontSize(Number(message.size), false, false);
  } else if (message.type === 'syncScroll') {
    applySyncedScroll(Number(message.line));
  } else if (message.type === 'renderError') {
    setStatus(String(message.message || 'Preview render failed.'), true);
  }
});

window.addEventListener('scroll', () => scheduleVisibleLineReport(false), { passive: true });
document.addEventListener('selectionchange', () => scheduleSelectionRefresh());
document.addEventListener('keydown', (event) => {
  if (event.altKey && !event.ctrlKey && !event.metaKey) {
    const key = String(event.key || '');
    if (key === '+' || key === '=') {
      event.preventDefault();
      adjustPreviewFontSize(1);
      return;
    }
    if (key === '-' || key === '_') {
      event.preventDefault();
      adjustPreviewFontSize(-1);
      return;
    }
  }
  if ((event.ctrlKey || event.metaKey) && String(event.key || '').toLowerCase() === 'f') {
    event.preventDefault();
    setSearchVisibility(true, { focus: true });
    return;
  }
  if (event.key === 'Escape') {
    hideContextMenu();
  }
});
document.addEventListener('mousedown', (event) => {
  if (contextMenu && !contextMenu.hidden && !contextMenu.contains(event.target)) {
    hideContextMenu();
  }
});
window.addEventListener('keyup', () => scheduleSelectionRefresh(), { passive: true });
window.addEventListener('mouseup', () => scheduleSelectionRefresh(), { passive: true });

content.addEventListener('click', (event) => {
  const target = event.target instanceof Element ? event.target : null;
  const anchor = target?.closest('a[data-mdext-href]');
  if (!anchor) {
    return;
  }
  event.preventDefault();
  vscode.postMessage({ type: 'openLink', href: anchor.getAttribute('data-mdext-href') || '' });
});

content.addEventListener('contextmenu', (event) => {
  refreshSelectionState();
  if (!latestSelectionInfo.hasSelection) {
    hideContextMenu();
    return;
  }
  event.preventDefault();
  showContextMenu(event.clientX, event.clientY);
});

content.addEventListener('dblclick', (event) => {
  const target = event.target instanceof Element ? event.target : null;
  const sourceNode = target?.closest('[data-md-line-start]');
  if (!sourceNode) {
    return;
  }
  vscode.postMessage({ type: 'revealLine', line: Number(sourceNode.getAttribute('data-md-line-start')) || 0 });
});

searchToggleButton?.addEventListener('click', () => {
  setSearchVisibility(!searchBarVisible, { focus: !searchBarVisible });
});

editButton?.addEventListener('click', () => {
  vscode.postMessage({ type: 'edit' });
});

searchCloseButton?.addEventListener('click', () => {
  setSearchVisibility(false);
});

searchInput?.addEventListener('input', () => {
  scheduleSearchUpdate(false);
});

searchInput?.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    if (searchInputTimer) {
      clearTimeout(searchInputTimer);
      searchInputTimer = null;
    }
    requestSearchUpdate(true);
  } else if (event.key === 'Escape') {
    event.preventDefault();
    if (searchInput.value) {
      searchInput.value = '';
      requestSearchUpdate(false);
    } else {
      setSearchVisibility(false);
    }
  }
});

for (const button of [highlightButton, highlightImportantButton]) {
  button?.addEventListener('mousedown', (event) => {
    event.preventDefault();
  });
}

highlightButton?.addEventListener('click', () => {
  handleHighlightRequest(PREVIEW_HIGHLIGHT_KIND_NORMAL);
});

highlightImportantButton?.addEventListener('click', () => {
  handleHighlightRequest(PREVIEW_HIGHLIGHT_KIND_IMPORTANT);
});

for (const button of [
  contextCopyRenderedButton,
  contextCopySourceButton,
  contextHighlightButton,
  contextHighlightImportantButton,
]) {
  button?.addEventListener('mousedown', (event) => event.preventDefault());
}
contextCopyRenderedButton?.addEventListener('click', () => {
  hideContextMenu();
  handleCopyRequest('rendered');
});
contextCopySourceButton?.addEventListener('click', () => {
  hideContextMenu();
  handleCopyRequest('source');
});
contextHighlightButton?.addEventListener('click', () => {
  hideContextMenu();
  handleHighlightRequest(PREVIEW_HIGHLIGHT_KIND_NORMAL);
});
contextHighlightImportantButton?.addEventListener('click', () => {
  hideContextMenu();
  handleHighlightRequest(PREVIEW_HIGHLIGHT_KIND_IMPORTANT);
});

refreshButton.addEventListener('click', () => vscode.postMessage({ type: 'refresh' }));
pdfButton?.addEventListener('click', () => {
  createPdfFromPreview();
});

compatHighlights()?.installMarkerOverlays?.();
refreshSelectionState();
