'use strict';

const vscode = acquireVsCodeApi();
const content = document.getElementById('preview-content');
const titleNode = document.getElementById('document-title');
const pathNode = document.getElementById('document-path');
const statusNode = document.getElementById('render-status');
const refreshButton = document.getElementById('refresh-button');
const pdfButton = document.getElementById('pdf-button');
const searchToggleButton = document.getElementById('search-toggle-button');
const searchBar = document.getElementById('preview-searchbar');
const searchInput = document.getElementById('preview-search-input');
const searchResultNode = document.getElementById('preview-search-result');
const searchCloseButton = document.getElementById('preview-search-close-button');
const highlightButton = document.getElementById('highlight-button');
const highlightImportantButton = document.getElementById('highlight-important-button');
const contextMenu = document.getElementById('preview-context-menu');
const contextHighlightButton = document.getElementById('context-highlight-button');
const contextHighlightImportantButton = document.getElementById('context-highlight-important-button');

const PREVIEW_HIGHLIGHT_KIND_NORMAL = 'normal';
const PREVIEW_HIGHLIGHT_KIND_IMPORTANT = 'important';
const PREVIEW_PERSISTENT_HIGHLIGHT_COLOR = 'rgba(102, 86, 178, 0.36)';
const PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_COLOR = 'rgba(225, 214, 255, 0.76)';
const PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_TEXT_COLOR = '#170534';
const PREVIEW_PERSISTENT_HIGHLIGHT_MARKER_COLOR = 'rgba(112, 90, 188, 0.92)';
const PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR = 'rgba(154, 132, 220, 0.96)';
const PDF_SAVE_RAW_CHUNK_BYTES = 48 * 1024;
const PDF_SAVE_YIELD_CHUNK_INTERVAL = 6;
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
let previewFontSize = Number(vscode.getState()?.previewFontSize) || null;
let nextPdfSaveId = 0;
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

function setPreviewFontSize(nextSize, announce = true) {
  const size = Math.max(8, Math.min(40, Number(nextSize) || currentPreviewFontSize()));
  previewFontSize = Math.round(size * 10) / 10;
  document.documentElement.style.setProperty('--mdext-preview-font-size', `${previewFontSize}px`);
  vscode.setState({ ...(vscode.getState() || {}), previewFontSize });
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

function imageNaturalSize(image) {
  const width = Math.max(1, Number(image?.naturalWidth) || Number(image?.width) || 0);
  const height = Math.max(1, Number(image?.naturalHeight) || Number(image?.height) || 0);
  return { width, height };
}

async function imageElementToDataUrl(image) {
  const { width, height } = imageNaturalSize(image);
  if (!(width > 0 && height > 0)) {
    return '';
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) {
    return '';
  }
  context.drawImage(image, 0, 0, width, height);
  return canvas.toDataURL('image/png');
}

async function imageSourceToDataUrl(image, source) {
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
    // Fall through to canvas extraction from the already-loaded preview image.
  }

  try {
    return await imageElementToDataUrl(image);
  } catch {
    return '';
  }
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
    const dataUrl = await imageSourceToDataUrl(image, source);
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

function byteSliceToBinaryString(bytes, start, end) {
  const clampStart = Math.max(0, Math.floor(Number(start) || 0));
  const clampEnd = Math.max(clampStart, Math.min(bytes.length, Math.floor(Number(end) || bytes.length)));
  const charChunk = 0x8000;
  let binary = '';
  for (let offset = clampStart; offset < clampEnd; offset += charChunk) {
    const nextOffset = Math.min(clampEnd, offset + charChunk);
    binary += String.fromCharCode(...bytes.subarray(offset, nextOffset));
  }
  return binary;
}

async function normalizePdfArrayBuffer(value) {
  if (value instanceof ArrayBuffer) {
    return value;
  }
  if (ArrayBuffer.isView(value)) {
    const view = value;
    return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
  }
  if (value && typeof value.arrayBuffer === 'function') {
    return value.arrayBuffer();
  }
  throw new Error('PDF renderer returned an unsupported output type');
}

async function postVsCodeMessage(message) {
  const delivered = await vscode.postMessage(message);
  if (delivered === false) {
    throw new Error('PDF export message delivery failed');
  }
}

async function postPdfSavePayloadFromArrayBuffer(bufferLike) {
  const arrayBuffer = await normalizePdfArrayBuffer(bufferLike);
  const bytes = new Uint8Array(arrayBuffer);
  if (!bytes.length) {
    throw new Error('PDF renderer returned empty output');
  }

  const saveId = `pdf-${Date.now().toString(16)}-${(nextPdfSaveId += 1).toString(16)}`;
  const totalChunks = Math.max(1, Math.ceil(bytes.length / PDF_SAVE_RAW_CHUNK_BYTES));
  await postVsCodeMessage({ type: 'savePdfStart', saveId, totalChunks });

  for (let index = 0; index < totalChunks; index += 1) {
    const start = index * PDF_SAVE_RAW_CHUNK_BYTES;
    const end = Math.min(bytes.length, start + PDF_SAVE_RAW_CHUNK_BYTES);
    await postVsCodeMessage({
      type: 'savePdfChunk',
      saveId,
      index,
      data: btoa(byteSliceToBinaryString(bytes, start, end)),
    });
    if ((index + 1) % PDF_SAVE_YIELD_CHUNK_INTERVAL === 0) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  await postVsCodeMessage({ type: 'savePdfEnd', saveId });
}

function resolvePdfFactory() {
  if (typeof window.html2pdf === 'function') {
    return window.html2pdf;
  }
  if (window.html2pdf && typeof window.html2pdf.default === 'function') {
    return window.html2pdf.default;
  }
  if (window.html2pdf && typeof window.html2pdf.html2pdf === 'function') {
    return window.html2pdf.html2pdf;
  }
  return null;
}

async function renderPdfArrayBuffer(pdfFactory) {
  const worker = pdfFactory()
    .set({
      margin: [0.55, 0.6, 0.65, 0.6],
      pagebreak: { mode: ['css', 'legacy'] },
      html2canvas: {
        scale: 1.5,
        useCORS: true,
        allowTaint: false,
        backgroundColor: '#ffffff',
        logging: false,
      },
      jsPDF: {
        unit: 'in',
        format: 'letter',
        orientation: 'portrait',
        compress: true,
      },
    })
    .from(content)
    .toPdf();

  if (typeof worker.outputPdf === 'function') {
    return normalizePdfArrayBuffer(await worker.outputPdf('arraybuffer'));
  }
  if (typeof worker.output === 'function') {
    return normalizePdfArrayBuffer(await worker.output('arraybuffer'));
  }
  if (typeof worker.get === 'function') {
    const pdfDocument = await worker.get('pdf');
    if (pdfDocument && typeof pdfDocument.output === 'function') {
      return normalizePdfArrayBuffer(pdfDocument.output('arraybuffer'));
    }
  }
  throw new Error('PDF renderer output API is unavailable');
}

async function createPdfFromPreview() {
  if (!pdfButton || pdfButton.disabled) {
    return;
  }
  const pdfFactory = resolvePdfFactory();
  if (!pdfFactory) {
    setStatus('PDF renderer is unavailable', true);
    return;
  }

  pdfButton.disabled = true;
  setStatus('Preparing PDF…', true);
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
    const pdfBuffer = await renderPdfArrayBuffer(pdfFactory);
    setStatus('Saving PDF…', true);
    await postPdfSavePayloadFromArrayBuffer(pdfBuffer);
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
if (Number.isFinite(previewFontSize)) {
  setPreviewFontSize(previewFontSize, false);
}

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

for (const button of [contextHighlightButton, contextHighlightImportantButton]) {
  button?.addEventListener('mousedown', (event) => event.preventDefault());
}
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
