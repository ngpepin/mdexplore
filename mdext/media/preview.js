'use strict';

const vscode = acquireVsCodeApi();
const content = document.getElementById('preview-content');
const titleNode = document.getElementById('document-title');
const pathNode = document.getElementById('document-path');
const statusNode = document.getElementById('render-status');
const refreshButton = document.getElementById('refresh-button');
const openSourceButton = document.getElementById('open-source-button');
const searchToggleButton = document.getElementById('search-toggle-button');
const searchBar = document.getElementById('preview-searchbar');
const searchInput = document.getElementById('preview-search-input');
const searchResultNode = document.getElementById('preview-search-result');
const searchCloseButton = document.getElementById('preview-search-close-button');
const highlightButton = document.getElementById('highlight-button');
const highlightImportantButton = document.getElementById('highlight-important-button');

const PREVIEW_HIGHLIGHT_KIND_NORMAL = 'normal';
const PREVIEW_HIGHLIGHT_KIND_IMPORTANT = 'important';
const PREVIEW_PERSISTENT_HIGHLIGHT_COLOR = 'rgba(102, 86, 178, 0.36)';
const PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_COLOR = 'rgba(225, 214, 255, 0.76)';
const PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_TEXT_COLOR = '#170534';
const PREVIEW_PERSISTENT_HIGHLIGHT_MARKER_COLOR = 'rgba(112, 90, 188, 0.92)';
const PREVIEW_PERSISTENT_HIGHLIGHT_IMPORTANT_MARKER_COLOR = 'rgba(154, 132, 220, 0.96)';

let renderRevision = 0;
let mermaidConfigured = false;
let statusTimer = null;
let scrollReportTimer = null;
let suppressScrollEventsUntil = 0;
let lastReportedLine = -1;
let searchInputTimer = null;
let selectionRefreshTimer = null;
let searchBarVisible = false;
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
    canvas.style.transform = `scale(${state.zoom})`;
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
    scheduleVisibleLineReport(true);
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

refreshButton.addEventListener('click', () => vscode.postMessage({ type: 'refresh' }));
openSourceButton.addEventListener('click', () => {
  reportVisibleLine(true);
  vscode.postMessage({ type: 'openSource' });
});

compatHighlights()?.installMarkerOverlays?.();
refreshSelectionState();
