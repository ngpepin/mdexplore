(function () {
  if (window.__pdfexploreBridge) {
    return window.__pdfexploreBridge;
  }

  const state = {
    installed: false,
    persistentEntries: [],
    searchTerms: [],
    lastClickedHighlightId: "",
    lastSelectionPayload: null,
    lastSelectionTimestamp: 0,
    refreshHandle: 0,
    observer: null,
  };

  function app() {
    return window.PDFViewerApplication || null;
  }

  function viewer() {
    const currentApp = app();
    return currentApp && currentApp.pdfViewer ? currentApp.pdfViewer : null;
  }

  function viewerContainer() {
    return document.getElementById("viewerContainer");
  }

  function escapeRegExp(value) {
    return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function normalizedText(value) {
    return String(value || "").replace(/\r\n?/g, "\n");
  }

  function pageElements() {
    return Array.from(document.querySelectorAll("#viewer .page[data-page-number]"));
  }

  function ensureInjectedStyle() {
    if (document.getElementById("__pdfexplore_embedded_style")) {
      return;
    }
    const style = document.createElement("style");
    style.id = "__pdfexplore_embedded_style";
    style.textContent = `
html, body {
  background: #0f1218 !important;
}
#toolbarContainer,
#secondaryToolbar,
#sidebarContainer,
#editorUndoBar,
#loadingBar,
.doorHanger,
.dialogButtonSpacer,
.toolbar,
#findbar {
  display: none !important;
}
#outerContainer,
#mainContainer,
#viewerContainer,
#viewer {
  inset: 0 !important;
  top: 0 !important;
}
#viewerContainer {
  background: #0f1218 !important;
}
.page {
  box-shadow: 0 18px 32px rgba(0, 0, 0, 0.26) !important;
}
.pdfexplore-overlay-host {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 6;
}
.pdfexplore-highlight-rect {
  position: absolute;
  border-radius: 3px;
}
.pdfexplore-highlight-rect.search {
  background: rgba(245, 211, 79, 0.42);
}
.pdfexplore-highlight-rect.normal {
  background: rgba(187, 157, 245, 0.33);
}
.pdfexplore-highlight-rect.important {
  background: rgba(239, 125, 125, 0.34);
}
`;
    document.head.appendChild(style);
  }

  function ensureOverlayHost(pageEl) {
    let host = pageEl.querySelector(":scope > .pdfexplore-overlay-host");
    if (!host) {
      host = document.createElement("div");
      host.className = "pdfexplore-overlay-host";
      pageEl.appendChild(host);
    }
    return host;
  }

  function clearOverlayClass(className) {
    for (const host of document.querySelectorAll(".pdfexplore-overlay-host")) {
      for (const node of Array.from(host.querySelectorAll(`.${className}`))) {
        node.remove();
      }
    }
  }

  function collectTextNodes(layer) {
    const walker = document.createTreeWalker(layer, NodeFilter.SHOW_TEXT);
    const nodes = [];
    let text = "";
    let node = null;
    while ((node = walker.nextNode())) {
      const value = normalizedText(node.nodeValue || "");
      if (!value) {
        continue;
      }
      nodes.push({
        node,
        start: text.length,
        end: text.length + value.length,
      });
      text += value;
    }
    return { text, nodes };
  }

  function pageIndex(pageEl) {
    const layer = pageEl.querySelector(".textLayer");
    if (!layer) {
      return null;
    }
    const cachedText = layer.__pdfexploreIndex;
    if (cachedText) {
      return cachedText;
    }
    const index = collectTextNodes(layer);
    layer.__pdfexploreIndex = index;
    return index;
  }

  function invalidatePageIndexes() {
    for (const layer of document.querySelectorAll(".textLayer")) {
      delete layer.__pdfexploreIndex;
    }
  }

  function nodeOffsetForAbsolute(index, absoluteOffset) {
    if (!index || !Array.isArray(index.nodes)) {
      return null;
    }
    for (const item of index.nodes) {
      if (absoluteOffset >= item.start && absoluteOffset <= item.end) {
        return {
          node: item.node,
          offset: Math.max(0, Math.min(item.node.nodeValue.length, absoluteOffset - item.start)),
        };
      }
    }
    const last = index.nodes[index.nodes.length - 1];
    if (!last) {
      return null;
    }
    return {
      node: last.node,
      offset: last.node.nodeValue.length,
    };
  }

  function rangeForOffsets(pageEl, start, end) {
    const index = pageIndex(pageEl);
    if (!index || !index.text || end <= start) {
      return null;
    }
    const startPos = nodeOffsetForAbsolute(index, start);
    const endPos = nodeOffsetForAbsolute(index, end);
    if (!startPos || !endPos) {
      return null;
    }
    const range = document.createRange();
    range.setStart(startPos.node, startPos.offset);
    range.setEnd(endPos.node, endPos.offset);
    return range;
  }

  function rectsForRange(pageEl, range) {
    if (!range) {
      return [];
    }
    const pageRect = pageEl.getBoundingClientRect();
    const rects = [];
    for (const rect of Array.from(range.getClientRects())) {
      const width = rect.width;
      const height = rect.height;
      if (width <= 0 || height <= 0) {
        continue;
      }
      rects.push({
        left: rect.left - pageRect.left,
        top: rect.top - pageRect.top,
        width,
        height,
      });
    }
    return rects;
  }

  function paintRects(pageEl, rects, kind, highlightId) {
    if (!rects.length) {
      return;
    }
    const host = ensureOverlayHost(pageEl);
    for (const rect of rects) {
      const node = document.createElement("div");
      node.className = `pdfexplore-highlight-rect ${kind}`;
      node.style.left = `${rect.left}px`;
      node.style.top = `${rect.top}px`;
      node.style.width = `${rect.width}px`;
      node.style.height = `${rect.height}px`;
      if (kind !== "search" && highlightId) {
        node.dataset.highlightId = highlightId;
      }
      host.appendChild(node);
    }
  }

  function pageNumberForNode(node) {
    const element =
      node instanceof Element ? node : node && node.parentElement ? node.parentElement : null;
    const pageEl = element ? element.closest(".page[data-page-number]") : null;
    if (!pageEl) {
      return null;
    }
    const raw = Number.parseInt(pageEl.dataset.pageNumber || "", 10);
    return Number.isFinite(raw) ? raw : null;
  }

  function rangeOffsetsForSelection(pageEl, range) {
    const index = pageIndex(pageEl);
    if (!index || !Array.isArray(index.nodes)) {
      return null;
    }
    const startContainer = range.startContainer;
    const endContainer = range.endContainer;
    let startOffset = -1;
    let endOffset = -1;
    for (const item of index.nodes) {
      if (item.node === startContainer) {
        startOffset = item.start + range.startOffset;
      }
      if (item.node === endContainer) {
        endOffset = item.start + range.endOffset;
      }
    }
    if (startOffset < 0 || endOffset < 0 || endOffset <= startOffset) {
      return null;
    }
    return { start: startOffset, end: endOffset };
  }

  function refreshPersistentHighlights() {
    clearOverlayClass("normal");
    clearOverlayClass("important");
    const entries = Array.isArray(state.persistentEntries) ? state.persistentEntries : [];
    for (const entry of entries) {
      const pageNum = Number.parseInt(entry.page, 10);
      if (!Number.isFinite(pageNum) || pageNum <= 0) {
        continue;
      }
      const pageEl = document.querySelector(`#viewer .page[data-page-number="${pageNum}"]`);
      if (!pageEl) {
        continue;
      }
      const start = Number.parseInt(entry.start, 10);
      const end = Number.parseInt(entry.end, 10);
      if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
        continue;
      }
      const range = rangeForOffsets(pageEl, start, end);
      if (!range) {
        continue;
      }
      paintRects(
        pageEl,
        rectsForRange(pageEl, range),
        String(entry.kind || "").toLowerCase() === "important" ? "important" : "normal",
        String(entry.id || ""),
      );
    }
  }

  function refreshSearchHighlights() {
    clearOverlayClass("search");
    const terms = Array.isArray(state.searchTerms) ? state.searchTerms : [];
    if (!terms.length) {
      return;
    }
    for (const pageEl of pageElements()) {
      const index = pageIndex(pageEl);
      if (!index || !index.text) {
        continue;
      }
      for (const term of terms) {
        const rawText = String(term && term.text ? term.text : "");
        if (!rawText.trim()) {
          continue;
        }
        const flags = term && term.caseSensitive ? "g" : "gi";
        const pattern = new RegExp(escapeRegExp(rawText), flags);
        let match = null;
        while ((match = pattern.exec(index.text))) {
          const start = match.index;
          const end = start + match[0].length;
          if (end <= start) {
            pattern.lastIndex = start + 1;
            continue;
          }
          const range = rangeForOffsets(pageEl, start, end);
          if (range) {
            paintRects(pageEl, rectsForRange(pageEl, range), "search", "");
          }
          if (pattern.lastIndex <= start) {
            pattern.lastIndex = start + 1;
          }
        }
      }
    }
  }

  function refreshHighlights() {
    invalidatePageIndexes();
    refreshPersistentHighlights();
    refreshSearchHighlights();
  }

  function scheduleRefresh() {
    if (state.refreshHandle) {
      window.cancelAnimationFrame(state.refreshHandle);
    }
    state.refreshHandle = window.requestAnimationFrame(() => {
      state.refreshHandle = 0;
      refreshHighlights();
    });
  }

  function locateClickedHighlightId(clientX, clientY) {
    for (const node of Array.from(document.querySelectorAll(".pdfexplore-highlight-rect[data-highlight-id]"))) {
      const rect = node.getBoundingClientRect();
      if (
        clientX >= rect.left &&
        clientX <= rect.right &&
        clientY >= rect.top &&
        clientY <= rect.bottom
      ) {
        return String(node.dataset.highlightId || "");
      }
    }
    return "";
  }

  function ensureObservers() {
    if (state.observer) {
      return;
    }
    const root = document.getElementById("viewer");
    if (!root) {
      return;
    }
    state.observer = new MutationObserver(() => scheduleRefresh());
    state.observer.observe(root, {
      childList: true,
      subtree: true,
      characterData: true,
    });
  }

  function install() {
    const currentApp = app();
    if (!currentApp || !viewer() || !viewerContainer()) {
      return false;
    }
    ensureInjectedStyle();
    ensureObservers();
    if (!state.installed) {
      document.addEventListener("contextmenu", (event) => {
        state.lastClickedHighlightId = locateClickedHighlightId(event.clientX, event.clientY);
      }, true);
      document.addEventListener("click", (event) => {
        state.lastClickedHighlightId = locateClickedHighlightId(event.clientX, event.clientY);
      }, true);
      document.addEventListener("selectionchange", () => {
        const snapshot = getSelectionInfo(0, 0);
        if (snapshot && snapshot.hasSelection) {
          state.lastSelectionPayload = {
            selectedText: String(snapshot.selectedText || ""),
            hasSelection: true,
            page: snapshot.page,
            start: snapshot.start,
            end: snapshot.end,
            multiPageSelection: Boolean(snapshot.multiPageSelection),
          };
          state.lastSelectionTimestamp = Date.now();
        }
      }, true);
      state.installed = true;
    }
    scheduleRefresh();
    return true;
  }

  function isReady() {
    const currentViewer = viewer();
    return !!(state.installed && currentViewer && currentViewer.pagesCount >= 1);
  }

  function getViewState() {
    const currentApp = app();
    const currentViewer = viewer();
    const container = viewerContainer();
    if (!currentApp || !currentViewer || !container) {
      return {};
    }
    const maxScrollTop = Math.max(1, container.scrollHeight - container.clientHeight);
    return {
      page: Number(currentApp.page || currentViewer.currentPageNumber || 1),
      pagesCount: Number(currentViewer.pagesCount || currentApp.pagesCount || 0),
      scale: String(currentViewer.currentScaleValue || "page-width"),
      scrollTop: Number(container.scrollTop || 0),
      scrollRatio: Number((container.scrollTop || 0) / maxScrollTop),
    };
  }

  function restoreViewState(stateValue) {
    const currentApp = app();
    const currentViewer = viewer();
    const container = viewerContainer();
    if (!currentApp || !currentViewer || !container || !stateValue || typeof stateValue !== "object") {
      return false;
    }
    const scale = String(stateValue.scale || "page-width");
    const page = Number.parseInt(stateValue.page, 10);
    currentViewer.currentScaleValue = scale || "page-width";
    if (Number.isFinite(page) && page > 0) {
      currentApp.page = page;
    }
    const applyScrollState = () => {
      const maxScrollTop = Math.max(0, container.scrollHeight - container.clientHeight);
      const ratio = Number(stateValue.scrollRatio);
      const scrollTop = Number(stateValue.scrollTop);
      if (Number.isFinite(ratio) && ratio >= 0) {
        container.scrollTop = Math.max(0, Math.min(maxScrollTop, ratio * maxScrollTop));
      } else if (Number.isFinite(scrollTop) && scrollTop >= 0) {
        container.scrollTop = Math.max(0, Math.min(maxScrollTop, scrollTop));
      }
    };
    window.requestAnimationFrame(() => {
      applyScrollState();
      window.requestAnimationFrame(applyScrollState);
      window.setTimeout(applyScrollState, 80);
      window.setTimeout(applyScrollState, 220);
    });
    return true;
  }

  function goToTop() {
    const currentApp = app();
    if (!currentApp) {
      return false;
    }
    currentApp.page = 1;
    const container = viewerContainer();
    if (container) {
      container.scrollTop = 0;
    }
    return true;
  }

  function getSelectionInfo(clickX, clickY) {
    const selected = window.getSelection();
    const selectedText = selected ? normalizedText(selected.toString()) : "";
    const payload = {
      selectedText,
      hasSelection: Boolean(selectedText),
      page: null,
      start: null,
      end: null,
      multiPageSelection: false,
      clickedHighlightId: locateClickedHighlightId(Number(clickX) || 0, Number(clickY) || 0) || state.lastClickedHighlightId || "",
    };
    if (!selected || !selected.rangeCount || selected.isCollapsed) {
      const cached = state.lastSelectionPayload;
      if (
        cached
        && cached.hasSelection
        && Date.now() - Number(state.lastSelectionTimestamp || 0) <= 1800
      ) {
        return {
          selectedText: String(cached.selectedText || ""),
          hasSelection: true,
          page: cached.page,
          start: cached.start,
          end: cached.end,
          multiPageSelection: Boolean(cached.multiPageSelection),
          clickedHighlightId: payload.clickedHighlightId,
        };
      }
      return payload;
    }
    const range = selected.getRangeAt(0);
    const startPage = pageNumberForNode(range.startContainer);
    const endPage = pageNumberForNode(range.endContainer);
    if (!startPage || !endPage || startPage !== endPage) {
      payload.multiPageSelection = Boolean(startPage || endPage);
      return payload;
    }
    const pageEl = document.querySelector(`#viewer .page[data-page-number="${startPage}"]`);
    if (!pageEl) {
      return payload;
    }
    const offsets = rangeOffsetsForSelection(pageEl, range);
    if (!offsets) {
      return payload;
    }
    payload.page = startPage;
    payload.start = offsets.start;
    payload.end = offsets.end;
    if (payload.hasSelection) {
      state.lastSelectionPayload = {
        selectedText: String(payload.selectedText || ""),
        hasSelection: true,
        page: payload.page,
        start: payload.start,
        end: payload.end,
        multiPageSelection: Boolean(payload.multiPageSelection),
      };
      state.lastSelectionTimestamp = Date.now();
    }
    return payload;
  }

  function setPersistentHighlights(entries) {
    state.persistentEntries = Array.isArray(entries) ? entries.slice() : [];
    scheduleRefresh();
    return true;
  }

  function setSearchTerms(terms) {
    state.searchTerms = Array.isArray(terms) ? terms.slice() : [];
    scheduleRefresh();
    return true;
  }

  function clearSearchTerms() {
    state.searchTerms = [];
    scheduleRefresh();
    return true;
  }

  window.__pdfexploreBridge = {
    install,
    isReady,
    getViewState,
    restoreViewState,
    goToTop,
    getSelectionInfo,
    setPersistentHighlights,
    setSearchTerms,
    clearSearchTerms,
  };
  return window.__pdfexploreBridge;
})();
