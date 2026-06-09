(function () {
  if (window.__pdfexploreBridge) {
    return window.__pdfexploreBridge;
  }

  const ScrollMode = Object.freeze({
    UNKNOWN: -1,
    VERTICAL: 0,
    HORIZONTAL: 1,
    WRAPPED: 2,
    PAGE: 3,
  });

  const SpreadMode = Object.freeze({
    UNKNOWN: -1,
    NONE: 0,
    ODD: 1,
    EVEN: 2,
  });

  const THREE_UP_DIVISOR = 3;
  const MIN_ZOOM_SCALE = 0.1;
  const MAX_ZOOM_SCALE = 10.0;
  const RESTORE_STABILIZE_MS = 2800;

  // Bridge runtime state. Search-indicator fields intentionally track both
  // completed and in-flight builds so marker rendering can stay progressive
  // while still supporting cancellation and interaction-priority interrupts.
  const state = {
    installed: false,
    eventBusHooksInstalled: false,
    persistentEntries: [],
    searchTerms: [],
    lastClickedHighlightId: "",
    lastSelectionPayload: null,
    lastSelectionTimestamp: 0,
    lastViewState: null,
    refreshHandle: 0,
    observer: null,
    threeUpActive: false,
    threeUpBaselineViewState: null,
    threeUpOnePageScale: 1,
    threeUpEntryOnePageScale: 1,
    threeUpBaselineScaleValue: "page-width",
    threeUpCenterPage: 1,
    threeUpNormalScrollMode: ScrollMode.VERTICAL,
    threeUpNormalSpreadMode: SpreadMode.NONE,
    pendingRestoreViewState: null,
    pendingRestoreUntil: 0,
    searchIndicatorBuildId: 0,
    searchIndicatorSignature: "",
    searchIndicatorPendingSignature: "",
    searchIndicatorEntries: [],
    searchIndicatorResumeHandle: 0,
    activeScrollHost: null,
    searchTextCacheKey: "",
    searchTextByPage: new Map(),
    searchTextPromiseByPage: new Map(),
  };

  function computeThreeUpScale(currentViewer) {
    if (!currentViewer) {
      return 1;
    }
    const rawEntryScale = Number(state.threeUpEntryOnePageScale || 0);
    const entryScale = Number.isFinite(rawEntryScale) && rawEntryScale > 0
      ? rawEntryScale
      : 1;
    const factor = clampZoomScale(state.threeUpOnePageScale) / entryScale;

    const originalScaleValue = String(currentViewer.currentScaleValue || "page-width");
    currentViewer.currentScaleValue = "page-width";
    const fitWidthScale = clampZoomScale(currentViewer.currentScale || 1);
    currentViewer.currentScaleValue = originalScaleValue || "page-width";

    return clampZoomScale((fitWidthScale / THREE_UP_DIVISOR) * factor);
  }

  function applyThreeUpLayout(currentViewer) {
    if (!currentViewer) {
      return;
    }
    currentViewer.spreadMode = SpreadMode.NONE;
    currentViewer.scrollMode = ScrollMode.WRAPPED;
    currentViewer.currentScale = computeThreeUpScale(currentViewer);
  }

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

  function isDocumentScrollHost(host) {
    return (
      host === window
      || host === document
      || host === document.documentElement
      || host === document.body
      || host === document.scrollingElement
    );
  }

  function scrollHostMetrics(host) {
    if (!host || isDocumentScrollHost(host)) {
      const doc = document.documentElement;
      const body = document.body;
      const clientWidth = Number(doc && doc.clientWidth ? doc.clientWidth : window.innerWidth || 0);
      const clientHeight = Number(doc && doc.clientHeight ? doc.clientHeight : window.innerHeight || 0);
      const scrollHeight = Math.max(
        Number(doc && doc.scrollHeight ? doc.scrollHeight : 0),
        Number(body && body.scrollHeight ? body.scrollHeight : 0),
      );
      const scrollTop = Number(window.scrollY || (doc && doc.scrollTop) || (body && body.scrollTop) || 0);
      return {
        isDocument: true,
        clientWidth,
        clientHeight,
        scrollHeight,
        scrollTop,
        rect: {
          top: 0,
          left: 0,
          right: Number(window.innerWidth || clientWidth),
          width: Number(window.innerWidth || clientWidth),
          height: Number(window.innerHeight || clientHeight),
        },
      };
    }

    const rect = typeof host.getBoundingClientRect === "function"
      ? host.getBoundingClientRect()
      : { top: 0, left: 0, right: 0, width: 0, height: 0 };
    return {
      isDocument: false,
      clientWidth: Number(host.clientWidth || rect.width || 0),
      clientHeight: Number(host.clientHeight || rect.height || 0),
      scrollHeight: Number(host.scrollHeight || 0),
      scrollTop: Number(host.scrollTop || 0),
      rect,
    };
  }

  function setScrollTopForHost(host, nextTop) {
    const target = Math.max(0, Number(nextTop || 0));
    if (!host || isDocumentScrollHost(host)) {
      window.scrollTo({ top: target, behavior: "auto" });
      return;
    }
    host.scrollTop = target;
  }

  function scrollHostCandidates() {
    return Array.from(new Set([
      document.scrollingElement,
      document.documentElement,
      document.body,
      document.getElementById("outerContainer"),
      document.getElementById("mainContainer"),
      viewerContainer(),
    ].filter((candidate) => !!candidate)));
  }

  function scrollRangeForHost(host) {
    const metrics = scrollHostMetrics(host);
    return Math.max(0, Number(metrics.scrollHeight || 0) - Number(metrics.clientHeight || 0));
  }

  function primaryScrollContainer() {
    const candidates = scrollHostCandidates();

    let best = null;
    let bestRange = -1;
    for (const candidate of candidates) {
      const range = scrollRangeForHost(candidate);
      if (range > bestRange) {
        bestRange = range;
        best = candidate;
      }
    }

    const preferred = state.activeScrollHost;
    if (preferred) {
      const preferredRange = scrollRangeForHost(preferred);
      const minimumActiveRange = Math.max(24, bestRange * 0.6);
      if (preferredRange >= minimumActiveRange) {
        return preferred;
      }
    }

    return best || document.scrollingElement || viewerContainer() || document.documentElement || document.body;
  }

  function isThreeUpToggleShortcutEvent(event) {
    if (!event || typeof event !== "object") {
      return false;
    }
    if (!event.ctrlKey || event.altKey || event.metaKey) {
      return false;
    }
    const key = String(event.key || "");
    const code = String(event.code || "");
    if (key === "\\" || key === "|") {
      return true;
    }
    if (code === "Backslash") {
      return true;
    }
    if (key === "(" || key === "9") {
      return true;
    }
    if (code === "Digit9") {
      return true;
    }
    return false;
  }

  function pageNearestViewportCenter(currentApp, currentViewer, container) {
    const fallbackPage = Number.parseInt(
      (currentApp && currentApp.page) || (currentViewer && currentViewer.currentPageNumber) || 1,
      10
    );
    if (!container) {
      return Number.isFinite(fallbackPage) && fallbackPage > 0 ? fallbackPage : 1;
    }
    const containerRect = container.getBoundingClientRect();
    const viewportCenterX = containerRect.left + (container.clientWidth / 2);
    const viewportCenterY = containerRect.top + (container.clientHeight / 2);

    let bestPage = Number.isFinite(fallbackPage) && fallbackPage > 0 ? fallbackPage : 1;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const pageEl of pageElements()) {
      const pageNumber = Number.parseInt(pageEl.dataset.pageNumber || "", 10);
      if (!Number.isFinite(pageNumber) || pageNumber <= 0) {
        continue;
      }
      const rect = pageEl.getBoundingClientRect();
      if (!rect || rect.width <= 0 || rect.height <= 0) {
        continue;
      }
      const pageCenterX = rect.left + (rect.width / 2);
      const pageCenterY = rect.top + (rect.height / 2);
      const dx = pageCenterX - viewportCenterX;
      const dy = pageCenterY - viewportCenterY;
      const distance = (dx * dx) + (dy * dy);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestPage = pageNumber;
      }
    }
    return bestPage;
  }

  function clampZoomScale(rawScale) {
    const parsed = Number(rawScale);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      return 1;
    }
    return Math.max(MIN_ZOOM_SCALE, Math.min(MAX_ZOOM_SCALE, parsed));
  }

  function cloneStateObject(rawValue) {
    if (!rawValue || typeof rawValue !== "object") {
      return {};
    }
    return Object.assign({}, rawValue);
  }

  function onePageScaleFromViewer(currentViewer) {
    return clampZoomScale(currentViewer && currentViewer.currentScale ? currentViewer.currentScale : 1);
  }

  function threeUpViewState() {
    const baseline = cloneStateObject(state.threeUpBaselineViewState);
    const currentViewer = viewer();
    const currentApp = app();
    const page = Number.parseInt(baseline.page, 10);
    const pagesCount = Number(
      baseline.pagesCount
      || (currentViewer && currentViewer.pagesCount)
      || (currentApp && currentApp.pagesCount)
      || 0
    );
    const scrollTop = Number(baseline.scrollTop || 0);
    const scrollRatio = Number(baseline.scrollRatio || 0);
    const onePageScale = clampZoomScale(state.threeUpOnePageScale);
    const baselineScaleValue = String(state.threeUpBaselineScaleValue || "").trim();
    const scaleValue = baselineScaleValue || String(onePageScale);
    return {
      page: Number.isFinite(page) && page > 0 ? page : Number(currentApp && currentApp.page ? currentApp.page : 1),
      pagesCount: Number.isFinite(pagesCount) && pagesCount > 0 ? pagesCount : 1,
      scale: scaleValue,
      scrollTop: Number.isFinite(scrollTop) && scrollTop >= 0 ? scrollTop : 0,
      scrollRatio: Number.isFinite(scrollRatio) && scrollRatio >= 0 ? scrollRatio : 0,
    };
  }

  function capturePersistedViewState() {
    if (state.threeUpActive) {
      return threeUpViewState();
    }
    return captureViewState();
  }

  function captureViewState() {
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
  position: relative !important;
  background: #0f1218 !important;
  scrollbar-width: none !important;
  -ms-overflow-style: none !important;
}
#viewerContainer::-webkit-scrollbar {
  width: 0 !important;
  height: 0 !important;
  display: none !important;
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
  background: rgba(102, 86, 178, 0.36);
}
.pdfexplore-highlight-rect.important {
  background: rgba(225, 214, 255, 0.76);
}
.pdfexplore-search-indicator-rail {
  position: fixed;
  top: 0;
  left: 0;
  width: 12px;
  height: 1px;
  pointer-events: auto;
  z-index: 2147483646;
  background: transparent;
  box-shadow: none;
  border-radius: 999px;
}
.pdfexplore-search-indicator {
  position: absolute;
  left: 1px;
  width: 10px;
  min-height: 7px;
  border-radius: 999px;
  pointer-events: auto;
  cursor: pointer;
  background: rgba(252, 227, 96, 1.0);
  box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.52);
  opacity: 0.98;
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

  function ensureSearchIndicatorRail() {
    let rail = document.querySelector("body > .pdfexplore-search-indicator-rail");
    if (!rail) {
      rail = document.createElement("div");
      rail.className = "pdfexplore-search-indicator-rail";
      document.body.appendChild(rail);
    }
    syncSearchIndicatorRailBounds(rail);
    return rail;
  }

  // Keep the right-side indicator rail aligned to whichever scroll container
  // currently owns the meaningful scroll range (document vs viewer internals).
  function syncSearchIndicatorRailBounds(rail) {
    const targetRail = rail || document.querySelector("body > .pdfexplore-search-indicator-rail");
    if (!targetRail) {
      return;
    }
    const container = primaryScrollContainer();
    const metrics = scrollHostMetrics(container);
    const railWidth = 12;
    const rightGap = 4;
    let left = Math.max(0, Math.round(window.innerWidth - railWidth - rightGap));
    let top = 0;
    let height = Math.max(1, Math.round(window.innerHeight || 1));
    if (metrics) {
      const rect = metrics.rect;
      const hasUsableRect = rect.width > 20 && rect.height > 20;
      if (hasUsableRect) {
        const containerScrollbarWidth = Math.max(
          0,
          Number(container && container.offsetWidth ? container.offsetWidth : metrics.clientWidth)
          - Number(metrics.clientWidth || 0),
        );
        const doc = document.documentElement;
        const outerScrollbarWidth = Math.max(
          0,
          Number(window.innerWidth || 0) - Number(doc && doc.clientWidth ? doc.clientWidth : window.innerWidth || 0),
        );
        const topPadding = 8;
        const bottomPadding = 8;
        const innerHeight = Math.max(1, Math.round((metrics.clientHeight || rect.height) - topPadding - bottomPadding));
        left = Math.max(
          0,
          Math.round(
            rect.left
            + Number(metrics.clientWidth || rect.width)
            - railWidth
            - rightGap
            - containerScrollbarWidth
            - outerScrollbarWidth,
          ),
        );
        top = Math.max(0, Math.round(rect.top + topPadding));
        height = innerHeight;
      }
    }
    targetRail.style.left = `${left}px`;
    targetRail.style.top = `${top}px`;
    targetRail.style.height = `${height}px`;
  }

  function clearSearchIndicators() {
    const rail = document.querySelector("body > .pdfexplore-search-indicator-rail");
    if (rail) {
      rail.remove();
    }
  }

  function clearSearchIndicatorBuildState() {
    state.searchIndicatorBuildId += 1;
    state.searchIndicatorSignature = "";
    state.searchIndicatorPendingSignature = "";
    state.searchIndicatorEntries = [];
    if (state.searchIndicatorResumeHandle) {
      window.clearTimeout(state.searchIndicatorResumeHandle);
      state.searchIndicatorResumeHandle = 0;
    }
    state.searchTextPromiseByPage = new Map();
  }

  // User clicks should win over marker throughput. Interrupt the active build,
  // allow navigation/rendering to settle, then resume using existing cache.
  function interruptSearchIndicatorBuildForInteraction() {
    if (!state.searchIndicatorPendingSignature) {
      return;
    }
    const activeTerms = normalizedSearchTerms(state.searchTerms);
    if (!activeTerms.length) {
      return;
    }
    state.searchIndicatorBuildId += 1;
    state.searchIndicatorPendingSignature = "";
    if (state.searchIndicatorResumeHandle) {
      window.clearTimeout(state.searchIndicatorResumeHandle);
      state.searchIndicatorResumeHandle = 0;
    }
    // Resume after the interaction settles so pdf.js can prioritize navigation/rendering work.
    state.searchIndicatorResumeHandle = window.setTimeout(() => {
      state.searchIndicatorResumeHandle = 0;
      scheduleSearchIndicatorBuild(state.searchTerms);
    }, 180);
  }

  function normalizedSearchTerms(rawTerms) {
    if (!Array.isArray(rawTerms)) {
      return [];
    }
    return rawTerms
      .map((entry) => {
        const text = String(entry && entry.text ? entry.text : "").trim();
        if (!text) {
          return null;
        }
        return {
          text,
          caseSensitive: Boolean(entry && entry.caseSensitive),
        };
      })
      .filter((entry) => !!entry);
  }

  function searchTermMatchers(rawTerms) {
    const terms = normalizedSearchTerms(rawTerms);
    return terms
      .map((term) => {
        const text = String(term.text || "");
        if (!text) {
          return null;
        }
        const caseSensitive = Boolean(term.caseSensitive);
        return {
          caseSensitive,
          needle: caseSensitive ? text : text.toLocaleLowerCase(),
        };
      })
      .filter((entry) => !!entry);
  }

  function currentSearchDocumentKey() {
    const currentApp = app();
    const currentDocument = currentApp && currentApp.pdfDocument ? currentApp.pdfDocument : null;
    const currentViewer = viewer();
    let documentKey = "";
    if (currentDocument && Array.isArray(currentDocument.fingerprints)) {
      documentKey = String(currentDocument.fingerprints[0] || "").trim();
    }
    if (!documentKey) {
      documentKey = String(
        (currentApp && (currentApp.url || (currentApp.baseUrl && currentApp.baseUrl.href)))
        || ""
      );
    }
    const pagesCount = Number(
      (currentViewer && currentViewer.pagesCount)
      || (currentApp && currentApp.pagesCount)
      || 0
    );
    return `${documentKey}|${pagesCount}`;
  }

  // A signature keys marker entries to document identity + normalized terms.
  // Matching signatures let us reuse built entries and avoid needless rebuilds.
  function currentSearchIndicatorSignature(terms) {
    const normalizedTerms = normalizedSearchTerms(terms);
    if (!normalizedTerms.length) {
      return "";
    }
    return `${currentSearchDocumentKey()}|${JSON.stringify(normalizedTerms)}`;
  }

  function countLiteralMatches(haystack, needle) {
    if (!haystack || !needle) {
      return 0;
    }
    let from = 0;
    let count = 0;
    while (from < haystack.length) {
      const index = haystack.indexOf(needle, from);
      if (index < 0) {
        break;
      }
      count += 1;
      from = index + 1;
    }
    return count;
  }

  function countPageMatches(pageText, matchers) {
    if (!pageText || !Array.isArray(matchers) || !matchers.length) {
      return 0;
    }
    let lowered = null;
    let total = 0;
    for (const matcher of matchers) {
      if (!matcher || !matcher.needle) {
        continue;
      }
      const haystack = matcher.caseSensitive
        ? pageText
        : (lowered || (lowered = pageText.toLocaleLowerCase()));
      total += countLiteralMatches(haystack, matcher.needle);
    }
    return total;
  }

  function resetSearchTextCache(cacheKey) {
    state.searchTextCacheKey = cacheKey;
    state.searchTextByPage = new Map();
    state.searchTextPromiseByPage = new Map();
  }

  // The page-text cache is document scoped. Any root/document switch resets the
  // cache key so stale text is never reused across different PDFs.
  function ensureSearchTextCacheForCurrentDocument() {
    const cacheKey = currentSearchDocumentKey();
    if (state.searchTextCacheKey !== cacheKey) {
      resetSearchTextCache(cacheKey);
    }
    return cacheKey;
  }

  async function pageSearchText(pdfDocument, pageNumber, cacheKey) {
    if (!pdfDocument || !Number.isFinite(pageNumber) || pageNumber <= 0) {
      return "";
    }
    if (state.searchTextCacheKey !== cacheKey) {
      return "";
    }
    if (state.searchTextByPage.has(pageNumber)) {
      return String(state.searchTextByPage.get(pageNumber) || "");
    }
    const existingPromise = state.searchTextPromiseByPage.get(pageNumber);
    if (existingPromise) {
      return String(await existingPromise);
    }

    // Deduplicate concurrent requests for the same page while a build is active.
    const task = (async () => {
      let pageText = "";
      try {
        const page = await pdfDocument.getPage(pageNumber);
        const textContent = await page.getTextContent();
        const chunks = Array.isArray(textContent && textContent.items)
          ? textContent.items
            .map((item) => String(item && typeof item.str === "string" ? item.str : ""))
            .filter((value) => !!value)
          : [];
        pageText = normalizedText(chunks.join("\n"));
      } catch (_error) {
        pageText = "";
      }
      if (state.searchTextCacheKey === cacheKey) {
        state.searchTextByPage.set(pageNumber, pageText);
      }
      return pageText;
    })();

    state.searchTextPromiseByPage.set(pageNumber, task);
    try {
      return String(await task);
    } finally {
      if (state.searchTextPromiseByPage.get(pageNumber) === task) {
        state.searchTextPromiseByPage.delete(pageNumber);
      }
    }
  }

  function tryCenterOnRenderedSearchHit(pageNumber, intraRatio) {
    const container = primaryScrollContainer();
    if (!container || !Number.isFinite(pageNumber) || pageNumber <= 0) {
      return false;
    }
    const metrics = scrollHostMetrics(container);
    const pageEl = document.querySelector(`#viewer .page[data-page-number="${pageNumber}"]`);
    if (!pageEl) {
      return false;
    }
    const searchRects = Array.from(pageEl.querySelectorAll(".pdfexplore-highlight-rect.search"));
    if (!searchRects.length) {
      return false;
    }
    const ratio = Number.isFinite(intraRatio) ? Math.max(0, Math.min(1, intraRatio)) : 0.5;
    const index = Math.max(0, Math.min(searchRects.length - 1, Math.round((searchRects.length - 1) * ratio)));
    const targetRect = searchRects[index].getBoundingClientRect();
    const containerRect = metrics.rect;
    const absoluteTop = metrics.scrollTop + (targetRect.top - containerRect.top);
    const desiredTop = Math.max(0, absoluteTop - (metrics.clientHeight * 0.35));
    setScrollTopForHost(container, desiredTop);
    return true;
  }

  function jumpToSearchIndicatorTarget(target) {
    if (!target || typeof target !== "object") {
      return;
    }
    interruptSearchIndicatorBuildForInteraction();
    const pageNumber = Number.parseInt(target.pageNumber, 10);
    const intraRatio = Number(target.intraRatio);
    const currentApp = app();
    const currentViewer = viewer();
    if (currentApp && currentViewer && Number.isFinite(pageNumber) && pageNumber > 0) {
      applyPageState(currentApp, currentViewer, pageNumber);
    }
    // Retry a few times because pdf.js may need a brief render delay before
    // search highlight rectangles exist on the target page.
    const tryFocus = () => {
      if (!Number.isFinite(pageNumber) || pageNumber <= 0) {
        return;
      }
      tryCenterOnRenderedSearchHit(pageNumber, intraRatio);
    };
    window.requestAnimationFrame(() => {
      if (tryCenterOnRenderedSearchHit(pageNumber, intraRatio)) {
        return;
      }
      window.setTimeout(() => {
        if (tryCenterOnRenderedSearchHit(pageNumber, intraRatio)) {
          return;
        }
        window.setTimeout(tryFocus, 180);
      }, 80);
    });
  }

  function renderedSearchIndicatorEntry(pageEl, rect) {
    const container = primaryScrollContainer();
    if (!container || !pageEl || !rect) {
      return null;
    }
    const metrics = scrollHostMetrics(container);
    const pageNumber = Number.parseInt(pageEl.dataset.pageNumber || "", 10);
    if (!Number.isFinite(pageNumber) || pageNumber <= 0) {
      return null;
    }
    const contentHeight = Number(metrics.scrollHeight || 0);
    if (!Number.isFinite(contentHeight) || contentHeight <= 1) {
      return null;
    }
    const pageHeight = Number(pageEl.clientHeight || 0);
    const pageRect = pageEl.getBoundingClientRect();
    const centerTop = Number(metrics.scrollTop || 0) + (Number(pageRect.top || 0) - Number(metrics.rect.top || 0)) + Number(rect.top || 0) + (Number(rect.height || 0) * 0.5);
    if (!Number.isFinite(centerTop)) {
      return null;
    }
    const intraRatio = pageHeight > 0
      ? Math.max(0, Math.min(1, (Number(rect.top || 0) + (Number(rect.height || 0) * 0.5)) / pageHeight))
      : 0.5;
    return {
      ratio: Math.max(0, Math.min(1, centerTop / contentHeight)),
      pageNumber,
      intraRatio,
    };
  }

  async function collectSearchIndicatorEntriesForTerms(terms, buildId, signature) {
    const currentApp = app();
    const currentViewer = viewer();
    const pdfDocument = currentApp && currentApp.pdfDocument ? currentApp.pdfDocument : null;
    const pagesCount = Number(
      (currentViewer && currentViewer.pagesCount)
      || (currentApp && currentApp.pagesCount)
      || 0
    );
    if (!pdfDocument || !Number.isFinite(pagesCount) || pagesCount <= 0) {
      return [];
    }
    const matchers = searchTermMatchers(terms);
    if (!matchers.length) {
      return [];
    }

    const cacheKey = ensureSearchTextCacheForCurrentDocument();

    // Build markers progressively in page order so users get early, clickable
    // coverage near the top of the document before full completion.
    const entries = [];
    const maxEntries = 2400;
    const concurrency = Math.max(2, Math.min(4, Number((navigator && navigator.hardwareConcurrency) || 4)));
    let processedBatches = 0;
    for (let batchStart = 1; batchStart <= pagesCount; batchStart += concurrency) {
      if (
        buildId !== state.searchIndicatorBuildId
        || state.searchIndicatorPendingSignature !== signature
      ) {
        return entries;
      }

      const batchEnd = Math.min(pagesCount, batchStart + concurrency - 1);
      const pageNumbers = [];
      for (let pageNumber = batchStart; pageNumber <= batchEnd; pageNumber += 1) {
        pageNumbers.push(pageNumber);
      }
      const batch = await Promise.all(
        pageNumbers.map(async (pageNumber) => {
          const pageText = await pageSearchText(pdfDocument, pageNumber, cacheKey);
          const pageHitCount = countPageMatches(pageText, matchers);
          return { pageNumber, pageHitCount };
        })
      );

      for (const item of batch) {
        if (
          buildId !== state.searchIndicatorBuildId
          || state.searchIndicatorPendingSignature !== signature
        ) {
          return entries;
        }
        const pageNumber = Number(item.pageNumber || 0);
        const pageHitCount = Number(item.pageHitCount || 0);
        if (pageHitCount <= 0 || pageNumber <= 0) {
          continue;
        }
        const markersForPage = Math.max(1, Math.min(pageHitCount, 48));
        for (let index = 0; index < markersForPage; index += 1) {
          if (entries.length >= maxEntries) {
            break;
          }
          const intraRatio = (index + 0.5) / markersForPage;
          const ratio = ((pageNumber - 1) + intraRatio) / pagesCount;
          entries.push({
            ratio: Math.max(0, Math.min(1, ratio)),
            pageNumber,
            intraRatio,
            pageHitCount,
          });
        }
      }

      // Publish partial entries after each batch to keep the rail interactive
      // throughout long scans.
      if (entries.length > 0) {
        state.searchIndicatorEntries = entries.slice();
        refreshSearchIndicators(state.searchIndicatorEntries);
      }

      processedBatches += 1;
      if (processedBatches % 3 === 0) {
        // Yield periodically so input/paint events are not starved by scan work.
        await new Promise((resolve) => window.setTimeout(resolve, 0));
      }

      if (entries.length >= maxEntries) {
        break;
      }
    }
    return entries;
  }

  function scheduleSearchIndicatorBuild(terms) {
    const normalizedTerms = normalizedSearchTerms(terms);
    if (!normalizedTerms.length) {
      clearSearchIndicatorBuildState();
      clearSearchIndicators();
      return;
    }
    const signature = currentSearchIndicatorSignature(normalizedTerms);
    if (!signature) {
      clearSearchIndicatorBuildState();
      clearSearchIndicators();
      return;
    }
    // Reuse completed entries when the query/document signature is unchanged.
    if (state.searchIndicatorSignature === signature && state.searchIndicatorEntries.length) {
      refreshSearchIndicators(state.searchIndicatorEntries);
      return;
    }
    if (state.searchIndicatorPendingSignature === signature) {
      return;
    }

    state.searchIndicatorPendingSignature = signature;
    const buildId = state.searchIndicatorBuildId + 1;
    state.searchIndicatorBuildId = buildId;
    state.searchIndicatorEntries = [];
    clearSearchIndicators();

    // Build asynchronously so paint/input can continue while markers appear.
    collectSearchIndicatorEntriesForTerms(normalizedTerms, buildId, signature)
      .then((entries) => {
        if (
          buildId !== state.searchIndicatorBuildId
          || state.searchIndicatorPendingSignature !== signature
        ) {
          return;
        }
        state.searchIndicatorEntries = Array.isArray(entries) ? entries.slice() : [];
        state.searchIndicatorSignature = signature;
        state.searchIndicatorPendingSignature = "";
        refreshSearchIndicators(state.searchIndicatorEntries);
      })
      .catch(() => {
        if (buildId !== state.searchIndicatorBuildId) {
          return;
        }
        state.searchIndicatorEntries = [];
        state.searchIndicatorSignature = "";
        state.searchIndicatorPendingSignature = "";
        clearSearchIndicators();
      });
  }

  function refreshSearchIndicators(entries) {
    // Merge nearby marker pixels into thicker clusters so dense hit regions stay
    // clickable and visually legible on long documents.
    const normalized = Array.isArray(entries)
      ? entries
        .map((entry) => {
          const ratio = Number(entry && entry.ratio);
          const pageNumber = Number.parseInt(entry && entry.pageNumber, 10);
          const intraRatio = Number(entry && entry.intraRatio);
          if (!Number.isFinite(ratio) || ratio < 0 || ratio > 1) {
            return null;
          }
          if (!Number.isFinite(pageNumber) || pageNumber <= 0) {
            return null;
          }
          return {
            ratio,
            pageNumber,
            intraRatio: Number.isFinite(intraRatio) ? Math.max(0, Math.min(1, intraRatio)) : 0.5,
          };
        })
        .filter((entry) => !!entry)
      : [];
    if (!normalized.length) {
      clearSearchIndicators();
      return;
    }
    const rail = ensureSearchIndicatorRail();
    if (!rail) {
      return;
    }
    syncSearchIndicatorRailBounds(rail);
    while (rail.firstChild) {
      rail.removeChild(rail.firstChild);
    }
    const railHeight = Math.max(1, rail.clientHeight || rail.getBoundingClientRect().height || 1);
    const maxPixel = Math.max(0, railHeight - 1);
    const markerPositions = normalized
      .map((entry) => {
        const top = Math.max(0, Math.min(maxPixel, Math.round(entry.ratio * maxPixel)));
        return {
          top,
          bottom: Math.max(top + 4, top),
          center: top + 2,
          target: entry,
        };
      })
      .sort((left, right) => left.top - right.top);
    const merged = [];
    for (const item of markerPositions) {
      const previous = merged.length ? merged[merged.length - 1] : null;
      if (previous && item.top <= previous.bottom + 2) {
        previous.bottom = Math.max(previous.bottom, item.bottom);
        previous.targets.push(item);
        continue;
      }
      merged.push({
        top: item.top,
        bottom: item.bottom,
        targets: [item],
      });
    }

    for (const item of merged) {
      const node = document.createElement("div");
      node.className = "pdfexplore-search-indicator";
      node.style.top = `${item.top}px`;
      node.style.height = `${Math.max(5, item.bottom - item.top)}px`;
      if (Array.isArray(item.targets) && item.targets.length) {
        const primary = item.targets[Math.floor(item.targets.length / 2)];
        node.dataset.pageNumber = String(primary.target.pageNumber);
        node.dataset.hitRatio = String(primary.target.ratio);
      }
      const activateMarker = (event) => {
        if (typeof event.button === "number" && event.button !== 0) {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        if (typeof event.stopImmediatePropagation === "function") {
          event.stopImmediatePropagation();
        }
        const railRect = rail.getBoundingClientRect();
        const clickY = Number.isFinite(event.clientY)
          ? event.clientY - railRect.top
          : (item.top + item.bottom) * 0.5;
        const candidate = Array.isArray(item.targets) && item.targets.length
          ? item.targets.reduce((best, next) => {
              if (!best) {
                return next;
              }
              return Math.abs(next.center - clickY) < Math.abs(best.center - clickY)
                ? next
                : best;
            }, null)
          : null;
        if (!candidate || !candidate.target) {
          return;
        }
        jumpToSearchIndicatorTarget(candidate.target);
      };
      node.addEventListener("mousedown", activateMarker);
      node.addEventListener("pointerdown", activateMarker);
      node.addEventListener("click", activateMarker);
      rail.appendChild(node);
    }
    window.requestAnimationFrame(() => syncSearchIndicatorRailBounds(rail));
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

  function rectsForRange(pageEl, range, host) {
    if (!range) {
      return [];
    }
    const baseRect = (host || pageEl).getBoundingClientRect();
    const rects = [];
    for (const rect of Array.from(range.getClientRects())) {
      const width = rect.width;
      const height = rect.height;
      if (width <= 0 || height <= 0) {
        continue;
      }
      rects.push({
        left: rect.left - baseRect.left,
        top: rect.top - baseRect.top,
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
      const host = ensureOverlayHost(pageEl);
      paintRects(
        pageEl,
        rectsForRange(pageEl, range, host),
        String(entry.kind || "").toLowerCase() === "important" ? "important" : "normal",
        String(entry.id || ""),
      );
    }
  }

  function refreshSearchHighlights() {
    clearOverlayClass("search");
    const terms = Array.isArray(state.searchTerms) ? state.searchTerms : [];
    // Fallback entries come from currently rendered highlight rects. They allow
    // immediate marker feedback before full-text scanning completes.
    const renderedFallbackEntries = [];
    if (!terms.length) {
      clearSearchIndicatorBuildState();
      clearSearchIndicators();
      return;
    }
    scheduleSearchIndicatorBuild(terms);
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
            const host = ensureOverlayHost(pageEl);
            const rects = rectsForRange(pageEl, range, host);
            paintRects(pageEl, rects, "search", "");
            for (const rect of rects) {
              const fallback = renderedSearchIndicatorEntry(pageEl, rect);
              if (fallback) {
                renderedFallbackEntries.push(fallback);
              }
            }
          }
          if (pattern.lastIndex <= start) {
            pattern.lastIndex = start + 1;
          }
        }
      }
    }
    if (state.searchIndicatorEntries.length) {
      refreshSearchIndicators(state.searchIndicatorEntries);
      return;
    }
    if (renderedFallbackEntries.length) {
      refreshSearchIndicators(renderedFallbackEntries);
      return;
    }
    clearSearchIndicators();
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
      document.addEventListener("keydown", (event) => {
        if (!isThreeUpToggleShortcutEvent(event)) {
          return;
        }
        toggleThreeUpMode();
        event.preventDefault();
        event.stopPropagation();
      }, true);
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
      const onAnyScroll = (event) => {
        // Track the dominant scroll host to avoid attaching marker semantics to
        // short-range internal scroll containers created by pdf.js layouts.
        const source = event && event.currentTarget ? event.currentTarget : null;
        if (source) {
          const mappedSource = source === window
            ? (document.scrollingElement || document.documentElement || document.body)
            : source;
          const sourceRange = scrollRangeForHost(mappedSource);
          const dominantRange = Math.max(
            0,
            ...scrollHostCandidates().map((candidate) => scrollRangeForHost(candidate)),
          );
          if (sourceRange >= Math.max(24, dominantRange * 0.5)) {
            state.activeScrollHost = mappedSource;
          }
        }
        syncSearchIndicatorRailBounds();
        if (state.threeUpActive) {
          const currentAppForCenter = app();
          const currentViewerForCenter = viewer();
          const activeContainer = primaryScrollContainer();
          if (currentAppForCenter && currentViewerForCenter && activeContainer) {
            state.threeUpCenterPage = pageNearestViewportCenter(
              currentAppForCenter,
              currentViewerForCenter,
              activeContainer
            );
          }
        }
        state.lastViewState = capturePersistedViewState();
        if (state.persistentEntries.length || state.searchTerms.length) {
          scheduleRefresh();
        }
      };
      const scrollTargets = new Set([
        viewerContainer(),
        document.getElementById("mainContainer"),
        document.getElementById("outerContainer"),
        document.scrollingElement,
        window,
      ]);
      for (const target of scrollTargets) {
        if (target && typeof target.addEventListener === "function") {
          target.addEventListener("scroll", onAnyScroll, { passive: true });
        }
      }
      window.addEventListener("resize", () => {
        syncSearchIndicatorRailBounds();
        if (state.searchTerms.length) {
          refreshSearchIndicators(state.searchIndicatorEntries);
        }
      });
      state.installed = true;
    }
    if (
      !state.eventBusHooksInstalled
      && currentApp.eventBus
      && typeof currentApp.eventBus.on === "function"
    ) {
      const refreshFromPdfJs = () => {
        state.lastViewState = capturePersistedViewState();
        scheduleRefresh();
      };
      currentApp.eventBus.on("updateviewarea", () => {
        state.lastViewState = capturePersistedViewState();
        if (state.persistentEntries.length || state.searchTerms.length) {
          scheduleRefresh();
        }
      });
      currentApp.eventBus.on("pagerendered", refreshFromPdfJs);
      currentApp.eventBus.on("pagesloaded", refreshFromPdfJs);
      currentApp.eventBus.on("textlayerrendered", refreshFromPdfJs);
      state.eventBusHooksInstalled = true;
    }
    state.lastViewState = capturePersistedViewState();
    scheduleRefresh();
    return true;
  }

  function isReady() {
    const currentViewer = viewer();
    return !!(state.installed && currentViewer && currentViewer.pagesCount >= 1);
  }

  function applyPageState(currentApp, currentViewer, page) {
    if (!Number.isFinite(page) || page <= 0) {
      return;
    }
    if (currentApp.pdfLinkService && typeof currentApp.pdfLinkService.goToPage === "function") {
      currentApp.pdfLinkService.goToPage(page);
      return;
    }
    if (typeof currentViewer.scrollPageIntoView === "function") {
      currentViewer.scrollPageIntoView({ pageNumber: page });
      return;
    }
    if ("page" in currentApp) {
      currentApp.page = page;
    }
    if ("currentPageNumber" in currentViewer) {
      currentViewer.currentPageNumber = page;
    }
  }

  function applyScrollState(container, stateValue) {
    const maxScrollTop = Math.max(0, container.scrollHeight - container.clientHeight);
    const ratio = Number(stateValue.scrollRatio);
    const scrollTop = Number(stateValue.scrollTop);
    if (Number.isFinite(scrollTop) && scrollTop >= 0) {
      container.scrollTop = Math.max(0, Math.min(maxScrollTop, scrollTop));
    } else if (Number.isFinite(ratio) && ratio >= 0) {
      container.scrollTop = Math.max(0, Math.min(maxScrollTop, ratio * maxScrollTop));
    }
  }

  function applyViewState(stateValue) {
    const currentApp = app();
    const currentViewer = viewer();
    const container = viewerContainer();
    if (!currentApp || !currentViewer || !container || !stateValue || typeof stateValue !== "object") {
      return false;
    }
    const scale = String(stateValue.scale || "page-width");
    const page = Number.parseInt(stateValue.page, 10);
    currentViewer.currentScaleValue = scale || "page-width";
    state.lastViewState = {
      page: Number.isFinite(page) && page > 0 ? page : Number(currentApp.page || currentViewer.currentPageNumber || 1),
      pagesCount: Number(currentViewer.pagesCount || currentApp.pagesCount || 0),
      scale: scale || "page-width",
      scrollTop: Number(stateValue.scrollTop || 0),
      scrollRatio: Number(stateValue.scrollRatio || 0),
    };
    state.pendingRestoreViewState = cloneStateObject(state.lastViewState);
    state.pendingRestoreUntil = Date.now() + RESTORE_STABILIZE_MS;
    applyPageState(currentApp, currentViewer, page);
    window.requestAnimationFrame(() => {
      applyPageState(currentApp, currentViewer, page);
      applyScrollState(container, stateValue);
      state.lastViewState = capturePersistedViewState();
      window.requestAnimationFrame(() => {
        applyPageState(currentApp, currentViewer, page);
        applyScrollState(container, stateValue);
        state.lastViewState = capturePersistedViewState();
      });
      window.setTimeout(() => {
        applyPageState(currentApp, currentViewer, page);
        applyScrollState(container, stateValue);
        state.lastViewState = capturePersistedViewState();
      }, 80);
      window.setTimeout(() => {
        applyPageState(currentApp, currentViewer, page);
        applyScrollState(container, stateValue);
        state.lastViewState = capturePersistedViewState();
      }, 220);
      window.setTimeout(() => {
        applyPageState(currentApp, currentViewer, page);
        applyScrollState(container, stateValue);
        state.lastViewState = capturePersistedViewState();
      }, 450);
      window.setTimeout(() => {
        applyPageState(currentApp, currentViewer, page);
        applyScrollState(container, stateValue);
        state.lastViewState = capturePersistedViewState();
      }, 900);
    });
    if (currentApp.eventBus && typeof currentApp.eventBus.on === "function") {
      const reapply = () => {
        applyPageState(currentApp, currentViewer, page);
        applyScrollState(container, stateValue);
        if (currentApp.eventBus && typeof currentApp.eventBus.off === "function") {
          currentApp.eventBus.off("pagerendered", reapply);
          currentApp.eventBus.off("pagesloaded", reapply);
        }
      };
      currentApp.eventBus.on("pagerendered", reapply);
      currentApp.eventBus.on("pagesloaded", reapply);
    }
    return true;
  }

  function enterThreeUpMode() {
    const currentApp = app();
    const currentViewer = viewer();
    if (!currentApp || !currentViewer) {
      return false;
    }
    if (state.threeUpActive) {
      return true;
    }
    const baseline = capturePersistedViewState();
    state.threeUpBaselineViewState = baseline && Object.keys(baseline).length
      ? baseline
      : {
          page: Number(currentApp.page || currentViewer.currentPageNumber || 1),
          pagesCount: Number(currentViewer.pagesCount || currentApp.pagesCount || 0),
          scale: String(currentViewer.currentScaleValue || "page-width"),
          scrollTop: 0,
          scrollRatio: 0,
        };
    state.threeUpOnePageScale = onePageScaleFromViewer(currentViewer);
    state.threeUpEntryOnePageScale = state.threeUpOnePageScale;
    state.threeUpBaselineScaleValue = String(
      currentViewer.currentScaleValue || state.threeUpOnePageScale
    );
    state.threeUpNormalScrollMode = Number.isInteger(currentViewer.scrollMode)
      ? currentViewer.scrollMode
      : ScrollMode.VERTICAL;
    state.threeUpNormalSpreadMode = Number.isInteger(currentViewer.spreadMode)
      ? currentViewer.spreadMode
      : SpreadMode.NONE;
    state.threeUpCenterPage = Number.parseInt(state.threeUpBaselineViewState.page, 10) || 1;
    state.pendingRestoreViewState = null;
    state.pendingRestoreUntil = 0;
    state.threeUpActive = true;
    applyThreeUpLayout(currentViewer);
    applyPageState(
      currentApp,
      currentViewer,
      Number.parseInt(state.threeUpBaselineViewState.page, 10)
    );
    state.lastViewState = threeUpViewState();
    scheduleRefresh();
    return true;
  }

  function leaveThreeUpMode(options) {
    const currentApp = app();
    const currentViewer = viewer();
    if (!currentApp || !currentViewer) {
      return false;
    }
    const rawOptions = options && typeof options === "object" ? options : {};
    const restoreBaseline = rawOptions.restoreBaseline !== false;
    const preferViewportCenterPage = rawOptions.preferViewportCenterPage === true;
    const container = viewerContainer();
    const trackedCenterPage = Number.parseInt(state.threeUpCenterPage, 10);
    const liveCenterPage = pageNearestViewportCenter(currentApp, currentViewer, container);
    const centeredExitPage = preferViewportCenterPage
      ? (
        Number.isFinite(liveCenterPage) && liveCenterPage > 0
          ? liveCenterPage
          : (Number.isFinite(trackedCenterPage) && trackedCenterPage > 0 ? trackedCenterPage : null)
      )
      : null;
    if (Number.isFinite(Number(rawOptions.onePageScale)) && Number(rawOptions.onePageScale) > 0) {
      state.threeUpOnePageScale = clampZoomScale(Number(rawOptions.onePageScale));
      state.threeUpBaselineScaleValue = String(state.threeUpOnePageScale);
    }
    if (!state.threeUpActive) {
      return true;
    }
    const restoreState = threeUpViewState();
    state.threeUpActive = false;
    currentViewer.spreadMode = Number.isInteger(state.threeUpNormalSpreadMode)
      ? state.threeUpNormalSpreadMode
      : SpreadMode.NONE;
    currentViewer.scrollMode = Number.isInteger(state.threeUpNormalScrollMode)
      ? state.threeUpNormalScrollMode
      : ScrollMode.VERTICAL;
    state.threeUpBaselineViewState = null;
    state.threeUpCenterPage = 1;
    state.pendingRestoreViewState = null;
    state.pendingRestoreUntil = 0;
    if (restoreBaseline) {
      if (Number.isFinite(centeredExitPage) && centeredExitPage > 0) {
        const onePageScale = clampZoomScale(state.threeUpOnePageScale);
        const centeredState = {
          page: centeredExitPage,
          pagesCount: Number(currentViewer.pagesCount || currentApp.pagesCount || 0),
          scale: String(onePageScale),
        };
        const restored = applyViewState(centeredState);
        if (restored) {
          scheduleRefresh();
        }
        return restored;
      }
      const restored = applyViewState(restoreState);
      if (restored) {
        scheduleRefresh();
      }
      return restored;
    }
    currentViewer.currentScale = clampZoomScale(state.threeUpOnePageScale);
    state.lastViewState = captureViewState();
    scheduleRefresh();
    return true;
  }

  function toggleThreeUpMode() {
    if (state.threeUpActive) {
      leaveThreeUpMode({ restoreBaseline: true, preferViewportCenterPage: true });
    } else {
      enterThreeUpMode();
    }
    const onePageScale = clampZoomScale(state.threeUpOnePageScale);
    return {
      active: Boolean(state.threeUpActive),
      threeUpActive: Boolean(state.threeUpActive),
      onePageScale,
      percent: Math.round(onePageScale * 100),
    };
  }

  function isThreeUpActive() {
    return Boolean(state.threeUpActive);
  }

  function setOnePageZoom100() {
    const resultPayload = {
      active: false,
      threeUpActive: false,
      onePageScale: 1,
      percent: 100,
      ok: true,
    };
    if (state.threeUpActive) {
      resultPayload.ok = leaveThreeUpMode({ restoreBaseline: true, onePageScale: 1.0 });
      return resultPayload;
    }
    const currentViewer = viewer();
    if (!currentViewer) {
      resultPayload.ok = false;
      return resultPayload;
    }
    currentViewer.currentScale = 1.0;
    state.threeUpOnePageScale = 1.0;
    state.threeUpBaselineScaleValue = "1";
    state.lastViewState = captureViewState();
    scheduleRefresh();
    return resultPayload;
  }

  function getViewState() {
    if (
      !state.threeUpActive
      &&
      state.pendingRestoreViewState
      && Date.now() <= Number(state.pendingRestoreUntil || 0)
    ) {
      const pending = cloneStateObject(state.pendingRestoreViewState);
      state.lastViewState = cloneStateObject(pending);
      return pending;
    }
    state.pendingRestoreViewState = null;
    state.pendingRestoreUntil = 0;
    const currentState = capturePersistedViewState();
    if (currentState && typeof currentState === "object" && Object.keys(currentState).length > 0) {
      state.lastViewState = cloneStateObject(currentState);
      return cloneStateObject(currentState);
    }
    return state.lastViewState && typeof state.lastViewState === "object"
      ? Object.assign({}, state.lastViewState)
      : {};
  }

  function restoreViewState(stateValue) {
    if (state.threeUpActive) {
      leaveThreeUpMode({ restoreBaseline: false });
    }
    return applyViewState(stateValue);
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

  function getZoomState() {
    const currentViewer = viewer();
    if (!currentViewer) {
      return {};
    }
    if (state.threeUpActive) {
      const onePageScale = clampZoomScale(state.threeUpOnePageScale);
      return {
        currentScale: onePageScale,
        currentScaleValue: String(onePageScale),
        percent: Math.round(onePageScale * 100),
        threeUpActive: true,
      };
    }
    const currentScale = Number(currentViewer.currentScale || 1);
    const currentScaleValue = String(currentViewer.currentScaleValue || "page-width");
    return {
      currentScale,
      currentScaleValue,
      percent: Math.round(currentScale * 100),
      threeUpActive: false,
    };
  }

  function setZoomScale(scaleValue) {
    const currentViewer = viewer();
    if (!currentViewer) {
      return false;
    }
    const nextScale = Number(scaleValue);
    if (!Number.isFinite(nextScale) || nextScale <= 0) {
      return false;
    }
    if (state.threeUpActive) {
      state.threeUpOnePageScale = clampZoomScale(nextScale);
      state.threeUpBaselineScaleValue = String(state.threeUpOnePageScale);
      state.threeUpBaselineViewState = threeUpViewState();
      state.threeUpBaselineViewState.scale = String(state.threeUpOnePageScale);
      applyThreeUpLayout(currentViewer);
      state.lastViewState = threeUpViewState();
      scheduleRefresh();
      return true;
    }
    currentViewer.currentScale = nextScale;
    state.lastViewState = captureViewState();
    scheduleRefresh();
    return true;
  }

  function resetZoom() {
    const currentViewer = viewer();
    if (!currentViewer) {
      return false;
    }
    if (state.threeUpActive) {
      leaveThreeUpMode({ restoreBaseline: false });
    }
    currentViewer.currentScaleValue = "page-width";
    state.threeUpOnePageScale = onePageScaleFromViewer(currentViewer);
    state.threeUpBaselineScaleValue = "page-width";
    state.lastViewState = captureViewState();
    scheduleRefresh();
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
      threeUpActive: Boolean(state.threeUpActive),
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
          threeUpActive: Boolean(state.threeUpActive),
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
    window.setTimeout(scheduleRefresh, 60);
    window.setTimeout(scheduleRefresh, 180);
    window.setTimeout(scheduleRefresh, 420);
    window.setTimeout(scheduleRefresh, 900);
    return true;
  }

  function setSearchTerms(terms) {
    state.searchTerms = Array.isArray(terms) ? terms.slice() : [];
    scheduleSearchIndicatorBuild(state.searchTerms);
    scheduleRefresh();
    return true;
  }

  function clearSearchTerms() {
    state.searchTerms = [];
    clearSearchIndicatorBuildState();
    clearSearchIndicators();
    scheduleRefresh();
    return true;
  }

  window.__pdfexploreBridge = {
    install,
    isReady,
    isThreeUpActive,
    toggleThreeUpMode,
    setOnePageZoom100,
    getViewState,
    restoreViewState,
    goToTop,
    getZoomState,
    setZoomScale,
    resetZoom,
    getSelectionInfo,
    setPersistentHighlights,
    setSearchTerms,
    clearSearchTerms,
  };
  return window.__pdfexploreBridge;
})();
