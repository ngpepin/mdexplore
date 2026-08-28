(() => {
  if (window.__mdexplorePageLayout) {
    return true;
  }

  const PAGE_WIDTH = 816;
  const PAGE_HEIGHT = 1056;
  const PAGE_PADDING_X = 48;
  const PAGE_PADDING_Y = 48;
  const CONTENT_HEIGHT = PAGE_HEIGHT - (PAGE_PADDING_Y * 2);
  const VALID_UP_COUNTS = new Set([2, 3, 6]);
  const STYLE_ID = "mdexplore-page-layout-style";
  const MAIN_CLASS = "mdexplore-page-layout";
  const PAGE_CLASS = "mdexplore-layout-page";
  const SURFACE_CLASS = "mdexplore-layout-page-surface";
  const CONTENT_CLASS = "mdexplore-layout-page-content";
  const EXIT_ANCHOR_CLASS = "mdexplore-layout-exit-anchor";

  const state = {
    activeCount: 0,
    repaginating: false,
    generation: 0,
    resizeObserver: null,
    delayedHandles: [],
    lastExitPageNumber: 0,
    lastLayoutAnchorPageNumber: 0,
    activityGeneration: 0,
    activityTrackingInstalled: false,
  };

  function mainElement() {
    return document.querySelector("main");
  }

  function installStyle() {
    if (document.getElementById(STYLE_ID)) {
      return;
    }
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      body.mdexplore-page-layout-active {
        background: color-mix(in srgb, var(--bg) 82%, #64748b 18%);
      }
      main.${MAIN_CLASS} {
        --mdexplore-up-count: 2;
        box-sizing: border-box;
        display: grid;
        grid-template-columns: repeat(var(--mdexplore-up-count), minmax(0, 1fr));
        align-items: start;
        gap: 12px;
        width: 100%;
        max-width: none;
        margin: 0;
        padding: 12px;
      }
      .${PAGE_CLASS} {
        box-sizing: border-box;
        position: relative;
        width: 100%;
        min-width: 0;
        overflow: hidden;
        background: var(--bg);
        border: 1px solid color-mix(in srgb, var(--border) 86%, transparent);
        border-radius: 3px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.24);
      }
      .${PAGE_CLASS}::after {
        content: attr(data-page-number);
        position: absolute;
        right: 5px;
        bottom: 3px;
        z-index: 2;
        color: color-mix(in srgb, var(--fg) 58%, transparent);
        font: 600 9px/1 "Noto Sans", "DejaVu Sans", sans-serif;
        pointer-events: none;
      }
      .${SURFACE_CLASS} {
        box-sizing: border-box;
        position: absolute;
        top: 0;
        left: 0;
        width: ${PAGE_WIDTH}px;
        height: ${PAGE_HEIGHT}px;
        padding: ${PAGE_PADDING_Y}px ${PAGE_PADDING_X}px;
        overflow: hidden;
        transform-origin: top left;
        background: var(--bg);
        color: var(--fg);
      }
      .${CONTENT_CLASS} {
        box-sizing: border-box;
        width: ${PAGE_WIDTH - (PAGE_PADDING_X * 2)}px;
        height: ${CONTENT_HEIGHT}px;
        overflow: hidden;
      }
      .${PAGE_CLASS}.mdexplore-layout-page-oversize .${CONTENT_CLASS} {
        overflow: auto;
      }
      @media print {
        body.mdexplore-page-layout-active {
          background: var(--bg);
        }
      }
    `;
    document.head.appendChild(style);
  }

  function installActivityTracking() {
    if (state.activityTrackingInstalled) {
      return;
    }
    const noteActivity = () => {
      state.activityGeneration += 1;
    };
    for (const eventName of ["wheel", "pointerdown", "touchstart", "keydown"]) {
      document.addEventListener(eventName, noteActivity, { capture: true, passive: true });
    }
    state.activityTrackingInstalled = true;
  }

  function isMeaningfulNode(node) {
    if (!node) {
      return false;
    }
    if (node.nodeType === Node.TEXT_NODE) {
      return Boolean(String(node.nodeValue || "").trim());
    }
    return (
      node.nodeType === Node.ELEMENT_NODE
      && !(node instanceof HTMLElement && node.classList.contains(EXIT_ANCHOR_CLASS))
    );
  }

  function pageElements(root) {
    if (!root) {
      return [];
    }
    return Array.from(root.children).filter(
      (node) => node instanceof HTMLElement && node.classList.contains(PAGE_CLASS)
    );
  }

  function pageContent(page) {
    return page ? page.querySelector(`.${CONTENT_CLASS}`) : null;
  }

  function viewportCentrePage(root) {
    const measuredPages = pageElements(root)
      .map((page) => ({ page, rect: page.getBoundingClientRect() }))
      .filter(({ rect }) => rect.width > 0 && rect.height > 0)
      .sort((left, right) => (
        Math.abs(left.rect.top - right.rect.top) < 3
          ? left.rect.left - right.rect.left
          : left.rect.top - right.rect.top
      ));
    if (!measuredPages.length) {
      return null;
    }

    const rows = [];
    for (const measured of measuredPages) {
      let row = rows[rows.length - 1];
      if (!row || Math.abs(row.top - measured.rect.top) >= 3) {
        row = {
          top: measured.rect.top,
          bottom: measured.rect.bottom,
          pages: [],
        };
        rows.push(row);
      }
      row.top = Math.min(row.top, measured.rect.top);
      row.bottom = Math.max(row.bottom, measured.rect.bottom);
      row.pages.push(measured);
    }

    const viewportCentreY = Math.max(
      1,
      Number(window.innerHeight || document.documentElement.clientHeight || 1)
    ) / 2;
    const verticalDistance = (row) => {
      if (viewportCentreY < row.top) {
        return row.top - viewportCentreY;
      }
      if (viewportCentreY > row.bottom) {
        return viewportCentreY - row.bottom;
      }
      return 0;
    };
    const selectedRow = rows.reduce((best, row) => (
      !best || verticalDistance(row) < verticalDistance(best) ? row : best
    ), null);
    if (!selectedRow || !selectedRow.pages.length) {
      return null;
    }

    selectedRow.pages.sort((left, right) => left.rect.left - right.rect.left);
    if (state.activeCount === 2) {
      // Facing pages use the page at the viewport's vertical centre-left.
      return selectedRow.pages[0].page;
    }

    const viewportCentreX = Math.max(
      1,
      Number(window.innerWidth || document.documentElement.clientWidth || 1)
    ) / 2;
    return selectedRow.pages.reduce((best, measured) => {
      if (!best) {
        return measured;
      }
      const centre = (measured.rect.left + measured.rect.right) / 2;
      const bestCentre = (best.rect.left + best.rect.right) / 2;
      const distance = Math.abs(centre - viewportCentreX);
      const bestDistance = Math.abs(bestCentre - viewportCentreX);
      if (distance < bestDistance - 0.5) {
        return measured;
      }
      if (
        Math.abs(distance - bestDistance) <= 0.5
        && measured.rect.left < best.rect.left
      ) {
        return measured;
      }
      return best;
    }, null).page;
  }

  function createExitAnchor(page) {
    const content = pageContent(page);
    if (!content) {
      return null;
    }
    const marker = document.createElement("span");
    marker.className = EXIT_ANCHOR_CLASS;
    marker.setAttribute("aria-hidden", "true");
    marker.style.cssText = (
      "display:block;width:0;height:0;margin:0;padding:0;border:0;overflow:hidden;"
    );
    content.insertBefore(marker, content.firstChild);
    return marker;
  }

  function restoreExitAnchor(marker, generation) {
    if (!(marker instanceof HTMLElement)) {
      return;
    }
    const position = () => {
      if (
        !marker.isConnected
        || generation !== state.generation
        || state.activeCount !== 0
      ) {
        marker.remove();
        return false;
      }
      marker.scrollIntoView({ block: "start", inline: "nearest", behavior: "instant" });
      return true;
    };
    window.requestAnimationFrame(() => {
      if (!position()) {
        return;
      }
      window.requestAnimationFrame(() => {
        if (!position()) {
          return;
        }
        // A final pass follows any host-side QWebEngine zoom restoration.
        window.setTimeout(() => {
          position();
          marker.remove();
        }, 80);
      });
    });
  }

  function pageSourceAnchor(page) {
    const content = pageContent(page);
    if (!content) {
      return null;
    }
    return Array.from(content.childNodes).find(
      (node) => node instanceof Element && isMeaningfulNode(node)
    ) || content.querySelector("*");
  }

  function continuousViewportAnchor(root) {
    if (!root) {
      return null;
    }
    const viewportWidth = Math.max(
      1,
      Number(window.innerWidth || document.documentElement.clientWidth || 1)
    );
    const viewportHeight = Math.max(
      1,
      Number(window.innerHeight || document.documentElement.clientHeight || 1)
    );
    let candidate = document.elementFromPoint(viewportWidth / 2, viewportHeight / 2);
    if (candidate && root.contains(candidate)) {
      while (candidate.parentElement && candidate.parentElement !== root) {
        candidate = candidate.parentElement;
      }
      if (
        candidate.parentElement === root
        && !candidate.classList.contains(EXIT_ANCHOR_CLASS)
      ) {
        return candidate;
      }
    }

    const centreX = viewportWidth / 2;
    const centreY = viewportHeight / 2;
    const measured = Array.from(root.children)
      .filter((node) => (
        node instanceof Element && !node.classList.contains(EXIT_ANCHOR_CLASS)
      ))
      .map((node) => ({ node, rect: node.getBoundingClientRect() }))
      .filter(({ rect }) => rect.width > 0 || rect.height > 0);
    const distance = ({ rect }) => {
      const dx = centreX < rect.left
        ? rect.left - centreX
        : centreX > rect.right
          ? centreX - rect.right
          : 0;
      const dy = centreY < rect.top
        ? rect.top - centreY
        : centreY > rect.bottom
          ? centreY - rect.bottom
          : 0;
      return Math.hypot(dx, dy);
    };
    return measured.reduce((best, item) => (
      !best || distance(item) < distance(best) ? item : best
    ), null)?.node || null;
  }

  function restoreLayoutAnchor(
    anchor,
    generation,
    activityGeneration = state.activityGeneration
  ) {
    if (!(anchor instanceof Element)) {
      return;
    }
    let positionedScrollY = null;
    const position = () => {
      if (
        !anchor.isConnected
        || generation !== state.generation
        || !VALID_UP_COUNTS.has(state.activeCount)
        || activityGeneration !== state.activityGeneration
      ) {
        return false;
      }
      if (
        positionedScrollY !== null
        && Math.abs(Number(window.scrollY || 0) - positionedScrollY) > 2
      ) {
        // Do not let a settling callback undo scrolling performed immediately
        // after the user changes layouts.
        return false;
      }
      const page = anchor.closest(`.${PAGE_CLASS}`);
      if (!(page instanceof HTMLElement)) {
        return false;
      }
      state.lastLayoutAnchorPageNumber = (
        Number.parseInt(page.dataset.pageNumber || "0", 10) || 0
      );
      const rect = page.getBoundingClientRect();
      const viewportHeight = Math.max(
        1,
        Number(window.innerHeight || document.documentElement.clientHeight || 1)
      );
      const targetScrollY = (
        Number(window.scrollY || 0)
        + rect.top
        + (rect.height / 2)
        - (viewportHeight / 2)
      );
      window.scrollTo({
        top: Math.max(0, targetScrollY),
        left: Number(window.scrollX || 0),
        behavior: "instant",
      });
      positionedScrollY = Number(window.scrollY || 0);
      return true;
    };
    window.requestAnimationFrame(() => {
      if (!position()) {
        return;
      }
      window.requestAnimationFrame(() => {
        position();
      });
    });
    // Chromium may not expose the final grid scroll extent until after the
    // first ResizeObserver/layout cycle. This pass is activity-cancelled so it
    // cannot undo scrolling performed after the shortcut.
    window.setTimeout(position, 160);
  }

  function sizePage(page) {
    if (!(page instanceof HTMLElement)) {
      return;
    }
    const width = Math.max(1, Number(page.clientWidth || 0));
    const scale = width / PAGE_WIDTH;
    page.style.height = `${PAGE_HEIGHT * scale}px`;
    const surface = page.querySelector(`.${SURFACE_CLASS}`);
    if (surface instanceof HTMLElement) {
      surface.style.transform = `scale(${scale})`;
    }
  }

  function sizeAllPages() {
    const root = mainElement();
    for (const page of pageElements(root)) {
      sizePage(page);
    }
  }

  function createPage(root) {
    const page = document.createElement("section");
    page.className = PAGE_CLASS;
    page.dataset.pageNumber = String(pageElements(root).length + 1);
    const surface = document.createElement("div");
    surface.className = SURFACE_CLASS;
    const content = document.createElement("div");
    content.className = CONTENT_CLASS;
    surface.appendChild(content);
    page.appendChild(surface);
    root.appendChild(page);
    return page;
  }

  function extractSourceNodes(root) {
    const fragment = document.createDocumentFragment();
    const pages = pageElements(root);
    if (pages.length) {
      for (const page of pages) {
        const content = pageContent(page);
        if (!content) {
          continue;
        }
        while (content.firstChild) {
          fragment.appendChild(content.firstChild);
        }
      }
      for (const page of pages) {
        page.remove();
      }
      return Array.from(fragment.childNodes);
    }
    while (root.firstChild) {
      fragment.appendChild(root.firstChild);
    }
    return Array.from(fragment.childNodes);
  }

  function contentOverflows(content) {
    return Boolean(
      content
      && Number(content.scrollHeight || 0) > Number(content.clientHeight || CONTENT_HEIGHT) + 1
    );
  }

  function paginate() {
    const root = mainElement();
    if (!root || !VALID_UP_COUNTS.has(state.activeCount) || state.repaginating) {
      return false;
    }
    state.repaginating = true;
    try {
      const sourceNodes = extractSourceNodes(root);
      let page = createPage(root);
      let content = pageContent(page);
      let meaningfulCount = 0;
      let forceNewPage = false;

      for (const node of sourceNodes) {
        const meaningful = isMeaningfulNode(node);
        if (forceNewPage && meaningful) {
          page = createPage(root);
          content = pageContent(page);
          meaningfulCount = 0;
          forceNewPage = false;
        }
        if (!content) {
          continue;
        }
        content.appendChild(node);
        if (meaningful) {
          meaningfulCount += 1;
        }
        if (!contentOverflows(content)) {
          continue;
        }
        if (meaningful && meaningfulCount > 1) {
          content.removeChild(node);
          page = createPage(root);
          content = pageContent(page);
          meaningfulCount = 1;
          if (content) {
            content.appendChild(node);
          }
        }
        if (contentOverflows(content)) {
          page.classList.add("mdexplore-layout-page-oversize");
          forceNewPage = true;
        }
      }

      const pages = pageElements(root);
      if (pages.length > 1) {
        const lastContent = pageContent(pages[pages.length - 1]);
        if (lastContent && !Array.from(lastContent.childNodes).some(isMeaningfulNode)) {
          pages[pages.length - 1].remove();
        }
      }
      pageElements(root).forEach((candidate, index) => {
        candidate.dataset.pageNumber = String(index + 1);
      });
      sizeAllPages();
      return true;
    } finally {
      state.repaginating = false;
    }
  }

  function cancelDelayedReflows() {
    for (const handle of state.delayedHandles) {
      window.clearTimeout(handle);
    }
    state.delayedHandles = [];
  }

  function scheduleDelayedReflows() {
    cancelDelayedReflows();
    const generation = state.generation;
    for (const delay of [120, 450, 1100, 2400]) {
      state.delayedHandles.push(window.setTimeout(() => {
        if (generation !== state.generation || !VALID_UP_COUNTS.has(state.activeCount)) {
          return;
        }
        const root = mainElement();
        const sourceAnchor = pageSourceAnchor(viewportCentrePage(root));
        paginate();
        sizeAllPages();
        restoreLayoutAnchor(sourceAnchor, generation);
      }, delay));
    }
  }

  function ensureResizeObserver() {
    if (state.resizeObserver || typeof ResizeObserver !== "function") {
      return;
    }
    state.resizeObserver = new ResizeObserver(() => sizeAllPages());
    const root = mainElement();
    if (root) {
      state.resizeObserver.observe(root);
    }
  }

  function setMode(rawCount) {
    const count = Number.parseInt(rawCount, 10);
    if (!VALID_UP_COUNTS.has(count)) {
      return clearMode();
    }
    installStyle();
    installActivityTracking();
    const root = mainElement();
    if (!root) {
      return { active: false, count: 0, pageCount: 0 };
    }
    const sourceAnchor = VALID_UP_COUNTS.has(state.activeCount)
      ? pageSourceAnchor(viewportCentrePage(root))
      : continuousViewportAnchor(root);
    state.activeCount = count;
    state.generation += 1;
    const generation = state.generation;
    document.body.classList.add("mdexplore-page-layout-active");
    root.classList.add(MAIN_CLASS);
    root.style.setProperty("--mdexplore-up-count", String(count));
    ensureResizeObserver();
    paginate();
    sizeAllPages();
    state.lastLayoutAnchorPageNumber = 0;
    restoreLayoutAnchor(sourceAnchor, generation);
    scheduleDelayedReflows();
    return {
      active: true,
      count,
      pageCount: pageElements(root).length,
      lastLayoutAnchorPageNumber: state.lastLayoutAnchorPageNumber,
    };
  }

  function clearMode(preserveViewportPage = true) {
    const root = mainElement();
    state.generation += 1;
    const generation = state.generation;
    cancelDelayedReflows();
    const centredPage = (
      root && preserveViewportPage && VALID_UP_COUNTS.has(state.activeCount)
        ? viewportCentrePage(root)
        : null
    );
    state.lastExitPageNumber = centredPage
      ? Number.parseInt(centredPage.dataset.pageNumber || "0", 10) || 0
      : 0;
    const exitAnchor = centredPage ? createExitAnchor(centredPage) : null;
    if (root) {
      const sourceNodes = extractSourceNodes(root);
      root.replaceChildren(...sourceNodes);
      root.classList.remove(MAIN_CLASS);
      root.style.removeProperty("--mdexplore-up-count");
    }
    document.body.classList.remove("mdexplore-page-layout-active");
    state.activeCount = 0;
    restoreExitAnchor(exitAnchor, generation);
    return {
      active: false,
      count: 0,
      pageCount: 0,
      lastExitPageNumber: state.lastExitPageNumber,
    };
  }

  function toggleMode(rawCount) {
    const count = Number.parseInt(rawCount, 10);
    if (state.activeCount === count) {
      return clearMode();
    }
    return setMode(count);
  }

  function getState() {
    const root = mainElement();
    return {
      active: VALID_UP_COUNTS.has(state.activeCount),
      count: state.activeCount,
      pageCount: pageElements(root).length,
      lastExitPageNumber: state.lastExitPageNumber,
      lastLayoutAnchorPageNumber: state.lastLayoutAnchorPageNumber,
    };
  }

  window.__mdexplorePageLayout = {
    setMode,
    clearMode,
    toggleMode,
    reflow: paginate,
    getState,
  };
  return true;
})()
