(() => {
  if (document.documentElement) {
    document.documentElement.classList.remove("mdexplore-pdf-export-mode");
  }
  document.body.classList.remove("mdexplore-pdf-export-mode");
  for (const token of Array.from(document.querySelectorAll(".mdexplore-pdf-landscape-token"))) {
    if (token instanceof HTMLElement) {
      token.remove();
    }
  }
  const pdfMermaidOverride = document.getElementById("__mdexplore_pdf_mermaid_light_override");
  if (pdfMermaidOverride && pdfMermaidOverride.parentNode) {
    pdfMermaidOverride.parentNode.removeChild(pdfMermaidOverride);
  }
  for (const toolbar of Array.from(document.querySelectorAll(".mdexplore-mermaid-toolbar[data-mdexplore-pdf-hidden='1']"))) {
    if (!(toolbar instanceof HTMLElement)) {
      continue;
    }
    toolbar.style.removeProperty("display");
    delete toolbar.dataset.mdexplorePdfHidden;
  }
  for (const viewport of Array.from(document.querySelectorAll(".mdexplore-mermaid-viewport[data-mdexplore-pdf-viewport-hidden='1']"))) {
    if (!(viewport instanceof HTMLElement)) {
      continue;
    }
    viewport.style.removeProperty("overflow");
    viewport.style.removeProperty("scrollbar-width");
    viewport.style.removeProperty("-ms-overflow-style");
    delete viewport.dataset.mdexplorePdfViewportHidden;
  }
  for (const svg of Array.from(document.querySelectorAll("svg.mdexplore-plantuml-inline[data-mdexplore-original-img]"))) {
    if (!(svg instanceof SVGElement)) {
      continue;
    }
    const original = String(svg.getAttribute("data-mdexplore-original-img") || "");
    if (!original) {
      continue;
    }
    try {
      const container = document.createElement("div");
      container.innerHTML = original;
      const replacement = container.firstElementChild;
      if (replacement instanceof HTMLImageElement) {
        svg.replaceWith(replacement);
      }
    } catch (_error) {
      // Ignore restore failures; a later preview rerender will recover.
    }
  }
  for (const fence of Array.from(document.querySelectorAll(".mdexplore-fence"))) {
    if (!(fence instanceof HTMLElement)) {
      continue;
    }
    fence.style.removeProperty("min-height");
    fence.style.removeProperty("width");
    fence.style.removeProperty("max-width");
    fence.style.removeProperty("overflow");
    delete fence.dataset.mdexplorePdfSectionWidth;
    for (const child of Array.from(fence.children)) {
      if (!(child instanceof HTMLElement)) {
        continue;
      }
      delete child.dataset.mdexplorePdfHeadingWidth;
      child.style.removeProperty("width");
      child.style.removeProperty("max-width");
      child.style.removeProperty("box-sizing");
      child.style.removeProperty("white-space");
    }
  }
  for (const block of Array.from(document.querySelectorAll(".mdexplore-print-block[data-mdexplore-print-block='1']"))) {
    if (!(block instanceof HTMLElement)) {
      continue;
    }
    const parent = block.parentNode;
    if (!parent) {
      continue;
    }
    while (block.firstChild) {
      parent.insertBefore(block.firstChild, block);
    }
    parent.removeChild(block);
  }
  const reapplyAll = () => {
    for (const shell of Array.from(document.querySelectorAll(".mdexplore-mermaid-shell"))) {
      const fn = shell && shell.__mdexploreReapplySavedState;
      if (typeof fn !== "function") {
        continue;
      }
      try {
        fn();
      } catch (_error) {
        // Ignore per-shell restore failures.
      }
    }
  };
  if (window.__mdexploreRunClientRenderers) {
    const maybePromise = window.__mdexploreRunClientRenderers({ mermaidMode: "auto", forceMermaid: true });
    Promise.resolve(maybePromise).then(() => reapplyAll()).catch(() => reapplyAll());
    return true;
  }
  if (window.__mdexploreRunMermaidWithMode) {
    const maybePromise = window.__mdexploreRunMermaidWithMode("auto", false);
    Promise.resolve(maybePromise).then(() => reapplyAll()).catch(() => reapplyAll());
    return true;
  }
  reapplyAll();
  return false;
})();
