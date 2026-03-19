(() => {
  const decodeSvgDataUriForPdf = (src) => {
    const raw = String(src || "");
    if (!raw.startsWith("data:image/svg+xml")) {
      return "";
    }
    const commaIndex = raw.indexOf(",");
    if (commaIndex < 0) {
      return "";
    }
    const meta = raw.slice(0, commaIndex).toLowerCase();
    const payload = raw.slice(commaIndex + 1);
    try {
      if (meta.includes(";base64")) {
        return atob(payload);
      }
      return decodeURIComponent(payload);
    } catch (_error) {
      return "";
    }
  };

  const intrinsicSvgSizeFromNodeForPdf = (svg) => {
    if (!(svg instanceof SVGElement)) {
      return { width: 0, height: 0 };
    }
    const viewBox = String(svg.getAttribute("viewBox") || "").trim();
    if (viewBox) {
      const parts = viewBox.split(/[\\s,]+/).map((part) => Number.parseFloat(part));
      if (parts.length === 4 && parts.every((num) => Number.isFinite(num))) {
        const width = Math.abs(parts[2]);
        const height = Math.abs(parts[3]);
        if (width > 0 && height > 0) {
          return { width, height };
        }
      }
    }
    const width = Number.parseFloat(String(svg.getAttribute("width") || "").replace(/px$/i, ""));
    const height = Number.parseFloat(String(svg.getAttribute("height") || "").replace(/px$/i, ""));
    if (Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0) {
      return { width, height };
    }
    return { width: 0, height: 0 };
  };

  const normalizePlantUmlInlineSvgForPdf = (svg) => {
    if (!(svg instanceof SVGElement)) {
      return;
    }
    svg.classList.add("plantuml", "mdexplore-plantuml-inline");
    svg.style.removeProperty("transform");
    svg.style.setProperty("display", "block", "important");
    svg.style.setProperty("width", "var(--mdexplore-print-diagram-width, auto)", "important");
    svg.style.setProperty("max-width", "var(--mdexplore-print-diagram-max-width, 100%)", "important");
    svg.style.setProperty("height", "var(--mdexplore-print-diagram-height, auto)", "important");
    svg.style.setProperty("max-height", "var(--mdexplore-print-diagram-max-height, none)", "important");
    svg.style.setProperty("margin", "0 auto", "important");
  };

  const replacePlantUmlImageWithInlineSvgForPdf = (img) => {
    if (!(img instanceof HTMLImageElement)) {
      return null;
    }
    if (img.dataset.mdexplorePlantumlInline === "1") {
      return null;
    }
    const markup = decodeSvgDataUriForPdf(img.currentSrc || img.src || "");
    if (!markup) {
      return null;
    }
    try {
      const template = document.createElement("template");
      template.innerHTML = markup.trim();
      const svg = template.content.firstElementChild;
      if (!(svg instanceof SVGElement) || String(svg.tagName || "").toLowerCase() !== "svg") {
        return null;
      }
      const replacement = svg;
      replacement.setAttribute("data-mdexplore-original-img", img.outerHTML);
      replacement.setAttribute("data-mdexplorePlantumlInline", "1");
      replacement.dataset.mdexplorePlantumlInline = "1";
      normalizePlantUmlInlineSvgForPdf(replacement);
      img.replaceWith(replacement);
      return replacement;
    } catch (_error) {
      return null;
    }
  };

  if (document.documentElement) {
    document.documentElement.classList.add("mdexplore-pdf-export-mode");
  }
  document.body.classList.add("mdexplore-pdf-export-mode");
  for (const fence of Array.from(document.querySelectorAll(".mdexplore-fence"))) {
    if (!(fence instanceof HTMLElement)) {
      continue;
    }
    const sectionWidth = String(fence.style.getPropertyValue("--mdexplore-print-section-width") || "").trim();
    if (sectionWidth) {
      fence.dataset.mdexplorePdfSectionWidth = sectionWidth;
      fence.style.setProperty("width", sectionWidth, "important");
      fence.style.setProperty("max-width", sectionWidth, "important");
      fence.style.setProperty("overflow", "visible", "important");
      for (const child of Array.from(fence.children)) {
        if (!(child instanceof HTMLElement)) {
          continue;
        }
        const tag = String(child.tagName || "").toLowerCase();
        if (!/^h[1-6]$/.test(tag)) {
          continue;
        }
        child.dataset.mdexplorePdfHeadingWidth = sectionWidth;
        child.style.setProperty("display", "block", "important");
        child.style.setProperty("width", sectionWidth, "important");
        child.style.setProperty("max-width", sectionWidth, "important");
        child.style.setProperty("box-sizing", "border-box", "important");
        child.style.setProperty("white-space", "normal", "important");
      }
    }
  }
  for (const shell of Array.from(document.querySelectorAll(".mdexplore-mermaid-shell"))) {
    if (!(shell instanceof HTMLElement)) {
      continue;
    }
    const host = shell.parentElement;
    if (!(host instanceof HTMLElement)) {
      continue;
    }
    host.style.setProperty("display", "block", "important");
    host.style.setProperty("width", "100%", "important");
    host.style.setProperty("max-width", "100%", "important");
    const viewport = shell.querySelector(".mdexplore-mermaid-viewport");
    const svg = viewport instanceof HTMLElement ? viewport.querySelector("svg") : shell.querySelector("svg");
    const plantImg =
      viewport instanceof HTMLElement ? viewport.querySelector("img.plantuml") : shell.querySelector("img.plantuml");
    const plantSvg =
      viewport instanceof HTMLElement
        ? viewport.querySelector("svg.mdexplore-plantuml-inline")
        : shell.querySelector("svg.mdexplore-plantuml-inline");
    if (svg instanceof SVGElement) {
      svg.style.removeProperty("transform");
      svg.removeAttribute("width");
      svg.removeAttribute("height");
      svg.style.removeProperty("width");
      svg.style.setProperty("width", "var(--mdexplore-print-diagram-width, auto)", "important");
      svg.style.setProperty("max-width", "var(--mdexplore-print-diagram-max-width, 100%)", "important");
      svg.style.setProperty("height", "var(--mdexplore-print-diagram-height, auto)", "important");
      svg.style.setProperty("max-height", "var(--mdexplore-print-diagram-max-height, none)", "important");
      host.innerHTML = "";
      host.appendChild(svg);
      continue;
    }
    const normalizedPlant =
      plantSvg instanceof SVGElement ? plantSvg : replacePlantUmlImageWithInlineSvgForPdf(plantImg);
    if (normalizedPlant instanceof SVGElement) {
      normalizePlantUmlInlineSvgForPdf(normalizedPlant);
      host.innerHTML = "";
      host.appendChild(normalizedPlant);
    }
  }
  for (const toolbar of Array.from(document.querySelectorAll(".mdexplore-mermaid-toolbar"))) {
    if (!(toolbar instanceof HTMLElement)) {
      continue;
    }
    toolbar.dataset.mdexplorePdfHidden = "1";
    toolbar.style.setProperty("display", "none", "important");
  }
  for (const viewport of Array.from(document.querySelectorAll(".mdexplore-mermaid-viewport"))) {
    if (!(viewport instanceof HTMLElement)) {
      continue;
    }
    viewport.dataset.mdexplorePdfViewportHidden = "1";
    viewport.scrollLeft = 0;
    viewport.scrollTop = 0;
    viewport.style.setProperty("overflow", "hidden", "important");
    viewport.style.setProperty("scrollbar-width", "none", "important");
    viewport.style.setProperty("-ms-overflow-style", "none", "important");
  }
  for (const img of Array.from(document.querySelectorAll("img.plantuml"))) {
    if (!(img instanceof HTMLImageElement)) {
      continue;
    }
    replacePlantUmlImageWithInlineSvgForPdf(img);
  }
  for (const svg of Array.from(document.querySelectorAll("svg.mdexplore-plantuml-inline"))) {
    normalizePlantUmlInlineSvgForPdf(svg);
  }
  for (const svg of Array.from(document.querySelectorAll(".mermaid svg"))) {
    if (!(svg instanceof SVGElement)) {
      continue;
    }
    svg.style.removeProperty("transform");
    svg.removeAttribute("width");
    svg.removeAttribute("height");
    svg.style.removeProperty("width");
    svg.style.removeProperty("max-width");
    svg.style.removeProperty("height");
    svg.style.setProperty("display", "block", "important");
    svg.style.setProperty("width", "var(--mdexplore-print-diagram-width, auto)", "important");
    svg.style.setProperty("max-width", "var(--mdexplore-print-diagram-max-width, 100%)", "important");
    svg.style.setProperty("height", "var(--mdexplore-print-diagram-height, auto)", "important");
    svg.style.setProperty("max-height", "var(--mdexplore-print-diagram-max-height, none)", "important");
  }
  return true;
})();
